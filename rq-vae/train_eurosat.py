"""
Standalone RQ-VAE training script for EuroSAT.
Works on CPU and MPS (Apple Silicon) — no CUDA/DDP required.

Usage:
    cd rq-vae
    python train_eurosat.py

Outputs:
    - Checkpoints saved to output/eurosat/
    - RQ codes saved to code/codes.txt (used by NAC)
"""
import os
import sys
import math
import logging
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add rq-vae to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rqvae.models.rqvae import RQVAE, get_rqvae
from rqvae.img_datasets.eurosat import EuroSAT
from rqvae.img_datasets.transforms import create_transforms
from rqvae.losses.vqgan.lpips import LPIPS
from rqvae.losses.vqgan.discriminator import NLayerDiscriminator, weights_init
from rqvae.losses.vqgan.gan_loss import hinge_d_loss, vanilla_g_loss

from omegaconf import OmegaConf
import yaml


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return OmegaConf.create(config)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def calculate_adaptive_weight(nll_loss, g_loss, last_layer):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'train.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train RQ-VAE on EuroSAT')
    parser.add_argument('-m', '--config', type=str,
                        default='configs/eurosat/stage1/eurosat-rqvae-8x8x4.yaml',
                        help='Path to config YAML')
    parser.add_argument('-o', '--output', type=str, default='output/eurosat',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Overrides
    if args.epochs is not None:
        config.experiment.epochs = args.epochs
    if args.batch_size is not None:
        config.experiment.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.init_lr = args.lr

    device = get_device()
    output_dir = args.output
    logger = setup_logging(output_dir)

    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")

    # Save config copy
    OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # ─── Dataset ───
    dataset_cfg = OmegaConf.create({'transforms': config.dataset.transforms})
    transforms_trn = create_transforms(dataset_cfg, split='train', is_eval=False)
    transforms_val = create_transforms(dataset_cfg, split='val', is_eval=True)

    root = config.dataset.get('root', '../EuroSAT_RGB')
    split_path = config.dataset.get('split_indices_path', '../eurosat_split_indices.pt')

    dataset_trn = EuroSAT(root, split='train', transform=transforms_trn,
                           split_indices_path=split_path)
    dataset_val = EuroSAT(root, split='val', transform=transforms_val,
                           split_indices_path=split_path)

    bs = config.experiment.batch_size
    loader_trn = DataLoader(dataset_trn, batch_size=bs, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=bs, shuffle=False,
                            num_workers=4, pin_memory=True)

    logger.info(f"Train: {len(dataset_trn)} images, Val: {len(dataset_val)} images")
    logger.info(f"Batch size: {bs}, Steps/epoch: {len(loader_trn)}")

    # ─── Model ───
    hps = dict(config.arch.hparams)
    ddconfig = dict(config.arch.ddconfig)
    model = RQVAE(**hps, ddconfig=ddconfig,
                   checkpointing=config.arch.get('checkpointing', False))
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"RQ-VAE parameters: {n_params:.2f}M")

    # ─── Discriminator ───
    gan_config = config.gan
    discriminator = NLayerDiscriminator(
        input_nc=gan_config.disc.arch.in_channels,
        n_layers=gan_config.disc.arch.num_layers,
        use_actnorm=gan_config.disc.arch.use_actnorm,
        ndf=gan_config.disc.arch.ndf,
    ).apply(weights_init).to(device)

    # ─── Losses ───
    perceptual_loss = LPIPS().to(device).eval()
    perceptual_weight = gan_config.loss.perceptual_weight
    disc_weight_factor = gan_config.loss.disc_weight
    gan_start_epoch = gan_config.loss.disc_start

    # ─── Optimizers ───
    lr = config.optimizer.init_lr
    betas = tuple(config.optimizer.betas)
    optimizer_g = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    # ─── LR Scheduler ───
    total_steps = config.experiment.epochs * len(loader_trn)
    warmup_steps = config.optimizer.warmup.epoch * len(loader_trn)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step / max(warmup_steps, 1), 1e-2)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.5 * (1 + math.cos(math.pi * progress)), 1e-2)

    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda)

    # ─── Training Loop ───
    epochs = config.experiment.epochs
    best_val_loss = float('inf')

    # Clear code file
    os.makedirs('code', exist_ok=True)
    code_file = 'code/codes.txt'
    if os.path.exists(code_file):
        os.remove(code_file)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        discriminator.train()
        use_gan = (epoch >= gan_start_epoch)

        train_loss_total = 0
        train_loss_recon = 0
        train_loss_latent = 0

        pbar = tqdm(loader_trn, desc=f"Epoch {epoch+1}/{epochs} [train]",
                    leave=False, ncols=120)

        for it, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)

            # --- Generator step ---
            optimizer_g.zero_grad()
            recon, quant_loss, codes = model(imgs)
            loss_out = model.compute_loss(recon, quant_loss, codes, xs=imgs)
            loss_rec_lat = loss_out['loss_total']
            loss_recon = loss_out['loss_recon']
            loss_latent = loss_out['loss_latent']

            loss_pcpt = perceptual_loss(imgs, recon)

            if use_gan:
                logits_fake, _ = discriminator(recon.contiguous(), None)
                loss_gen = vanilla_g_loss(logits_fake)
                g_weight = calculate_adaptive_weight(
                    loss_recon + perceptual_weight * loss_pcpt,
                    loss_gen, last_layer=model.get_last_layer()
                )
            else:
                loss_gen = torch.zeros((), device=device)
                g_weight = torch.zeros((), device=device)

            loss_g_total = loss_rec_lat + perceptual_weight * loss_pcpt + \
                           g_weight * disc_weight_factor * loss_gen
            loss_g_total.backward()
            optimizer_g.step()
            scheduler_g.step()

            # --- Discriminator step ---
            if use_gan:
                optimizer_d.zero_grad()
                logits_fake, logits_real = discriminator(
                    recon.contiguous().detach(), imgs.contiguous().detach()
                )
                loss_disc = hinge_d_loss(logits_real, logits_fake)
                (disc_weight_factor * loss_disc).backward()
                optimizer_d.step()
                scheduler_d.step()

            train_loss_total += loss_g_total.item()
            train_loss_recon += loss_recon.item()
            train_loss_latent += loss_latent.item()

            pbar.set_postfix({
                'recon': f'{loss_recon.item():.4f}',
                'lat': f'{loss_latent.item():.4f}',
                'pcpt': f'{loss_pcpt.item():.4f}',
                'lr': f'{scheduler_g.get_last_lr()[0]:.2e}',
            })

        n_steps = len(loader_trn)
        logger.info(
            f"Epoch {epoch+1}/{epochs} [train] "
            f"loss_total: {train_loss_total/n_steps:.4f}, "
            f"loss_recon: {train_loss_recon/n_steps:.4f}, "
            f"loss_latent: {train_loss_latent/n_steps:.4f}, "
            f"lr: {scheduler_g.get_last_lr()[0]:.2e}"
        )

        # --- Validation ---
        if (epoch + 1) % config.experiment.get('test_freq', 1) == 0:
            model.eval()
            val_loss_total = 0
            val_loss_recon = 0

            with torch.no_grad():
                for imgs, _ in loader_val:
                    imgs = imgs.to(device)
                    recon, quant_loss, codes = model(imgs)
                    loss_out = model.compute_loss(recon, quant_loss, codes,
                                                   xs=imgs, valid=True)
                    val_loss_total += loss_out['loss_total'].item()
                    val_loss_recon += loss_out['loss_recon'].item()

            n_val = len(loader_val)
            avg_val_loss = val_loss_recon / max(n_val, 1)
            logger.info(
                f"Epoch {epoch+1}/{epochs} [val]   "
                f"loss_total: {val_loss_total/max(n_val,1):.4f}, "
                f"loss_recon: {avg_val_loss:.4f}"
            )

            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                ckpt_path = os.path.join(output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer_g.state_dict(),
                }, ckpt_path)
                logger.info(f"  -> New best model saved (val_recon={avg_val_loss:.4f})")

            # Save reconstructions
            if (epoch + 1) % 10 == 0 or epoch == 0:
                with torch.no_grad():
                    sample_imgs = imgs[:8]
                    sample_recon = model(sample_imgs)[0]
                    sample_imgs_vis = sample_imgs * 0.5 + 0.5
                    sample_recon_vis = torch.clamp(sample_recon * 0.5 + 0.5, 0, 1)
                    grid = torch.cat([sample_imgs_vis, sample_recon_vis], dim=0)
                    grid = torchvision.utils.make_grid(grid, nrow=8)
                    img_path = os.path.join(output_dir, f'recon_epoch{epoch+1:03d}.png')
                    torchvision.utils.save_image(grid, img_path)

        # --- Save checkpoint periodically ---
        if (epoch + 1) % config.experiment.get('save_ckpt_freq', 5) == 0:
            ckpt_path = os.path.join(output_dir, f'epoch{epoch+1}_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer_g.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, ckpt_path)
            logger.info(f"  Checkpoint saved: {ckpt_path}")

    # ─── Extract codes for NAC ───
    logger.info("=" * 60)
    logger.info("Extracting RQ codes for NAC...")
    model.eval()

    # Clear any codes written during training
    if os.path.exists(code_file):
        os.remove(code_file)

    code_shape = config.arch.hparams.code_shape
    depth = code_shape[-1]

    all_codes_loader = DataLoader(dataset_trn, batch_size=bs, shuffle=False,
                                  num_workers=4, pin_memory=True)
    n_images = 0
    with torch.no_grad():
        for imgs, _ in tqdm(all_codes_loader, desc="Extracting codes"):
            imgs = imgs.to(device)
            codes = model.get_codes(imgs)  # [B, h, w, D]
            codes_np = codes.cpu().numpy()
            with open(code_file, 'a') as f:
                for i in range(codes_np.shape[0]):
                    flat = codes_np[i].reshape(-1).astype(int)
                    f.write(' '.join(map(str, flat)) + '\n')
            n_images += codes_np.shape[0]

    # Also extract val codes
    with torch.no_grad():
        for imgs, _ in tqdm(loader_val, desc="Extracting val codes"):
            imgs = imgs.to(device)
            codes = model.get_codes(imgs)
            codes_np = codes.cpu().numpy()
            with open(code_file, 'a') as f:
                for i in range(codes_np.shape[0]):
                    flat = codes_np[i].reshape(-1).astype(int)
                    f.write(' '.join(map(str, flat)) + '\n')
            n_images += codes_np.shape[0]

    h, w = code_shape[0], code_shape[1]
    logger.info(f"Saved {n_images} code sequences to {code_file}")
    logger.info(f"Code shape per image: {h}x{w}x{depth} = {h*w*depth} indices")

    # Copy to nac/data/ for convenience
    nac_code_file = os.path.join('..', 'nac', 'data', f'codes{h}x{w}x{depth}.txt')
    os.makedirs(os.path.dirname(nac_code_file), exist_ok=True)
    import shutil
    shutil.copy2(code_file, nac_code_file)
    logger.info(f"Copied codes to {nac_code_file}")
    logger.info("Done! You can now run: cd ../nac && python nac_eurosat.py")


if __name__ == '__main__':
    main()
