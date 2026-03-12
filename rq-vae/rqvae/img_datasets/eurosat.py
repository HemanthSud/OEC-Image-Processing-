# rqvae/img_datasets/eurosat.py
"""
EuroSAT dataset loader for RQ-VAE training.
EuroSAT contains 64×64 Sentinel-2 satellite images across 10 land-use classes.
"""
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


class EuroSAT(Dataset):
    """
    EuroSAT RGB Dataset for image reconstruction tasks.

    Expected directory structure (ImageFolder-style):
        root/
        ├── AnnualCrop/
        ├── Forest/
        ├── HerbaceousVegetation/
        ├── Highway/
        ├── Industrial/
        ├── Pasture/
        ├── PermanentCrop/
        ├── Residential/
        ├── River/
        └── SeaLake/

    For train/val splitting, use the pre-computed split indices
    (eurosat_split_indices.pt) and wrap with torch.utils.data.Subset.
    """

    def __init__(self, root, split='train', transform=None, max_samples=None,
                 split_indices_path=None):
        """
        Args:
            root: Path to EuroSAT_RGB directory.
            split: 'train', 'val', or 'all'. If 'all', loads everything.
            transform: torchvision transforms to apply.
            max_samples: Optional cap on number of samples.
            split_indices_path: Path to eurosat_split_indices.pt for train/val split.
                                If None, loads all images (split='all' behaviour).
        """
        self.root = root
        self.split = split
        self.transform = transform

        # Collect all images recursively
        all_files = glob(os.path.join(root, '**', '*.*'), recursive=True)
        all_files = [f for f in all_files if os.path.splitext(f)[
            1].lower() in IMG_EXTS]
        all_files.sort()

        if len(all_files) == 0:
            raise RuntimeError(f'No images found in {root}')

        # Apply train/val split if indices are provided
        if split_indices_path is not None and split != 'all':
            import torch
            indices = torch.load(split_indices_path, weights_only=True)
            if split in ('train',):
                idx_list = indices['train']
            elif split in ('val', 'valid'):
                idx_list = indices['val']
            elif split in ('test',):
                idx_list = indices['test']
            else:
                raise ValueError(
                    f"Unknown split '{split}'. Use 'train', 'val', 'test', or 'all'.")
            # Handle both list and tensor types
            if hasattr(idx_list, 'tolist'):
                idx_list = idx_list.tolist()
            self.files = [all_files[i] for i in idx_list]
        else:
            self.files = all_files

        # Limit samples
        if max_samples is not None and max_samples > 0:
            self.files = self.files[:max_samples]
            print(f'[EuroSAT] Limited to {len(self.files)} images ({split})')
        else:
            print(f'[EuroSAT] Found {len(self.files)} images ({split})')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # Return (image, dummy_label) to match trainer interface
        return (img, 0)

    def get_image_path(self, idx):
        return self.files[idx]
