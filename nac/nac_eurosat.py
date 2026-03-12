"""
NAC (N-gram Arithmetic Coding) for EuroSAT RQ-VAE codes.

This script:
  1. Reads RQ-VAE generated codes (8×8×D flattened per image)
  2. Builds an N-gram frequency table from training codes
  3. Encodes/decodes test codes using arithmetic coding
  4. Reports compression rates and timing

Usage:
    cd nac/
    python nac_eurosat.py
"""
from ngram import NGramModel
from arithmetic_coding import ArithmeticEncoder
import sys
import logging
import os
import time


def readcode(filename, n=None):
    result = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                break
            line = line.strip()
            if line:
                numbers = [int(x) for x in line.split()]
                result.append(numbers)
    return result


# =================== Configuration ===================
# N-gram order
N = 2
# K smoothing constant
K = 0.1
# Depth of RQ-VAE code
D = 4
# Spatial dimensions of EuroSAT latent (8×8 vs vehicle's 23×40)
H, W = 8, 8
# Train/test split
N_TRAIN = 900
N_TOTAL = 1000
# ======================================================


logger = logging.getLogger()
logger.setLevel(logging.INFO)

os.makedirs("logs", exist_ok=True)

logfile = f"logs/{N}gram_eurosat_{H}x{W}x{D}_log.txt"
print("Log:", logfile)
file_handler = logging.FileHandler(logfile, mode='w', encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Code file generated from RQ-VAE training on EuroSAT
filename = f"data/codes{H}x{W}x{D}.txt"
logger.info(f"N={N}, K={K}")
logger.info(f"Code shape: {H}×{W}×{D} = {H*W*D} codes per image")
logger.info(f"Data: {filename}")

training_sequences = readcode(filename, N_TRAIN)

model = NGramModel(n=N, k=K, start_token=-1, end_token=-2)
model.fit(training_sequences)

os.makedirs("models", exist_ok=True)
model.save(f"models/{N}gram_eurosat_{H}x{W}x{D}.pkl")

encoder = ArithmeticEncoder(ngram_model=model, bits=32)
print("Encoder Created")

codes = readcode(filename, N_TOTAL)[N_TRAIN:N_TOTAL]

avgrate = []
encode_times_ms = []
decode_times_ms = []

# Uncompressed size: each code index uses 11 bits (log2(2048) = 11)
BITS_PER_CODE = 11

for i in range(len(codes)):
    logger.info(f"Code: {i}")
    test_sequence = codes[i]

    t0 = time.perf_counter()
    encoded_bits = encoder.encode(test_sequence)
    t1 = time.perf_counter()
    encode_ms = (t1 - t0) * 1000.0
    encode_times_ms.append(encode_ms)

    logger.info(
        f"Encoded: {len(encoded_bits)} bits, encode_time={encode_ms:.4f} ms")
    rate = len(encoded_bits) / (len(test_sequence) * BITS_PER_CODE)
    avgrate.append(rate)
    logger.info(f"Compression Rate: {rate:.2%}")

    # Uncompressed image size: 64×64×3×8 = 98304 bits
    raw_image_bits = 64 * 64 * 3 * 8
    compression_ratio = raw_image_bits / len(encoded_bits)
    logger.info(f"vs Raw Image Compression Ratio: {compression_ratio:.1f}×")

    t0 = time.perf_counter()
    decoded_sequence = encoder.decode(encoded_bits)
    t1 = time.perf_counter()
    decode_ms = (t1 - t0) * 1000.0
    decode_times_ms.append(decode_ms)

    logger.info(f"Decode_time={decode_ms:.4f} ms")
    logger.info(
        f"Verification: {'Correct' if decoded_sequence == test_sequence else 'Wrong'}")

logger.info("")
logger.info("=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)
logger.info(
    f"Average Compression Rate (vs uncompressed codes): {sum(avgrate)/len(avgrate):.2%}")
avg_raw_ratio = sum(
    64*64*3*8 / (len(encoder.encode(codes[i])))
    for i in range(len(codes))
) / len(codes)
logger.info(f"Average Compression Ratio (vs raw RGB): {avg_raw_ratio:.1f}×")
logger.info("")
logger.info("Timing summary (ms):")
avg_encode = sum(encode_times_ms) / len(encode_times_ms)
avg_decode = sum(decode_times_ms) / len(decode_times_ms)
logger.info(f"  Average encode time: {avg_encode:.4f} ms")
logger.info(f"  Average decode time: {avg_decode:.4f} ms")
logger.info(
    f"  encode: min={min(encode_times_ms):.4f}, max={max(encode_times_ms):.4f}")
logger.info(
    f"  decode: min={min(decode_times_ms):.4f}, max={max(decode_times_ms):.4f}")
