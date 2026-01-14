import os
import numpy as np
from tqdm import tqdm

# ==============================
# CONFIG
# ==============================
SRC_DIR = "DATA_KEYPOINTS_PACKED_50PLUS"
DST_DIR = "DATA_PACKED_FINAL"

SEQ_LEN = 30
FEATURES = 1662 * 2  # position + velocity

os.makedirs(DST_DIR, exist_ok=True)

# ==============================
# NORMALIZATION (PER SEQUENCE)
# ==============================
def normalize_sequence(seq):
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std

# ==============================
# PACK DATASET
# ==============================
labels = [l for l in os.listdir(SRC_DIR) if os.path.isdir(os.path.join(SRC_DIR, l))]
print(f"📂 Found {len(labels)} labels")

for label in tqdm(labels, desc="Packing labels"):
    src_label = os.path.join(SRC_DIR, label)
    dst_label = os.path.join(DST_DIR, label)
    os.makedirs(dst_label, exist_ok=True)

    for file in os.listdir(src_label):
        if not file.endswith(".npy"):
            continue

        src_path = os.path.join(src_label, file)
        dst_path = os.path.join(dst_label, file)

        # ✅ resume-safe
        if os.path.exists(dst_path):
            continue

        try:
            seq = np.load(src_path)

            # shape check
            if seq.shape != (SEQ_LEN, FEATURES):
                print(f"⚠️ Skipping {file}: wrong shape {seq.shape}")
                continue

            seq = normalize_sequence(seq)
            np.save(dst_path, seq)

        except Exception as e:
            print(f"❌ Failed {file}: {e}")

print("✅ Packing complete")
