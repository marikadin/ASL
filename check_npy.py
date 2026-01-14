import os
import numpy as np
import random

# ==========================
# CONFIG
# ==========================
DATA_PATH = r"D:\OneDrive\DATA_KEYPOINTS"   # change if needed
NUM_FILES_TO_CHECK = 20
EXPECTED_FEATURE_DIM = 1662

# ==========================
# COLLECT ALL NPY FILES
# ==========================
npy_files = []

for label in os.listdir(DATA_PATH):
    label_path = os.path.join(DATA_PATH, label)
    if not os.path.isdir(label_path):
        continue

    for seq in os.listdir(label_path):
        seq_path = os.path.join(label_path, seq)
        if not os.path.isdir(seq_path):
            continue

        for f in os.listdir(seq_path):
            if f.endswith(".npy"):
                npy_files.append(os.path.join(seq_path, f))

print(f"📦 Found {len(npy_files)} total .npy files")

if len(npy_files) == 0:
    raise RuntimeError("No .npy files found!")

# ==========================
# SAMPLE FILES
# ==========================
sample_files = random.sample(
    npy_files,
    min(NUM_FILES_TO_CHECK, len(npy_files))
)

print(f"\n🔍 Checking {len(sample_files)} random .npy files:\n")

# ==========================
# CHECK FILES
# ==========================
bad_files = []

for path in sample_files:
    print(f"📄 {path}")

    try:
        arr = np.load(path, allow_pickle=True)

        arr = np.asarray(arr).reshape(-1)

        print(f"   shape: {arr.shape}")
        print(f"   dtype: {arr.dtype}")
        print(f"   min / max: {arr.min():.4f} / {arr.max():.4f}")
        print(f"   mean / std: {arr.mean():.4f} / {arr.std():.4f}")

        if arr.shape[0] != EXPECTED_FEATURE_DIM:
            print(f"   ⚠️ WARNING: feature length != {EXPECTED_FEATURE_DIM}")

        if not np.isfinite(arr).all():
            print("   ❌ ERROR: contains NaN or Inf values")
            bad_files.append(path)

    except Exception as e:
        print(f"   ❌ FAILED TO LOAD: {e}")
        bad_files.append(path)

    print("-" * 60)

# ==========================
# SUMMARY
# ==========================
print("\n📊 SUMMARY")
print(f"Total checked: {len(sample_files)}")
print(f"Problematic files: {len(bad_files)}")

if bad_files:
    print("\n❌ Bad files:")
    for f in bad_files:
        print(" ", f)
else:
    print("\n✅ All sampled files look OK")
