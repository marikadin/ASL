import os
import numpy as np

DATA_PATH = r"D:\ASL\DATA_KEYPOINTS_PACKED_50PLUS"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 3324   # ✅ positions + velocity


def find_sequence():
    labels = sorted(
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    )

    for label in labels:
        label_path = os.path.join(DATA_PATH, label)
        npys = sorted(
            f for f in os.listdir(label_path)
            if f.endswith(".npy")
        )

        for f in npys:
            p = os.path.join(label_path, f)
            try:
                arr = np.load(p)
                if arr.shape == (SEQUENCE_LENGTH, FEATURE_DIM):
                    return label, p, arr
            except Exception:
                continue

    return None, None, None


label, path, arr = find_sequence()

if path is None:
    print("❌ No valid packed sequence found")
    raise SystemExit(1)

print("✅ Found valid sequence")
print("Label:", label)
print("Path:", path)
print("Shape:", arr.shape)
print()

# ===== stats =====
print("Overall stats")
print("mean:", arr.mean())
print("std :", arr.std())
print("min :", arr.min())
print("max :", arr.max())
print()

print("Per-frame mean (first 10):")
print(arr.mean(axis=1)[:10])

print("Per-frame std (first 10):")
print(arr.std(axis=1)[:10])

zero_frames = np.sum(np.all(arr == 0, axis=1))
print("Zero frames:", int(zero_frames))

print()
print("Per-feature mean (first 10):")
print(arr.mean(axis=0)[:10])

print("Per-feature std (first 10):")
print(arr.std(axis=0)[:10])

# sanity checks
print()
print("NaNs:", np.isnan(arr).sum())
print("Infs:", np.isinf(arr).sum())
