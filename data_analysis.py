import os
import numpy as np
from collections import Counter, defaultdict
from statistics import mean

# ============================
# CONFIG (must match training)
# ============================
DATA_PATH = r"DATA_KEYPOINTS_PACKED_50PLUS"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 1662
CATEGORY_SIZE = 150

# ============================
# HELPERS
# ============================
def safe_npy_shape(path):
    try:
        arr = np.load(path, allow_pickle=True)
        return arr.size
    except Exception:
        return None

# ============================
# ANALYSIS STORAGE
# ============================
label_sequence_counts = Counter()
category_sequence_counts = Counter()
frame_counts = []
feature_lengths = []
corrupted_npy_files = []
short_sequences = []
missing_frames = []
label_frame_stats = defaultdict(list)

# ============================
# SCAN DATASET
# ============================
labels = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])

categories = [
    labels[i:i + CATEGORY_SIZE]
    for i in range(0, len(labels), CATEGORY_SIZE)
]

print("\n📁 DATASET STRUCTURE")
print(f"Total labels: {len(labels)}")
print(f"Total categories: {len(categories)}")
print(f"Category size (target): {CATEGORY_SIZE}")

total_sequences = 0

for cat_idx, cat_labels in enumerate(categories):
    cat_name = f"category_{cat_idx}"
    for label in cat_labels:
        label_path = os.path.join(DATA_PATH, label)
        if not os.path.isdir(label_path):
            continue

        sequences = [
            s for s in os.listdir(label_path)
            if os.path.isdir(os.path.join(label_path, s))
        ]

        for seq in sequences:
            seq_path = os.path.join(label_path, seq)
            frames = sorted([
                f for f in os.listdir(seq_path)
                if f.endswith(".npy")
            ])

            frame_count = len(frames)
            frame_counts.append(frame_count)
            label_frame_stats[label].append(frame_count)

            if frame_count < SEQUENCE_LENGTH:
                short_sequences.append((label, seq, frame_count))

            # Check missing frame indices
            expected = {f"{i}.npy" for i in range(SEQUENCE_LENGTH)}
            missing = expected - set(frames)
            if missing:
                missing_frames.append((label, seq, sorted(missing)))

            # Analyze frame features
            for f in frames[:SEQUENCE_LENGTH]:
                fpath = os.path.join(seq_path, f)
                size = safe_npy_shape(fpath)
                if size is None:
                    corrupted_npy_files.append(fpath)
                else:
                    feature_lengths.append(size)

            label_sequence_counts[label] += 1
            category_sequence_counts[cat_name] += 1
            total_sequences += 1

# ============================
# SUMMARY REPORT
# ============================
print("\n🎞️ SEQUENCE STATS")
print(f"Total sequences: {total_sequences}")
print(f"Sequences per label (min / mean / max): "
      f"{min(label_sequence_counts.values(), default=0)} / "
      f"{mean(label_sequence_counts.values()):.2f} / "
      f"{max(label_sequence_counts.values(), default=0)}")

print("\n📊 FRAME STATS")
print(f"Frames per sequence (min / mean / max): "
      f"{min(frame_counts, default=0)} / "
      f"{mean(frame_counts):.2f} / "
      f"{max(frame_counts, default=0)}")

print("\n🧠 FEATURE STATS")
if feature_lengths:
    print(f"Feature length (min / mean / max): "
          f"{min(feature_lengths)} / "
          f"{mean(feature_lengths):.1f} / "
          f"{max(feature_lengths)}")
else:
    print("No valid feature vectors found.")

print("\n⚖️ CLASS IMBALANCE")
print("Top 10 most common labels:")
for lbl, cnt in label_sequence_counts.most_common(10):
    print(f"  {lbl}: {cnt}")

print("\nTop 10 least common labels:")
for lbl, cnt in label_sequence_counts.most_common()[-10:]:
    print(f"  {lbl}: {cnt}")

# ============================
# WARNINGS
# ============================
print("\n🚨 DATASET WARNINGS")

print(f"Sequences shorter than {SEQUENCE_LENGTH}: {len(short_sequences)}")
print(f"Sequences with missing frames: {len(missing_frames)}")
print(f"Corrupted .npy files: {len(corrupted_npy_files)}")

if corrupted_npy_files[:5]:
    print("Example corrupted files:")
    for f in corrupted_npy_files[:5]:
        print(" ", f)

# ============================
# LABEL QUALITY CHECK
# ============================
print("\n🔍 LABEL QUALITY CHECK")
low_data_labels = [l for l, c in label_sequence_counts.items() if c < 10]

print(f"Labels with <10 sequences: {len(low_data_labels)}")
if low_data_labels[:10]:
    print("Examples:", low_data_labels[:10])

print("\n✅ DATASET ANALYSIS COMPLETE")
