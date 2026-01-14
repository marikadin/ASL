import os
import json
from collections import Counter
from sklearn.model_selection import train_test_split

DATA_DIR = r"D:\ONEDRIVE\DATA_KEYPOINTS"

# Build label map
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

with open("labels.json", "w") as f:
    json.dump(label_map, f)

# Scan dataset
sample_paths = []
sample_labels = []

for label in labels:
    label_folder = os.path.join(DATA_DIR, label)
    for sample_folder in os.listdir(label_folder):
        sample_folder_path = os.path.join(label_folder, sample_folder)
        if os.path.isdir(sample_folder_path):
            sample_paths.append(sample_folder_path)
            sample_labels.append(label_map[label])

print(f"Total samples found: {len(sample_paths)}")

# -----------------------------
# Filter singleton classes and split
# -----------------------------
label_counts = Counter(sample_labels)

filtered_paths = []
filtered_labels = []

for path, label in zip(sample_paths, sample_labels):
    if label_counts[label] > 1:
        filtered_paths.append(path)
        filtered_labels.append(label)

removed_labels = [label for label, count in label_counts.items() if count == 1]
print(f"Removed labels with only 1 sample: {removed_labels}")
print(f"Total samples after filtering: {len(filtered_paths)}")

train_paths, test_paths, train_labels, test_labels = train_test_split(
    filtered_paths, filtered_labels, test_size=0.2, stratify=filtered_labels, random_state=42
)

print(f"Training samples: {len(train_paths)}, Test samples: {len(test_paths)}")
