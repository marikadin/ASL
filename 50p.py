import os
import shutil

DATA_DIR = "DATA_KEYPOINTS_FIXED"
FILTERED_DIR = "DATA_KEYPOINTS_PACKED_50PLUS"

os.makedirs(FILTERED_DIR, exist_ok=True)

for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_path):
        continue

    # Count .npy files in the label folder
    npy_files = [f for f in os.listdir(label_path) if f.endswith(".npy")]
    
    if len(npy_files) >= 50:
        # Copy the entire folder to the new directory
        shutil.copytree(label_path, os.path.join(FILTERED_DIR, label))
        print(f"Kept '{label}' with {len(npy_files)} samples.")
    else:
        print(f"Skipped '{label}' with {len(npy_files)} samples.")

print(f"\nFiltered dataset created at: {FILTERED_DIR}")
