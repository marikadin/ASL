import os
import shutil

BASE_DIR = "DATA_KEYPOINTS_FIXED"

folders = [
    f for f in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, f))
]

for folder in folders:
    if not folder.startswith("SEED"):
        continue

    seed_path = os.path.join(BASE_DIR, folder)
    target_name = folder[len("SEED"):]  # remove SEED
    target_path = os.path.join(BASE_DIR, target_name)

    # Ensure target exists
    os.makedirs(target_path, exist_ok=True)

    for file in os.listdir(seed_path):
        if not file.endswith(".npy"):
            continue

        src = os.path.join(seed_path, file)
        dst = os.path.join(target_path, file)

        # ✅ resume-safe
        if os.path.exists(dst):
            continue

        shutil.move(src, dst)

    # remove empty SEED folder
    try:
        os.rmdir(seed_path)
        print(f"🧹 Removed empty folder: {folder}")
    except OSError:
        print(f"⚠️ Could not remove {folder} (not empty)")

print("✅ SEED folders merged correctly")
