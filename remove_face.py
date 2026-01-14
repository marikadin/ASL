import os
import numpy as np
from tqdm import tqdm

# =====================
# CONFIG
# =====================
SRC_DIR = r"D:\ASL\DATA_KEYPOINTS_PACKED_50PLUS"
DST_DIR = r"D:\ASL\DATA_KEYPOINTS_HANDS_ONLY"
KEEP_VELOCITY = True   # set False if you want ONLY positions

os.makedirs(DST_DIR, exist_ok=True)

# =====================
# MEDIAPIPE CONSTANTS
# =====================
POSE_DIM = 33 * 3        # 99
FACE_DIM = 468 * 3       # 1404
HAND_DIM = 21 * 3        # 63

# indices in original feature vector
POSE_START = 0
FACE_START = POSE_START + POSE_DIM
LHAND_START = FACE_START + FACE_DIM
RHAND_START = LHAND_START + HAND_DIM

HANDS_SLICE = slice(LHAND_START, RHAND_START + HAND_DIM)

# =====================
# CONVERT
# =====================
print("🔄 Converting dataset to HANDS ONLY...\n")

for label in os.listdir(SRC_DIR):
    src_label_dir = os.path.join(SRC_DIR, label)
    if not os.path.isdir(src_label_dir):
        continue

    dst_label_dir = os.path.join(DST_DIR, label)
    os.makedirs(dst_label_dir, exist_ok=True)

    for fname in tqdm(os.listdir(src_label_dir), desc=f"Processing {label}"):
        if not fname.endswith(".npy"):
            continue

        src_path = os.path.join(src_label_dir, fname)
        dst_path = os.path.join(dst_label_dir, fname)

        seq = np.load(src_path)

        # expected shape: (T, F)
        T, F = seq.shape

        # detect velocity
        has_velocity = F > RHAND_START + HAND_DIM

        # positions
        hands_pos = seq[:, HANDS_SLICE]

        if KEEP_VELOCITY and has_velocity:
            vel_offset = F // 2
            hands_vel = seq[:, vel_offset + HANDS_SLICE.start : vel_offset + HANDS_SLICE.stop]
            new_seq = np.concatenate([hands_pos, hands_vel], axis=-1)
        else:
            new_seq = hands_pos

        np.save(dst_path, new_seq)

print("\n✅ DONE")
