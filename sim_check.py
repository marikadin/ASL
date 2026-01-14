import os
import cv2
import numpy as np
from main import MediapipeHelper
from numpy.linalg import norm

# ============================
# CONFIG
# ============================
SEQ_LEN = 30
BASE_FEATURES = 1662        # original full feature length (positions only)
FEATURE_DIM = 126           # hands-only positions (2 hands * 21 landmarks * 3)
INPUT_DIR = r"DATA"         # original raw videos
FINAL_DIR = r"D:\ASL\DATA_KEYPOINTS_HANDS_ONLY"  # final processed .npy
VIDEO_NAME = "9811020872319085-ABOUT 2.mp4"  # change to your video

# ============================
# MEDIAPIPE
# ============================
mp_helper = MediapipeHelper()

# ============================
# HAND-CENTER NORMALIZATION
# ============================
def center_on_hands(keypoints):
    kp = keypoints.reshape(-1, 3)
    left_hand = kp[468:489]
    right_hand = kp[489:510]

    if np.any(left_hand):
        center = left_hand.mean(axis=0)
    elif np.any(right_hand):
        center = right_hand.mean(axis=0)
    else:
        return keypoints

    kp = kp - center
    return kp.flatten()

# ============================
# EXTRACT SEQUENCE FROM VIDEO
# ============================
def extract_sequence(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frames = []
    while len(frames) < SEQ_LEN:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        _, results = mp_helper.detect(frame)
        keypoints = mp_helper.extract_keypoints(results)
        keypoints = center_on_hands(keypoints)
        frames.append(keypoints.astype(np.float32))

    cap.release()
    if len(frames) < 5:
        raise ValueError("Video too short / not enough frames")

    # pad frames
    while len(frames) < SEQ_LEN:
        frames.append(np.zeros(BASE_FEATURES, dtype=np.float32))

    frames = np.stack(frames)
    return frames

# ============================
# HANDS-ONLY (positions only)
# ============================
def hands_only(seq):
    POSE_DIM = 33*3
    FACE_DIM = 468*3
    HAND_DIM = 21*3
    POSE_START = 0
    FACE_START = POSE_START + POSE_DIM
    LHAND_START = FACE_START + FACE_DIM
    RHAND_START = LHAND_START + HAND_DIM
    HANDS_SLICE = slice(LHAND_START, RHAND_START+HAND_DIM)
    return seq[:, HANDS_SLICE]

# ============================
# COSINE SIMILARITY
# ============================
def cosine_similarity(a, b):
    return np.sum(a*b, axis=1) / (norm(a, axis=1) * norm(b, axis=1) + 1e-6)

# ============================
# RMSE
# ============================
def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2, axis=1))

# ============================
# MAIN
# ============================
if __name__ == "__main__":
    # 1️⃣ Process raw video (hands-only, raw positions)
    video_path = os.path.join(INPUT_DIR, VIDEO_NAME)
    seq_raw = extract_sequence(video_path)
    seq_hands = hands_only(seq_raw)
    # NO normalization → high similarity mode
    seq_norm = seq_hands

    # 2️⃣ Load final processed file (positions only)
    final_path = None
    for lbl in os.listdir(FINAL_DIR):
        folder = os.path.join(FINAL_DIR, lbl)
        candidate = os.path.join(folder, VIDEO_NAME.replace(".mp4", ".npy"))
        if os.path.exists(candidate):
            final_path = candidate
            break
    if final_path is None:
        raise FileNotFoundError("Final processed file not found!")

    seq_final = np.load(final_path)
    seq_final_pos = seq_final[:, :FEATURE_DIM]  # first 126 = positions only

    # Ensure shapes match
    if seq_final_pos.shape != seq_norm.shape:
        min_len = min(seq_final_pos.shape[0], seq_norm.shape[0])
        seq_norm = seq_norm[:min_len]
        seq_final_pos = seq_final_pos[:min_len]

    # 3️⃣ Compare
    cos_sims = cosine_similarity(seq_norm, seq_final_pos)
    rmses = rmse(seq_norm, seq_final_pos)

    print(f"✅ Mean frame-wise cosine similarity: {cos_sims.mean():.4f}")
    print(f"✅ Mean frame-wise RMSE: {rmses.mean():.4f}")
    print(f"Frame-wise cosine similarity:\n{cos_sims}")
    print(f"Frame-wise RMSE:\n{rmses}")
