import cv2
import numpy as np
import os
import re
from tqdm import tqdm
from main import MediapipeHelper

# ==============================
# CONFIG
# ==============================
SEQ_LEN = 30
BASE_FEATURES = 1662
OUTPUT_DIR = r"D:\ASL\DATA_KEYPOINTS_FIXED"
INPUT_DIR = "DATA"

# ==============================
# LABEL CLEANING
# ==============================
def extract_label(filename):
    name = os.path.splitext(filename.upper())[0]
    name = re.sub(r"[0-9\.\-\+e_]", "", name)
    return name.strip()

# ==============================
# MEDIAPIPE
# ==============================
mp_helper = MediapipeHelper()

# ==============================
# HAND-CENTER NORMALIZATION
# ==============================
def center_on_hands(keypoints):
    """
    Centers landmarks around the active hand
    """
    kp = keypoints.reshape(-1, 3)

    # MediaPipe hand landmark indices (relative)
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

# ==============================
# VELOCITY FEATURES
# ==============================
def compute_velocity(sequence):
    velocity = np.zeros_like(sequence)
    velocity[1:] = sequence[1:] - sequence[:-1]
    return velocity

# ==============================
# EXTRACT ONE VIDEO → ONE ARRAY
# ==============================
def extract_sequence(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None  # corrupted video

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
        return None  # useless sample

    # Pad
    while len(frames) < SEQ_LEN:
        frames.append(np.zeros(BASE_FEATURES, dtype=np.float32))

    frames = np.stack(frames)
    velocity = compute_velocity(frames)

    # Final shape: (SEQ_LEN, BASE_FEATURES * 2)
    return np.concatenate([frames, velocity], axis=-1)

# ==============================
# DATASET PROCESSING (RESUMABLE)
# ==============================
def process_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    videos = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".mp4")
    ])

    print(f"🎬 Found {len(videos)} videos")

    for video in tqdm(videos, desc="Extracting", unit="video"):
        label = extract_label(video)
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)

        # 🔑 deterministic output name
        video_id = os.path.splitext(video)[0]
        out_path = os.path.join(label_dir, f"{video_id}.npy")

        # ✅ resume support
        if os.path.exists(out_path):
            continue

        seq = extract_sequence(os.path.join(input_dir, video))
        if seq is None:
            continue

        np.save(out_path, seq)

    print("✅ Dataset extraction complete")

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    process_folder()
