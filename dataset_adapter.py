import cv2
import numpy as np
import os
import re
from main import MediapipeHelper
from tqdm import tqdm

# -------------------------------
# 1) Clean label
# -------------------------------
def extract_label_from_filename(filename):
    name = os.path.splitext(filename.upper())[0]
    name = re.sub(r"[0-9\.\-\+e_]", "", name)
    return name.strip()

# -------------------------------
# 2) Extract keypoints
# -------------------------------
MP_HELPER = MediapipeHelper()

def extract_keypoints_from_video(video_path, save_folder, sequence_length=30):
    os.makedirs(save_folder, exist_ok=True)

    existing = os.listdir(save_folder)
    sequence_id = len(existing)
    seq_folder = os.path.join(save_folder, str(sequence_id))
    os.makedirs(seq_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frames_keypoints = []

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        _, results = MP_HELPER.detect(frame)
        keypoints = MP_HELPER.extract_keypoints(results)
        frames_keypoints.append(keypoints)
        frame_num += 1

    cap.release()


    # Normalize to exactly sequence_length frames
    if len(frames_keypoints) >= sequence_length:
        frames_keypoints = frames_keypoints[:sequence_length]
    else:
        missing = sequence_length - len(frames_keypoints)
        frames_keypoints.extend([np.zeros(1662)] * missing)

    # Save frames
    for i, kp in enumerate(frames_keypoints):
        np.save(os.path.join(seq_folder, f"{i}.npy"), kp)

    #print(f"✅ Saved sequence {sequence_id} → {seq_folder}")

# -------------------------------
# 3) Process single video
# -------------------------------
def process_single_video(file_path, input_folder, output_folder):
    input_path = os.path.join(input_folder, file_path)

    label = extract_label_from_filename(file_path)
    label_folder = os.path.join(output_folder, label)
    os.makedirs(label_folder, exist_ok=True)

    extract_keypoints_from_video(input_path, label_folder)

    return f"✅ Processed: {file_path}"

# -------------------------------
# 4) Process folder sequentially (robust)
# -------------------------------
def process_folder(input_folder, output_folder="DATA_KEYPOINTS"):
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(".mp4")]
    print(f"🎬 Found {len(files)} videos.")

    results = []
    for file in tqdm(files, desc="Processing Videos"):
        result = process_single_video(file, input_folder, output_folder)
        results.append(result)

    print("\n".join(results))
    print("\n🎉 DONE — All videos processed!")

# -------------------------------
# 5) Run
# -------------------------------
if __name__ == "__main__":
    process_folder("DATA", output_folder="DATA_KEYPOINTS")
