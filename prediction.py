import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = r"D:\ASL\models\asl_hands_model_full.weights.h5"
SEQ_LEN = 30
FEATURE_DIM = 252  # 21 landmarks * 3 coords * 2 hands (pos + velocity)
DATA_KEYPOINTS_DIR = r"D:\ASL\DATA_KEYPOINTS_HANDS_ONLY"
BUFFER_MAX = 60     # Capture ~2 seconds (if 30 FPS videos)
CAPTURE_FPS = 30    # Assumed live FPS for scaling

# Labels
labels = sorted([d for d in os.listdir(DATA_KEYPOINTS_DIR)
                 if os.path.isdir(os.path.join(DATA_KEYPOINTS_DIR, d))])
id_to_label = {i: label for i, label in enumerate(labels)}

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded.")

# =========================
# MEDIA PIPE SETUP
# =========================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# =========================
# HELPER FUNCTIONS
# =========================
def extract_hand_keypoints(results):
    """Flattened right + left hand keypoints"""
    right = np.zeros(21*3, dtype=np.float32)
    left  = np.zeros(21*3, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, h in zip(results.multi_hand_landmarks, results.multi_handedness):
            arr = np.array([[l.x, l.y, l.z] for l in lm.landmark], dtype=np.float32).flatten()
            if h.classification[0].label == "Left":
                left = arr
            else:
                right = arr
    return np.concatenate([right, left])

def center_on_hands(keypoints):
    """Center landmarks around detected hand (like training)"""
    kp = keypoints.reshape(2, 21, 3)  # 0: right, 1: left
    left, right = kp[1], kp[0]

    if np.any(left):
        center = left.mean(axis=0)
    elif np.any(right):
        center = right.mean(axis=0)
    else:
        return keypoints
    kp -= center
    return kp.flatten()

def compute_velocity(seq):
    """Compute velocity like training"""
    seq = np.stack(seq)
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    return vel

def compress_sequence(seq, target_len=SEQ_LEN):
    """Interpolate/compress to SEQ_LEN frames"""
    seq = np.stack(seq)
    if len(seq) == target_len:
        return seq
    indices = np.linspace(0, len(seq)-1, target_len)
    compressed = np.stack([seq[int(round(i))] for i in indices])
    return compressed.astype(np.float32)

def normalize_sequence(seq):
    """Normalize each feature to zero mean, unit variance"""
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + 1e-6
    return (seq - mean) / std

# =========================
# LIVE WEBCAM LOOP
# =========================
cap = cv2.VideoCapture(0)
frame_buffer = []
prev_keypoints = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)

    # Draw landmarks
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    # Extract features
    if results.multi_hand_landmarks:
        keypoints = extract_hand_keypoints(results)
        keypoints = center_on_hands(keypoints)
        if prev_keypoints is None:
            velocity = np.zeros_like(keypoints)
        else:
            velocity = keypoints - prev_keypoints
        prev_keypoints = keypoints.copy()

        feature = np.concatenate([keypoints, velocity], axis=0)  # 252-dim
        frame_buffer.append(feature)

        # Keep buffer size limited
        if len(frame_buffer) > BUFFER_MAX:
            frame_buffer.pop(0)
    else:
        prev_keypoints = None

    # Only predict if we have enough frames
    if len(frame_buffer) >= 5:
        seq = compress_sequence(frame_buffer, SEQ_LEN)
        seq_input = normalize_sequence(seq)
        seq_input = np.expand_dims(seq_input, axis=0)
        predictions = model.predict(seq_input, verbose=0)
        pred_id = np.argmax(predictions)
        confidence = predictions[0][pred_id]
        cv2.putText(frame, f"{id_to_label[pred_id]} ({confidence*100:.1f}%)",
                    (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("ASL Live Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
