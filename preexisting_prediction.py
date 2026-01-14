import os
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
MODEL_PATH = r"D:\ASL\models\asl_hands_model_full.weights.h5"
SEQ_LEN = 30
FEATURE_DIM = 252  # positions + velocity
DATA_DIR = r"D:\ASL\DATA_KEYPOINTS_HANDS_ONLY"
VIDEO_NAME = r"9811020872319085-ABOUT 2.mp4"  # exact video from training

# Map labels to ids
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
id_to_label = {i: label for i, label in enumerate(labels)}

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded.")

# =========================
# LOAD TRAINING SEQUENCE
# =========================
seq_path = None
for label in labels:
    candidate = os.path.join(DATA_DIR, label, VIDEO_NAME.replace(".mp4", ".npy"))
    if os.path.exists(candidate):
        seq_path = candidate
        break

if seq_path is None:
    raise FileNotFoundError("Could not find processed .npy for this video in DATA_KEYPOINTS_HANDS_ONLY!")

seq_input = np.load(seq_path)

# Ensure shape is correct
if seq_input.shape != (SEQ_LEN, FEATURE_DIM):
    raise ValueError(f"Sequence shape mismatch: {seq_input.shape}")

# Add batch dimension
seq_input = np.expand_dims(seq_input, axis=0)

# =========================
# PREDICT
# =========================
predictions = model.predict(seq_input, verbose=0)
predicted_id = np.argmax(predictions)
confidence = predictions[0][predicted_id]

print(f"🎯 Prediction: {id_to_label[predicted_id]}, Confidence: {confidence:.3f}")
