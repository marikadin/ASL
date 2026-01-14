import numpy as np
from tensorflow.keras.models import load_model
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = r"D:\ASL\models\asl_hands_model_full.weights.h5"
TEST_SEQUENCE_PATH = r"DATA_KEYPOINTS_HANDS_ONLY/SLED/2555772799617253-SLED 1.npy"  # replace with your test sequence
DATA_DIR = r"D:\ASL\DATA_KEYPOINTS_HANDS_ONLY"

# Map labels to ids
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
id_to_label = {i: label for i, label in enumerate(labels)}

# =========================
# LOAD MODEL
# =========================
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded.")

# =========================
# LOAD SEQUENCE
# =========================
seq_input = np.load(TEST_SEQUENCE_PATH)  # already processed sequence
print(f"✅ Sequence loaded: {TEST_SEQUENCE_PATH}, shape: {seq_input.shape}")

# Ensure batch dimension
seq_input = np.expand_dims(seq_input, axis=0)  # shape: (1, 30, 252)

# =========================
# PREDICTION
# =========================
predictions = model.predict(seq_input, verbose=0)
predicted_id = np.argmax(predictions)
confidence = predictions[0][predicted_id]

print(f"🎯 Prediction: {id_to_label[predicted_id]}, Confidence: {confidence:.3f}")
