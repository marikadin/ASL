# =========================================
# ASL HANDS-ONLY TRAINING SCRIPT (WORKING)
# =========================================

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import (
    Input, Dense, Dropout,
    LSTM, Bidirectional,
    GlobalAveragePooling1D, Attention
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# =========================================
# CONFIG
# =========================================

DATA_DIR = r"D:\ASL\DATA_KEYPOINTS_HANDS_ONLY"
SEQ_LEN = 30
FEATURE_DIM = 252      # 21 landmarks * 3 coords * 2 hands
BATCH_SIZE = 32
EPOCHS = 100

MODEL_PATH = r"D:\ASL\models\asl_hands_model_full.weights.h5"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# =========================================
# INDEX DATASET
# =========================================

labels = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])

label_to_id = {label: i for i, label in enumerate(labels)}
num_classes = len(labels)

samples = []
for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    for f in os.listdir(label_dir):
        if f.endswith(".npy"):
            samples.append((os.path.join(label_dir, f), label_to_id[label]))

print("Classes:", num_classes)
print("Samples:", len(samples))

train_samples, val_samples = train_test_split(
    samples,
    test_size=0.1,
    shuffle=True,
    random_state=42
)

# =========================================
# DATA GENERATOR
# =========================================

class ASLSequence(Sequence):
    def __init__(self, samples, batch_size, num_classes):
        self.samples = samples
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.indices = np.arange(len(samples))
        np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.samples) // self.batch_size

    def __getitem__(self, idx):
        batch_idx = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        X = np.zeros((len(batch_idx), SEQ_LEN, FEATURE_DIM), dtype=np.float32)
        y = np.zeros((len(batch_idx), self.num_classes), dtype=np.float32)

        for i, j in enumerate(batch_idx):
            path, label = self.samples[j]
            seq = np.load(path)

            if seq.shape != (SEQ_LEN, FEATURE_DIM):
                raise ValueError(f"Invalid shape {seq.shape} in {path}")

            X[i] = seq
            y[i, label] = 1.0

        return X, y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

train_gen = ASLSequence(train_samples, BATCH_SIZE, num_classes)
val_gen   = ASLSequence(val_samples,   BATCH_SIZE, num_classes)

# =========================================
# MODEL
# =========================================

def build_model(num_classes):
    inputs = Input(shape=(SEQ_LEN, FEATURE_DIM))

    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    x = Attention()([x, x])
    x = GlobalAveragePooling1D()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)

model = build_model(num_classes)
model.summary()

# =========================================
# COMPILE
# =========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[
        "categorical_accuracy",
        tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
    ]
)

# =========================================
# CALLBACKS
# =========================================

callbacks = [
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_top5",
        save_best_only=True,
        save_weights_only=False,
        mode="max",
        verbose=1
    ),
    EarlyStopping(
        monitor="val_top5",
        patience=10,
        restore_best_weights=True,
        mode="max"
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# =========================================
# TRAIN
# =========================================

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)
