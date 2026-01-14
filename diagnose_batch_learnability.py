import os, random
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam

DATA_PATH = r"D:\OneDrive\DATA_KEYPOINTS"
SEQUENCE_LENGTH = 30
FEATURE_DIM = 1662
CATEGORY_SIZE = 150

# build categories like train.py
all_labels = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
categories = []
for i in range(0, len(all_labels), CATEGORY_SIZE):
    categories.append(all_labels[i:i+CATEGORY_SIZE])
num_cats = len(categories)
print('num categories:', num_cats)

# For each category, find one valid sequence
samples = []  # (cat_idx, seq_path)
for cat_idx, cat_labels in enumerate(categories):
    found = False
    for label in cat_labels:
        label_path = os.path.join(DATA_PATH, label)
        if not os.path.isdir(label_path):
            continue
        seq_folders = sorted([f for f in os.listdir(label_path) if os.path.isdir(os.path.join(label_path, f))])
        for seq in seq_folders:
            seq_path = os.path.join(label_path, seq)
            npys = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')])
            if len(npys) >= SEQUENCE_LENGTH:
                samples.append((cat_idx, seq_path))
                found = True
                break
        if found:
            break

# Limit to first 16 categories if more
samples = samples[:16]
if len(samples) < 2:
    print('Not enough samples found for diagnosis')
    raise SystemExit(1)

X = []
y = []
for cat_idx, seq_path in samples:
    frames = []
    for i in range(SEQUENCE_LENGTH):
        p = os.path.join(seq_path, f"{i}.npy")
        a = np.load(p, allow_pickle=True)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        if a.size < FEATURE_DIM:
            pad = np.zeros(FEATURE_DIM, dtype=np.float32)
            pad[:a.size] = a
            a = pad
        else:
            a = a[:FEATURE_DIM]
        frames.append(a)
    arr = np.stack(frames, axis=0)
    # per-sequence normalize
    mean = arr.mean(); std = arr.std()
    if std > 0:
        arr = (arr-mean)/(std+1e-6)
    else:
        arr = arr - mean
    X.append(arr)
    y.append(cat_idx)

X = np.array(X, dtype=np.float32)
num_classes = len(set(y))
Y = np.zeros((len(y), num_classes), dtype=np.float32)
for i, cls in enumerate(y):
    Y[i, cls] = 1.0

print('X shape', X.shape, 'Y shape', Y.shape)
print('X mean,std', X.mean(), X.std())

# Simple MLP
inp = Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM))
flat = Flatten()(inp)
fc = Dense(512, activation='relu')(flat)
out = Dense(num_classes, activation='softmax')(fc)
simple = Model(inp, out)
simple.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

simple.fit(X, Y, epochs=100, verbose=2)
loss, acc = simple.evaluate(X, Y, verbose=0)
print('Final MLP train acc:', acc, 'loss:', loss)

# Simple LSTM test (single LSTM layer)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout
model_lstm = Sequential([
    Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
    LSTM(128, return_sequences=True),
    LSTM(128, return_sequences=False),
    Dense(256, activation='relu'),
    Dropout(0.25),
    Dense(num_classes, activation='softmax')
])
model_lstm.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print('\nTraining simple LSTM (2-layer with dropout)...')
model_lstm.fit(X, Y, epochs=50, verbose=2)
loss_l, acc_l = model_lstm.evaluate(X, Y, verbose=0)
print('Final simple LSTM train acc:', acc_l, 'loss:', loss_l)
