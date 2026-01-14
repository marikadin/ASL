import cv2
import numpy as np
import os
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import random

# Mediapipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class MediapipeHelper:
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def detect(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    @staticmethod
    def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    @staticmethod
    def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]
                        ).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]
                        ).flatten() if results.face_landmarks else np.zeros(468 * 3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]
                      ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]
                      ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return np.concatenate([pose, face, lh, rh])

class DataCollector:
    def __init__(self, actions, no_sequences=30, sequence_length=30, data_path='MP_DATA'):
        self.actions = actions
        self.no_sequences = no_sequences
        self.sequence_length = sequence_length
        self.data_path = data_path
        self.mp_helper = MediapipeHelper()
        self._create_folders()

    def _create_folders(self):
        for action in self.actions:
            for sequence in range(self.no_sequences):
                try:
                    os.makedirs(os.path.join(self.data_path, action, str(sequence)))
                except:
                    pass

    def collect(self):
        cap = cv2.VideoCapture(0)
        for action in self.actions:
            for sequence in range(self.no_sequences):
                for frame_num in range(self.sequence_length):
                    ret, frame = cap.read()
                    frame = cv2.flip(frame, 1)
                    image, results = self.mp_helper.detect(frame)
                    self.mp_helper.draw_landmarks(image, results)

                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting for {action} | Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('Collecting', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting for {action} | Video {sequence}', (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    keypoints = self.mp_helper.extract_keypoints(results)
                    np.save(os.path.join(self.data_path, action, str(sequence), f"{frame_num}.npy"), keypoints)

                    cv2.imshow('Collecting', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
        cap.release()
        cv2.destroyAllWindows()


class ActionModel:
    def __init__(self, sequence_length=30, data_path=r'D:\OneDrive\DATA_KEYPOINTS', batch_size=32):
        self.sequence_length = sequence_length
        self.data_path = data_path
        self.batch_size = batch_size

        # Detect labels
        self.actions = np.array(sorted([
            d for d in os.listdir(self.data_path)
            if os.path.isdir(os.path.join(self.data_path, d))
        ]))
        print("Detected actions (count):", len(self.actions))

        # Label → number
        self.label_map = {label: idx for idx, label in enumerate(self.actions)}

        # Build sequence list once
        self.all_sequences = self._index_dataset()
        print(f"Total sequences indexed: {len(self.all_sequences)}")

        # Split train/test (lists of tuples (action, seq_path))
        self.train_seqs, self.test_seqs = train_test_split(
            self.all_sequences, test_size=0.05, shuffle=True
        )

        print(f"Train sequences: {len(self.train_seqs)}")
        print(f"Test sequences:  {len(self.test_seqs)}\n")

    # ---------------------------------------------------------
    # Build index: list of (action, sequence_path)
    # ---------------------------------------------------------
    def _index_dataset(self):
        index = []
        from tqdm import tqdm

        print("Indexing dataset...")
        for action in tqdm(self.actions):
            action_folder = os.path.join(self.data_path, action)
            try:
                seq_folders = sorted(os.listdir(action_folder))
            except FileNotFoundError:
                continue

            for seq in seq_folders:
                seq_path = os.path.join(action_folder, seq)
                # guard: must be a folder and contain at least sequence_length files
                if os.path.isdir(seq_path):
                    try:
                        if len([f for f in os.listdir(seq_path) if f.endswith('.npy')]) >= self.sequence_length:
                            index.append((action, seq_path))
                    except Exception:
                        # ignore unreadable folders
                        continue
        return index

    # ---------------------------------------------------------
    # Generator – loads batch sequences on demand
    # - Shuffles order at each epoch pass
    # ---------------------------------------------------------
    def generator(self, sequence_list):
        num_classes = len(self.actions)
        seqs = list(sequence_list)

        while True:
            # shuffle each epoch to avoid same-order bias
            random.shuffle(seqs)
            X_batch, y_batch = [], []

            for action, seq_path in seqs:
                frames = []
                # load exactly sequence_length frames; if missing -> pad with zeros
                for i in range(self.sequence_length):
                    npy_path = os.path.join(seq_path, f"{i}.npy")
                    if os.path.exists(npy_path):
                        frames.append(np.load(npy_path))
                    else:
                        frames.append(np.zeros(1662, dtype=np.float32))

                X_batch.append(frames)
                y_batch.append(self.label_map[action])

                if len(X_batch) == self.batch_size:
                    yield (
                        np.array(X_batch, dtype=np.float32),
                        to_categorical(y_batch, num_classes=num_classes)
                    )
                    X_batch, y_batch = [], []

            # if there's remainder that didn't make a full batch, yield it too
            if len(X_batch) > 0:
                yield (
                    np.array(X_batch, dtype=np.float32),
                    to_categorical(y_batch, num_classes=num_classes)
                )

    # ---------------------------------------------------------
    # Build LSTM model
    # ---------------------------------------------------------
    def build_model(self):
        model = Sequential([
            Input(shape=(self.sequence_length, 1662)),
            LSTM(64, return_sequences=True, activation='relu'),
            LSTM(128, return_sequences=True, activation='relu'),
            LSTM(64, return_sequences=False, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(len(self.actions), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy']
        )

        return model

    # ---------------------------------------------------------
    # Train with generator
    # ---------------------------------------------------------
    def train(self, epochs=30, use_tensorboard=True, checkpoint_path='checkpoints/action-{epoch:02d}-{val_categorical_accuracy:.3f}.h5'):
        # Basic checks
        if len(self.train_seqs) == 0:
            print("No training sequences found. Abort.")
            return

        train_gen = self.generator(self.train_seqs)
        test_gen = self.generator(self.test_seqs) if len(self.test_seqs) > 0 else None

        steps_train = max(1, len(self.train_seqs) // self.batch_size)
        steps_val = max(1, len(self.test_seqs) // self.batch_size) if test_gen else 0

        print(f"Starting training: epochs={epochs}, steps_per_epoch={steps_train}, val_steps={steps_val}")

        model = self.build_model()

        # Callbacks
        callbacks = []
        if use_tensorboard:
            log_dir = os.path.join("Logs", time.strftime("%Y%m%d-%H%M%S"))
            callbacks.append(TensorBoard(log_dir=log_dir))

        # checkpoint
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks.append(ModelCheckpoint(filepath=checkpoint_path, save_best_only=False, save_weights_only=False))

        # early stopping to avoid long wasted runs (optional but helpful)
        callbacks.append(EarlyStopping(monitor='val_categorical_accuracy' if test_gen else 'categorical_accuracy',
                                       patience=6, restore_best_weights=False, verbose=1))

        try:
            model.fit(
                train_gen,
                validation_data=test_gen,
                steps_per_epoch=steps_train,
                validation_steps=steps_val if test_gen else None,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (KeyboardInterrupt). Saving model as 'action_interrupted.h5' ...")
            model.save('action_interrupted.h5')
            print("Saved.")
            return

        model.save("action.h5")
        print("Model training complete. Saved as 'action.h5'.")



class LivePredictor:
    def __init__(self, actions, sequence_length=30, threshold=0.6):
        self.actions = actions
        self.sequence_length = sequence_length
        self.threshold = threshold
        self.mp_helper = MediapipeHelper()
        self.model = load_model('action.h5')

    def predict(self):
        sequence = []
        sentence = []
        predictions = []
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image, results = self.mp_helper.detect(frame)
            self.mp_helper.draw_landmarks(image, results)

            keypoints = self.mp_helper.extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-self.sequence_length:]

            if len(sequence) == self.sequence_length:
                res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > self.threshold:
                        if len(sentence) > 0:
                            if self.actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(self.actions[np.argmax(res)])
                        else:
                            sentence.append(self.actions[np.argmax(res)])
                if len(sentence) > 5:
                    sentence = sentence[-5:]

            # Display
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Live Prediction', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

class SingleVideoCollector:
    def __init__(self, video_path, action="test", sequence_length=30, save_path="MP_DATA"):
        self.video_path = video_path
        self.action = action
        self.sequence_length = sequence_length
        self.save_path = save_path
        self.mp_helper = MediapipeHelper()

        # Create folder MP_DATA/test/0
        os.makedirs(os.path.join(self.save_path, action, "0"), exist_ok=True)

    def collect(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"❌ Unable to open video: {self.video_path}")
            return

        sequence = []
        frame_index = 0

        print("🎥 Starting visual keypoint extraction...\n")
        print("Press 'q' to quit early.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("📌 End of video reached.")
                break

            # Flip horizontally to match webcam behavior
            frame = cv2.flip(frame, 1)

            # Detect and draw landmarks
            image, results = self.mp_helper.detect(frame)
            self.mp_helper.draw_landmarks(image, results)

            # Extract keypoints for saving
            keypoints = self.mp_helper.extract_keypoints(results)
            sequence.append(keypoints)

            # Display frame number
            cv2.putText(image, f"Frame: {frame_index}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show result visually
            cv2.imshow("Video Keypoint Viewer", image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("⛔ User stopped the process.")
                break

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

        # Normalize to exactly sequence_length
        sequence = self._normalize_sequence(sequence)

        # Save frames as .npy
        for i, frame_keypoints in enumerate(sequence):
            np.save(os.path.join(self.save_path, self.action, "0", f"{i}.npy"), frame_keypoints)

        print(f"\n✅ Saved {self.sequence_length} keypoint files to: {self.save_path}/{self.action}/0/")

    def _normalize_sequence(self, frames):
        # Too long → cut
        if len(frames) >= self.sequence_length:
            return frames[:self.sequence_length]

        # Too short → pad with zeros
        missing = self.sequence_length - len(frames)
        zero_frame = np.zeros(1662)
        frames.extend([zero_frame] * missing)

        return frames



if __name__ == "__main__":
    actions = np.array(['hello', 'iloveyou', 'thanks'])
    print("Select mode:")
    print("1: Collect data")
    print("2: Train model")
    print("3: Predict live")
    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "1":
        SingleVideoCollector(video_path="output_1sec.mp4").collect()
    elif mode == "2":
        ActionModel().train()
    elif mode == "3":
        LivePredictor(actions).predict()
    else:
        print("Invalid selection.")
