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
    def __init__(self, actions, sequence_length=30, data_path='MP_DATA'):
        self.actions = actions
        self.sequence_length = sequence_length
        self.data_path = data_path
        self.label_map = {label: num for num, label in enumerate(actions)}

    def load_data(self):
        sequences, labels = [], []
        for action in self.actions:
            for sequence in range(30):
                window = []
                for frame_num in range(self.sequence_length):
                    path = os.path.join(self.data_path, action, str(sequence), f"{frame_num}.npy")
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Missing file: {path}")
                    window.append(np.load(path))
                sequences.append(window)
                labels.append(self.label_map[action])
        X = np.array(sequences)
        y = to_categorical(labels).astype(int)
        return train_test_split(X, y, test_size=0.05)

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(30, 1662)))
        model.add(LSTM(64, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.actions.shape[0], activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()
        model = self.build_model()
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
        model.save('action.h5')
        print("✅ Model trained and saved as 'action.h5'")

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

if __name__ == "__main__":
    actions = np.array(['hello', 'iloveyou', 'thanks'])
    print("Select mode:")
    print("1: Collect data")
    print("2: Train model")
    print("3: Predict live")
    mode = input("Enter 1, 2, or 3: ").strip()

    if mode == "1":
        DataCollector(actions).collect()
    elif mode == "2":
        ActionModel(actions).train()
    elif mode == "3":
        LivePredictor(actions).predict()
    else:
        print("Invalid selection.")
