import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Input
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                              )    

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])



  
DATA_PATH = os.path.join('MP_DATA')
actions = np.array(['hello','iloveyou','thanks'])
no_sequances = 30
sequance_length = 30

label_map = {label:num for num, label in enumerate(actions)}
sequances, lables = [], []
for action in actions:
    for sequance in range(no_sequances):
        window = []
        for frame_num in range(sequance_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequance), f"{frame_num}.npy"))
            window.append(res)
        sequances.append(window)
        lables.append(label_map[action])

X = np.array(sequances)
y = to_categorical(lables).astype(int)   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)   
model = Sequential()
model.add(Input(shape=(30, 1662))) 
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

def create():
    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])
    model.save('action.h5')
 
 
def load():
    model.load_weights('action.h5')  

load()

sequance = []
sentence = []
predictions = []
threshold = 0.6

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image, results = mediapipe_detection(frame, holistic)         
        draw_styled_landmarks(image, results)        
        
        keypoints = extract_keypoints(results)
        sequance.append(keypoints)
        sequance = sequance[-30:]
        
        if len(sequance) == 30:
            res = model.predict(np.expand_dims(sequance,axis=0))[0]
            predictions.append(np.argmax(res))
        
            if len(predictions) >=10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                max_idx = np.argmax(res)
                if max_idx < len(actions) and res[max_idx] > threshold:
                    if len(sentence) > 0:
                        if actions[max_idx] != sentence[-1]:
                            sentence.append(actions[max_idx])
                    else:
                        sentence.append(actions[max_idx])
                                
            if len(sentence) > 5:
                sentence = sentence[-5:]
        
        cv2.rectangle(image, (0,0), (640,40), (245, 117, 26), -1)
        cv2.putText(image, ' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
'''
for action in actions:
    for sequance in range(no_sequances):
        try:
            os.makedirs(os.path.join(DATA_PATH,action,str(sequance)))
        except:
            pass

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:    
    
    for action in actions:
        
        for sequance in range(no_sequances):
            
            for frame_num in range(sequance_length):
                
                ret, frame = cap.read()
                frame = cv2.flip(frame,1)
                image, results = mediapipe_detection(frame, holistic)

                draw_styled_landmarks(image, results)
                
                if frame_num == 0:
                    cv2.putText(image,'STARTING COLLECTION',(120,200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 4, cv2.LINE_AA)
                    cv2.putText(image,f'collecting frames for {action} video number {sequance}',(15,12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 1, cv2.LINE_AA)
                    cv2.waitKey(500)
                else:
                    cv2.putText(image,f'collecting frames for {action} video number {sequance}',(15,12),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 1, cv2.LINE_AA)                    
                
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequance),str(frame_num))
                np.save(npy_path,keypoints)
                cv2.imshow('Camera Video', image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
    cap.release()
    cv2.destroyAllWindows()
'''