import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

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