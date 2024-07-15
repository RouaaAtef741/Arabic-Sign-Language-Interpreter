import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        for hand_id in range(2):  
            data_aux = []

            x_ = []
            y_ = []

            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:  
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y

                        x_.append(x)
                        y_.append(y)

                    min_x = min(x_)
                    min_y = min(y_)
                    for x, y in zip(x_, y_):
                        data_aux.append(x - min_x)
                        data_aux.append(y - min_y)

                    data.append(data_aux)
                    labels.append(dir_)

hands.close()

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
