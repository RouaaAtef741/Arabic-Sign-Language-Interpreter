import pickle
import cv2
import mediapipe as mp
import numpy as np
import arabic_reshaper #from here is where the arabic text being displayed correctly libraries were imported. 
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

#open the model for prediction.
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#open the camera to capture the stream.
cap = cv2.VideoCapture(0)

#initialize mediapipe.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#the labels now correspond with the arabic alphabet
labels_dict = {0:'ا', 1:'ب', 2:'ت', 3:'ث', 4:'ج', 5:'ح', 6:'خ', 7:'د', 8:'ذ', 9:'ر', 10:'ز', 11:'س', 12:'ش', 
               13:'ص', 14:'ض', 15:'ط', 16:'ظ', 17:'ع', 18:'غ', 19:'ف', 20:'ق', 21:'ك', 22:'ل', 23:'م', 24:'ن', 25:'ه', 26:'و', 
               27:'ي', 28:'ة', 29:'ء'}

#intialize the font path.
font_path = "C:\\Users\\justa\\OneDrive\\Desktop\\Grad-proj\\Noto_Naskh_Arabic\\NotoNaskhArabic-VariableFont_wght.ttf"
font = ImageFont.truetype(font_path, 32)

while True:
    data = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

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

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])

            predicted_label = labels_dict[int(prediction[0])]
            reshaped_label = arabic_reshaper.reshape(predicted_label)
            bidi_label = get_display(reshaped_label)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text((x1, y1 - 40), bidi_label, font=font, fill=(0, 0, 0))

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
