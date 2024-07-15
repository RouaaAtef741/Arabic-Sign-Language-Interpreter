#NOTE: I've connected the following script using networking method which means for this code to work both front-end and back-end NEED to be connected on the same 
#internet connection. The speed of the internet will also play a factor in the speed of the response between devices.


#This script takes the image from the front end in the form of string and converts it back to bytes and preforms the prediction and sends that back.
from flask import Flask, request
import sys
import base64
import os
import glob
import io
import pickle
import cv2
import numpy as np
import PIL.Image
import mediapipe as mp

#so that the output supports arabic output and you dont need arabic reshaper since arabic reshaper doesn't allow the letters to flow and connect how they normally would in arabic text.
sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def main():
    try:
        data = request.json
        #print(f"Received data: {data}")

        #correpond the dictionary keys to te arabic alphabet
        labels_dict = {0:'ا', 1:'ب', 2:'ت', 3:'ث', 4:'ج', 5:'ح', 6:'خ', 7:'د', 8:'ذ', 9:'ر', 10:'ز', 11:'س', 12:'ش', 
               13:'ص', 14:'ض', 15:'ط', 16:'ظ', 17:'ع', 18:'غ', 19:'ف', 20:'ق', 21:'ك', 22:'ل', 23:'م', 24:'ن', 25:'ه', 26:'و', 
               27:'ي', 28:'ة', 29:'ء'}

        image_data = data.get('Image') if data else None

        if not image_data:
            return "Error: No image data received"

        pil_image = PIL.Image.open(io.BytesIO(base64.b64decode(image_data)))
        pil_image = pil_image.resize((640, 480)) #if you know that the image that will be inputed is the same size as the image you trained on skip this
                        #Also check if the resolution of your image matches this one if not this can cause accuracy issues.

        img_array = np.array(pil_image)

        # Convert RGB to BGR for opencv
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Convert for mediapipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        #this code detects only one hand and will return an error if two hands are present in the image.
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            x_ = []
            y_ = []

        
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            input_data = np.array(data_aux).reshape(1, -1)

            # Perform prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

            return predicted_character
        else:
            #I have assigned the space bar to no hands detected if you made a different character change it to:
            # return "No hand landmarks detected!"
            return " "
    except FileNotFoundError as fnf_error:
        return f"Error: {fnf_error}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    #The IPv4 can be found out by opening the cnd and typing "ipconfig"
    #the IPv4 also changes when you connect to a different network so watch out for that.
    app.run(host='write your IPv4 address', port=5500, debug=True)
