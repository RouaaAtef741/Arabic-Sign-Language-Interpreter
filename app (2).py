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

        image_data = data.get('Image') if data else None

        if not image_data:
            return "Error: No image data received"

        labels_dict = {0: 'ا', 1: 'ب', 2: 'ت'}
        # Convert the image bytes to a PIL Image
        pil_image = PIL.Image.open(io.BytesIO(base64.b64decode(image_data)))
        pil_image = pil_image.resize((640, 480))

        # Convert the image to a NumPy array
        img_array = np.array(pil_image)

        # Convert RGB to BGR for OpenCV if needed
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Process the image with MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
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

            # Ensure the input has the correct shape
            input_data = np.array(data_aux).reshape(1, -1)

            # Perform prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

            return predicted_character
        else:
            return " "
    except FileNotFoundError as fnf_error:
        return f"Error: {fnf_error}"
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(host='172.20.10.13', port=5500, debug=True)
