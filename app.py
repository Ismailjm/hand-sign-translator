# importing necessary libraries
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64
import time

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')  # Set async_mode to 'eventlet' for asynchronous processing

# Load the alphabet and expression models
try:
    alphabet_model_dict = pickle.load(open('./model/model_alpha.p', 'rb'))
    alphabet_model = alphabet_model_dict['model']

    expression_model_dict = pickle.load(open('./model/model_expressions.p', 'rb'))
    expression_model = expression_model_dict['model']

    # logging.info("Models loaded successfully.")
except Exception as e:
    # logging.error("Failed to load model: %s", str(e))
    raise ValueError("Failed to load model: ", str(e))

# Initialize MediaPipe for hand detection with real-time processing
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Label dictionaries
alphabet_labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 
    17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 
    25: 'Z', 26: 'space', 27: 'delete'
}
expression_labels_dict = {0: "Hello ", 1: "Thank You ", 2: "Yes ", 3: "No ", 4: "I Love You "}

last_detection_time = 0  # To control detection frequency

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/draw_hand', methods=['POST'])
def draw_hand():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        # Check if the image_data is a valid base64 string
        if ',' not in image_data:
            return jsonify({'error': 'Invalid image data'}), 400

        # Decode base64 image to OpenCV format
        image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' part
        decoded_image = base64.b64decode(image_data)
        
        if not decoded_image:
            return jsonify({'error': 'Failed to decode base64 image'}), 400

        np_arr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=3, circle_radius=3)
                )
            
        # Assuming you generate an image to return, encode it as base64 again to send to the client
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{image_base64}"

        return jsonify({'image': image_data_url})

    except Exception as e:
        print(f"Error in /draw_hand: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_detection_time
    try:
        data = request.json
        image_data = data.get('image')
        selected_model = data.get('model', 'alphabet')  # Default to alphabet if not specified

        # Decode base64 image to OpenCV format
        image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' part
        decoded_image = base64.b64decode(image_data)
        np_arr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'prediction': 'Error: Failed to decode image'})

        current_time = time.time()
        predicted_character = ""

        # Perform detection every second
        if current_time - last_detection_time >= 1:
            last_detection_time = current_time
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = []
                    y_ = []
                    data_aux = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                        data_aux.append(hand_landmarks.landmark[i].y - min(y_))

                    # Adjust the number of features
                    if selected_model == 'alphabet':
                        expected_features = 42
                    else : 
                        expected_features = 84
                        
                    current_features = len(data_aux)

                    if current_features < expected_features:
                        data_aux.extend([0] * (expected_features - current_features))
                    elif current_features > expected_features:
                        data_aux = data_aux[:expected_features]

                    try:
                        # Choose the model and label dictionary based on the selected model
                        if selected_model == 'alphabet':
                            prediction = alphabet_model.predict([np.asarray(data_aux)])
                            predicted_character = alphabet_labels_dict[int(prediction[0])]
                        else:
                            prediction = expression_model.predict([np.asarray(data_aux)])
                            predicted_character = expression_labels_dict[int(prediction[0])]

                    except Exception as e:
                        predicted_character = ""
                        # logging.error(f"Prediction error: {e}")

                    break  # Only predict and draw for one hand

        return jsonify({'prediction': predicted_character})

    except Exception as e:
        # logging.error(f"Error in /process_frame: {e}")
        return jsonify({'prediction':predicted_character})

if __name__ == '__main__':
    # Use eventlet for production readiness and async support
    socketio.run(app, debug=True)
