# # import warnings

# # # Suppress all UserWarning warnings
# # warnings.filterwarnings("ignore", category=UserWarning)

# # from flask import Flask, render_template, Response
# # from flask_socketio import SocketIO, emit
# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import pickle
# # import time

# # app = Flask(__name__)
# # socketio = SocketIO(app)

# # model_dict = pickle.load(open('./model/model.p', 'rb'))
# # model = model_dict['model']

# # mp_hands = mp.solutions.hands
# # mp_drawing = mp.solutions.drawing_utils
# # mp_drawing_styles = mp.solutions.drawing_styles

# # hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
# #                12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W',
# #                23: 'X', 24: 'Y', 25: 'Z', 27: 'delete', 26: 'space'}

# # cap = cv2.VideoCapture(0)

# # # Global variables to store the sentence and the last detected sign
# # sentence = ""
# # last_detected_sign = ""
# # last_detection_time = time.time()  # To track the last detection time

# # def generate_frames():
# #     global sentence, last_detected_sign, last_detection_time
# #     while True:
# #         data_aux = []
# #         x_ = []
# #         y_ = []

# #         ret, frame = cap.read()

# #         if not ret:
# #             break

# #         H, W, _ = frame.shape

# #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #         # current_time = time.time()
# #         # if current_time - last_detection_time >= 1:  # Check if 1 second has passed
# #         results = hands.process(frame_rgb)
# #         # last_detection_time = current_time  # Update the last detection time

# #         if results.multi_hand_landmarks:
# #             for hand_landmarks in results.multi_hand_landmarks:
# #                 mp_drawing.draw_landmarks(
# #                     frame,  # image to draw
# #                     hand_landmarks,  # model output
# #                     mp_hands.HAND_CONNECTIONS,  # hand connections
# #                     mp_drawing_styles.get_default_hand_landmarks_style(),
# #                     mp_drawing_styles.get_default_hand_connections_style())

# #             for hand_landmarks in results.multi_hand_landmarks:
# #                 for i in range(len(hand_landmarks.landmark)):
# #                     x = hand_landmarks.landmark[i].x
# #                     y = hand_landmarks.landmark[i].y

# #                     x_.append(x)
# #                     y_.append(y)

# #                 for i in range(len(hand_landmarks.landmark)):
# #                     x = hand_landmarks.landmark[i].x
# #                     y = hand_landmarks.landmark[i].y
# #                     data_aux.append(x - min(x_))
# #                     data_aux.append(y - min(y_))

# #             expected_features = 42
# #             current_features = len(data_aux)

# #             if current_features < expected_features:
# #                 data_aux.extend([0] * (expected_features - current_features))
# #             elif current_features > expected_features:
# #                 data_aux = data_aux[:expected_features]

# #             x1 = int(min(x_) * W) - 10
# #             y1 = int(min(y_) * H) - 10

# #             x2 = int(max(x_) * W) - 10
# #             y2 = int(max(y_) * H) - 10

# #             prediction = model.predict([np.asarray(data_aux)])

# #             predicted_character = labels_dict[int(prediction[0])]

# #             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
# #             cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3,
# #                         cv2.LINE_AA)

# #             # Update the sentence and print it only if the detected sign is different from the last one
# #             # if predicted_character != last_detected_sign:
# #             #     last_detected_sign = predicted_character
# #             current_time = time.time()
# #             if current_time - last_detection_time >= 1:  # Check if 1 second has passed
# #                 last_detection_time = current_time  # Update the last detection time
# #                 if predicted_character == 'delete':
# #                     sentence = sentence[:-1]
# #                 elif predicted_character == 'space':
# #                     sentence += ' '
# #                 else:
# #                     sentence += predicted_character
# #                 print(f"Current sentence: {sentence}")

# #                 # Emit the updated sentence to the frontend
# #                 socketio.emit('update_sentence', {'sentence': sentence})

# #         ret, buffer = cv2.imencode('.jpg', frame)
# #         frame = buffer.tobytes()

# #         yield (b'--frame\r\n'
# #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # @app.route('/get_sentence')
# # def get_sentence():
# #     global current_sentence
# #     print(f"Current sentence: {current_sentence}")  # Add this line to print to the console
# #     return {'sentence': current_sentence}


# # if __name__ == '__main__':
# #     socketio.run(app, debug=True)
    

import warnings
from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import pickle
import base64

# Suppress all UserWarning warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
socketio = SocketIO(app)

# Load the hand sign model
model_dict = pickle.load(open('./model/model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map predicted labels to letters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
               11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
               21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 27: 'delete', 26: 'space'}

# Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('index.html')
import psutil

@app.route('/monitor')
def monitor_resources():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    return jsonify({
        'cpu_percent': cpu_percent,
        'memory_used': memory_info.used,
        'memory_percent': memory_info.percent
    })
    
# Endpoint to process the image sent from the client-side webcam
@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    image_data = data['image']

    # Decode base64 image to OpenCV format
    image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' part
    decoded_image = base64.b64decode(image_data)
    np_arr = np.frombuffer(decoded_image, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Process the frame to detect hand landmarks
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

            expected_features = 42
            current_features = len(data_aux)

            if current_features < expected_features:
                data_aux.extend([0] * (expected_features - current_features))
            elif current_features > expected_features:
                data_aux = data_aux[:expected_features]

            # Predict the hand sign
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            return jsonify({'prediction': predicted_character})

    # In case no hand landmarks are detected
    return jsonify({'prediction': ''})

if __name__ == '__main__':
    socketio.run(app, debug=True)


# import warnings
# from flask import Flask, request, render_template, jsonify, Response
# from flask_socketio import SocketIO
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import base64

# # Suppress all UserWarning warnings
# warnings.filterwarnings("ignore", category=UserWarning)

# app = Flask(__name__)
# socketio = SocketIO(app)

# # Load the hand sign model
# model_dict = pickle.load(open('./model/model.p', 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe for hand detection
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# # Dictionary to map predicted labels to letters
# labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 
#                11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
#                21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 27: 'delete', 26: 'space'}

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_frame', methods=['POST'])
# def process_frame():
#     data = request.json
#     image_data = data['image']

#     # Decode base64 image to OpenCV format
#     image_data = image_data.split(',')[1]  # Remove the 'data:image/jpeg;base64,' part
#     decoded_image = base64.b64decode(image_data)
#     np_arr = np.frombuffer(decoded_image, np.uint8)

#     if np_arr.size == 0:
#         return jsonify({'error': 'Image data is empty'}), 400

#     frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#     if frame is None:
#         return jsonify({'error': 'Failed to decode image'}), 400

#     # Process the frame to detect hand landmarks
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
    
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Draw hand landmarks and connections
#             mp_drawing.draw_landmarks(
#                 frame,  # image to draw on
#                 hand_landmarks,  # model output
#                 mp_hands.HAND_CONNECTIONS,  # hand connections
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
        
#         # Encode the frame to base64
#         _, buffer = cv2.imencode('.jpg', frame)
#         img_str = base64.b64encode(buffer).decode('utf-8')
#         return jsonify({'image': 'data:image/jpeg;base64,' + img_str})
    
#     # In case no hand landmarks are detected
#     return jsonify({'image': ''})

# if __name__ == '__main__':
#     socketio.run(app, debug=True)
