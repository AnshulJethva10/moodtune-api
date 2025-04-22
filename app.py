from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import base64
import os
import io
from PIL import Image

app = Flask(__name__)
CORS(app, origins=["*"])  # Enable CORS for all routes

# Load face detector
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load pre-trained model
classifier = load_model('Emotion_final_model.h5')

# Define emotion classes
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_emotion', methods=['POST', 'OPTIONS'])
def detect_emotion():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get image data from request
        image_data = request.json['image']
        # Remove data:image/png;base64, prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'message': 'No face detected'
            })
        
        # Process the first face found
        x, y, w, h = faces[0]
        
        # Extract face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Preprocess the image
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make prediction using TensorFlow
        with tf.device('/CPU:0'):  # Fallback to CPU if GPU not available
            preds = classifier.predict(roi, verbose=0)
        
        # Get emotion with highest probability
        label = class_labels[np.argmax(preds[0])]
        confidence = float(np.max(preds[0]) * 100)
        
        # Create face box coordinates (relative to image size)
        height, width = frame.shape[:2]
        face_box = {
            'x': int(x / width * 100),  # Convert to percentage
            'y': int(y / height * 100),
            'width': int(w / width * 100),
            'height': int(h / height * 100)
        }
        
        return jsonify({
            'success': True,
            'emotion': label,
            'confidence': confidence,
            'face_box': face_box
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

