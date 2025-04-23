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
import logging
from werkzeug.middleware.proxy_fix import ProxyFix
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Apply ProxyFix to handle reverse proxies (important for production)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# Configure CORS properly
CORS(app, resources={r"/*": {"origins": "*"}})

# Load models on startup
try:
    # Check if models exist
    if not os.path.exists('haarcascade_frontalface_default.xml'):
        logger.error("Face classifier model not found!")
        raise FileNotFoundError("haarcascade_frontalface_default.xml not found")
    
    if not os.path.exists('Emotion_final_model.h5'):
        logger.error("Emotion classifier model not found!")
        raise FileNotFoundError("Emotion_final_model.h5 not found")
    
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    # Use global TensorFlow settings for better performance
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.warning(f"GPU memory growth setting failed: {e}")
    
    classifier = load_model('Emotion_final_model.h5')
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    exit(1)

# Define emotion classes
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
    })

@lru_cache(maxsize=32)  # Cache recent predictions to improve performance
def predict_emotion(image_data_hash, image_array):
    """Make prediction with TensorFlow model and cache results"""
    with tf.device('/CPU:0'):  # Fallback to CPU if GPU not available
        predictions = classifier.predict(image_array, verbose=0)
    
    # Get emotion with highest probability
    emotion_idx = np.argmax(predictions[0])
    return {
        'label': class_labels[emotion_idx],
        'confidence': float(predictions[0][emotion_idx] * 100)
    }

@app.route('/detect_emotion', methods=['POST', 'OPTIONS'])
def detect_emotion():
    if request.method == 'OPTIONS':
        # Explicitly handle OPTIONS requests
        response = app.make_default_options_response()
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    if not request.json or 'image' not in request.json:
        logger.warning("No image provided in request")
        return jsonify({'success': False, 'message': 'No image provided'}), 400
    
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
        
        if frame is None or frame.size == 0:
            logger.warning("Decoded image is empty or invalid")
            return jsonify({
                'success': False,
                'message': 'Invalid image data. Please try a different image.'
            }), 400
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            logger.info("No face detected in image")
            return jsonify({
                'success': False,
                'message': 'No face detected. Please try with a clearer image showing your face.'
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
        
        # Create a hash for caching
        image_hash = hash(roi.tobytes())
        
        # Get prediction (cached if the same face was recently processed)
        result = predict_emotion(image_hash, roi)
        
        # Create face box coordinates (relative to image size)
        height, width = frame.shape[:2]
        face_box = {
            'x': int(x / width * 100),  # Convert to percentage
            'y': int(y / height * 100),
            'width': int(w / width * 100),
            'height': int(h / height * 100)
        }
        
        logger.info(f"Detected emotion: {result['label']} with confidence: {result['confidence']:.2f}%")
        
        return jsonify({
            'success': True,
            'emotion': result['label'],
            'confidence': result['confidence'],
            'face_box': face_box
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Error processing image. Please try again.'
        }), 500

if __name__ == '__main__':
    # Use environment variables for configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug, host='0.0.0.0', port=port)s
