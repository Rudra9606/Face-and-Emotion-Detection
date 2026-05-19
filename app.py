from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

# Load the model
try:
    model = load_model('Models/model_v_47.hdf5')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the face cascade
try:
    face_cascade = cv2.CascadeClassifier('Harcascade/haarcascade_frontalface_default.xml')
except Exception as e:
    print(f"Error loading cascade: {e}")
    face_cascade = None

# Define emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

@app.route('/api/detect', methods=['POST'])
def detect_emotion():
    if model is None or face_cascade is None:
        return jsonify({'error': 'Model or cascade classifier not loaded'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Read image file
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # For simplicity, we'll only process the first face found
            (fX, fY, fW, fH) = faces[0]
            
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = model.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            return jsonify({'emotion': label, 'probability': float(emotion_probability)})
        else:
            return jsonify({'error': 'No face detected'})

@app.route('/')
def index():
    return "<h1>Emotion Detection API</h1><p>Use the /api/detect endpoint to post an image.</p>"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
