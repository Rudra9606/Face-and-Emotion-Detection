from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

app = Flask(__name__)

# Debug: Print working directory and file structure
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
if os.path.exists('Models'):
    print(f"Files in Models/: {os.listdir('Models')}")
if os.path.exists('Harcascade'):
    print(f"Files in Harcascade/: {os.listdir('Harcascade')}")

# Load the model
model = None
try:
    model_path = os.path.abspath('Models/model_v_47.hdf5')
    print(f"Attempting to load model from: {model_path}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    if os.path.exists(model_path):
        print(f"Model file size: {os.path.getsize(model_path)} bytes")
        # Load with safe=False to handle older model formats
        try:
            from tensorflow.keras.saving import load_model as keras_load_model
            model = keras_load_model(model_path, safe_mode=False)
        except:
            # Fallback if safe_mode parameter doesn't exist
            model = load_model(model_path)
        print(f"✓ Model loaded successfully")
    else:
        print(f"✗ Model file not found at {model_path}")
except Exception as e:
    print(f"✗ Error loading model: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Load the face cascade
face_cascade = None
try:
    cascade_path = os.path.abspath('Harcascade/haarcascade_frontalface_default.xml')
    print(f"Attempting to load cascade from: {cascade_path}")
    print(f"Cascade file exists: {os.path.exists(cascade_path)}")
    if os.path.exists(cascade_path):
        print(f"Cascade file size: {os.path.getsize(cascade_path)} bytes")
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print(f"✗ Cascade classifier is empty after loading")
            face_cascade = None
        else:
            print(f"✓ Cascade loaded successfully")
    else:
        print(f"✗ Cascade file not found at {cascade_path}")
except Exception as e:
    print(f"✗ Error loading cascade: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Define emotion labels
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering template: {e}")
        error_msg = f"Model loaded: {model is not None}, Cascade loaded: {face_cascade is not None}"
        return f"<h1>Emotion Detection API</h1><p>{error_msg}</p>", 500

@app.route('/api/status')
def status():
    """Debug endpoint to check if models are loaded"""
    import glob
    
    model_files = glob.glob('Models/*') if os.path.exists('Models') else []
    cascade_files = glob.glob('Harcascade/*') if os.path.exists('Harcascade') else []
    
    return jsonify({
        'model_loaded': model is not None,
        'cascade_loaded': face_cascade is not None,
        'working_directory': os.getcwd(),
        'model_exists': os.path.exists('Models/model_v_47.hdf5'),
        'cascade_exists': os.path.exists('Harcascade/haarcascade_frontalface_default.xml'),
        'models_directory_exists': os.path.exists('Models'),
        'harcascade_directory_exists': os.path.exists('Harcascade'),
        'files_in_models': model_files,
        'files_in_harcascade': cascade_files,
        'all_files_in_cwd': os.listdir('.')
    })

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
