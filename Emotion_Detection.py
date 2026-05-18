import argparse

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

class_labels={0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

def build_model():
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Conv2D(64, (5, 5), activation='relu', padding='valid'),
        Conv2D(64, (5, 5), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.4),
        Conv2D(128, (3, 3), activation='relu', padding='valid'),
        Conv2D(128, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax'),
    ])
    model.load_weights('./Models/model_v_47.hdf5')
    return model


def predict_face(classifier, face_gray):
    roi = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi.astype('float32') / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = classifier.predict(roi, verbose=0)[0]
    return class_labels[int(preds.argmax())], preds


def run_on_image(classifier, image_path):
    face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        faces = [(0, 0, img.shape[1], img.shape[0])]

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label, _ = predict_face(classifier, gray[y:y + h, x:x + w])
        label_position = (x + int(w / 2), max(0, y - 10))
        cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f'{image_path}: {label}')

    output_path = './Test_Images/annotated_output.jpg'
    cv2.imwrite(output_path, img)
    print(f'Saved annotated image to {output_path}')


def run_on_webcam(classifier):
    face_classifier = cv2.CascadeClassifier('./Harcascade/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Could not open webcam')

    while True:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            label, _ = predict_face(classifier, gray[y:y + h, x:x + w])
            label_position = (x + int(w / 2), abs(y - 10))
            cv2.putText(img, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', img)
        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Face and emotion detection')
    parser.add_argument('--image', help='Run detection on a single image instead of the webcam')
    args = parser.parse_args()

    classifier = build_model()
    if args.image:
        run_on_image(classifier, args.image)
    else:
        run_on_webcam(classifier)


if __name__ == '__main__':
    main()


