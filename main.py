import os
import cv2
import dlib
import numpy as np
import sqlite3
from imutils import face_utils
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.camera import Camera

# Load Dlib's pre-trained face detector and the 68-point shape predictor model
detector = dlib.get_frontal_face_detector()
predictor_path = 'E:/NIT-W/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

DB_PATH = 'face_data.db'

# Ensure the database and table creation
def create_database():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create table for facial landmarks
    c.execute('''
    CREATE TABLE IF NOT EXISTS landmarks (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        part TEXT NOT NULL,
        x INTEGER NOT NULL,
        y INTEGER NOT NULL
    )
    ''')
    
    # Create table for facial distances
    c.execute('''
    CREATE TABLE IF NOT EXISTS distances (
        id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        eye_distance REAL NOT NULL,
        nose_mouth_distance REAL NOT NULL,
        left_eye_nose_distance REAL NOT NULL,
        right_eye_nose_distance REAL NOT NULL,
        left_eye_mouth_distance REAL NOT NULL,
        right_eye_mouth_distance REAL NOT NULL
    )
    ''')
    
    conn.commit()
    conn.close()

# Function to extract facial landmarks and calculate distances
def extract_facial_landmarks_and_distances(image_path):
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 1)
        distances = []
        landmarks_list = []

        for (i, rect) in enumerate(faces):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye_center = np.mean(shape[36:42], axis=0)
            right_eye_center = np.mean(shape[42:48], axis=0)
            nose_tip = shape[30]
            mouth_center = np.mean(shape[48:68], axis=0)

            eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
            nose_mouth_distance = np.linalg.norm(nose_tip - mouth_center)
            left_eye_nose_distance = np.linalg.norm(left_eye_center - nose_tip)
            right_eye_nose_distance = np.linalg.norm(right_eye_center - nose_tip)
            left_eye_mouth_distance = np.linalg.norm(left_eye_center - mouth_center)
            right_eye_mouth_distance = np.linalg.norm(right_eye_center - mouth_center)

            distances.extend([eye_distance, nose_mouth_distance,
                              left_eye_nose_distance, right_eye_nose_distance,
                              left_eye_mouth_distance, right_eye_mouth_distance])

            landmarks_list = [(f'face_{i}', part, int(x), int(y)) for part, (x, y) in zip(range(68), shape)]

        return distances, landmarks_list
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return [], []

# Function to store features in the database
def store_features_in_db(name, features, landmarks):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Insert distances
    c.execute('''
    INSERT INTO distances (
        label, eye_distance, nose_mouth_distance, left_eye_nose_distance,
        right_eye_nose_distance, left_eye_mouth_distance, right_eye_mouth_distance
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (name, *features))
    
    # Insert landmarks
    c.executemany('''
    INSERT INTO landmarks (label, part, x, y) VALUES (?, ?, ?, ?)
    ''', [(name, part, x, y) for label, part, x, y in landmarks])

    conn.commit()
    conn.close()

# Function to authenticate face using extracted features
def authenticate_face(image_path):
    extracted_features, _ = extract_facial_landmarks_and_distances(image_path)
    if not extracted_features:
        print("No face detected or error in feature extraction.")
        return False

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
    SELECT eye_distance, nose_mouth_distance, left_eye_nose_distance, right_eye_nose_distance, left_eye_mouth_distance, right_eye_mouth_distance
    FROM distances
    ''')
    stored_features = c.fetchall()
    conn.close()

    threshold = 5.0
    for stored in stored_features:
        stored = np.array(stored)
        similarity = np.linalg.norm(np.array(extracted_features) - stored)
        if similarity < threshold:
            print("Face authenticated. Match found in database.")
            return True

    print("No matching face found in the database.")
    return False

class FaceRecognitionApp(App):
    def build(self):
        create_database()

        self.layout = BoxLayout(orientation='vertical')

        self.label = Label(text='Face Recognition App', font_size=30)
        self.layout.add_widget(self.label)

        self.camera = Camera(resolution=(640, 480), play=True)
        self.layout.add_widget(self.camera)

        self.capture_button = Button(text='Capture Image', on_release=self.capture_image)
        self.layout.add_widget(self.capture_button)

        self.register_button = Button(text='Register Face', on_release=self.register_face)
        self.layout.add_widget(self.register_button)

        self.authenticate_button = Button(text='Authenticate Face', on_release=self.authenticate_face)
        self.layout.add_widget(self.authenticate_button)

        self.img_display = Image(source='path_to_default_image.png')
        self.layout.add_widget(self.img_display)

        self.result_label = Label(text='Result: ', font_size=20)
        self.layout.add_widget(self.result_label)

        self.name_input = TextInput(hint_text='Enter Name for Registration')
        self.layout.add_widget(self.name_input)

        return self.layout

    def capture_image(self, *args):
        self.image_path = 'captured_image.png'
        self.camera.export_to_png(self.image_path)
        self.img_display.source = self.image_path
        self.result_label.text = 'Image captured. Ready for authentication or registration.'

    def register_face(self, *args):
        name = self.name_input.text.strip()
        if not name:
            self.result_label.text = 'Please enter a name for registration.'
            return

        if os.path.exists(self.image_path):
            features, landmarks = extract_facial_landmarks_and_distances(self.image_path)
            if features:
                store_features_in_db(name, features, landmarks)
                self.result_label.text = f'{name} registered successfully.'
            else:
                self.result_label.text = 'Face extraction failed. Try again.'
        else:
            self.result_label.text = 'No captured image found. Capture an image first.'

    def authenticate_face(self, *args):
        if os.path.exists(self.image_path):
            is_authenticated = authenticate_face(self.image_path)
            self.result_label.text = f"Result: {'Authenticated' if is_authenticated else 'Not Authenticated'}"
        else:
            self.result_label.text = 'No captured image found. Capture an image first.'

if __name__ == '__main__':
    FaceRecognitionApp().run()
