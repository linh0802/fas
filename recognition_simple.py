import numpy as np
from deepface import DeepFace
import h5py
import pickle
import cv2
import os

class RecognitionSimple:
    def __init__(self, model_path='models/train_FN.h5', threshold=0.65):
        self.FACE_RECOGNITION_THRESHOLD = threshold
        self.train_data, self.label_encoder = self._load_training_data(model_path)
        # Dùng cascade mặc định của OpenCV để phát hiện mặt
        haar_path = os.path.join(os.path.dirname(__import__('cv2').__file__), 'data', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haar_path)

    def _load_training_data(self, model_path):
        try:
            with h5py.File(model_path, 'r') as f:
                train_data = {'embeddings': np.array(f['embeddings']), 'labels': np.array(f['labels'])}
            with open(model_path.replace('.h5', '_label_encoder.pkl'), 'rb') as f_le:
                label_encoder = pickle.load(f_le)
            return train_data, label_encoder
        except Exception as e:
            print(f"Lỗi tải dữ liệu training: {e}")
            return None, None

    def get_face_embedding(self, face_img):
        try:
            # Phát hiện khuôn mặt trước khi lấy embedding
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                print("Không phát hiện khuôn mặt trong ảnh.")
                return None, "Không phát hiện khuôn mặt"
            embedding = DeepFace.represent(
                face_img,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='skip',
                align=False,
                normalization='base'
            )[0]['embedding']
            return embedding, "Nhận diện thành công"
        except Exception as e:
            print(f"Lỗi lấy embedding: {e}")
            return None, f"Lỗi lấy embedding: {e}" 