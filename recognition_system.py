import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import logging
import threading
import queue
from datetime import datetime
import time
from pyzbar import pyzbar
import gspread
from google.oauth2.service_account import Credentials
import hashlib
import h5py
import torch
import torch.nn.functional as F
import urllib.request
import mediapipe as mp
import signal
import sys
import json
from mediapipe.python.solutions.face_detection import FaceDetection
from pir_sensor import PIRSensor

# Tắt warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Biến global
running = True
known_qr_codes = set()
face_queue = queue.Queue(maxsize=10)
known_persons = {}
embedding_cache = {}
DETECT_QR_INTERVAL = 3.0
FRAME_PROCESS_INTERVAL = 0.1
stats = {'faces_detected': 0, 'qrs_detected': 0, 'start_time': time.time()}
DETECTION_RESULT = None
DETECTION_TIMESTAMP = time.time()  # Khởi tạo giá trị ban đầu

# Thêm các thông số mới
FACE_RECOGNITION_THRESHOLD = 0.7  # Tăng ngưỡng nhận diện lên 60%
ANTISPOOF_THRESHOLD = 0.7  # Ngưỡng anti-spoofing (0.5 = 50% - ngưỡng mặc định của Fasnet)
MIN_FACE_SIZE = 50  # Kích thước tối thiểu của khuôn mặt
FACE_MARGIN = 0.3    # Margin khi cắt khuôn mặt (30%)
kalman_filters = {}  # Dictionary để lưu trữ các bộ lọc Kalman
FACE_DETECTION_INTERVAL = 0.1  # Khoảng thời gian giữa các lần phát hiện khuôn mặt
FACE_RECOGNITION_INTERVAL = 1.0  # Khoảng thời gian giữa các lần nhận diện
DEBUG_IMAGE_INTERVAL = 1.0  # Khoảng thời gian giữa các lần lưu ảnh debug

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        # Giảm process noise để làm mượt hơn
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0], 
                                              [0, 0, 1, 0], 
                                              [0, 0, 0, 1]], np.float32) * 0.01
        self.last_measurement = None
        self.last_prediction = None
        self.skip_count = 0  # Thêm biến đếm để bỏ qua một số frame

    def update(self, measurement):
        if self.last_measurement is None:
            self.last_measurement = measurement
            self.last_prediction = measurement
            return measurement

        # Bỏ qua một số frame để tăng tốc
        self.skip_count += 1
        if self.skip_count % 2 != 0:  # Chỉ cập nhật mỗi 2 frame
            return self.last_prediction

        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        self.last_measurement = measurement
        self.last_prediction = prediction
        return prediction

def get_kalman_filter(face_id):
    if face_id not in kalman_filters:
        kalman_filters[face_id] = KalmanFilter()
    return kalman_filters[face_id]

# Lớp Fasnet cho anti-spoofing
class Fasnet:
    def __init__(self):
        logging.info("Khởi tạo Fasnet...")
        try:
            import torch
        except Exception as err:
            logging.error("Cần cài đặt torch: `pip install torch`")
            raise ValueError("Cần cài đặt torch") from err

        home = os.path.expanduser("~")
        device = torch.device("cpu")
        self.device = device

        model_path = f"{home}/.deepface/weights/2.7_80x80_MiniFASNetV2.pth"
        model_url = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"

        try:
            self.download_model(model_path, model_url)
        except Exception as e:
            logging.error(f"Lỗi tải mô hình MiniFASNetV2: {str(e)}")
            raise

        try:
            from deepface.models.spoofing import FasNetBackbone
            self.model = FasNetBackbone.MiniFASNetV2(conv6_kernel=(5, 5)).to(device)
            state_dict = torch.load(model_path, map_location=device)
            if next(iter(state_dict)).find("module.") >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                self.model.load_state_dict(new_state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.model.eval()
            logging.info("Đã tải mô hình MiniFASNetV2 thành công")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo mô hình MiniFASNetV2: {str(e)}")
            raise

    def download_model(self, file_path, url):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            logging.info(f"Đang tải mô hình từ {url}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                logging.info(f"Đã tải mô hình vào {file_path}")
            except Exception as e:
                logging.error(f"Lỗi tải mô hình: {str(e)}")
                raise
        else:
            logging.info(f"Mô hình đã tồn tại: {file_path}")

    def analyze(self, img, facial_area: tuple):
        """
        Phân tích khuôn mặt để phát hiện giả mạo
        Args:
            img: Ảnh đầu vào
            facial_area: Tọa độ khuôn mặt (x, y, w, h)
        Returns:
            is_real: True nếu là khuôn mặt thật
            score: Độ tin cậy của kết quả
        """
        try:
            x, y, w, h = facial_area
            
            # Kiểm tra kích thước khuôn mặt
            if w < 50 or h < 50:
                logging.warning(f"[ANTI-SPOOF] Khuôn mặt quá nhỏ: {w}x{h}")
                return False, 0.0
                
            # Cắt và resize khuôn mặt
            img = self.crop(img, (x, y, w, h), 2.7, 80, 80)
            
            # Chuyển đổi ảnh
            test_transform = self.Compose([self.ToTensor(self)])
            img = test_transform(img)
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            img = img.unsqueeze(0).to(self.device)

            # Phân tích
            with torch.no_grad():
                result = self.model.forward(img)
                result = F.softmax(result).cpu().numpy()

            # Lấy kết quả
            label = np.argmax(result)
            is_real = label == 1
            score = result[0][label]
            return is_real, score
            
        except Exception as e:
            logging.error(f"[ANTI-SPOOF] Lỗi phân tích: {str(e)}")
            return False, 0.0

    def to_tensor(self, pic):
        if pic.ndim == 2:
            pic = pic.reshape((pic.shape[0], pic.shape[1], 1))
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float()

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img

    class ToTensor:
        def __init__(self, outer):
            self.outer = outer
        def __call__(self, pic):
            return Fasnet.to_tensor(self.outer, pic)

    def _get_new_box(self, src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2
        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1
        return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h):
        src_h, src_w, _ = np.shape(org_img)
        left_top_x, left_top_y, right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)
        img = org_img[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]
        try:
            dst_img = cv2.resize(img, (out_w, out_h))
            return dst_img
        except Exception as e:
            logging.error(f"Lỗi resize ảnh: {str(e)}")
            return org_img

def ensure_dir(directory):
    os.makedirs(directory, exist_ok=True)

def setup_logging(log_filename=None, log_dir='logs', max_logs=5):
    try:
        ensure_dir(log_dir)
        log_files = [f for f in os.listdir(log_dir) if f.startswith('face_recognition_') and f.endswith('.log')]
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True)
        for old_log in log_files[max_logs:]:
            try:
                os.remove(os.path.join(log_dir, old_log))
            except Exception as e:
                logging.info(f"Lỗi khi xóa file log {old_log}: {str(e)}")
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'face_recognition_{current_time}.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('mediapipe').setLevel(logging.ERROR)
        logging.getLogger('PIL').setLevel(logging.ERROR)
        logging.info("="*50)
        logging.info(f"Bắt đầu ghi log vào file: {log_file}")
        logging.info("="*50)
        return log_file
    except Exception as e:
        logging.info(f"Lỗi thiết lập logging: {str(e)}")
        return None

def setup_google_sheets(sheet_name="Attendance", credentials_path='credentials/face-attendance.json'):
    try:
        if not os.path.exists(credentials_path):
            logging.error(f"Không tìm thấy file credentials: {credentials_path}")
            return None
            
        # Kiểm tra thư mục credentials
        if not os.path.exists('credentials'):
            os.makedirs('credentials')
            logging.error("Thư mục credentials không tồn tại. Đã tạo thư mục mới.")
            return None
            
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/spreadsheets"
        ]
        
        try:
            creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
            client = gspread.authorize(creds)
            
            # Thử mở spreadsheet
            try:
                sheet = client.open(sheet_name).worksheet("Attendance")
                logging.info("Kết nối Google Sheets thành công")
                return sheet
            except gspread.exceptions.SpreadsheetNotFound:
                logging.error(f"Không tìm thấy spreadsheet: {sheet_name}")
                return None
            except gspread.exceptions.WorksheetNotFound:
                logging.error(f"Không tìm thấy worksheet: Attendance")
                return None
                
        except Exception as e:
            logging.error(f"Lỗi xác thực Google Sheets: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Lỗi thiết lập Google Sheets: {str(e)}")
        return None

def clean_debug_files(debug_dir, max_files=10):
    try:
        files = sorted(os.listdir(debug_dir), key=lambda x: os.path.getmtime(os.path.join(debug_dir, x)))
        for f in files[:-max_files]:
            os.remove(os.path.join(debug_dir, f))
    except Exception as e:
        logging.error(f"Lỗi dọn dẹp thư mục {debug_dir}: {str(e)}")

def detect_faces(frame, mp_face_detection):
    try:
        # Chuyển đổi frame sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = []
        processed_faces = set()
        
        # Lấy kích thước frame
        h, w, _ = frame.shape
        
        # Tính toán ngưỡng kích thước khuôn mặt (1/4 frame thay vì 1/3)
        face_size_threshold = min(w, h) / 4
        
        # Thêm biến đếm khuôn mặt
        face_count = 0
        
        # Sử dụng model 0 cho khuôn mặt nhỏ (< 1/4 frame)
        mp_face_detection.model_selection = 0
        results = mp_face_detection.process(frame_rgb)
        
        if hasattr(results, "detections") and results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Tính toán tọa độ thực tế
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Giảm kích thước tối thiểu xuống 40x40
                if box_w < 40 or box_h < 40:
                    logging.debug(f"Bỏ qua khuôn mặt quá nhỏ: {box_w}x{box_h}")
                    continue
                
                # Nếu khuôn mặt lớn hơn 1/4 frame, bỏ qua model 0
                if box_w > face_size_threshold or box_h > face_size_threshold:
                    logging.debug(f"Bỏ qua khuôn mặt quá lớn cho model 0: {box_w}x{box_h}")
                    continue
                
                # Xử lý khuôn mặt với model 0 (khoảng cách xa)
                margin = 0.1  # 10% margin cho khuôn mặt xa
                margin_x = int(box_w * margin)
                margin_y = int(box_h * margin)
                
                # Điều chỉnh tọa độ với margin
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                box_w = min(w - x, box_w + 2 * margin_x)
                box_h = min(h - y, box_h + 2 * margin_y)
                
                # Kiểm tra tỷ lệ khung hình (nới lỏng điều kiện)
                aspect_ratio = box_w / box_h
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                    logging.debug(f"Bỏ qua khuôn mặt có tỷ lệ không phù hợp: {aspect_ratio:.2f}")
                    continue
                    
                # Kiểm tra trùng lặp
                face_key = f"{x}_{y}_{box_w}_{box_h}"
                if face_key in processed_faces:
                    logging.debug("Bỏ qua khuôn mặt trùng lặp")
                    continue
                processed_faces.add(face_key)
                
                # Lưu tọa độ gốc (không có margin)
                original_x = x + margin_x
                original_y = y + margin_y
                original_w = box_w - 2 * margin_x
                original_h = box_h - 2 * margin_y
                
                detections.append({
                    'facial_area': {
                        'x': original_x,
                        'y': original_y,
                        'w': original_w,
                        'h': original_h
                    },
                    'model': 0
                })
                
                logging.debug(f"Phát hiện khuôn mặt (model 0): x={original_x}, y={original_y}, w={original_w}, h={original_h}")
                
                # Lưu ảnh debug với index
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                face_roi = debug_frame[y:y+box_h, x:x+box_w]
                timestamp = int(time.time())
                cv2.imwrite(f"debug_faces/detection_{timestamp}_{face_count}.jpg", face_roi)
                
                face_count += 1
        
        # Sử dụng model 1 cho khuôn mặt lớn (> 1/4 frame)
        mp_face_detection.model_selection = 1
        results = mp_face_detection.process(frame_rgb)
        
        if hasattr(results, "detections") and results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Tính toán tọa độ thực tế
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                # Nếu khuôn mặt quá nhỏ, bỏ qua model 1
                if box_w < face_size_threshold and box_h < face_size_threshold:
                    logging.debug(f"Bỏ qua khuôn mặt quá nhỏ cho model 1: {box_w}x{box_h}")
                    continue
                
                # Xử lý khuôn mặt với model 1 (khoảng cách gần)
                margin = 0.2  # 20% margin cho khuôn mặt gần
                margin_x = int(box_w * margin)
                margin_y = int(box_h * margin)
                
                # Điều chỉnh tọa độ với margin
                x = max(0, x - margin_x)
                y = max(0, y - margin_y)
                box_w = min(w - x, box_w + 2 * margin_x)
                box_h = min(h - y, box_h + 2 * margin_y)
                
                # Kiểm tra tỷ lệ khung hình (nới lỏng điều kiện)
                aspect_ratio = box_w / box_h
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                    logging.debug(f"Bỏ qua khuôn mặt có tỷ lệ không phù hợp: {aspect_ratio:.2f}")
                    continue
                    
                # Kiểm tra trùng lặp
                face_key = f"{x}_{y}_{box_w}_{box_h}"
                if face_key in processed_faces:
                    logging.debug("Bỏ qua khuôn mặt trùng lặp")
                    continue
                processed_faces.add(face_key)
                
                # Lưu tọa độ gốc (không có margin)
                original_x = x + margin_x
                original_y = y + margin_y
                original_w = box_w - 2 * margin_x
                original_h = box_h - 2 * margin_y
                
                detections.append({
                    'facial_area': {
                        'x': original_x,
                        'y': original_y,
                        'w': original_w,
                        'h': original_h
                    },
                    'model': 1
                })
                
                logging.debug(f"Phát hiện khuôn mặt (model 1): x={original_x}, y={original_y}, w={original_w}, h={original_h}")
                
                # Lưu ảnh debug với index
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
                face_roi = debug_frame[y:y+box_h, x:x+box_w]
                timestamp = int(time.time())
                cv2.imwrite(f"debug_faces/detection_{timestamp}_{face_count}.jpg", face_roi)
                
                face_count += 1
        
        return detections
    except Exception as e:
        logging.error(f"Lỗi phát hiện khuôn mặt với MediaPipe: {str(e)}")
        return []

def preprocess_face(img, facial_area, target_size=(160, 160)):
    try:
        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
        
        # Cắt khuôn mặt
        face_img = img[y:y+h, x:x+w].copy()
        
        # Resize về kích thước chuẩn
        face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
        
        # Chuyển đổi màu BGR sang RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa về [0, 1] giống như trong train
        face_img = face_img.astype(np.float32)
        face_img = (face_img - face_img.min()) / (face_img.max() - face_img.min() + 1e-8)
        
        # Lưu ảnh debug với index
        debug_img = (face_img * 255).astype(np.uint8)
        timestamp = int(time.time())
        face_index = facial_area.get('index', 0)  # Lấy index từ facial_area
        cv2.imwrite(f"debug_faces/face_normalized_{timestamp}_{face_index}.jpg", 
                   cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
        return face_img
        
    except Exception as e:
        logging.error(f"Lỗi tiền xử lý khuôn mặt: {str(e)}")
        return None

def get_face_embedding(face_img):
    try:
        img_hash = hashlib.md5(face_img.tobytes()).hexdigest()
        if img_hash in embedding_cache:
            return embedding_cache[img_hash]
        embedding = DeepFace.represent(
            face_img,
            model_name='Facenet512',
            enforce_detection=False,
            detector_backend='skip',
            align=False,
            normalization='base'
        )[0]['embedding']
        embedding_cache[img_hash] = embedding
        return embedding
    except Exception as e:
        logging.error(f"Lỗi lấy embedding: {str(e)}")
        return None

def recognize_face_directly(frame, facial_area, fasnet, threshold=FACE_RECOGNITION_THRESHOLD):
    global DETECTION_TIMESTAMP
    
    try:
        # 1. Kiểm tra kích thước khuôn mặt
        if facial_area['w'] < MIN_FACE_SIZE or facial_area['h'] < MIN_FACE_SIZE:
            logging.warning(f"[RECOGNITION] Khuôn mặt quá nhỏ: w={facial_area['w']}, h={facial_area['h']}")
            return None, 0.0, 0.0, False, 0.0

        # 2. Kiểm tra anti-spoofing trước
        current_time = time.time()
        face_coords = (facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h'])
        is_real, antispoof_score = fasnet.analyze(frame, face_coords)
        DETECTION_TIMESTAMP = current_time        
        # Nếu là khuôn mặt giả, trả về ngay lập tức
        if not is_real or antispoof_score < ANTISPOOF_THRESHOLD:
            logging.warning(f"[RECOGNITION] Phát hiện khuôn mặt giả - Score: {antispoof_score:.4f}")
            return None, 0.0, 0.0, False, antispoof_score

        # 3. Tiền xử lý khuôn mặt
        face_img = preprocess_face(frame, facial_area)
        if face_img is None:
            logging.warning("[RECOGNITION] Không thể tiền xử lý khuôn mặt")
            return None, 0.0, 0.0, True, antispoof_score
            
        # 4. Tạo embedding
        target_embedding = get_face_embedding(face_img)
        if target_embedding is None:
            logging.warning("[RECOGNITION] Không thể lấy embedding từ khuôn mặt")
            return None, 0.0, 0.0, True, antispoof_score
            
        # 5. So sánh với embeddings trong train data
        if train_data is None:
            logging.error("train_data chưa được khởi tạo hoặc load thất bại.")
            return None, 0.0, 0.0, False, antispoof_score
        similarities = cosine_similarity([target_embedding], train_data['embeddings'])[0]
        best_match_index = np.argmax(similarities)
        confidence = similarities[best_match_index]        
        if confidence >= threshold:
            predicted_label = train_data['labels'][best_match_index]
            name = str(predicted_label)
            if 'label_encoder' in train_data:
                name = train_data['label_encoder'].inverse_transform([int(predicted_label)])[0]
            logging.info(f"[RECOGNITION] Đã nhận diện thành công: {name} (confidence={confidence:.4f}, anti-spoof={antispoof_score:.4f})")
            return name, confidence, 0.0, True, antispoof_score
        else:
            logging.warning(f"[RECOGNITION] Khuôn mặt không vượt qua ngưỡng độ tin cậy: {confidence:.4f} < {threshold}")
            return None, confidence, 0.0, True, antispoof_score
            
    except Exception as e:
        logging.error(f"[RECOGNITION] Lỗi nhận diện: {str(e)}", exc_info=True)
        return None, 0.0, 0.0, False, 0.0

def save_to_google_sheets(sheet, data_type, data, confidence=None, max_retries=3, offline_syncer=None):
    if not sheet:
        logging.info("[OFFLINE] Google Sheets không được thiết lập, lưu offline!")
        if offline_syncer:
            offline_syncer.add_attendance(data_type, data, confidence)
        return
    for attempt in range(max_retries):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row = [timestamp, str(data), data_type, f"{confidence:.4f}" if confidence else ""]
            sheet.append_row(row)
            logging.info(f"[ONLINE] Đã lưu Google Sheets: {data_type} - {data}")
            if offline_syncer:
                offline_syncer.sync_to_google_sheets(sheet)
            return
        except Exception as e:
            logging.info(f"[ONLINE] Lỗi lưu Google Sheets (thử {attempt+1}/{max_retries}): {str(e)}")
            time.sleep(2)
    # Nếu lưu online thất bại, lưu offline
    if offline_syncer:
        offline_syncer.add_attendance(data_type, data, confidence)

def save_debug_image(img, state, name=None):
    # state: 'pre_recognition', 'recognized', ...
    ensure_dir(f'debug_faces/{state}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if name:
        filename = f'debug_faces/{state}/{name}_{timestamp}.jpg'
    else:
        filename = f'debug_faces/{state}/{timestamp}.jpg'
    cv2.imwrite(filename, img)
    logging.debug(f"Đã lưu ảnh debug: {filename}")

def detect_qr_code(frame, last_qr_time, sheet):
    global known_qr_codes
    if time.time() - last_qr_time < DETECT_QR_INTERVAL:
        return frame, last_qr_time
    try:
        qr_frame = frame.copy()
        gray = cv2.cvtColor(qr_frame, cv2.COLOR_BGR2GRAY)
        try:
            detected = pyzbar.decode(gray)
            if detected:
                for qr in detected:
                    try:
                        # Lấy tọa độ và kích thước của mã QR
                        (x, y, w, h) = qr.rect
                        
                        # Vẽ khung màu đỏ xung quanh mã QR
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        
                        qr_data = qr.data.decode('utf-8')
                        if qr_data in known_qr_codes:
                            continue
                        known_qr_codes.add(qr_data)
                        logging.info(f"Phát hiện mã QR: {qr_data}")
                        save_to_google_sheets(sheet, "QR", qr_data)
                        stats['qrs_detected'] += 1
                    except Exception as e:
                        logging.error(f"Lỗi xử lý QR code: {str(e)}", exc_info=True)
                        continue
        except Exception as e:
            logging.error(f"Lỗi phát hiện QR: {str(e)}", exc_info=True)
        return frame, time.time()
    except Exception as e:
        logging.error(f"Lỗi phát hiện QR: {str(e)}", exc_info=True)
        return frame, last_qr_time

def open_camera_multi_backend():
    try:
        # Thử các backend camera khác nhau
        backends = [
            cv2.CAP_V4L2,  # Video4Linux2 (Linux)
            cv2.CAP_ANY   # Bất kỳ backend nào
        ]
        
        for backend in backends:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                # Cấu hình tối ưu cho Raspberry Pi
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer size
                
                # Kiểm tra xem cấu hình có được áp dụng không
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                
                logging.info(f"Đã mở webcam thành công với backend {backend}")
                logging.info(f"Độ phân giải thực tế: {actual_width}x{actual_height}, FPS: {actual_fps}")
                return cap
                
        logging.error("Không thể mở webcam với bất kỳ backend nào")
        return None
    except Exception as e:
        logging.error(f"Lỗi khi mở webcam: {str(e)}")
        return None

def face_detection_thread(cap, mp_face_detection, fasnet, sheet, pir_sensor):
    global running
    try:
        logging.info("Bắt đầu thread phát hiện khuôn mặt và giám sát PIR")
        
        # Cấu hình window
        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        
        last_qr_time = time.time()
        last_clean_time = time.time()
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        # Bắt đầu ở trạng thái nghỉ để chờ tín hiệu đầu tiên từ cảm biến
        MOTION_INACTIVITY_TIMEOUT = 60 # Tạm dừng sau 60 giây không có chuyển động
        is_idle = True
        last_motion_time = 0 # Đặt về 0 để hệ thống vào trạng thái nghỉ ngay lập tức
        logging.info("Hệ thống khởi động ở trạng thái nghỉ, đang chờ chuyển động đầu tiên.")

        while running:
            # Kiểm tra cảm biến PIR
            if pir_sensor.is_motion():
                if is_idle:
                    logging.info("PIR: Phát hiện chuyển động, hệ thống hoạt động trở lại.")
                    is_idle = False
                last_motion_time = time.time()

            # Kiểm tra nếu không có chuyển động trong một khoảng thời gian
            if time.time() - last_motion_time > MOTION_INACTIVITY_TIMEOUT:
                if not is_idle:
                    logging.info(f"PIR: Không có chuyển động trong {MOTION_INACTIVITY_TIMEOUT} giây. Tạm dừng hệ thống.")
                    # Xóa hàng đợi khuôn mặt để không xử lý ảnh cũ khi hoạt động lại
                    with face_queue.mutex:
                        face_queue.queue.clear()
                    is_idle = True
                
                # Hiển thị màn hình chờ
                idle_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(idle_frame, "He thong tam dung. Dang cho chuyen dong...", 
                            (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                try:
                    cv2.imshow('Webcam', idle_frame)
                    key = cv2.waitKey(100) # Chờ 100ms
                    if key == ord('q'):
                        running = False
                except Exception as e:
                    logging.warning(f"Lỗi hiển thị màn hình chờ: {e}")

                time.sleep(0.5) # Giảm tải CPU khi hệ thống tạm dừng
                continue
            
            # --- Phần xử lý frame khi hệ thống hoạt động ---
            ret, frame = cap.read()
            if not ret or frame is None:
                logging.error("Không thể đọc frame từ webcam (ret=False hoặc frame=None)")
                time.sleep(0.1)
                continue
                
            current_time = time.time()
            if current_time - start_time > 1.0/30: # Giới hạn FPS
                frame_count += 1
            
                if frame_count >= 30:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    logging.debug(f"FPS hiện tại: {fps:.1f}")
                    frame_count = 0
                    start_time = end_time
                
                original_frame = frame.copy()
                original_frame, last_qr_time = detect_qr_code(original_frame, last_qr_time, sheet)
                
                detections = detect_faces(frame, mp_face_detection)
                logging.debug(f"Số khuôn mặt phát hiện được: {len(detections)}")
                
                for detection in detections:
                    facial_area = detection['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                        logging.info(f"BỎ QUA nhận diện: Khuôn mặt quá nhỏ (w={w}, h={h}), model={detection.get('model')}")
                        continue
                        
                    cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    try:
                        face_queue.put_nowait((frame, facial_area))
                    except queue.Full:
                        # Nếu hàng đợi đầy, bỏ cái cũ nhất, thêm cái mới nhất
                        face_queue.get_nowait()
                        face_queue.put_nowait((frame, facial_area))
                        
                cv2.putText(original_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                try:
                    cv2.imshow('Webcam', original_frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        logging.info("Người dùng yêu cầu thoát")
                        running = False
                except Exception as e:
                    logging.error(f"Lỗi hiển thị frame: {str(e)}")
                    
            if time.time() - last_clean_time > 3600: # Dọn dẹp mỗi giờ
                clean_debug_files('debug_faces')
                last_clean_time = time.time()
                
    except Exception as e:
        logging.error(f"Lỗi trong thread phát hiện khuôn mặt: {str(e)}", exc_info=True)
    finally:
        logging.info("Đóng thread phát hiện khuôn mặt")
        # Việc dọn dẹp cap và cv2 sẽ được xử lý trong hàm cleanup_resources

def face_recognition_thread(fasnet, sheet):
    global running
    last_recognition_time = {}  # Dictionary để theo dõi thời gian nhận diện
    
    logging.info("[RECOGNITION] Thread nhận diện khuôn mặt đã khởi động")
    
    while running:
        try:
            logging.debug("[RECOGNITION] Đang chờ khuôn mặt từ queue...")
            frame, facial_area = face_queue.get(timeout=1.0)
            current_time = time.time()
            
            # Kiểm tra xem có nên nhận diện khuôn mặt này không
            face_id = f"{facial_area['x']}_{facial_area['y']}"
            if face_id in last_recognition_time:
                if current_time - last_recognition_time[face_id] < 1.0:  # Chỉ nhận diện mỗi 1 giây
                    logging.debug(f"[RECOGNITION] Bỏ qua khuôn mặt {face_id} - quá sớm để nhận diện lại")
                    continue
            name, confidence, antispoof_confidence, is_unknown, antispoof_score = recognize_face_directly(frame, facial_area, fasnet)
            
            if name:
                logging.info(f"[RECOGNITION] Đã nhận diện thành công: {name} (confidence={confidence:.4f}, anti-spoof={antispoof_score:.4f})")
                if name not in known_persons:
                    known_persons[name] = datetime.now()
                    save_to_google_sheets(sheet, "Face", name, confidence)
                    stats['faces_detected'] += 1
                    logging.info(f"[ATTENDANCE] Đã điểm danh: {name}")
            else:         
                last_recognition_time[face_id] = current_time
                    
        except queue.Empty:
            logging.debug("[RECOGNITION] Queue trống, đang chờ...")
            continue
        except Exception as e:
            logging.error(f"[RECOGNITION] Lỗi trong thread nhận diện: {str(e)}", exc_info=True)

def clean_debug_directories():
    try:
        debug_dirs = ['debug_faces']
        for dir_name in debug_dirs:
            if os.path.exists(dir_name):
                for file in os.listdir(dir_name):
                    file_path = os.path.join(dir_name, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception:
                        pass
    except Exception:
        pass

def signal_handler(signum, frame):
    global running
    logging.info(f"Nhận tín hiệu {signal.Signals(signum).name}, đang dọn dẹp...")
    running = False
    # Khối finally trong main() sẽ xử lý việc dọn dẹp
    sys.exit(0)

def cleanup_resources(cap=None, pir_sensor=None):
    try:
        if cap and cap.isOpened():
            cap.release()
            logging.info("Đã giải phóng tài nguyên webcam.")
        if pir_sensor:
            pir_sensor.stop() # Sử dụng phương thức stop() của class
        cv2.destroyAllWindows()
        logging.info("Đã giải phóng tất cả tài nguyên.")
    except Exception as e:
        print(f"Lỗi khi dọn dẹp tài nguyên: {str(e)}")

def main():
    # Làm trống hoàn toàn file offline_attendance.json khi khởi động
    open('offline_attendance.json', 'w').close()
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        log_file = setup_logging()
        if log_file is None:
            logging.info("Không thể thiết lập logging. Chương trình sẽ thoát.")
            return
        clean_debug_directories()
        logging.info("Khởi động hệ thống nhận diện khuôn mặt")
        
        # Kiểm tra thư mục models
        if not os.path.exists('models'):
            logging.error("Không tìm thấy thư mục models")
            return
            
        # Kiểm tra file model
        if not os.path.exists('models/train_FN.h5'):
            logging.error("Không tìm thấy file models/train_FN.h5")
            return
        if not os.path.exists('models/train_FN_label_encoder.pkl'):
            logging.error("Không tìm thấy file models/train_FN_label_encoder.pkl")
            return
            
        global train_data
        train_data = None
        try:
            logging.info("Đang load dữ liệu train...")
            with h5py.File('models/train_FN.h5', 'r') as f:
                train_data = {
                    'embeddings': np.array(f['embeddings']).astype(np.float32),
                    'labels': np.array(f['labels']),
                    'embedding_model': f.attrs['embedding_model'],
                    'embedding_size': f.attrs['embedding_size']
                }
            with open('models/train_FN_label_encoder.pkl', 'rb') as f:
                train_data['label_encoder'] = pickle.load(f)
            norms = np.linalg.norm(train_data['embeddings'], axis=1, keepdims=True)
            train_data['embeddings'] = train_data['embeddings'] / norms
            logging.info(f"Đã load dữ liệu train: {len(np.unique(train_data['labels']))} nhãn")
            logging.info(f"Model embedding: {train_data['embedding_model']}, kích thước: {train_data['embedding_size']}")
        except Exception as e:
            logging.error(f"Lỗi load train_FN.h5: {str(e)}", exc_info=True)
            return
            
        # Kiểm tra thư mục credentials
        if not os.path.exists('credentials'):
            logging.warning("Không tìm thấy thư mục credentials, tạo thư mục mới")
            os.makedirs('credentials')
            
        sheet = setup_google_sheets()
        if sheet is None:
            logging.warning("Không thể kết nối Google Sheets. Hệ thống sẽ chạy mà không lưu dữ liệu.")
            
        pir_sensor = None
        cap = None
        try:
            logging.info("Đang khởi tạo cảm biến PIR...")
            pir_sensor = PIRSensor(pin_signal=17)
            pir_sensor.start()
        except Exception as e:
            logging.error(f"Không thể khởi tạo cảm biến PIR: {e}. Hệ thống sẽ chạy không có cảm biến.")
            # Nếu không có cảm biến, tạo một đối tượng giả để is_motion() luôn trả về True
            class MockPIR:
                def is_motion(self): return True
                def stop(self): pass
            pir_sensor = MockPIR()

        logging.info("Đang khởi tạo FaceDetector...")
        # Tối ưu hóa MediaPipe Face Detection cho Raspberry Pi
        mp_face_detection = FaceDetection(
            model_selection=0,
            min_detection_confidence=0.7
        )
        logging.info("Đã khởi tạo FaceDetector")
        
        logging.info("Đang khởi tạo Fasnet anti-spoofing...")
        fasnet = Fasnet()
        logging.info("Đã khởi tạo Fasnet anti-spoofing")
        
        logging.info("Đang mở webcam...")
        cap = open_camera_multi_backend()
        if not cap or not cap.isOpened():
            logging.error("Không thể mở webcam")
            cleanup_resources(pir_sensor=pir_sensor)
            return
            
        logging.info("Khởi động các thread...")
        detection_thread = threading.Thread(target=face_detection_thread, args=(cap, mp_face_detection, fasnet, sheet, pir_sensor))
        recognition_thread = threading.Thread(target=face_recognition_thread, args=(fasnet, sheet))
        
        detection_thread.start()
        logging.info("Thread phát hiện khuôn mặt đã khởi động")
        
        recognition_thread.start()
        logging.info("Thread nhận diện khuôn mặt đã khởi động")
        
        detection_thread.join()
        recognition_thread.join()
        logging.info("Hệ thống nhận diện khuôn mặt đã dừng")
        
    except Exception as e:
        logging.error(f"Lỗi trong main: {str(e)}", exc_info=True)
    finally:
        cleanup_resources(
            cap=locals().get('cap'), 
            pir_sensor=locals().get('pir_sensor')
        )
        logging.info(f"Thống kê: {stats['faces_detected']} khuôn mặt, {stats['qrs_detected']} mã QR, thời gian chạy: {time.time() - stats['start_time']:.1f}s")

class OfflineAttendanceSync:
    def __init__(self, offline_file='offline_attendance.json'):
        self.offline_file = offline_file
        self.load_offline_data()

    def load_offline_data(self):
        if os.path.exists(self.offline_file):
            try:
                with open(self.offline_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []
        else:
            self.data = []

    def save_offline_data(self):
        try:
            with open(self.offline_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_attendance(self, data_type, data, confidence=None):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'timestamp': timestamp,
            'data': str(data),
            'data_type': data_type,
            'confidence': float(confidence) if confidence is not None else None
        }
        self.data.append(row)
        self.save_offline_data()
        logging.info(f"[OFFLINE] Đã lưu offline: {row}")

    def sync_to_google_sheets(self, sheet):
        if not sheet:
            logging.info("[OFFLINE] Không có kết nối Google Sheets, không thể đồng bộ.")
            return False
        synced = 0
        for row in self.data:
            try:
                sheet.append_row([
                    row['timestamp'],
                    row['data'],
                    row['data_type'],
                    f"{row['confidence']:.4f}" if row['confidence'] is not None else ''
                ])
                logging.info(f"[OFFLINE->ONLINE] Đã đồng bộ lên Google Sheets: {row}")
                synced += 1
            except Exception as e:
                logging.info(f"[OFFLINE->ONLINE] Lỗi đồng bộ: {e}")
                break
        if synced > 0:
            self.save_offline_data()
        return synced > 0

if __name__ == "__main__":
    main()