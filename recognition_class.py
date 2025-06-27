import os
import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import logging
import time
import gspread
from google.oauth2.service_account import Credentials
import h5py
import torch
import torch.nn.functional as F
import urllib.request
from pyzbar import pyzbar
from datetime import datetime
import json
import queue
from typing import Callable, Optional
try:
    from pir_sensor import PIRSensor
except (ImportError, RuntimeError) as e:
    PIRSensor = None # Sẽ xử lý nếu không có thư viện RPi.GPIO
from facenet_pytorch import MTCNN

# Tắt warnings TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

_face_attendance_logging_configured = False

class RecognitionSystem:
    def __init__(self, sheet_name="Attendance", credentials_path='credentials/face-attendance.json', pir_pin=17):
        self.setup_logging()
        logging.info("Khởi tạo Hệ thống Nhận diện...")
        
        self.known_persons = set()
        self.known_qr_codes = set()
        
        # Cấu hình
        self.FACE_RECOGNITION_THRESHOLD = 0.65
        self.ANTISPOOF_THRESHOLD = 0.6
        self.MIN_FACE_SIZE = 50
        self.PIR_PIN = pir_pin
        
        # Biến trạng thái cho GUI
        self.system_status = "Đang khởi tạo..."
        self.log_messages = []
        self.running = False # Sẽ được set True khi run() được gọi
        self.cap = None

        # Hàng đợi và callback để giao tiếp với GUI
        self.frame_for_gui = queue.Queue(maxsize=2)
        self.attendance_callback: Optional[Callable] = None
        
        # Khởi tạo các thành phần
        self._update_status_and_logs("Khởi tạo model chống giả mạo...")
        self.fasnet = self._initialize_fasnet()
        self._update_status_and_logs("Khởi tạo model phát hiện khuôn mặt...")
        self.mtcnn = MTCNN(keep_all=True, device='cpu')
        self._update_status_and_logs("Tải dữ liệu training...")
        self.train_data, self.label_encoder = self._load_training_data()
        self._update_status_and_logs("Kết nối Google Sheets...")
        self.sheet = self.setup_google_sheets(sheet_name, credentials_path)
        
        self.last_motion_time = None

        # SỬA LẠI LOGIC KHỞI TẠO PIR
        self.pir_sensor = None
        if PIRSensor:
            try:
                self._update_status_and_logs(f"Khởi tạo cảm biến PIR trên chân GPIO {self.PIR_PIN}...")
                self.pir_sensor = PIRSensor(pin_signal=self.PIR_PIN)
                self.pir_sensor.start()
                self._update_status_and_logs("PIR đã khởi động và sẵn sàng.")
            except Exception as e:
                self._update_status_and_logs(f"Lỗi khởi tạo PIR: {e}", "error")
        else:
            self._update_status_and_logs("Không tìm thấy thư viện cảm biến, bỏ qua PIR.", "warning")

        self.offline_syncer = OfflineAttendanceSync()
        self.system_status = "Sẵn sàng"
        self._update_status_and_logs("Hệ thống nhận diện đã khởi tạo xong.")

    def setup_logging(self, log_dir='logs', max_logs=5):
        global _face_attendance_logging_configured
        os.makedirs(log_dir, exist_ok=True)
        log_files = sorted(
            [f for f in os.listdir(log_dir) if f.startswith('face_recognition_') and f.endswith('.log')],
            key=lambda x: os.path.getmtime(os.path.join(log_dir, x)), reverse=True
        )
        for old_log in log_files[max_logs:]:
            os.remove(os.path.join(log_dir, old_log))
        log_filename = os.path.join(log_dir, f"face_recognition_{time.strftime('%Y%m%d_%H%M%S')}.log")
        # --- Sửa cấu hình logging ---
        root_logger = logging.getLogger()
        if not _face_attendance_logging_configured:
            root_logger.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            # Xóa handler cũ để tránh log trùng
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            file_handler = logging.FileHandler(log_filename, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)
            # Đánh dấu đã cấu hình để không bị lặp lại
            _face_attendance_logging_configured = True
        logging.info("Đã cấu hình logging.")

    def _initialize_fasnet(self):
        logging.info("Đang khởi tạo model Fasnet anti-spoofing...")
        return Fasnet()

    def _load_training_data(self, model_path='models/train_FN.h5'):
        logging.info("Đang tải dữ liệu training...")
        try:
            with h5py.File(model_path, 'r') as f:
                train_data = {'embeddings': np.array(f['embeddings']), 'labels': np.array(f['labels'])}
            with open(model_path.replace('.h5', '_label_encoder.pkl'), 'rb') as f_le:
                label_encoder = pickle.load(f_le)
            logging.info("Đã tải dữ liệu training và label encoder thành công.")
            return train_data, label_encoder
        except Exception as e:
            logging.error(f"Lỗi tải dữ liệu training: {e}")
            return None, None

    def setup_google_sheets(self, sheet_name, credentials_path):
        logging.info("Đang kết nối Google Sheets...")
        try:
            scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                     "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
            creds = Credentials.from_service_account_file(credentials_path, scopes=scope)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key("1CbHHt5gwfOJzEMzrzGoEpvOyjMXYOk_AeEVtfRx_8pk")
            sheet = spreadsheet.worksheet(sheet_name)
            logging.info(f"Kết nối Google Sheets tới sheet '{sheet_name}' thành công.")
            return sheet
        except Exception as e:
            logging.error(f"Lỗi kết nối Google Sheets: {e}. Hệ thống sẽ hoạt động ở chế độ offline.")
            return None
            
    def detect_and_recognize(self, frame):
        """
        Phương thức chính để xử lý một frame: phát hiện, chống giả mạo, nhận diện và quét QR.
        Trả về: (danh sách khuôn mặt, danh sách mã QR)
        """
        all_faces_info = []

        if self.mtcnn is None or self.train_data is None:
            logging.warning("MTCNN hoặc train_data chưa được khởi tạo.")
            return all_faces_info, []

        # 1. Phát hiện khuôn mặt bằng MTCNN
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detect_result = self.mtcnn.detect(image_rgb)
        if isinstance(detect_result, tuple) and len(detect_result) >= 2:
            boxes, probs = detect_result[:2]
        else:
            boxes, probs = None, None
        img_h, img_w = frame.shape[:2]
        if boxes is not None:
            logging.info(f"MTCNN phát hiện {len(boxes)} khuôn mặt.")
            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                w, h = x2 - x1, y2 - y1
                # Bỏ qua box có tọa độ âm hoặc vượt quá biên ảnh
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    continue
                # Bỏ qua box quá nhỏ
                if w < 60 or h < 60:
                    continue
                if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                    continue
                facial_area = {'x': x1, 'y': y1, 'w': w, 'h': h}
                # 2. Chống giả mạo (Anti-spoofing)
                is_real, antispoof_score = self.fasnet.analyze(frame, (x1, y1, w, h))
                if not is_real or antispoof_score < self.ANTISPOOF_THRESHOLD:
                    all_faces_info.append({'name': 'Fake', 'facial_area': facial_area, 'confidence': antispoof_score})
                    continue
                # 3. Nhận diện khuôn mặt thật
                face_img = self.preprocess_face(frame, (x1, y1, w, h))
                if face_img is None:
                    logging.info(f"Face {idx}: Không thể tiền xử lý khuôn mặt.")
                    continue
                embedding = self.get_face_embedding(face_img)
                if embedding is None:
                    logging.info(f"Face {idx}: Không thể lấy embedding.")
                    continue
                similarities = cosine_similarity([embedding], self.train_data['embeddings'])[0]
                best_match_index = np.argmax(similarities)
                confidence = similarities[best_match_index]
                if confidence >= self.FACE_RECOGNITION_THRESHOLD:
                    predicted_label = self.train_data['labels'][best_match_index]
                    name = str(predicted_label)
                    if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                        try:
                            name = self.label_encoder.inverse_transform([int(predicted_label)])[0]
                        except Exception:
                            name = str(predicted_label)
                    logging.info(f"Face {idx}: ĐÃ NHẬN DIỆN: {name} (confidence={confidence})")
                    is_duplicate = name in self.known_persons
                    if not is_duplicate:
                        self.known_persons.add(name)
                        self.log_attendance(name, confidence, "FACE", is_duplicate=False)
                    else:
                        logging.info(f"Đã điểm danh (trùng): {name} (FACE)")
                        pass
                    all_faces_info.append({'name': name, 'facial_area': facial_area, 'confidence': confidence})
                else:
                    all_faces_info.append({'name': 'Unknown', 'facial_area': facial_area, 'confidence': confidence})
        # 5. Phát hiện mã QR
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_qrs = pyzbar.decode(gray)
            for qr in detected_qrs:
                qr_data = qr.data.decode('utf-8')
                if qr_data not in self.known_qr_codes:
                    if self.attendance_callback:
                        self.attendance_callback({
                            'type': 'QR-CAPTURE-REQUEST',
                            'qr_data': qr_data,
                            'frame': frame.copy()
                        })
        except Exception as e:
            logging.error(f"Lỗi xử lý QR code: {e}")

        return all_faces_info, []

    def save_qr_attendance(self, qr_data, confidence=1.0):
        is_duplicate = qr_data in self.known_qr_codes
        if not is_duplicate:
            self.known_qr_codes.add(qr_data)
            self.log_attendance(qr_data, confidence, "QR", is_duplicate=False)
            if self.attendance_callback:
                self.attendance_callback({'type': 'QR_DONE', 'data': qr_data})
        else:
            self._update_status_and_logs(f"Đã điểm danh (trùng): {qr_data}", "info")
            logging.info(f"Đã điểm danh (trùng): {qr_data}")
            if self.attendance_callback:
                try:
                    self.attendance_callback({
                        'type': 'DUPLICATE_QR_ATTENDANCE',
                        'qr_data': qr_data
                    })
                except Exception as e:
                    logging.error(f"Lỗi callback duplicate QR: {e}")

    def preprocess_face(self, img, facial_area, target_size=(160, 160)):
        try:
            # facial_area là (x1, y1, w, h) hoặc (x1, y1, x2, y2)
            if len(facial_area) == 4:
                x, y, w, h = facial_area
                x2, y2 = x + w, y + h
            else:
                x, y, x2, y2 = facial_area
            # Đảm bảo không vượt quá biên ảnh
            x, y = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            face_img = img[y:y2, x:x2]
            if face_img.size == 0: return None
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            # Resize giữ tỉ lệ, thêm padding đen để thành hình vuông
            h, w = face_img.shape[:2]
            target_w, target_h = target_size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Tạo ảnh nền đen
            result = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            return result
        except Exception as e:
            logging.error(f"Lỗi tiền xử lý khuôn mặt: {e}")
            return None

    def get_face_embedding(self, face_img):
        try:
            embedding = DeepFace.represent(
                face_img,
                model_name='Facenet512',
                enforce_detection=False,
                detector_backend='skip',
                align=False,
                normalization='base'
            )[0]['embedding']
            return embedding
        except Exception as e:
            logging.error(f"Lỗi lấy embedding: {e}")
            return None

    def log_attendance(self, name, confidence, mode, is_duplicate=False):
        # Định dạng confidence thành phần trăm
        try:
            confidence_str = f"{float(confidence)*100:.1f} %"
        except Exception:
            confidence_str = str(confidence)
        record = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, mode, confidence_str)
        if not is_duplicate and self.sheet:
            try:
                self.sheet.append_row(list(record))
                logging.info(f"Đã lưu lên Sheets: {name} ({mode})")
            except Exception as e:
                logging.error(f"Lỗi ghi Google Sheets: {e}")
        elif is_duplicate:
            logging.info(f"Đã điểm danh (trùng): {name} ({mode})")

    def stop(self):
        logging.info("Yêu cầu dừng hệ thống nhận diện...")
        self.running = False
        
        # Không cần join thread ở đây vì vòng lặp sẽ tự kết thúc
        # GUI sẽ quản lý việc join thread nếu cần

    def run(self):
        """Vòng lặp chính của hệ thống nhận diện, được chạy trong một thread riêng."""
        self._update_status_and_logs("Bắt đầu vòng lặp nhận diện...")
        self.running = True

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Không thể mở webcam.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._update_status_and_logs("Webcam đã được kết nối.", "info")

            last_check_time = time.time()
            is_idle = False
            self.last_motion_time = time.time()

            while self.running:
                # 1. Quản lý trạng thái PIR và camera
                if self.pir_sensor:
                    if self.pir_sensor.is_motion():
                        if is_idle:
                            self._update_status_and_logs("Phát hiện chuyển động, bật camera.", "info")
                            if not self.cap.isOpened(): self.cap.open(0)
                        is_idle = False
                        self.last_motion_time = time.time()
                    elif not is_idle and (time.time() - self.last_motion_time > 10):
                        self._update_status_and_logs("Không có chuyển động, tạm dừng camera.", "info")
                        is_idle = True
                
                if is_idle:
                    self.cap.release()
                    time.sleep(0.5)
                    continue

                if not self.cap.isOpened():
                    time.sleep(0.5)
                    continue

                # 2. Đọc và xử lý frame
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue

                # Chỉ xử lý 5-10 frame mỗi giây để giảm tải CPU
                if time.time() - last_check_time < 0.15:
                    # Vẫn đẩy frame gốc lên GUI để không bị giật
                    try: self.frame_for_gui.put_nowait(cv2.resize(frame, (640, 480)))
                    except queue.Full: pass
                    continue
                
                last_check_time = time.time()
                
                # 3. Phát hiện và nhận diện
                faces, _ = self.detect_and_recognize(frame)

                # 4. Vẽ lên frame để hiển thị
                for face in faces:
                    x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                    name = face.get('name', 'Error')
                    conf = face.get('confidence', 0)
                    color = (0, 255, 0) # Green
                    if name == 'Fake': color = (0, 0, 255) # Red
                    elif name == 'Unknown': color = (255, 0, 0) # Blue
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{name} ({conf:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 5. Gửi frame đã xử lý cho GUI
                try:
                    self.frame_for_gui.put_nowait(cv2.resize(frame, (640, 480)))
                except queue.Full:
                    pass # Bỏ qua nếu GUI đang bận

        except Exception as e:
            self._update_status_and_logs(f"Lỗi trong vòng lặp chính: {e}", "error")
            logging.error(f"Lỗi trong vòng lặp chính: {e}", exc_info=True)
        finally:
            self._update_status_and_logs("Đã thoát vòng lặp nhận diện.", "info")
            if self.cap and self.cap.isOpened():
                self.cap.release()
                logging.info("Camera đã được giải phóng trong finally.")
            if self.pir_sensor:
                self.pir_sensor.release()
                logging.info("PIR sensor đã được giải phóng trong finally.")
            self.running = False

    def _update_status_and_logs(self, message, level="info"):
        """Cập nhật trạng thái hệ thống và thêm log message cho GUI."""
        self.system_status = message
        
        log_entry = f"[{level.upper()}] {message}"
        self.log_messages.append(log_entry)
        # Giới hạn số lượng log để tránh dùng quá nhiều bộ nhớ
        if len(self.log_messages) > 100:
            self.log_messages.pop(0)

        # Ghi log ra file
        if level == "info":
            logging.info(message)
        elif level == "warning":
            logging.warning(message)
        elif level == "error":
            logging.error(message)

# =============================================================================
# Lớp lưu trữ offline
# =============================================================================
class OfflineAttendanceSync:
    def __init__(self, offline_file='offline_attendance.json'):
        self.offline_file = offline_file
        self.data = []
        self.load_offline_data()

    def load_offline_data(self):
        if os.path.exists(self.offline_file):
            try:
                with open(self.offline_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content:
                        self.data = json.loads(content)
            except (json.JSONDecodeError, IOError) as e:
                logging.error(f"Lỗi đọc file offline, sẽ tạo file mới: {e}")
                self.data = []

    def save_offline_data(self):
        try:
            with open(self.offline_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            logging.error(f"Lỗi ghi file offline: {e}")

    def add_attendance(self, name, confidence, data_type):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = {'timestamp': timestamp, 'data': str(name), 'data_type': data_type, 'confidence': float(confidence)}
        self.data.append(row)
        self.save_offline_data()
        logging.info(f"[OFFLINE] Đã lưu điểm danh offline: {row}")

    def sync_to_google_sheets(self, sheet):
        if not self.data:
            return
        
        logging.info(f"Bắt đầu đồng bộ {len(self.data)} mục offline lên Google Sheets...")
        remaining_data = self.data[:]
        
        for row in self.data:
            try:
                sheet.append_row([row['timestamp'], row['data'], row['data_type'], f"{row.get('confidence', 0):.4f}"])
                remaining_data.remove(row)
            except Exception as e:
                logging.error(f"Lỗi đồng bộ, sẽ thử lại lần sau: {e}")
                break
        
        if len(remaining_data) != len(self.data):
            self.data = remaining_data
            self.save_offline_data()
            logging.info(f"Đồng bộ hoàn tất. Còn lại {len(self.data)} mục chưa đồng bộ.")

# =============================================================================
# Lớp Fasnet cho anti-spoofing (giữ nguyên, không thay đổi)
# =============================================================================
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

        self.download_model(model_path, model_url)
        
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
        except Exception as e:
            logging.error(f"Lỗi khởi tạo mô hình MiniFASNetV2: {str(e)}")
            raise

    def download_model(self, file_path, url):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not os.path.exists(file_path):
            logging.info(f"Đang tải mô hình từ {url}...")
            urllib.request.urlretrieve(url, file_path)

    def analyze(self, img, facial_area: tuple):
        try:
            x, y, w, h = facial_area
            if w < 50 or h < 50: return False, 0.0

            img_cropped = self.crop(img, (x, y, w, h), 2.7, 80, 80)
            
            test_transform = self.Compose([self.ToTensor(self)])
            img_tensor = test_transform(img_cropped)
            if isinstance(img_tensor, np.ndarray):
                img_tensor = torch.from_numpy(img_tensor)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                result = self.model.forward(img_tensor)
                result = F.softmax(result, dim=1).cpu().numpy()

            label = np.argmax(result)
            is_real = label == 1
            score = result[0][label]
            return is_real, score
            
        except Exception as e:
            # Ghi log lỗi chi tiết hơn
            logging.error(f"[ANTI-SPOOF] Lỗi phân tích: {e}", exc_info=True)
            return False, 0.0

    def to_tensor(self, pic):
        if pic.ndim == 2: pic = pic.reshape((pic.shape[0], pic.shape[1], 1))
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img.float()

    class Compose:
        def __init__(self, transforms): self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms: img = t(img)
            return img

    class ToTensor:
        def __init__(self, outer): self.outer = outer
        def __call__(self, pic): return self.outer.to_tensor(pic)

    def _get_new_box(self, src_w, src_h, bbox, scale):
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w / 2 + x, box_h / 2 + y
        
        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        
        # Tính toán toạ độ bottom-right từ center, giống với script gốc
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        # Đảm bảo box nằm trong ảnh (logic từ script gốc, an toàn hơn)
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
            h, w = img.shape[:2]
            # Resize giữ tỉ lệ, cạnh lớn hơn sẽ bị cắt
            scale = max(out_w / w, out_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Crop chính giữa để thành hình vuông
            start_x = (new_w - out_w) // 2 if new_w > out_w else 0
            start_y = (new_h - out_h) // 2 if new_h > out_h else 0
            img_cropped = img_resized[start_y:start_y+out_h, start_x:start_x+out_w]
            return img_cropped
        except Exception as e:
            logging.error(f"Lỗi resize/crop ảnh: {str(e)}")
            return org_img

if __name__ == '__main__':
    rec_system = RecognitionSystem()
    rec_system.run() 