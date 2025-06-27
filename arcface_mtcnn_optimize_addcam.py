import pickle
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
import cv2
from mtcnn import MTCNN
import logging
import gc
import threading
import queue
from datetime import datetime
# Tạo thư mục logs nếu chưa tồn tại
if not os.path.exists('logs'):
    os.makedirs('logs')

# Tạo tên file log dựa trên thời gian
log_filename = f"logs/face_recognition_{datetime.now().strftime('%H%M%S')}.log"

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Khởi tạo MTCNN với các tham số tối ưu cho Raspberry Pi 5
detector = MTCNN(
)

# Queue để lưu trữ các khuôn mặt đã phát hiện
face_queue = queue.Queue()

# Thời gian giữa các lần detect (giây)
DETECT_INTERVAL = 1.0

# Thư mục lưu ảnh khuôn mặt
FACE_SAVE_DIR = "detected_faces"
if not os.path.exists(FACE_SAVE_DIR):
    os.makedirs(FACE_SAVE_DIR)

# Dictionary để lưu trữ danh sách người đã nhận diện
known_persons = set()

# Biến đếm số lượng ảnh đã lưu
face_count = 0

def preprocess_face(img, face_box, target_size=(112, 112)):
    """
    Tiền xử lý ảnh khuôn mặt cho một khuôn mặt cụ thể
    """
    try:
        x, y, w, h = face_box
        
        # Mở rộng vùng khuôn mặt
        x = max(0, x - int(w * 0.1))
        y = max(0, y - int(h * 0.1))
        w = min(img.shape[1] - x, w + int(w * 0.2))
        h = min(img.shape[0] - y, h + int(h * 0.2))
        
        # Cắt khuôn mặt
        face_img = img[y:y+h, x:x+w]
        
        # Resize về kích thước mục tiêu
        face_img = cv2.resize(face_img, target_size, interpolation=cv2.INTER_AREA)
        
        # Chuyển đổi sang RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Chuẩn hóa pixel về khoảng [-1, 1]
        face_img = (face_img.astype(np.float32) - 127.5) / 128.0
        
        return face_img, (x, y, w, h)
    except Exception as e:
        logging.error(f"Loi tien xu ly khuon mat: {str(e)}")
        return None, None
    finally:
        gc.collect()

def load_train_data(train_file):
    """
    Load dữ liệu train và tối ưu hóa cho Raspberry Pi
    """
    try:
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)
            
        # Chuyển đổi sang float32 để tiết kiệm bộ nhớ
        train_data['X_train'] = train_data['X_train'].astype(np.float32)
        
        return train_data
    except Exception as e:
        logging.error(f"Loi khi load du lieu train: {str(e)}")
        raise e

def recognize_face(image_path, train_file='models/train_data.pkl', 
                   embedding_model='ArcFace',
                   threshold=0.75):
    """
    Nhận diện khuôn mặt từ một ảnh sử dụng dữ liệu train đã lưu.
    Tối ưu hóa cho Raspberry Pi 5 với tốc độ xử lý nhanh nhất.

    Args:
        image_path (str): Đường dẫn đến ảnh chứa khuôn mặt cần nhận diện.
        train_file (str): Đường dẫn đến file train (.pkl).
        embedding_model (str): Tên mô hình embedding đã sử dụng khi train 
        threshold (float): Ngưỡng cho độ tương đồng cosine.

    Returns:
        tuple: (Tên người được nhận diện, độ tin cậy, thời gian xử lý)
    """
    start_time = time.time()
    
    try:
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Khong tim thay file train: {train_file}")

        # Load dữ liệu train đã tối ưu
        train_data = load_train_data(train_file)
        train_embeddings = train_data['X_train']
        label_encoder = train_data['label_encoder']
        train_labels = train_data['y_train']

        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Khong the doc anh: {image_path}")

        # Trích xuất embedding với các tham số tối ưu
        target_embedding = DeepFace.represent(
            img, 
            model_name=embedding_model, 
            enforce_detection=True,
            detector_backend='mtcnn',
            align=True,
            normalization='base'
        )[0]['embedding']
        
        # Chuẩn hóa embedding
        target_embedding = target_embedding / np.linalg.norm(target_embedding)
        target_embedding = np.expand_dims(target_embedding, axis=0)

        # Tính toán độ tương đồng
        similarities = cosine_similarity(target_embedding, train_embeddings)[0]
        best_match_index = np.argmax(similarities)
        confidence = similarities[best_match_index]

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        if confidence >= threshold:
            best_label = train_labels[best_match_index]
            predicted_label = label_encoder.inverse_transform([best_label])[0]
            return predicted_label, confidence, processing_time
        else:
            logging.info(f"Do tin cay thap: {confidence:.2%} < {threshold:.2%}")
            return None, confidence, processing_time

    except Exception as e:
        logging.error(f"Loi nhan dien: {str(e)}")
        return None, 0.0, 0.0
    finally:
        # Giải phóng bộ nhớ
        gc.collect()

def save_face(face_img):
    """
    Lưu ảnh khuôn mặt vào thư mục
    """
    try:
        # Đếm số lượng ảnh hiện có trong thư mục
        existing_files = [f for f in os.listdir(FACE_SAVE_DIR) if f.startswith('face_')]
        next_number = len(existing_files) + 1
        
        filename = f"{FACE_SAVE_DIR}/face_{next_number}.jpg"
        cv2.imwrite(filename, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        global face_count
        face_count += 1
        logging.info(f"Da luu khuon mat: {filename} (Tong so: {face_count})")
        return filename
    except Exception as e:
        logging.error(f"Loi khi luu khuon mat: {str(e)}")
        return None

def load_known_persons():
    """
    Load danh sách người đã nhận diện từ file
    """
    try:
        if os.path.exists("recognition_results.txt"):
            with open("recognition_results.txt", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        known_persons.add(parts[2])  # Tên người
    except Exception as e:
        logging.error(f"Loi khi load danh sach nguoi da nhan dien: {str(e)}")

def save_recognition_result(face_path, person_name, confidence):
    """
    Lưu kết quả nhận diện vào file
    """
    try:
        with open("recognition_results.txt", "a", encoding="utf-8") as f:
            result = f"{face_path},{person_name},{confidence:.4f}\n"
            f.write(result)
        known_persons.add(person_name)
        logging.info(f"Da luu ket qua nhan dien: {person_name} (do tin cay: {confidence:.2%})")
    except Exception as e:
        logging.error(f"Loi khi luu ket qua nhan dien: {str(e)}")

def face_detection_thread():
    """
    Luồng phát hiện khuôn mặt từ webcam
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Khong the mo webcam")
        return

    # Đặt kích thước frame nhỏ hơn để tăng tốc độ
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_detect_time = time.time()
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Đọc frame từ webcam
            ret, frame = cap.read()
            if not ret:
                logging.error("Khong the doc frame tu webcam")
                continue

            # Lưu frame gốc để vẽ khung
            original_frame = frame.copy()
            
            # Chỉ detect nếu đã qua khoảng thời gian quy định
            if current_time - last_detect_time >= DETECT_INTERVAL:
                # Giảm kích thước ảnh để tăng tốc độ xử lý
                scale_percent = 50
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                small_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                
                # Chuyển đổi sang RGB để detect khuôn mặt
                small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Phát hiện tất cả khuôn mặt
                faces = detector.detect_faces(small_frame_rgb)
                
                if faces:
                    logging.info(f"Phat hien duoc {len(faces)} khuon mat")
                    
                    for i, face in enumerate(faces):
                        # Lấy tọa độ khuôn mặt
                        x, y, w, h = face['box']
                        
                        # Nhân đôi tọa độ vì đã giảm kích thước ảnh 50%
                        x = x * 2
                        y = y * 2
                        w = w * 2
                        h = h * 2
                        
                        # Vẽ khung cho khuôn mặt
                        cv2.rectangle(original_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(original_frame, f"Face {i+1}", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # Tiền xử lý và lưu ảnh khuôn mặt
                        face_img, bbox = preprocess_face(frame, (x, y, w, h))
                        if face_img is not None:
                            face_path = save_face(face_img)
                            if face_path:
                                face_queue.put((face_path, bbox))
                                logging.info(f"Da luu khuon mat {i+1}")
                
                # Cập nhật thời gian detect cuối cùng
                last_detect_time = current_time
            
            # Hiển thị frame
            cv2.imshow('Face Detection', original_frame)
            
            # Thoát nếu nhấn 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except Exception as e:
        logging.error(f"Loi trong qua trinh phat hien khuon mat: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        os._exit(0)

def face_recognition_thread():
    """
    Luồng nhận diện khuôn mặt từ ảnh đã lưu
    """
    try:
        while True:
            if not face_queue.empty():
                face_path, bbox = face_queue.get()
                
                try:
                    # Nhận diện khuôn mặt
                    predicted_name, confidence, processing_time = recognize_face(face_path)
                    
                    if predicted_name:
                        # Kiểm tra xem người này đã được nhận diện chưa
                        if predicted_name in known_persons:
                            logging.info(f"Da nhan dien nguoi da biet: {predicted_name}")
                            continue
                        
                        # Lưu kết quả nhận diện mới
                        save_recognition_result(face_path, predicted_name, confidence)
                
                except Exception as rec_error:
                    # Log specific error for face detection failure
                    if "Face could not be detected" in str(rec_error):
                        logging.warning(f"Khong the nhan dien khuon mat trong anh {face_path}. Co the do chat luong anh thap.")
                        # Record failed image path
                        with open("failed_images.log", "a") as f:
                            f.write(f"{face_path}\n")
                    else:
                        logging.error(f"Loi trong qua trinh nhan dien: {str(rec_error)}")
                    
                    # Try to remove the failed image to save space
                    try:
                        if os.path.exists(face_path):
                            os.remove(face_path)
                            logging.info(f"Da xoa anh loi: {face_path}")
                    except Exception as del_error:
                        logging.error(f"Khong the xoa anh loi {face_path}: {str(del_error)}")
                
            time.sleep(0.1)
            
    except Exception as e:
        logging.error(f"Loi trong qua trinh nhan dien: {str(e)}")

def main():
    """
    Hàm chính để chạy cả hai luồng
    """
    logging.info("Bat dau chuong trinh nhan dien khuon mat")
    
    # Load danh sách người đã nhận diện
    load_known_persons()
    
    # Tạo và chạy luồng phát hiện khuôn mặt
    detection_thread = threading.Thread(target=face_detection_thread)
    detection_thread.daemon = True
    detection_thread.start()
    
    # Tạo và chạy luồng nhận diện khuôn mặt
    recognition_thread = threading.Thread(target=face_recognition_thread)
    recognition_thread.daemon = True
    recognition_thread.start()
    
    # Đợi các luồng kết thúc
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Dang thoat chuong trinh...")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
