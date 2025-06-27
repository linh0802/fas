import os
import cv2
import numpy as np
import logging
import h5py
import pickle
import gc
from deepface import DeepFace
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN
from PIL import Image

def clear_log_file():
    """Xóa nội dung file training.log."""
    with open('training.log', 'w') as f:
        f.write('')
    logging.info("Đã xóa nội dung file training.log")

# Thiết lập logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ensure_directories():
    """Tạo các thư mục cần thiết."""
    os.makedirs('processed_faces', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logging.info("Đã tạo các thư mục cần thiết")

def clear_processed_faces():
    """Xóa thư mục processed_faces."""
    import shutil
    if os.path.exists('processed_faces'):
        shutil.rmtree('processed_faces')
    os.makedirs('processed_faces')
    logging.info("Đã xóa và tạo lại thư mục processed_faces")


def extract_and_save_faces(images_dir, target_size=(160, 160)):
    """
    Trích xuất khuôn mặt từ ảnh và lưu vào processed_faces, bảo toàn độ sáng và màu sắc.
    Args:
        images_dir: Thư mục chứa ảnh gốc
        target_size: Kích thước đầu ra của ảnh khuôn mặt
    Returns:
        processed_faces_dir: Đường dẫn đến thư mục chứa ảnh đã xử lý
    """
    mtcnn = MTCNN(
        image_size=target_size[0],
        margin=30,
        min_face_size=50,
        thresholds=[0.7, 0.8, 0.9],
        device='cpu'
    )
    
    processed_faces_dir = 'processed_faces'
    os.makedirs(processed_faces_dir, exist_ok=True)
    
    # Đếm tổng số ảnh cần xử lý
    total_images = 0
    for person_name in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        total_images += len(os.listdir(person_dir))
    
    total_processed = 0
    total_failed = 0
    
    for person_name in os.listdir(images_dir):
        person_dir = os.path.join(images_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        output_person_dir = os.path.join(processed_faces_dir, person_name)
        os.makedirs(output_person_dir, exist_ok=True)
        
        logging.info(f"Đang xử lý thư mục: {person_name}")
        
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            
            try:
                # Đọc và kiểm tra ảnh đầu vào
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                if img_array.size == 0 or img_array.shape[-1] != 3:
                    logging.error(f"Ảnh đầu vào không hợp lệ: {img_path}")
                    total_failed += 1
                    continue
                
                logging.info(f"Kênh màu ảnh gốc {img_name}: {img_array[0, 0]}")
                
                # Phát hiện khuôn mặt
                face = mtcnn(img)
                
                if face is not None:
                    # Chuyển tensor về numpy array
                    face = face.permute(1, 2, 0).numpy()
                    if face.shape[-1] != 3:
                        logging.error(f"Tensor từ MTCNN không có 3 kênh màu: {img_path}")
                        total_failed += 1
                        continue
                    
                    # Chuẩn hóa và bảo toàn độ sáng
                    logging.info(f"Giá trị pixel trước chuẩn hóa {img_name}: min={face.min()}, max={face.max()}")
                    face = (face - face.min()) / (face.max() - face.min() + 1e-8)  # Chuẩn hóa về [0, 1]
                    
                    # Bảo toàn tỷ lệ màu sắc
                    face = face * 255.0  # Đưa về [0, 255]
                    face = face.astype(np.uint8)
                    
                    # Lưu ảnh bằng PIL
                    face_pil = Image.fromarray(face)
                    output_path = os.path.join(output_person_dir, f"processed_{img_name}")
                    face_pil.save(output_path)
                    
                    total_processed += 1
                    # Chỉ log % từ 0-49% ở bước này
                    percent = int(total_processed / total_images * 49) if total_images else 0
                    logging.info(f"Tiến trình: {total_processed}/{total_images} ({percent}%)")
                    print(f"Tiến trình: {total_processed}/{total_images} ({percent}%)", flush=True)
                    if total_processed % 10 == 0:
                        logging.info(f"Đã xử lý {total_processed} ảnh")
                else:
                    logging.warning(f"Không phát hiện được khuôn mặt trong ảnh: {img_path}")
                    total_failed += 1
                    
            except Exception as e:
                logging.error(f"Lỗi xử lý ảnh {img_path}: {str(e)}")
                total_failed += 1
                continue
    
    logging.info(f"Hoàn thành xử lý ảnh:")
    logging.info(f"- Tổng số ảnh đã xử lý thành công: {total_processed}")
    logging.info(f"- Tổng số ảnh thất bại: {total_failed}")
    # Log 50% sau khi xử lý xong toàn bộ ảnh hợp lệ
    logging.info(f"Tiến trình: {total_processed}/{total_images} (50%)")
    print(f"Tiến trình: {total_processed}/{total_images} (50%)", flush=True)
    
    return processed_faces_dir

def process_batch(batch, embedding_model='Facenet512'):
    """Xử lý batch ảnh với FaceNet-512d."""
    batch_embeddings = []
    
    try:
        for img in batch:
            embedding_result = DeepFace.represent(
                img,
                model_name=embedding_model,
                enforce_detection=False,
                detector_backend='skip',
                align=False,
                normalization='base',
                anti_spoofing=False
            )[0]
            
            embedding = embedding_result['embedding']
            batch_embeddings.append(embedding)
            
    except Exception as e:
        logging.error(f"Lỗi xử lý batch: {str(e)}")
        return []
        
    return batch_embeddings

def create_training_data(
    processed_faces_dir,
    output_train_file='models/train_FN.h5',
    embedding_model='Facenet512',
    batch_size=4,
    test_size=0.2,
    random_state=42,
    target_size=(160, 160)
):
    """
    Tạo dữ liệu train từ ảnh khuôn mặt đã được cắt.
    Args:
        embedding_model: 'Facenet' (128 chiều) hoặc 'Facenet512' (512 chiều)
        batch_size: Kích thước batch, đề xuất 4 cho Raspberry Pi 5
    """
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    
    embeddings = []
    labels = []
    failed_images = []
    
    if embedding_model not in ['Facenet', 'Facenet512']:
        raise ValueError("embedding_model phải là 'Facenet' hoặc 'Facenet512'")
    
    try:
        person_dirs = [d for d in os.listdir(processed_faces_dir)
                      if os.path.isdir(os.path.join(processed_faces_dir, d))]
        
        if not person_dirs:
            raise ValueError("Không tìm thấy thư mục người nào")
            
        logging.info(f"Tìm thấy {len(person_dirs)} thư mục người")
        
        for person_name in person_dirs:
            person_path = os.path.join(processed_faces_dir, person_name)
            logging.info(f"Đang xử lý thư mục: {person_name}")
            
            image_files = [
                f for f in os.listdir(person_path)
                if f.startswith(('processed_', 'augmented_'))
            ]
            
            if len(image_files) < 5:
                logging.warning(f"Thư mục {person_name} có ít hơn 5 ảnh, bỏ qua")
                continue
            
            # Xử lý ảnh theo batch
            batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
            
            for batch in batches:
                batch_images = []
                batch_paths = []
                
                for image_file in batch:
                    image_path = os.path.join(person_path, image_file)
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            failed_images.append((image_file, "Không thể đọc ảnh"))
                            continue
                            
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = img.astype(np.float32) / 255.0
                        batch_images.append(img)
                        batch_paths.append(image_file)
                        
                    except Exception as e:
                        failed_images.append((image_file, str(e)))
                        continue
                
                if batch_images:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        batch_embeddings = executor.submit(process_batch, batch_images, embedding_model).result()
                    
                    for embedding, image_file in zip(batch_embeddings, batch_paths):
                        if embedding:
                            embeddings.append(embedding)
                            labels.append(person_name)
                        else:
                            failed_images.append((image_file, "Lỗi tạo embedding"))
                            
                    # Giải phóng bộ nhớ
                    del batch_images
                    gc.collect()
        
        # Sau khi trích xuất embedding xong, log 90%
        logging.info(f"Tiến trình: 0/0 (90%)")
        print(f"Tiến trình: 0/0 (90%)", flush=True)
        if embeddings:
            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels)
            
            X_train, X_val, y_train, y_val = train_test_split(
                np.array(embeddings),
                encoded_labels,
                test_size=test_size,
                random_state=random_state,
                stratify=encoded_labels
            )
            
            X_train = X_train / np.linalg.norm(X_train, axis=1, keepdims=True)
            X_val = X_val / np.linalg.norm(X_val, axis=1, keepdims=True)
            
            with h5py.File(output_train_file, 'w') as f:
                f.create_dataset('embeddings', data=X_train)
                f.create_dataset('labels', data=y_train)
                f.create_dataset('X_val', data=X_val)
                f.create_dataset('y_val', data=y_val)
                f.attrs['embedding_model'] = embedding_model
                f.attrs['embedding_size'] = 512 if embedding_model == 'Facenet512' else 128
                
            with open(output_train_file.replace('.h5', '_label_encoder.pkl'), 'wb') as f_le:
                pickle.dump(label_encoder, f_le)
            
            logging.info(f"Đã tạo file train tại: {output_train_file}")
            logging.info(f"Số lượng ảnh train: {len(y_train)}")
            logging.info(f"Số lượng ảnh validation: {len(y_val)}")
            
            if failed_images:
                logging.warning(f"Tổng số ảnh lỗi: {len(failed_images)}")
                with open('failed_images.log', 'w', encoding='utf-8') as f:
                    for img, reason in failed_images:
                        f.write(f"{img}: {reason}\n")
            # Log 100% sau khi lưu file train thành công
            logging.info(f"Tiến trình: 0/0 (100%)")
            print(f"Tiến trình: 0/0 (100%)", flush=True)
        else:
            raise ValueError("Không có dữ liệu khuôn mặt nào được xử lý.")
            
    except Exception as e:
        logging.error(f"Lỗi tạo dữ liệu train: {str(e)}")
        raise

def main():
    try:
        clear_log_file()  # Xóa log cũ trước khi bắt đầu
        ensure_directories()
        clear_processed_faces()
        processed_faces_dir = extract_and_save_faces("images_attendance", target_size=(160, 160))
        create_training_data(
            processed_faces_dir=processed_faces_dir,
            output_train_file="models/train_FN.h5",
            embedding_model='Facenet512',
            batch_size=4,
            test_size=0.2
        )
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình train: {str(e)}")
        raise

if __name__ == "__main__":
    main()