# Hệ Thống Điểm Danh Khuôn Mặt - FaceAttendance

## Giới thiệu
Hệ thống điểm danh khuôn mặt FaceAttendance là một giải pháp AI/IoT tích hợp, giúp tự động nhận diện, quản lý và lưu trữ thông tin điểm danh bằng khuôn mặt. Hệ thống hỗ trợ giao diện người dùng trực quan (Tkinter), backend API (Flask), đồng bộ dữ liệu với Google Sheets, hỗ trợ mã QR, cảm biến chuyển động PIR, và nhiều tính năng nâng cao khác.

## Kiến trúc tổng thể
- **Giao diện người dùng (Tkinter):** Quản lý, thêm mới, điểm danh, xem dữ liệu, huấn luyện lại model.
- **Nhận diện khuôn mặt (AI):** Sử dụng DeepFace, Facenet, MTCNN, anti-spoofing (Fasnet), nhận diện real-time.
- **Quản lý dữ liệu:** Lưu trữ thông tin người dùng, ảnh, embedding vào SQLite3, đồng bộ với Google Sheets.
- **Web backend (Flask):** API phục vụ dữ liệu điểm danh, quản lý người dùng, upload ảnh, xác thực, gửi email QR.
- **Các thành phần phụ trợ:** TTS (phát âm tên), cảm biến PIR, QR code, logging, xuất/nhập CSV.

## Sơ đồ luồng hoạt động
```
[Webcam] -> [Nhận diện AI] -> [Tkinter GUI] <-> [SQLite3 DB] <-> [Flask API] <-> [Google Sheets]
                                            |                |
                                         [QR, TTS, PIR]   [Xuất/nhập CSV]
```

## Các tính năng chính
- **Nhận diện khuôn mặt real-time** với AI, chống giả mạo (anti-spoofing).
- **Thêm người mới**: Tự động nhận diện tên, chụp ảnh, lưu vào DB, kiểm tra trùng lặp.
- **Điểm danh tự động**: Giao diện trực quan, log chi tiết, phát âm tên, lưu lịch sử.
- **Huấn luyện lại mô hình**: Tự động trích xuất khuôn mặt, sinh embedding, cập nhật model.
- **Quản lý dữ liệu**: Xem, xuất, nhập, đồng bộ dữ liệu người dùng và ảnh.
- **API backend**: Cung cấp dữ liệu điểm danh, xác thực, upload ảnh, gửi email QR.
- **Đồng bộ Google Sheets**: Lưu lịch sử điểm danh lên cloud, hỗ trợ offline sync.
- **Tích hợp cảm biến PIR**: Tự động bật/tắt nhận diện khi có chuyển động.
- **Hỗ trợ mã QR**: Đăng ký khách, gửi email QR, xác nhận điểm danh qua QR.
- **Phát âm tên (TTS)**: Thông báo trạng thái, cảnh báo, xác nhận.
- **Logging chuyên sâu**: Ghi log hệ thống, log huấn luyện, log điểm danh.

## Hướng dẫn cài đặt
### 1. Yêu cầu hệ thống
- **Phần cứng:** Raspberry Pi 4/5 hoặc PC, webcam, cảm biến PIR (tùy chọn), loa (TTS), kết nối internet.
- **Phần mềm:** Python 3.8+, pip, các thư viện trong `requirements.txt`.

### 2. Cài đặt
```bash
# Clone mã nguồn
$ git clone <repo_url>
$ cd faceattendance

# Tạo môi trường ảo (khuyến nghị)
$ python3 -m venv .venv
$ source .venv/bin/activate

# Cài đặt thư viện
$ pip install -r requirements.txt

# Khởi tạo database (chỉ lần đầu)
$ python db.py

# Chạy backend Flask (API)
$ python web_app/app.py

# Chạy giao diện người dùng (GUI)
$ python gui_tkinter.py
```

### 3. Cấu hình Google Sheets, Email, Ngrok
- Đặt file credentials Google API vào `credentials/face-attendance.json`.
- Cấu hình biến môi trường email, ngrok nếu dùng tính năng gửi QR/email/điểm danh từ xa.

## Hướng dẫn sử dụng
- **Thêm người mới:** Vào giao diện, nhập/tự động nhận diện tên, chụp ảnh, lưu, huấn luyện lại model.
- **Điểm danh:** Chọn tab nhận diện, hệ thống tự động nhận diện, log, phát âm tên, lưu lịch sử.
- **Xem dữ liệu:** Tab dữ liệu điểm danh, có thể xuất CSV, xem log, lọc dữ liệu.
- **Huấn luyện lại:** Sau khi thêm người mới, nhấn "Huấn luyện" để cập nhật model.
- **Quản lý ảnh:** Ảnh lưu tại `images_attendance/`, embedding tại `models/`.
- **API:** Truy cập các endpoint Flask để lấy dữ liệu, upload ảnh, xác thực, v.v.

## Nguyên lý kỹ thuật
- **Nhận diện khuôn mặt:**
  - Phát hiện khuôn mặt bằng MTCNN.
  - Chống giả mạo bằng Fasnet (anti-spoofing).
  - Sinh embedding bằng Facenet512, so sánh cosine similarity với dữ liệu train.
  - Gán nhãn bằng label encoder, xác suất >0.65 mới xác nhận.
- **Quản lý dữ liệu:**
  - Lưu thông tin người dùng, ảnh, embedding vào SQLite3.
  - Tự động đồng bộ với Google Sheets (nếu có mạng).
  - Hỗ trợ xuất/nhập CSV, backup dữ liệu.
- **Huấn luyện lại:**
  - Trích xuất khuôn mặt từ ảnh gốc, sinh embedding, lưu model mới.
  - Tự động bỏ qua ảnh lỗi, log chi tiết quá trình train.
- **Điểm danh:**
  - Khi nhận diện thành công, lưu lịch sử vào DB, đồng bộ cloud, phát âm tên.
  - Hỗ trợ điểm danh qua QR code, xác nhận qua API/web.
- **Tích hợp cảm biến PIR:**
  - Tự động bật/tắt webcam khi có chuyển động, tiết kiệm tài nguyên.
- **TTS:**
  - Phát âm tên, cảnh báo, trạng thái hệ thống qua loa.

## Các file/thư mục quan trọng
- `gui_tkinter.py`: Giao diện người dùng, quản lý toàn bộ luồng chính.
- `recognition_class.py`: Nhận diện khuôn mặt, anti-spoofing, pipeline AI.
- `db.py`: Quản lý database, khởi tạo, đồng bộ, xuất/nhập dữ liệu.
- `finish_train.py`: Huấn luyện lại model, trích xuất khuôn mặt, sinh embedding.
- `web_app/app.py`: Backend Flask, API, xác thực, upload ảnh, gửi email QR.
- `models/`: Chứa model AI, label encoder.
- `images_attendance/`: Lưu ảnh khuôn mặt gốc.
- `processed_faces/`: Ảnh đã cắt khuôn mặt, phục vụ train.
- `logs/`, `training.log`: Log hệ thống, log huấn luyện.
- `requirements.txt`: Danh sách thư viện cần cài đặt.
- `credentials/`: Thông tin xác thực Google API.

## Yêu cầu phần mềm/phần cứng
- Python >= 3.8, pip
- Raspberry Pi 4/5 hoặc PC, webcam, loa, cảm biến PIR (tùy chọn)
- Kết nối internet để đồng bộ cloud, gửi email, Google Sheets

## Tài liệu tham khảo
- [DeepFace](https://github.com/serengil/deepface)
- [Facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- [Flask](https://flask.palletsprojects.com/)
- [Google Sheets API](https://developers.google.com/sheets/api)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [OpenCV](https://opencv.org/)

---

## Nguyên lý hoạt động chi tiết

### 1. Giao diện người dùng (Tkinter GUI)
- **Quản lý luồng chính:**
  - Khi khởi động, GUI tạo các frame: Nhận diện, Thêm người mới, Xem dữ liệu.
  - Điều hướng giữa các frame bằng các nút chức năng.
- **Luồng thêm người mới:**
  - Tự động nhận diện tên qua 10 frame webcam đầu, lấy tên xuất hiện nhiều nhất (nếu có).
  - Cho phép nhập tay nếu không nhận diện được.
  - Chụp 10 ảnh, kiểm tra trùng lặp, lưu vào thư mục và database.
  - Sau khi lưu, có thể huấn luyện lại model ngay trên giao diện.
- **Luồng nhận diện:**
  - Hiển thị webcam real-time, nhận diện khuôn mặt liên tục.
  - Khi phát hiện khuôn mặt hợp lệ, hiển thị thông tin, phát âm tên, lưu lịch sử.
  - Hỗ trợ log chi tiết, xem lại lịch sử nhận diện.
- **Luồng xem dữ liệu:**
  - Lấy dữ liệu điểm danh từ API backend, hiển thị dạng bảng.
  - Hỗ trợ xuất CSV, lọc, tìm kiếm, xem log hệ thống.

### 2. Nhận diện khuôn mặt (AI/Deep Learning) - Chuyên sâu

#### 2.1. Phát hiện khuôn mặt với MTCNN (chi tiết kiến trúc)
- **MTCNN** gồm 3 mạng con, mỗi mạng là một deep CNN nhỏ:
  - **P-Net (Proposal Network):**
    - Là một fully convolutional network nhỏ, quét sliding window trên ảnh đầu vào.
    - Trả về các bounding box đề xuất (proposal) và xác suất là khuôn mặt.
    - Có thể phát hiện nhiều box chồng lấn, nhiều tỷ lệ.
    - Kiến trúc: 3 conv layers + PReLU, output 2 nhánh (probability, bounding box regression).
  - **R-Net (Refine Network):**
    - Nhận các box từ P-Net, crop và resize về 24x24, lọc lại các box có xác suất thấp.
    - Kiến trúc: 3 conv layers + 1 fully connected, output 2 nhánh (probability, bounding box regression).
    - Loại bỏ false positive, refine lại vị trí box.
  - **O-Net (Output Network):**
    - Nhận các box từ R-Net, crop và resize về 48x48, xác định chính xác vị trí, kích thước.
    - Kiến trúc: 4 conv layers + 2 fully connected, output 3 nhánh (probability, bounding box regression, facial landmarks).
    - Dự đoán thêm 5 điểm landmark (2 mắt, mũi, 2 miệng) để căn chỉnh khuôn mặt.
- **Quy trình forward:**
  1. Ảnh đầu vào -> P-Net (proposal) -> NMS (loại box trùng lặp)
  2. Box còn lại -> R-Net (refine) -> NMS
  3. Box còn lại -> O-Net (output, landmark) -> NMS
- **Non-Maximum Suppression (NMS):**
  - Thuật toán loại bỏ các box trùng lặp, chỉ giữ lại box có xác suất cao nhất.

#### 2.2. Chống giả mạo (Anti-spoofing)
- **Fasnet** là model kiểm tra khuôn mặt thật/giả (ảnh, video, mask).
- Ảnh cắt từ webcam được đưa vào Fasnet, trả về xác suất là mặt thật.
- Nếu xác suất thấp, hệ thống loại bỏ, cảnh báo qua TTS.

#### 2.3. Sinh embedding & nhận diện với Facenet (chi tiết deep CNN & embedding)
- **Facenet** là một deep CNN với kiến trúc dựa trên Inception-ResNet hoặc Inception-v3:
  - **Các lớp chính:**
    - Nhiều tầng convolution (Conv2D), batch normalization, activation (ReLU), pooling.
    - Các block Inception cho phép trích xuất đặc trưng đa tỷ lệ.
    - Cuối mạng là global average pooling, flatten, fully connected để ra vector embedding.
  - **Loss function:**
    - Sử dụng **Triplet Loss**: Đảm bảo embedding của cùng một người gần nhau, khác người thì xa nhau.
    - Triplet = (anchor, positive, negative):
      - |f(anchor) - f(positive)|^2 + margin < |f(anchor) - f(negative)|^2
    - Nhờ đó, embedding học được không gian metric tốt cho so sánh.
  - **Đầu ra:**
    - Vector embedding 128 hoặc 512 chiều, chuẩn hóa L2 (norm = 1).
    - Mỗi vector là "dấu vân tay" duy nhất cho mỗi khuôn mặt.
- **Ý nghĩa embedding:**
  - Là vector đặc trưng, chứa thông tin hình học, cấu trúc, tỷ lệ khuôn mặt.
  - Hai ảnh cùng người sẽ có embedding gần nhau (cosine similarity ~1), khác người thì xa nhau.
  - Hệ thống chỉ cần lưu embedding, không cần lưu ảnh gốc để so khớp.
- **So khớp:**
  - Khi nhận diện, tính cosine similarity giữa embedding mới và embedding đã lưu.
  - Nếu similarity > ngưỡng (0.65), xác nhận là cùng người.
  - Ưu điểm: mở rộng dễ dàng, thêm người mới chỉ cần thêm embedding mới, không cần train lại toàn bộ mạng.

#### Liên hệ thực tế với hệ thống
- **MTCNN** giúp hệ thống phát hiện chính xác, đa khuôn mặt, căn chỉnh tốt cho bước nhận diện.
- **Facenet** đảm bảo nhận diện chính xác, bền vững, chống giả mạo nhờ embedding metric learning.
- **Embedding** giúp hệ thống mở rộng, lưu trữ, so khớp nhanh, tiết kiệm tài nguyên, bảo mật (không cần lưu ảnh gốc).

#### 2.4. Pipeline train dữ liệu (finish_train.py)
- **Bước 1: Trích xuất khuôn mặt**
  - Duyệt toàn bộ ảnh gốc trong database.
  - Dùng MTCNN cắt khuôn mặt, lưu vào `processed_faces/`.
  - Bỏ qua ảnh không phát hiện được mặt hoặc chất lượng kém.
- **Bước 2: Sinh embedding**
  - Với mỗi ảnh đã cắt, dùng Facenet512 sinh vector embedding.
  - Lưu embedding và nhãn (user_id) vào file HDF5 (`models/train_FN.h5`).
- **Bước 3: Train label encoder**
  - Sử dụng sklearn LabelEncoder để ánh xạ user_id <-> tên.
  - Lưu encoder vào file `models/train_FN_label_encoder.pkl`.
- **Bước 4: Đánh giá dữ liệu**
  - Thống kê số ảnh hợp lệ, số ảnh lỗi, số người có ít hơn 5 ảnh.
  - Log chi tiết quá trình train vào `training.log`.
- **Bước 5: Cập nhật hệ thống**
  - Sau khi train xong, hệ thống tự động nạp lại model mới cho nhận diện.

#### 2.5. Nhận diện real-time
- Mỗi frame webcam được xử lý qua pipeline: MTCNN -> Fasnet -> Facenet -> So khớp -> Gán nhãn.
- Nếu nhận diện thành công, phát âm tên, lưu lịch sử, đồng bộ cloud.
- Nếu phát hiện giả mạo hoặc không rõ mặt, cảnh báo qua TTS, không lưu lịch sử.

### 3. Quản lý dữ liệu & Database (SQLite3)
- **Cấu trúc:**
  - Bảng `users`: Lưu thông tin người dùng (user_id, username, full_name, password...)
  - Bảng `face_profiles`: Lưu đường dẫn ảnh, embedding, user_id liên kết.
- **Thêm người mới:**
  - Khi lưu ảnh, tạo user mới nếu chưa có, lưu ảnh vào `images_attendance/user_{id}`.
  - Cập nhật bảng `face_profiles` với đường dẫn ảnh, user_id.
- **Đồng bộ dữ liệu:**
  - Hỗ trợ xuất/nhập CSV, backup, đồng bộ ảnh từ thư mục vào DB.
  - Có thể đồng bộ dữ liệu với Google Sheets qua backend.

### 4. Backend/API (Flask)
- **API điểm danh:**
  - `/api/attendance`: Trả về dữ liệu điểm danh dạng JSON cho GUI.
- **Quản lý người dùng:**
  - Đăng nhập, xác thực, quản lý session qua Flask session.
- **Upload ảnh, lưu dữ liệu:**
  - Nhận ảnh từ giao diện web, kiểm tra hợp lệ, lưu vào DB.
- **Gửi email QR:**
  - Tạo mã QR cho khách, gửi email xác nhận, lưu lịch sử đăng ký.
- **Đồng bộ Google Sheets:**
  - Khi có internet, backend tự động ghi lịch sử điểm danh lên Google Sheets.
  - Nếu offline, lưu vào file tạm, tự động sync khi online.

### 5. Google Sheets & Cloud Sync
- **Kết nối:**
  - Sử dụng Google API credentials, xác thực OAuth2.
- **Đồng bộ:**
  - Khi có điểm danh mới, ghi thêm dòng vào Google Sheets.
  - Nếu offline, lưu vào file tạm, tự động sync khi online.
- **Quản lý lịch sử:**
  - Có thể xem, xuất dữ liệu điểm danh trên cloud, backup dễ dàng.

### 6. QR Code
- **Đăng ký khách:**
  - Giao diện web cho phép khách đăng ký, sinh mã QR cá nhân.
- **Điểm danh qua QR:**
  - Khi quét QR, hệ thống kiểm tra thông tin, xác nhận điểm danh, lưu lịch sử.
- **Gửi email:**
  - Tự động gửi mã QR qua email cho khách, xác nhận tham dự.

### 7. TTS (Text-to-Speech)
- **Phát âm tên:**
  - Khi nhận diện thành công, phát âm tên người qua loa.
- **Cảnh báo:**
  - Phát âm cảnh báo khi phát hiện giả mạo, khuôn mặt lạ, trạng thái hệ thống.
- **Tùy chỉnh:**
  - Có thể bật/tắt log chi tiết, phát âm theo sự kiện.

### 8. Cảm biến PIR
- **Tiết kiệm tài nguyên:**
  - Khi không có chuyển động, tự động tắt webcam, dừng nhận diện.
  - Khi phát hiện chuyển động, tự động bật lại webcam, nhận diện tiếp.
- **Tích hợp sâu với GUI:**
  - Trạng thái cảm biến hiển thị trực tiếp trên giao diện.
  - Có thể phát âm thông báo khi chuyển trạng thái.

---

##### Triplet Loss (Facenet)
- **Triplet Loss** là hàm mất mát đặc biệt dùng để huấn luyện mạng nhận diện khuôn mặt:
  - Mỗi lần huấn luyện, lấy 3 ảnh: anchor (gốc), positive (cùng người), negative (khác người).
  - Mục tiêu: khoảng cách (embedding(anchor), embedding(positive)) phải nhỏ hơn (embedding(anchor), embedding(negative)) ít nhất một giá trị margin.
  - Công thức:
    - L = max(0, ||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin)
    - Trong đó: f là hàm embedding, a: anchor, p: positive, n: negative.
  - Nhờ đó, mạng học được không gian đặc trưng mà các khuôn mặt cùng người "tụ lại", khác người "cách xa".
  - **Ý nghĩa thực tế:**
    - Giúp hệ thống nhận diện tốt cả khi có người mới, chỉ cần thêm embedding mới, không cần train lại toàn bộ mạng.
    - Tăng khả năng phân biệt, giảm nhầm lẫn giữa các khuôn mặt giống nhau.

##### Inception Block (Facenet)
- **Inception block** là kiến trúc CNN đặc biệt, cho phép trích xuất đặc trưng ở nhiều tỷ lệ (multi-scale):
  - Mỗi block gồm nhiều nhánh song song:
    - Conv 1x1, Conv 3x3, Conv 5x5, MaxPooling 3x3.
    - Kết quả các nhánh được nối lại (concatenate) thành một tensor lớn.
  - **Ưu điểm:**
    - Mỗi nhánh học đặc trưng khác nhau: cạnh nhỏ, cạnh lớn, texture, pattern tổng thể.
    - Giảm số tham số nhờ Conv 1x1 (bottleneck), tăng tốc độ train.
  - **Trong Facenet:**
    - Các block Inception giúp mạng học được đặc trưng khuôn mặt ở nhiều tỷ lệ, góc nhìn, điều kiện ánh sáng khác nhau.

##### Kỹ thuật Augmentation trong train dữ liệu
- **Augmentation** là quá trình biến đổi ảnh gốc để tạo ra nhiều mẫu train đa dạng hơn:
  - **Các kỹ thuật thường dùng:**
    - Xoay (rotation), lật ngang (flip), thay đổi độ sáng/tương phản, crop, zoom, thêm nhiễu (noise).
    - Dịch chuyển (shift), co giãn (shear), làm mờ (blur).
  - **Ý nghĩa:**
    - Giúp model học được các biến thiên thực tế của khuôn mặt (góc nghiêng, ánh sáng, biểu cảm).
    - Giảm overfitting, tăng độ bền vững khi nhận diện ngoài thực tế.
  - **Trong hệ thống:**
    - Khi train lại, có thể sinh thêm ảnh augmented từ ảnh gốc để tăng số lượng mẫu cho mỗi người.
    - Đặc biệt quan trọng với các user có ít ảnh, giúp model nhận diện ổn định hơn.

##### Cách tính toán embedding và so khớp
- **Chuẩn hóa embedding:**
  - Sau khi qua Facenet, vector embedding được chuẩn hóa L2:
    - \( \text{embedding}_{norm} = \frac{\text{embedding}}{\|\text{embedding}\|_2} \ )
    - Đảm bảo mọi vector đều có độ dài (norm) = 1, giúp so sánh công bằng.
- **Cosine similarity:**
  - Đo độ tương đồng giữa hai vector embedding:
    - \( \text{cosine	sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} \ )
    - Vì đã chuẩn hóa, công thức rút gọn còn: \( a \cdot b \) (tích vô hướng).
    - Giá trị gần 1: rất giống nhau (cùng người), gần 0: khác nhau.
- **Ý nghĩa toán học:**
  - Embedding là điểm trong không gian nhiều chiều, mỗi chiều là một đặc trưng khuôn mặt.
  - So khớp = tìm vector gần nhất (cosine similarity lớn nhất) trong tập embedding đã lưu.
  - Ngưỡng (ví dụ 0.65) được chọn thực nghiệm để cân bằng giữa nhận diện đúng và tránh nhầm lẫn.
- **L2 distance (Euclidean):**
  - Ngoài cosine, có thể dùng khoảng cách Euclid: \( \|a-b\|_2 \ ).
  - Tuy nhiên, cosine thường ổn định hơn khi embedding đã chuẩn hóa.

##### Giải thích chi tiết các mạng con của MTCNN
- **P-Net (Proposal Network):**
  - **Input:** Ảnh gốc, nhiều tỷ lệ (image pyramid) để phát hiện mặt ở nhiều kích thước.
  - **Kiến trúc:**
    - 3 lớp convolution liên tiếp (Conv2D + PReLU), kernel nhỏ (3x3), stride 1.
    - 2 nhánh output:
      - 1 nhánh: xác suất là khuôn mặt (classification, sigmoid).
      - 1 nhánh: tọa độ box (bounding box regression, linear).
  - **Vai trò:**
    - Quét toàn bộ ảnh, đề xuất rất nhiều box có khả năng chứa mặt (có thể dư thừa, trùng lặp).
    - Rất nhanh, ưu tiên recall cao.
  - **Output:**
    - Danh sách box đề xuất, xác suất, tọa độ.

- **R-Net (Refine Network):**
  - **Input:** Crop các box từ P-Net, resize về 24x24.
  - **Kiến trúc:**
    - 3 lớp convolution + 1 fully connected, activation PReLU.
    - 2 nhánh output: xác suất là mặt, tọa độ box refined.
  - **Vai trò:**
    - Lọc lại các box có xác suất thấp, refine lại vị trí, loại bỏ false positive.
    - Tăng độ chính xác, giảm số box cần xử lý tiếp.
  - **Output:**
    - Danh sách box refined, xác suất, tọa độ.

- **O-Net (Output Network):**
  - **Input:** Crop các box từ R-Net, resize về 48x48.
  - **Kiến trúc:**
    - 4 lớp convolution + 2 fully connected, activation PReLU.
    - 3 nhánh output:
      - Xác suất là mặt (classification).
      - Tọa độ box refined cuối cùng (regression).
      - 5 điểm landmark (2 mắt, mũi, 2 miệng) (regression).
  - **Vai trò:**
    - Xác định chính xác vị trí, kích thước box cuối cùng.
    - Dự đoán landmark để căn chỉnh khuôn mặt (align), giúp bước nhận diện chính xác hơn.
  - **Output:**
    - Danh sách box cuối cùng, xác suất, tọa độ, landmark.

- **Quy trình tổng thể:**
  1. Ảnh gốc -> P-Net (đề xuất box) -> NMS (loại trùng lặp)
  2. Box còn lại -> R-Net (lọc, refine) -> NMS
  3. Box còn lại -> O-Net (refine, landmark) -> NMS
  4. Kết quả: các box mặt đã căn chỉnh, sẵn sàng cho nhận diện embedding.

### 8. So sánh với các phương pháp nhận diện khuôn mặt khác

#### 8.1. Phương pháp truyền thống
- **Eigenfaces, Fisherfaces (PCA/LDA):**
  - Dựa trên phân tích thành phần chính, giảm chiều dữ liệu ảnh thành vector đặc trưng.
  - **Ưu điểm:** Nhanh, dễ cài đặt, không cần nhiều tài nguyên.
  - **Nhược điểm:** Nhạy cảm với ánh sáng, góc nhìn, biểu cảm; độ chính xác thấp với dữ liệu thực tế.
- **LBPH (Local Binary Pattern Histogram):**
  - Trích xuất đặc trưng texture cục bộ, so khớp bằng histogram.
  - **Ưu điểm:** Chịu được thay đổi ánh sáng, đơn giản, chạy tốt trên thiết bị yếu.
  - **Nhược điểm:** Không phân biệt tốt với khuôn mặt giống nhau, kém hiệu quả với ảnh chất lượng thấp.

#### 8.2. Phương pháp học sâu (Deep Learning)
- **CNN truyền thống (LeNet, VGG, SVM+CNN):**
  - Dùng CNN để trích xuất đặc trưng, phân loại bằng softmax hoặc SVM.
  - **Ưu điểm:** Độ chính xác cao hơn truyền thống, học được đặc trưng phức tạp.
  - **Nhược điểm:** Không mở rộng tốt, muốn thêm người mới phải train lại toàn bộ mạng.
- **DeepID, DeepFace:**
  - Các mạng deep CNN đầu tiên cho nhận diện khuôn mặt, dùng softmax hoặc contrastive loss.
  - **Ưu điểm:** Độ chính xác cao, mở đầu cho kỷ nguyên deep learning trong nhận diện mặt.
  - **Nhược điểm:** Chưa tối ưu cho open-set, khó mở rộng linh hoạt.
- **ArcFace, CosFace, SphereFace:**
  - Các phương pháp embedding hiện đại, dùng loss function đặc biệt (ArcMargin, CosMargin) để tăng độ phân biệt.
  - **Ưu điểm:** Độ chính xác rất cao, robust với dữ liệu lớn, nhiều người.
  - **Nhược điểm:** Cần nhiều tài nguyên, khó triển khai real-time trên thiết bị nhúng.

#### 8.3. So sánh với pipeline Facenet + MTCNN của hệ thống
- **Ưu điểm:**
  - Kết hợp MTCNN (phát hiện, căn chỉnh) + Facenet (embedding, metric learning) cho phép nhận diện chính xác, mở rộng linh hoạt, phù hợp real-time.
  - Chỉ cần lưu embedding, không cần train lại khi thêm người mới.
  - Chạy tốt trên Raspberry Pi, PC cấu hình thấp, tối ưu cho ứng dụng thực tế.
  - Chống giả mạo tốt nhờ anti-spoofing, robust với nhiều điều kiện môi trường.
- **Nhược điểm:**
  - Độ chính xác có thể thấp hơn ArcFace trên tập dữ liệu cực lớn (>10.000 người).
  - Cần nhiều ảnh chất lượng tốt cho mỗi người để đạt hiệu quả tối ưu.

#### 8.4. Lý do chọn pipeline hiện tại cho hệ thống
- **Cân bằng giữa độ chính xác, tốc độ, khả năng mở rộng và tài nguyên phần cứng.**
- **Dễ triển khai, bảo trì, mở rộng, phù hợp với ứng dụng thực tế, giáo dục, doanh nghiệp nhỏ.**
- **Có thể tích hợp thêm các phương pháp hiện đại (ArcFace, CosFace) nếu cần nâng cấp về sau.**

#### 8.5. Bảng so sánh các mô hình phát hiện và nhận diện khuôn mặt

##### Bảng so sánh các mô hình **phát hiện khuôn mặt** (bổ sung)
| Mô hình      | Độ chính xác | Tốc độ | Hỗ trợ landmark | Tài nguyên | Chống giả mạo | Nhiều người cùng lúc | Ưu điểm | Nhược điểm |
|--------------|--------------|--------|-----------------|------------|---------------|---------------------|---------|------------|
| **MTCNN**    | Rất cao      | Trung bình | Có           | Trung bình | Không (cần anti-spoofing ngoài) | Có (multi-face) | Phát hiện đa mặt, căn chỉnh tốt | Chậm hơn Haar, cần GPU để real-time nhiều mặt |
| **Haar**     | Thấp-Trung bình | Rất nhanh | Không      | Thấp       | Không         | Có (multi-face)     | Nhẹ, dễ triển khai | Nhạy cảm ánh sáng, góc nhìn, nhiều false positive |
| **Dlib HOG** | Trung bình   | Trung bình | Không         | Thấp       | Không         | Có (multi-face)     | Nhẹ, không cần GPU | Không phát hiện landmark, kém với ảnh nghiêng |
| **SSD**      | Cao          | Cao    | Không           | Cao        | Không         | Có (multi-face)     | Nhanh, chính xác | Cần GPU, không căn chỉnh landmark |
| **YOLO**     | Cao          | Rất cao| Không           | Cao        | Không         | Có (multi-face)     | Nhanh, phát hiện nhiều đối tượng | Cần GPU, không chuyên biệt cho mặt |

##### Bảng so sánh các mô hình **nhận diện khuôn mặt** (bổ sung)
| Mô hình      | Độ chính xác (LFW) | Tốc độ | Kích thước embedding | Tài nguyên | Chống giả mạo | Nhiều người cùng lúc | Ưu điểm | Nhược điểm |
|--------------|--------------------|--------|----------------------|------------|---------------|---------------------|---------|------------|
| **ArcFace**  | 99.83%             | Trung bình | 512              | Cao        | Không (cần anti-spoofing ngoài) | Có (multi-face) | Độ chính xác rất cao, robust | Cần nhiều tài nguyên, khó real-time nhúng |
| **Facenet**  | 99.65%             | Trung bình | 128/512           | Trung bình | Không (cần anti-spoofing ngoài) | Có (multi-face) | Mở rộng dễ, nhận diện tốt | Cần nhiều ảnh chất lượng tốt |
| **VGGFace**  | 98.95%             | Thấp    | 2622                | Cao        | Không         | Có (multi-face)     | Độ chính xác tốt, phổ biến | Embedding lớn, chậm |
| **DeepFace** | 98.97%             | Trung bình | 4096              | Cao        | Không         | Có (multi-face)     | Dễ dùng, tích hợp sẵn | Embedding lớn, chậm |
| **OpenFace** | 93.80%             | Cao     | 128                 | Thấp       | Không         | Có (multi-face)     | Nhẹ, nhanh, dễ triển khai | Độ chính xác thấp hơn |
| **Dlib ResNet** | 99.38%          | Cao     | 128                 | Thấp       | Không         | Có (multi-face)     | Nhẹ, nhanh | Độ chính xác thấp hơn ArcFace, Facenet |
| **SFace**    | 99.55%             | Trung bình | 128              | Trung bình | Không         | Có (multi-face)     | Cân bằng tốc độ/độ chính xác | Chưa phổ biến rộng rãi |

> **Lưu ý:** Chống giả mạo (anti-spoofing) thường phải tích hợp thêm module chuyên biệt (ví dụ: Fasnet, liveness detection) vào pipeline nhận diện.

---

#### Giải thích học thuật về chống giả mạo và nhận diện nhiều người cùng lúc

- **Chống giả mạo (Anti-spoofing / Liveness Detection):**
  - Là khả năng phân biệt khuôn mặt thật (người thật) với các hình thức giả mạo như ảnh in, video, mặt nạ.
  - Các phương pháp phổ biến:
    - **Texture-based:** Phân tích đặc trưng texture (LBP, CNN) để phát hiện ảnh in, màn hình.
    - **Motion-based:** Phân tích chuyển động tự nhiên (nháy mắt, cử động môi, head pose).
    - **Depth-based:** Sử dụng camera 3D, IR để đo chiều sâu khuôn mặt.
    - **Deep learning:** Sử dụng các mạng CNN chuyên biệt (ví dụ: Fasnet) để học đặc trưng giả mạo.
  - **Trong hệ thống:** Sử dụng Fasnet (anti-spoofing) kết hợp với pipeline nhận diện để tăng độ an toàn.

- **Nhận diện nhiều người cùng lúc (Multi-face Recognition):**
  - Là khả năng phát hiện và nhận diện đồng thời nhiều khuôn mặt trong cùng một ảnh/frame.
  - **Yêu cầu:**
    - Mô hình phát hiện phải trả về tất cả bounding box khuôn mặt (multi-face detection).
    - Pipeline nhận diện phải xử lý song song embedding cho từng khuôn mặt.
    - Hệ thống phải tối ưu tốc độ để đảm bảo real-time.
  - **Trong hệ thống:**
    - MTCNN phát hiện tất cả khuôn mặt, trả về list box.
    - Facenet sinh embedding cho từng box, so khớp từng người độc lập.
    - Có thể nhận diện, điểm danh nhiều người cùng lúc trong một khung hình.

---

### Giải thích học thuật chi tiết từng khái niệm trong nhận diện/phát hiện khuôn mặt

#### 1. Phát hiện khuôn mặt (Face Detection)
- **Phát hiện khuôn mặt** là quá trình xác định vị trí của tất cả các khuôn mặt xuất hiện trong một bức ảnh hoặc một khung hình video. Kết quả là các hình chữ nhật (bounding box) bao quanh từng khuôn mặt.
- **Ví dụ thực tế:** Khi bạn mở camera điện thoại, hệ thống sẽ vẽ các khung vuông quanh mặt người để lấy nét – đó là phát hiện khuôn mặt.
- **Cách hoạt động:**
  - Ảnh đầu vào được quét qua nhiều vị trí và kích thước khác nhau để tìm ra vùng có khả năng là khuôn mặt.
  - Các mô hình hiện đại dùng mạng nơ-ron tích chập (CNN) để học ra đặc trưng khuôn mặt từ dữ liệu lớn, giúp phát hiện chính xác hơn trong nhiều điều kiện ánh sáng, góc nhìn.

#### 2. Landmark (Điểm đặc trưng khuôn mặt)
- **Landmark** là các điểm đặc biệt trên khuôn mặt như khóe mắt, chóp mũi, khóe miệng, v.v. Việc xác định các điểm này giúp căn chỉnh khuôn mặt về cùng một hướng, giúp nhận diện chính xác hơn.
- **Ví dụ:** Khi bạn thấy các chấm nhỏ trên mặt trong ứng dụng chỉnh sửa ảnh, đó là các landmark.
- **Cách hoạt động:** Mô hình sẽ dự đoán vị trí các điểm này dựa trên hình dạng tổng thể của khuôn mặt.

#### 3. Nhận diện khuôn mặt (Face Recognition)
- **Nhận diện khuôn mặt** là quá trình xác định danh tính của từng khuôn mặt đã được phát hiện. Tức là, hệ thống sẽ trả lời: "Đây là ai?".
- **Cách hoạt động:**
  - Đầu tiên, khuôn mặt được căn chỉnh và cắt ra từ ảnh gốc.
  - Sau đó, khuôn mặt này được chuyển thành một dãy số đặc trưng (gọi là embedding).
  - Dãy số này sẽ được so sánh với các dãy số đã lưu trong cơ sở dữ liệu để tìm ra người giống nhất.

#### 4. Embedding là gì?
- **Embedding** là một dãy số (vector) đại diện cho các đặc trưng duy nhất của khuôn mặt. Mỗi khuôn mặt sẽ có một embedding khác nhau, nhưng các khuôn mặt của cùng một người sẽ có embedding gần giống nhau.
- **Ví dụ:** Nếu coi mỗi khuôn mặt là một điểm trong không gian 128 chiều, thì embedding chính là tọa độ của điểm đó.
- **Cách tạo embedding:** Mô hình học sâu (deep learning) sẽ học cách chuyển ảnh khuôn mặt thành embedding sao cho các khuôn mặt giống nhau thì embedding gần nhau, khác nhau thì embedding cách xa nhau.

#### 5. So khớp embedding là gì?
- **So khớp embedding** là quá trình so sánh hai dãy số (embedding) để xác định xem hai khuôn mặt có phải cùng một người hay không.
- **Cách hoạt động:**
  - Tính khoảng cách giữa hai embedding (thường dùng khoảng cách cosine hoặc Euclidean).
  - Nếu khoảng cách nhỏ hơn một ngưỡng nhất định, hệ thống kết luận hai khuôn mặt là cùng một người.
- **Ví dụ:** Nếu embedding của A là [0.1, 0.2, ...] và của B là [0.11, 0.19, ...], khoảng cách nhỏ → có thể là cùng người.

#### 6. Metric learning là gì?
- **Metric learning** là phương pháp dạy cho mô hình biết cách đo "độ giống nhau" giữa các đối tượng (ở đây là khuôn mặt).
- **Cách hoạt động:**
  - Mô hình được huấn luyện để embedding của cùng một người thì gần nhau, embedding của người khác thì xa nhau.
  - Sử dụng các hàm mất mát đặc biệt như Triplet Loss để tối ưu hóa khoảng cách này.
- **Ví dụ thực tế:** Nếu bạn có nhiều ảnh của từng người, mô hình sẽ học để các ảnh của cùng một người luôn gần nhau trong không gian embedding.

#### 7. Closed-set và Open-set là gì?
- **Closed-set:** Hệ thống chỉ nhận diện trong tập người đã biết trước (có trong database). Nếu gặp người lạ, hệ thống sẽ cố gắng gán vào một người đã biết.
- **Open-set:** Hệ thống có thể phát hiện người lạ (không có trong database) bằng cách kiểm tra khoảng cách embedding. Nếu khoảng cách lớn hơn ngưỡng, sẽ trả về "Unknown".
- **Ví dụ:**
  - Closed-set: Nhận diện học sinh trong lớp, chỉ có danh sách cố định.
  - Open-set: Nhận diện khách ra vào công ty, có thể gặp người mới.

#### 8. Cosine similarity là gì?
- **Cosine similarity** là một cách đo độ giống nhau giữa hai vector bằng cách tính góc giữa chúng. Nếu hai vector cùng hướng, cosine similarity gần 1 (rất giống nhau); nếu vuông góc, gần 0 (khác nhau).
- **Cách tính:**
  - Lấy tích vô hướng của hai vector, chia cho tích độ dài của chúng.
- **Ứng dụng:** Dùng để so sánh embedding khuôn mặt.

#### 9. Triplet Loss là gì?
- **Triplet Loss** là một hàm mất mát dùng để huấn luyện mô hình sao cho embedding của cùng một người thì gần nhau, embedding của người khác thì xa nhau.
- **Cách hoạt động:**
  - Mỗi lần huấn luyện, lấy 3 ảnh: ảnh gốc (anchor), ảnh cùng người (positive), ảnh khác người (negative).
  - Mục tiêu: khoảng cách giữa anchor và positive nhỏ hơn anchor và negative ít nhất một giá trị nhất định (margin).
- **Ví dụ:**
  - Ảnh A1 (anchor), A2 (positive) cùng người, B1 (negative) khác người. Mô hình sẽ học để embedding(A1) gần embedding(A2) và xa embedding(B1).

#### 10. Augmentation là gì?
- **Augmentation** là kỹ thuật tạo ra nhiều phiên bản khác nhau của ảnh gốc bằng cách xoay, lật, thay đổi sáng/tối, thêm nhiễu, v.v. Mục đích là giúp mô hình học được nhiều tình huống thực tế, tăng khả năng tổng quát.
- **Ví dụ:** Một ảnh khuôn mặt có thể được xoay 10 độ, làm mờ nhẹ, đổi màu để tạo thành nhiều ảnh mới.

#### 11. Regularization là gì?
- **Regularization** là các kỹ thuật giúp mô hình không bị học thuộc lòng dữ liệu huấn luyện (overfitting), từ đó hoạt động tốt hơn với dữ liệu mới.
- **Các phương pháp phổ biến:**
  - **Dropout:** Ngẫu nhiên bỏ qua một số neuron khi huấn luyện.
  - **Batch normalization:** Chuẩn hóa đầu ra của mỗi lớp giúp mô hình ổn định hơn.
  - **Augmentation:** Như đã giải thích ở trên.

#### 12. NMS (Non-Maximum Suppression) là gì?
- **NMS** là thuật toán loại bỏ các khung phát hiện trùng lặp, chỉ giữ lại khung có xác suất cao nhất cho mỗi đối tượng.
- **Cách hoạt động:**
  - Nếu nhiều khung bao quanh cùng một khuôn mặt, chỉ giữ lại khung có điểm số cao nhất, loại bỏ các khung còn lại nếu chúng chồng lấn quá nhiều.
- **Ví dụ:** Khi phát hiện 3 khung quanh cùng một mặt, NMS sẽ chỉ giữ lại 1 khung tốt nhất.

#### 13. Chống giả mạo (Anti-spoofing / Liveness Detection) là gì?
- **Chống giả mạo** là khả năng phân biệt khuôn mặt thật với các hình thức giả mạo như ảnh in, video, mặt nạ.
- **Cách hoạt động:**
  - Phân tích chuyển động tự nhiên (nháy mắt, cử động môi), kiểm tra texture da, hoặc dùng camera đo chiều sâu để xác định khuôn mặt có phải người thật không.
- **Ví dụ:** Nếu bạn đưa ảnh chụp khuôn mặt trước camera, hệ thống sẽ phát hiện đó là ảnh tĩnh, không phải người thật.

#### 14. Nhận diện nhiều người cùng lúc là gì?
- **Nhận diện nhiều người cùng lúc** là khả năng phát hiện và nhận diện đồng thời tất cả khuôn mặt xuất hiện trong một ảnh hoặc khung hình.
- **Cách hoạt động:**
  - Mô hình phát hiện sẽ trả về danh sách các khung khuôn mặt.
  - Mỗi khuôn mặt được cắt ra, chuyển thành embedding và so khớp độc lập.
- **Ví dụ:** Ảnh lớp học có 10 người, hệ thống sẽ phát hiện và nhận diện đủ cả 10 người trong một lần xử lý.

---
