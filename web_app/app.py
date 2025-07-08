### Các thư viện cần thiết cho hệ thống điểm danh
# gspread - Thư viện thao tác với Google Sheets API
import gspread
# Credentials - Xác thực tài khoản dịch vụ Google
from google.oauth2.service_account import Credentials
# pandas - Phân tích và xử lý dữ liệu dạng bảng
import pandas as pd
# Flask - Framework phát triển web
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
# base64 - Mã hóa/giải mã dữ liệu hình ảnh
import base64
# numpy - Xử lý mảng số học và ma trận (cần cho xử lý ảnh)
import numpy as np
# cv2 (OpenCV) - Thư viện xử lý hình ảnh và thị giác máy tính
import cv2
# datetime - Xử lý thời gian và ngày tháng
from datetime import datetime, timedelta
# qrcode - Tạo mã QR cho khách tham dự
import qrcode
# BytesIO - Xử lý dữ liệu nhị phân trong bộ nhớ
from io import BytesIO
# wraps - Decorator cho chức năng xác thực
from functools import wraps
# smtplib - Gửi email thông báo
import smtplib
# MIMEMultipart, MIMEText, MIMEImage - Tạo nội dung email có định dạng
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
# load_dotenv - Đọc biến môi trường từ file .env
from dotenv import load_dotenv
# os - Thao tác với hệ thống tập tin và đường dẫn
import os 
# subprocess - Thực thi và quản lý tiến trình con
import subprocess
# time - Xử lý thời gian và tạo độ trễ
import time
# atexit - Đăng ký hàm được gọi khi chương trình kết thúc
import atexit
# unicodedata - Xử lý chuỗi Unicode (loại bỏ dấu tiếng Việt)
import unicodedata
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qrcode.constants import ERROR_CORRECT_M
from qrcode.image.pil import PilImage
from db import get_db_connection
from recognition_simple import RecognitionSimple

#sys.stdout = open('/tmp/webapp.log', 'a')
#sys.stderr = open('/tmp/webapp.log', 'a')

app = Flask(__name__, template_folder='./templates')
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'linh_vo_dich_2025')  # Sử dụng biến môi trường, mặc định là 'linh_vo_dich_2025'

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=1)  

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMPLOYEE_IMAGES_FOLDER = os.path.join(PROJECT_ROOT, 'images_attendance')
os.makedirs(EMPLOYEE_IMAGES_FOLDER, exist_ok=True)

# Tài khoản và mật khẩu cứng mã (CHỈ DÙNG CHO MỤC ĐÍCH DEMO)
VALID_USERNAME = 'admin'
VALID_PASSWORD = '1'
# Thông tin xác thực Google Sheets
CREDENTIALS_FILE = os.path.join(PROJECT_ROOT, 'credentials', 'face-attendance.json')
SPREADSHEET_ID = '1CbHHt5gwfOJzEMzrzGoEpvOyjMXYOk_AeEVtfRx_8pk'  # Thay thế bằng ID Google Sheet của bạn

# Thông tin tài khoản Gmail của bạn
GMAIL_EMAIL = os.environ.get("GMAIL_EMAIL")
GMAIL_PASSWORD = os.environ.get("GMAIL_PASSWORD")
NGROK_URL = os.getenv("NGROK_URL", "")  # Thêm biến môi trường NGROK_URL

# Khởi tạo RecognitionSimple dùng riêng cho nhận diện ảnh upload
recognition_simple = None

def get_recognition_simple():
    global recognition_simple
    if recognition_simple is None:
        recognition_simple = RecognitionSimple(model_path="models/train_FN.h5")
    return recognition_simple

def start_ngrok_tunnel():
    try:
        # Dừng tất cả các tunnel đang chạy
        subprocess.run(['ngrok', 'stop'], capture_output=True)
        print("Đã dừng tất cả tunnel ngrok đang chạy")

        # Chạy lệnh ngrok start linh
        print("Đang khởi động ngrok tunnel 'linh'...")
        ngrok_process = subprocess.Popen(['ngrok', 'start', 'linh'], 
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       text=True)
        
        # Đợi một chút để ngrok khởi động
        time.sleep(5)  # Tăng thời gian chờ lên 5s
        
        # Kiểm tra xem ngrok đã chạy thành công chưa
        if ngrok_process.poll() is not None:
            print("Lỗi: Không thể khởi động ngrok. Vui lòng kiểm tra lại cấu hình.")
            return None
            
        # Lấy URL ngrok từ API
        try:
            import requests
            # Thử nhiều lần để đảm bảo ngrok đã sẵn sàng
            max_retries = 3
            for i in range(max_retries):
                try:
                    response = requests.get("http://localhost:4040/api/tunnels")
                    if response.status_code == 200:
                        tunnels = response.json()["tunnels"]
                        if tunnels:
                            ngrok_url = tunnels[0]["public_url"]
                            os.environ["NGROK_URL"] = ngrok_url
                            print(f"Ngrok đã được khởi động thành công!")
                            print(f"URL ngrok: {ngrok_url}")
                            return ngrok_process
                except requests.exceptions.ConnectionError:
                    if i < max_retries - 1:
                        print(f"Đang thử lại lần {i+1}...")
                        time.sleep(2)
                    continue
                    
            print("Không thể lấy URL ngrok sau nhiều lần thử")
            return ngrok_process
            
        except Exception as e:
            print(f"Lỗi khi lấy URL ngrok: {e}")
            return ngrok_process
            
    except Exception as e:
        print(f"Lỗi khi khởi động ngrok: {e}")
        return None

def cleanup():
    try:
        # Dừng tất cả các tunnel ngrok
        subprocess.run(['ngrok', 'stop'], capture_output=True)
        print("Đã dừng tất cả tunnel ngrok")
    except Exception as e:
        print(f"Lỗi khi dừng ngrok: {e}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session or not session['logged_in']:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username'].strip().lower()
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE LOWER(username)=? AND password=?", (username, password))
        user = cur.fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['user_id'] = user['user_id']
            session['username'] = user['username']
            session['full_name'] = user['full_name']
            session.permanent = True
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                return redirect(url_for('index'))
        else:
            error = 'Tài khoản hoặc mật khẩu không đúng'

    return render_template('login.html', error=error)

@app.route('/logout')  # Thêm route để đăng xuất thủ công
def logout():
    session.pop('logged_in', None)
    return render_template('login.html')

@app.route('/train_face')
@login_required
def train_face_page():
    full_name = session.get('full_name', '')
    return render_template('train_face.html', full_name=full_name)

@app.route('/save_face', methods=['POST'])
@login_required
def save_face():
    data = request.get_json()
    if not data or 'images' not in data:
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

    images = data['images']
    user_id = session.get('user_id')
    full_name = session.get('full_name')
    if not user_id or not full_name:
        return jsonify({'error': 'Không xác định được user'}), 400

    recog = get_recognition_simple()

    # Kiểm tra dữ liệu training
    if recog.train_data is None or recog.label_encoder is None:
        return jsonify({'error': 'Hệ thống chưa có dữ liệu training hoặc label encoder. Vui lòng huấn luyện lại mô hình.'}), 500

    # Kiểm tra user đã có trong model chưa
    user_in_model = False
    if recog.label_encoder is not None:
        all_names = set(recog.label_encoder.inverse_transform(recog.train_data['labels']))
        user_in_model = full_name.strip().lower() in [n.strip().lower() for n in all_names]

    valid_images = []
    invalid_indices = []
    log_messages = []
    for idx, image_data_url in enumerate(images):
        try:
            image_data = base64.b64decode(image_data_url.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                invalid_indices.append(idx)
                log_messages.append(f"Ảnh {idx}: Không đọc được ảnh.")
                continue
            face_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_img = cv2.resize(face_img, (160, 160))
            name, confidence = recog.predict_name(face_img)
            log_messages.append(f"Ảnh {idx}: {name} (conf={confidence:.2f})")
            if name == "Unknown" or confidence <= 0.65:
                invalid_indices.append(idx)
                continue
            # Nếu user đã có trong model, chỉ lưu nếu đúng tên
            if user_in_model:
                if confidence < recog.FACE_RECOGNITION_THRESHOLD:
                    invalid_indices.append(idx)
                    log_messages.append(f"Ảnh {idx}: Không đủ độ tin cậy.")
                    continue
                if name.strip().lower() != full_name.strip().lower():
                    invalid_indices.append(idx)
                    log_messages.append(f"Ảnh {idx}: Không trùng tên user.")
                    continue
                valid_images.append((idx, img))
                log_messages.append(f"Ảnh {idx}: Nhận diện thành công cho user {full_name}.")
            else:
                # Nếu user chưa có trong model, chỉ lưu nếu không trùng ai
                if confidence >= recog.FACE_RECOGNITION_THRESHOLD:
                    invalid_indices.append(idx)
                    log_messages.append(f"Ảnh {idx}: Gương mặt này đã trùng user ({name}), không được phép lưu.")
                    continue
                valid_images.append((idx, img))
                log_messages.append(f"Ảnh {idx}: Đã lưu ảnh thành công cho user mới.")
        except Exception as e:
            log_messages.append(f"Ảnh {idx}: Lỗi nhận diện: {e}")
            invalid_indices.append(idx)

    if not valid_images:
        return jsonify({'error': f'Không có ảnh nào hợp lệ với tài khoản ({full_name}).', 'log': log_messages}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    user_folder = os.path.join(EMPLOYEE_IMAGES_FOLDER, f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)
    saved_paths = []
    for idx, img in valid_images:
        try:
            filename = os.path.join(user_folder, f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{idx}.jpg")
            cv2.imwrite(filename, img)
            saved_paths.append(filename)
            # Cập nhật DB: thêm vào bảng face_profiles
            cur.execute("INSERT INTO face_profiles(user_id, image_path) VALUES (?, ?)", (user_id, filename))
        except Exception as e:
            log_messages.append(f"Ảnh {idx}: Lỗi lưu ảnh: {e}")
    conn.commit()
    conn.close()

    # In log chi tiết ra terminal
    for log in log_messages:
        print(log)

    return jsonify({'success': True, 'message': f'Đã lưu {len(saved_paths)} ảnh hợp lệ, loại bỏ {len(invalid_indices)} ảnh không đúng.', 'log': log_messages})

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',  # Chỉ đọc
]

def get_attendance_data():
    """Lấy dữ liệu điểm danh từ Google Sheet."""
    try:
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
        print(f"Đã load credentials từ {CREDENTIALS_FILE}")
        gc = gspread.authorize(creds)
        print("Đã authorize với Google Sheets")
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)
        print(f"Đã mở spreadsheet {SPREADSHEET_ID}")
        worksheet = spreadsheet.worksheet('Attendance')  # Thay 'attendance' bằng 'sheet1' hoặc tên sheet cụ thể
        print("Đã chọn worksheet")
        data = worksheet.get_all_values()
        print(f"Đã lấy dữ liệu: {len(data)} dòng")
        if data:
            headers = list(map(str, data[0]))
            values = [list(map(str, row)) for row in data[1:]]
            df = pd.DataFrame(values, columns=np.array(headers))
            return df
        return None
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ Google Sheet: {e}")
        return None

def send_qr_email(recipient_email, guest_name, qrcode_image_bytes, guest_info):
    """Gửi mã QR và thông tin khách đến email."""
    print("Gửi email tới:", recipient_email)
    print("GMAIL_EMAIL:", GMAIL_EMAIL)
    print("GMAIL_PASSWORD:", GMAIL_PASSWORD)
    sender_email = GMAIL_EMAIL
    sender_password = GMAIL_PASSWORD

    if not sender_email or not recipient_email:
        print("Thiếu thông tin email người gửi hoặc người nhận.")
        return False

    if not sender_email or not sender_password:
        print("Thiếu thông tin tài khoản Gmail hoặc mật khẩu.")
        return False

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = 'Mã QR Tham Dự Sự Kiện'

    body = f"""
    Xin chào {guest_name},

    Cảm ơn bạn đã đăng ký tham dự sự kiện của chúng tôi.
    Đây là mã QR của bạn, vui lòng sử dụng nó để điểm danh khi đến.

    Thông tin bạn đã đăng ký:
    Họ và tên: {guest_info['guest_name']}
    Email: {guest_info['guest_email']}
    Thông tin thêm: {guest_info['guest_info']}

    Vui lòng giữ mã QR này cẩn thận.

    Trân trọng,
    Hệ thống Điểm danh
    """
    message.attach(MIMEText(body, 'plain'))

    # Đính kèm ảnh QR
    image = MIMEImage(qrcode_image_bytes)
    image.add_header('Content-Disposition', 'attachment', filename='guest_qrcode.png')
    message.attach(image)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True
    except Exception as e:
        print(f"Lỗi khi gửi email: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/guest')
def guest_register():
    return render_template('guest.html')

@app.route('/generate_qr', methods=['POST'])
def generate_qr():
    if request.method == 'POST':
        guest_name = request.form['guest_name']
        guest_email = request.form['guest_email']
        guest_info_raw = request.form['guest_info']
        
        # Tạo chuỗi dữ liệu với encoding UTF-8
        guest_data = f"Tên: {guest_name}\nEmail: {guest_email}\nThông tin: {guest_info_raw}"
        
        # Tạo QR code với các tham số tối ưu cho tiếng Việt
        qr = qrcode.QRCode(
            version=1,  # Version cố định để QR code nhỏ gọn
            error_correction=ERROR_CORRECT_M,  # Mức độ sửa lỗi trung bình
            box_size=10, 
            border=4,
        )
        
        # Thêm dữ liệu với encoding UTF-8
        qr.add_data(guest_data.encode('utf-8'))
        qr.make(fit=True)

        # Tạo ảnh QR với độ tương phản cao
        img = qr.make_image(fill_color="black", back_color="white", image_factory=PilImage)
        
        # Tăng kích thước ảnh để dễ đọc hơn
        img = img.resize((400, 400), resample=1)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        qrcode_image_bytes = buffer.getvalue()
        qrcode_image_base64 = base64.b64encode(qrcode_image_bytes).decode('utf-8')
        qrcode_image_data_url = f"data:image/png;base64,{qrcode_image_base64}"

        guest_info = {
            'guest_name': guest_name,
            'guest_email': guest_email,
            'guest_info': guest_info_raw
        }

        # Gửi email chứa mã QR
        if send_qr_email(guest_email, guest_name, qrcode_image_bytes, guest_info):
            return render_template('guest.html', qrcode_image=qrcode_image_data_url, guest_info=guest_info, message='Mã QR đã được gửi đến email của bạn!')
        else:
            return render_template('guest.html', qrcode_image=qrcode_image_data_url, guest_info=guest_info, error=f'Có lỗi xảy ra khi gửi email.')

    return redirect(url_for('guest_register'))

@app.route('/face_data')
def face_data():
    print("Đang truy cập route /face_data")
    attendance_data = get_attendance_data()
    print("Đã lấy dữ liệu điểm danh")
    return render_template('face_data.html', attendance_data=attendance_data)

@app.route('/api/attendance')
def api_attendance():
    """API endpoint để trả về dữ liệu điểm danh dưới dạng JSON."""
    try:
        attendance_data = get_attendance_data()
        if attendance_data is not None:
            # Chuyển DataFrame thành list of dictionaries
            data_list = attendance_data.to_dict('records')
            return jsonify(data_list)
        else:
            return jsonify({'error': 'Không thể lấy dữ liệu điểm danh'}), 500
    except Exception as e:
        print(f"Lỗi API attendance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    # Đăng ký hàm cleanup để chạy khi thoát chương trình
    atexit.register(cleanup)
    
    # Khởi động ngrok
    ngrok_process = start_ngrok_tunnel()
    if ngrok_process is None:
        print("Không thể khởi động ngrok. Ứng dụng sẽ chạy ở chế độ localhost.")
    
    # Khởi động Flask app với debug mode
    app.run(host='127.0.0.1', port=5000)
