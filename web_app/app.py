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
import sys
from qrcode.constants import ERROR_CORRECT_M
from qrcode.image.pil import PilImage

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

print("ENV GMAIL_EMAIL:", os.environ.get("GMAIL_EMAIL"))
print("ENV GMAIL_PASSWORD:", os.environ.get("GMAIL_PASSWORD"))

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
        username = request.form['username']
        password = request.form['password']

        if username == VALID_USERNAME and password == VALID_PASSWORD:
            session['logged_in'] = True
            session.permanent = True  # Kích hoạt session vĩnh viễn với thời gian sống đã cấu hình
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
    return render_template('train_face.html')

@app.route('/save_face', methods=['POST'])
def save_face():
    data = request.get_json()
    if not data or 'name' not in data or 'images' not in data:
        print("Lỗi: Dữ liệu không hợp lệ")
        return jsonify({'error': 'Dữ liệu không hợp lệ'}), 400

    name = data['name'].strip()
    images = data['images']
    
    try:
        # Chuyển đổi tên thư mục thành không dấu
        safe_name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
        # Thay thế khoảng trắng bằng dấu gạch dưới
        safe_name = safe_name.replace(' ', '_')
        employee_folder = os.path.join(EMPLOYEE_IMAGES_FOLDER, safe_name)
        print(f"Đường dẫn thư mục: {employee_folder}")
        
        # Kiểm tra xem thư mục đã tồn tại chưa
        if not os.path.exists(employee_folder):
            print(f"Tạo thư mục mới: {employee_folder}")
            os.makedirs(employee_folder, exist_ok=True)
        
        for idx, image_data_url in enumerate(images):
            print(f"Đang xử lý ảnh {idx + 1}/{len(images)}")
            try:
                image_data = base64.b64decode(image_data_url.split(',')[1])
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f"Lỗi: Không thể giải mã ảnh {idx + 1}")
                    continue
                    
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(employee_folder, f'{timestamp}.jpg')
                print(f"Đang lưu ảnh vào: {filename}")
                
                # Kiểm tra quyền ghi trước khi lưu
                if not os.access(employee_folder, os.W_OK):
                    print(f"Không có quyền ghi vào thư mục: {employee_folder}")
                    return jsonify({'error': 'Không có quyền ghi vào thư mục'}), 500
                
                success = cv2.imwrite(filename, img)
                if not success:
                    print(f"Lỗi: Không thể lưu ảnh {filename}")
                else:
                    print(f"Đã lưu ảnh thành công: {filename}")
                    
            except Exception as e:
                print(f"Lỗi khi xử lý ảnh {idx + 1}: {str(e)}")
                continue

        return jsonify({'success': True})

    except Exception as e:
        print(f"Lỗi khi lưu hình ảnh: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
