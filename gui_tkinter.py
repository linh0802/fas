import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
from PIL import Image, ImageTk
import cv2
import threading
import subprocess
import time
import requests
import os
import sys
import queue
import pandas as pd
import numpy as np
from datetime import datetime
from recognition_class import RecognitionSystem
import gc
import signal
import psutil
import textwrap
import logging
from smart_tts import play_name_smart
# Tạo thư mục logs nếu chưa có
os.makedirs('logs', exist_ok=True)
log_filename = datetime.now().strftime('logs/face_recognition_%Y%m%d_%H%M%S.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Nếu muốn log ra cả console
    ]
)

# Đường dẫn các file/script
TRAIN_SCRIPT = 'finish_train.py'
API_ATTENDANCE_URL = 'http://127.0.0.1:5000/api/attendance'
WEB_APP_SCRIPT = 'web_app/app.py'

DARK_BG = '#181a20'
DARK_PANEL = '#23272f'
DARK_ACCENT = '#00bcd4'
DARK_TEXT = '#f1f1f1'
DARK_BTN = '#222c36'
DARK_BTN_HOVER = '#00bcd4'

class EnlargedFaceWindow(tk.Toplevel):
    def __init__(self, parent, name, image):
        super().__init__(parent)
        self.title(f"Khuôn mặt: {name}")
        self.configure(bg=DARK_PANEL)
        self.image_reference = None # Thuộc tính để giữ tham chiếu ảnh

        # Lấy kích thước màn hình
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        w, h = img.size
        # Phóng to nhưng không vượt quá màn hình (trừ lề 100px)
        max_w = screen_w - 100
        max_h = screen_h - 100
        scale = min(max_w / w, max_h / h, 4)  # Không phóng quá 4 lần, và không vượt màn hình
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_reference = imgtk # Giữ tham chiếu

        label = tk.Label(self, image=self.image_reference, bg=DARK_PANEL)
        label.pack(padx=20, pady=20)
        
        self.transient(parent)
        try:
            self.grab_set()
        except tk.TclError:
            pass  # Bỏ qua lỗi grab nếu window không viewable

class AttendanceDataFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=DARK_BG)
        self.controller = controller
        # --- Layout ---
        title = tk.Label(self, text='DỮ LIỆU ĐIỂM DANH', font=('Trebuchet MS', 22, 'bold'), fg='white', bg=DARK_ACCENT)
        title.pack(fill='x', pady=(0, 5))

        # Bảng dữ liệu điểm danh
        data_frame = tk.Frame(self, bg=DARK_BG)
        data_frame.pack(fill='x', padx=10, pady=5)
        y_scrollbar = ttk.Scrollbar(data_frame, orient='vertical')
        self.attendance_display = tk.Text(
            data_frame,
            wrap="none",
            font=("Courier", 16),
            bg=DARK_PANEL,
            fg=DARK_TEXT,
            borderwidth=0,
            highlightthickness=0,
            height=15,
            yscrollcommand=y_scrollbar.set
        )
        self.attendance_display.tag_configure('bold_header', font=("Courier", 16, "bold"))
        y_scrollbar.config(command=self.attendance_display.yview)
        y_scrollbar.pack(side='right', fill='y')
        self.attendance_display.pack(fill="x", expand=False, padx=5, pady=5)
        self.attendance_display.config(state="disabled")
        # Hỗ trợ kéo bằng cảm ứng
        self.attendance_display.bind('<Button-1>', self.on_text_click)
        self.attendance_display.bind('<B1-Motion>', self.on_text_drag)
        self.attendance_display.bind('<ButtonRelease-1>', self.on_text_release)
        self.scroll_start_y = 0

        # Nút điều khiển luôn ở dưới
        btn_frame = tk.Frame(self, bg=DARK_BG)
        btn_frame.pack(fill='x', pady=5)
        self.refresh_btn = ttk.Button(btn_frame, text='Làm mới', command=self.load_attendance_data)
        self.refresh_btn.pack(side='left', padx=10, ipadx=10, ipady=5)
        self.new_person_btn = ttk.Button(btn_frame, text='Thêm người mới', command=lambda: controller.show_frame(DataEntryFrame))
        self.new_person_btn.pack(side='left', padx=10, ipadx=10, ipady=5)
        self.back_btn = ttk.Button(btn_frame, text='Quay lại nhận diện', command=lambda: controller.show_frame(RecognitionFrame))
        self.back_btn.pack(side='right', padx=10, ipadx=10, ipady=5)

        # Log hệ thống luôn ở dưới cùng
        self.log_text = scrolledtext.ScrolledText(self, font=('Consolas', 12), height=4, state='disabled', bg='#23272f', fg=DARK_TEXT)
        self.log_text.pack(side='bottom', fill='x', padx=10, pady=5)
        # Thêm hỗ trợ vuốt cảm ứng cho log_text
        self.log_text.bind('<Button-1>', self._log_on_click)
        self.log_text.bind('<B1-Motion>', self._log_on_drag)
        self.log_text.bind('<ButtonRelease-1>', self._log_on_release)
        self._log_scroll_start_y = 0
        self._log_scroll_start_view = 0

    def start_processes(self):
        self.load_attendance_data()

    def stop_processes(self):
        pass

    def write_log(self, msg):
        self.controller.write_log(msg)

    def update_log(self):
        try:
            while True:
                msg = self.controller.log_queue.get_nowait()
                if msg:
                    self.log_text.config(state='normal')
                    self.log_text.insert('end', msg + '\n')
                    self.log_text.see('end')
                    self.log_text.config(state='disabled')
        except queue.Empty:
            pass
        self.after(100, self.update_log)

    def load_attendance_data(self):
        self.write_log('Bắt đầu làm mới dữ liệu điểm danh...')
        if not self.controller.web_running:
            self.write_log(' Web app chưa sẵn sàng, sẽ thử lại sau...')
            self.after(3000, self.load_attendance_data)
            return
        def load_task():
            try:
                response = requests.get(API_ATTENDANCE_URL, timeout=10)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                            self.after(0, self.display_attendance_data, df)
                            self.write_log('Tải và hiển thị dữ liệu điểm danh thành công.')
                        else:
                            self.write_log(' Lỗi: Dữ liệu không đúng định dạng.')
                    except ValueError:
                        dfs = pd.read_html(response.text)
                        if dfs:
                            df = dfs[0]
                            self.after(0, self.display_attendance_data, df)
                            self.write_log('Tải và hiển thị dữ liệu điểm danh thành công.')
                        else:
                            self.write_log('Lỗi: Không tìm thấy bảng dữ liệu.')
                else:
                    self.write_log(f'Lỗi HTTP: {response.status_code}')
            except requests.exceptions.ConnectionError:
                self.write_log('Lỗi kết nối: Web app chưa sẵn sàng, sẽ thử lại sau...')
                self.after(3000, self.load_attendance_data)
            except Exception as e:
                self.write_log(f' Lỗi khi tải dữ liệu: {e}')
        threading.Thread(target=load_task, daemon=True).start()

    def display_attendance_data(self, df):
        if df.empty:
            self.write_log('Không có dữ liệu điểm danh.')
            return
        column_mapping = {
            'Lo\u1ea1i': 'Type', 'Ng\u00e0y/ gi\u1edd': 'Timestamp', 
            'Th\u00f4ng tin': 'Data', '\u0110\u1ed9 tin c\u1eady': 'Confidence',
            'Ngày/ giờ': 'Timestamp', 'Thông tin': 'Data', 'Loại': 'Type',
            'Độ tin cậy': 'Confidence', 'timestamp': 'Timestamp', 'data': 'Data',
            'type': 'Type', 'confidence': 'Confidence'
        }
        df = df.rename(columns=lambda c: column_mapping.get(c, c))
        required_cols = ['Timestamp', 'Data', 'Type', 'Confidence']
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
        df = df[required_cols]
        self.attendance_display.config(state="normal")
        self.attendance_display.delete("1.0", tk.END)
        header = "Ngày/giờ                              Thông tin                                                   Loại                 Độ tin cậy\n"
        self.attendance_display.insert(tk.END, header, 'bold_header')
        self.attendance_display.insert(tk.END, "\n")  # Thêm dòng trống giữa header và dữ liệu
        self.attendance_display.tag_configure('bold_header', font=("Arial", 16, "bold"))
        data_col_width = 30
        timestamp_padding = " " * 20
        for _, row in df.iterrows():
            timestamp = str(row.get('Timestamp', ''))
            data = str(row.get('Data', ''))
            record_type = str(row.get('Type', ''))
            confidence = row.get('Confidence', '')
            # Định dạng confidence thành phần trăm nếu là số
            try:
                conf_val = float(confidence)
                confidence_str = f"{conf_val*100:.1f} %"
            except Exception:
                confidence_str = str(confidence)
            original_lines = data.splitlines()
            all_display_lines = []
            for line in original_lines:
                wrapped_lines = textwrap.wrap(line, width=data_col_width, replace_whitespace=False, drop_whitespace=False)
                if not wrapped_lines:
                    all_display_lines.append('')
                else:
                    all_display_lines.extend(wrapped_lines)
            if not all_display_lines:
                all_display_lines = ['']
            first_display_line = all_display_lines[0]
            line_to_print = f"{timestamp:<20} {first_display_line:<30} {record_type:<10} {confidence_str:<10}\n"
            self.attendance_display.insert(tk.END, line_to_print)
            if len(all_display_lines) > 1:
                for subsequent_line in all_display_lines[1:]:
                    padding_end = " " * (10 + 10 + 2)
                    indented_line = f"{timestamp_padding} {subsequent_line:<30}{padding_end}\n"
                    self.attendance_display.insert(tk.END, indented_line)
        self.attendance_display.config(state="disabled")

    def on_text_click(self, event):
        self.scroll_start_y = event.y
        self.scroll_start_view = self.attendance_display.yview()[0]

    def on_text_drag(self, event):
        if hasattr(self, 'scroll_start_y') and hasattr(self, 'scroll_start_view'):
            delta_y = self.scroll_start_y - event.y
            scroll_amount = delta_y / 100 
            self.attendance_display.yview_scroll(int(scroll_amount), "units")

    def on_text_release(self, event):
        self.scroll_start_y = None
        self.scroll_start_view = None

    def _log_on_click(self, event):
        self._log_scroll_start_y = event.y
        self._log_scroll_start_view = self.log_text.yview()[0]

    def _log_on_drag(self, event):
        if hasattr(self, '_log_scroll_start_y') and hasattr(self, '_log_scroll_start_view'):
            delta_y = self._log_scroll_start_y - event.y
            scroll_amount = delta_y / 40
            self.log_text.yview_scroll(int(scroll_amount), "units")

    def _log_on_release(self, event):
        self._log_scroll_start_y = None
        self._log_scroll_start_view = None

class AttendanceGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Hệ thống điểm danh khuôn mặt')
        self.geometry('1020x590')
        self.configure(bg=DARK_BG)
        self.resizable(False, False)

        # Quản lý webcam tập trung
        self.shared_cap = None
        self.webcam_device_id = None
        self.webcam_lock = threading.Lock()
        
        self.web_process = None
        self.web_running = False
        self.log_queue = queue.Queue()

        # Đăng ký signal handler để tắt hoàn toàn
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 16, 'bold'), padding=5, background=DARK_BTN, foreground=DARK_TEXT)
        style.map('TButton', background=[('active', DARK_BTN_HOVER)])
        style.configure('TCheckbutton', font=('Arial', 14), background=DARK_BG, foreground=DARK_TEXT, indicatorcolor=DARK_ACCENT, padding=5)
        
        title_frame = tk.Frame(self, bg=DARK_ACCENT, height=70)
        title_frame.pack(fill='x', side='top')
        title_label = tk.Label(title_frame, text='HỆ THỐNG ĐIỂM DANH KHUÔN MẶT', font=('Trebuchet MS', 22, 'bold'), fg='white', bg=DARK_ACCENT)
        title_label.pack(pady=2)

        container = tk.Frame(self, bg=DARK_BG)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        for F in (RecognitionFrame, DataEntryFrame, AttendanceDataFrame):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Manually raise the first frame without starting processes
        first_frame = self.frames[RecognitionFrame]
        first_frame.reset_state(log=False) # Set initial state correctly
        first_frame.tkraise()
        
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.update_log()
        
        threading.Thread(target=self.start_web_app, daemon=True).start()

        self.attributes('-fullscreen', True)
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

    def initialize_webcam(self):
        """Khởi tạo webcam tập trung"""
        with self.webcam_lock:
            if self.shared_cap is not None:
                return True  # Webcam đã được khởi tạo
                
            # Thử nhiều video device khác nhau
            video_devices = [0, 1, 2, 10, 20]
            
            for device_id in video_devices:
                try:
                    self.write_log(f'[WEBCAM] Đang thử mở webcam device {device_id}...')
                    cap = cv2.VideoCapture(device_id)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.shared_cap = cap
                        self.webcam_device_id = device_id
                        self.write_log(f'Đã kết nối thành công với device {device_id}.')
                        return True
                    else:
                        cap.release()
                except Exception as e:
                    self.write_log(f'Lỗi với device {device_id}: {e}')
            
            self.write_log('Không thể mở webcam với bất kỳ device nào.')
            return False

    def get_webcam(self):
        """Lấy webcam đã được khởi tạo"""
        with self.webcam_lock:
            return self.shared_cap

    def release_webcam(self):
        """Giải phóng webcam"""
        with self.webcam_lock:
            if self.shared_cap:
                self.shared_cap.release()
                self.shared_cap = None
                self.webcam_device_id = None
                self.write_log('Đã giải phóng webcam.')

    def read_webcam_frame(self):
        """Đọc frame từ webcam"""
        with self.webcam_lock:
            if self.shared_cap and self.shared_cap.isOpened():
                return self.shared_cap.read()
            return False, None

    def show_frame(self, cont):
        # Dừng các tiến trình của frame cũ trước khi chuyển
        current_frame_key = next((key for key, value in self.frames.items() if value.winfo_ismapped()), None)
        if current_frame_key:
            current_frame_obj = self.frames[current_frame_key]
            if hasattr(current_frame_obj, 'stop_processes'):
                current_frame_obj.stop_processes()

        # Hiển thị frame mới
        frame = self.frames[cont]

        # Reset lại frame nhận diện khi quay lại
        if cont == RecognitionFrame:
            if hasattr(frame, 'reset_state'):
                frame.reset_state()

        frame.tkraise()
        
        # Bắt đầu tiến trình của frame mới
        if hasattr(frame, 'start_processes'):
            frame.start_processes()

    def write_log(self, msg):
        timestamp = time.strftime('%H:%M:%S')
        log_line = f'[{timestamp}] {msg}'
        self.log_queue.put(log_line)
        logging.info(msg)

    def update_log(self):
        try:
            while True: 
                msg = self.log_queue.get_nowait()
                if msg:
                    for frame in self.frames.values():
                        if frame.winfo_ismapped() and hasattr(frame, 'log_text'):
                            frame.log_text.config(state='normal')
                            frame.log_text.insert('end', msg + '\n')
                            frame.log_text.see('end')
                            frame.log_text.config(state='disabled')
        except queue.Empty:
            pass
        self.after(100, self.update_log)

    def start_web_app(self):
        if self.web_running:
            return
            
        try:
            self.write_log('Đang khởi động web app...')
            self.web_process = subprocess.Popen([
                sys.executable, 'web_app/app.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Chờ web app khởi động
            time.sleep(3)  # Chờ 3 giây để Flask start
            self.web_running = True
            self.write_log('Web app đã khởi động thành công.')
            
        except Exception as e:
            self.write_log(f'Lỗi khởi động web app: {e}')
            self.web_running = False

    def on_exit(self):
        self.write_log('Nhận được yêu cầu thoát. Đang dọn dẹp...')
        if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát chương trình?"):
            # Dừng tất cả các frame
            for frame in self.frames.values():
                 if hasattr(frame, 'stop_processes'):
                    frame.stop_processes()

            # Tắt web app một cách mạnh mẽ
            if self.web_process:
                self.write_log('Đang tắt tiến trình web app...')
                try:
                    # Thử terminate trước
                    self.web_process.terminate()
                    self.web_process.wait(timeout=3)
                    self.write_log('Web app đã được tắt thành công.')
                except subprocess.TimeoutExpired:
                    self.write_log('Web app không phản hồi, buộc dừng...')
                    try:
                        self.web_process.kill()
                        self.web_process.wait(timeout=2)
                    except:
                        pass
                
                # Đảm bảo tắt hoàn toàn bằng cách tìm và kill tất cả process liên quan
                try:
                    # Tìm tất cả process Python chạy web app
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            cmdline = proc.info['cmdline']
                            if cmdline and any('web_app/app.py' in arg for arg in cmdline):
                                self.write_log(f'Tìm thấy process web app PID {proc.info["pid"]}, đang tắt...')
                                proc.terminate()
                                proc.wait(timeout=2)
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                            pass
                except Exception as e:
                    self.write_log(f'Lỗi khi tìm process web app: {e}')
            
            # Tắt tất cả process ngrok nếu có
            try:
                subprocess.run(['ngrok', 'stop'], capture_output=True, timeout=5)
                self.write_log('Đã tắt ngrok.')
            except:
                pass
            
            self.web_running = False
            self.write_log('Đã dọn dẹp xong. Tạm biệt!')
            self.destroy()

    def signal_handler(self, signum, frame):
        """Xử lý signal từ hệ thống để tắt hoàn toàn"""
        self.write_log(f'Nhận signal {signum}, đang tắt ứng dụng...')
        self.force_exit()

    def force_exit(self):
        """Tắt ứng dụng một cách mạnh mẽ"""
        try:
            # Dừng tất cả các frame
            for frame in self.frames.values():
                 if hasattr(frame, 'stop_processes'):
                    frame.stop_processes()

            # Tắt web app
            if self.web_process:
                try:
                    self.web_process.terminate()
                    self.web_process.wait(timeout=2)
                except:
                    try:
                        self.web_process.kill()
                    except:
                        pass

            # Tìm và tắt tất cả process web app
            try:
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info['cmdline']
                        if cmdline and any('web_app/app.py' in arg for arg in cmdline):
                            proc.terminate()
                            proc.wait(timeout=1)
                    except:
                        pass
            except:
                pass

            # Tắt ngrok
            try:
                subprocess.run(['ngrok', 'stop'], capture_output=True, timeout=3)
            except:
                pass

        except Exception as e:
            print(f"Lỗi khi tắt ứng dụng: {e}")
        finally:
            os._exit(0)

# =================================================================================
# === MÀN HÌNH NHẬN DIỆN ==========================================================
# =================================================================================
class RecognitionFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=DARK_BG)
        self.controller = controller
        # --- Main layout ---
        main_frame = tk.Frame(self, bg=DARK_BG)
        main_frame.pack(side='top', fill='both', expand=False, padx=5, pady=2)
        
        # --- Webcam ---
        webcam_frame = tk.LabelFrame(main_frame, text='Webcam', font=('Arial', 16, 'bold'), bg=DARK_PANEL, fg=DARK_ACCENT, relief='groove', bd=1, labelanchor='n', width=640, height=480)
        webcam_frame.pack(side='left')
        webcam_frame.pack_propagate(False)
        self.webcam_status_var = tk.StringVar(value='Hệ thống đang tắt')
        self.webcam_status_label = tk.Label(
            webcam_frame,
            textvariable=self.webcam_status_var, 
            font=('Arial', 16, 'bold'), 
            fg=DARK_ACCENT, 
            bg=DARK_PANEL)
        self.webcam_status_label.pack(side='bottom', fill='x', pady=(1, 1))
        self.video_label = tk.Label(webcam_frame, bg='#11151c', width=640, height=480)
        self.video_label.pack(expand=True, fill='both', side='top')

        # --- Cột phải ---
        right_col = tk.Frame(main_frame, bg=DARK_BG, width=380, height=480)
        right_col.pack(side='left')
        right_col.pack_propagate(False)
        # --- Khuôn mặt đã nhận diện (trên) ---
        faces_frame = tk.LabelFrame(right_col, text='Khuôn mặt đã nhận diện', font=('Arial', 16, 'bold'), bg=DARK_PANEL, fg=DARK_ACCENT, relief='groove', bd=1, labelanchor='n', height=360)
        faces_frame.pack(side='top', fill='x')
        faces_frame.pack_propagate(False)
        # Thêm scrollbar dọc cho faces_canvas
        faces_canvas_frame = tk.Frame(faces_frame, bg=DARK_PANEL)
        faces_canvas_frame.pack(fill='both', expand=True)
        self.faces_canvas = tk.Canvas(faces_canvas_frame, bg=DARK_PANEL, highlightthickness=0, height=340)
        self.faces_canvas.pack(side='left', fill='both', expand=True)
        faces_scrollbar = ttk.Scrollbar(faces_canvas_frame, orient='vertical', command=self.faces_canvas.yview)
        faces_scrollbar.pack(side='right', fill='y')
        self.faces_canvas.configure(yscrollcommand=faces_scrollbar.set)
        # Hỗ trợ cuộn cảm ứng
        self.faces_canvas.bind('<Button-1>', self._faces_canvas_on_click)
        self.faces_canvas.bind('<B1-Motion>', self._faces_canvas_on_drag)
        self.faces_canvas.bind('<ButtonRelease-1>', self._faces_canvas_on_release)
        self._faces_scroll_start_y = 0
        self._faces_scroll_start_view = 0
        # --- Log (dưới cùng cột phải) ---
        log_frame = tk.Frame(right_col, bg=DARK_BG, height=120)
        log_frame.pack(side='bottom', fill='x')
        log_frame.pack_propagate(False)
        self.log_text = scrolledtext.ScrolledText(log_frame, font=('Consolas', 12), height=4, state='disabled', bg='#23272f', fg=DARK_TEXT)
        self.log_text.pack(side='bottom', fill='both', expand=True)
        # Thêm hỗ trợ vuốt cảm ứng cho log_text
        self.log_text.bind('<Button-1>', self._log_on_click)
        self.log_text.bind('<B1-Motion>', self._log_on_drag)
        self.log_text.bind('<ButtonRelease-1>', self._log_on_release)
        self._log_scroll_start_y = 0
        self._log_scroll_start_view = 0
        # --- Nút điều khiển webcam (dưới cùng, kéo dài toàn bộ chiều ngang) ---
        webcam_btn_frame = tk.Frame(self, bg=DARK_BG)
        # Thêm padding phía trên để các nút không dính sát mép dưới
        webcam_btn_frame.pack(side='bottom', fill='x', pady=(0, 10))
        for i in range(6):
            webcam_btn_frame.grid_columnconfigure(i, weight=1)
        self.start_btn = ttk.Button(webcam_btn_frame, text='Khởi động', command=self.start_recognition_system)
        self.start_btn.grid(row=0, column=0, padx=2, ipadx=2, ipady=2)
        self.stop_btn = ttk.Button(webcam_btn_frame, text='Tạm dừng', command=self.stop_recognition_system, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=2, ipadx=2, ipady=2)
        self.new_person_btn = ttk.Button(webcam_btn_frame, text='Thêm người mới', command=self.switch_to_data_entry)
        self.new_person_btn.grid(row=0, column=2, padx=2, ipadx=2, ipady=2)
        self.attendance_data_btn = ttk.Button(webcam_btn_frame, text='Dữ liệu điểm danh', command=lambda: controller.show_frame(AttendanceDataFrame))
        self.attendance_data_btn.grid(row=0, column=3, padx=2, ipadx=2, ipady=2)
        self.exit_btn = ttk.Button(webcam_btn_frame, text='Thoát', command=controller.on_exit)
        self.exit_btn.grid(row=0, column=4, padx=2, ipadx=2, ipady=2)
        self.verbose_logging = tk.BooleanVar(value=False)
        self.verbose_log_check = ttk.Checkbutton(webcam_btn_frame, text='Log chi tiết', variable=self.verbose_logging, command=self.toggle_verbose_log, style='Verbose.TCheckbutton')
        self.verbose_log_check.grid(row=0, column=5, padx=2)
        
        self.recognition_system = None
        self.webcam_thread = None
        self.recognition_thread = None
        self.running = False
        self.pir_sensor = None
        self.face_thumbs = []
        self.original_faces = {}
        self.recognized_names = set()
        self.recognized_faces = []
        self.pir_last_motion_time = 0
        self.pir_idle = True
        self.pir_timeout = 30
        self.fps = 0
        self._fps_frame_count = 0
        self._fps_start_time = time.time()
        self._image_references = {}
        self.frame_queue = queue.Queue(maxsize=2)
        # Thêm flag để chỉ phát âm một lần khi vào trạng thái chờ PIR
        self.pir_wait_announced = False
        
        # Hàng đợi mới cho xử lý đa luồng
        self.recognition_results_queue = queue.Queue(maxsize=5)

        self.reset_state(log=False)

        self.qr_capture_popup = None
        self.qr_capture_in_progress = False
    def _log_on_click(self, event):
        self._log_scroll_start_y = event.y
        self._log_scroll_start_view = self.log_text.yview()[0]

    def _log_on_drag(self, event):
        if hasattr(self, '_log_scroll_start_y') and hasattr(self, '_log_scroll_start_view'):
            delta_y = self._log_scroll_start_y - event.y
            scroll_amount = delta_y / 40
            self.log_text.yview_scroll(int(scroll_amount), "units")

    def _log_on_release(self, event):
        self._log_scroll_start_y = None
        self._log_scroll_start_view = None
    def switch_to_data_entry(self):
        """Chuyển sang màn hình thêm người mới và dừng hệ thống nhận diện"""
        if self.running:
            self.write_log("[CHUYỂN] Dừng hệ thống nhận diện để chuyển sang thêm người mới...")
            self.stop_recognition_system()
        self.controller.show_frame(DataEntryFrame)

    def reset_state(self, log=True):
        self.stop_recognition_system()
        
        self.face_thumbs.clear()
        self.original_faces.clear()
        self.recognized_names.clear()
        self.recognized_faces.clear()
        
        self.pir_last_motion_time = 0
        self.pir_idle = True
        self.fps = 0
        self.scroll_start_y = 0
        self._fps_frame_count = 0
        self._fps_start_time = time.time()
        
        self.webcam_status_var.set('Hệ thống sẵn sàng. Nhấn "Khởi động" để bắt đầu.')
        self.faces_canvas.delete('all')
        
        # Xóa hình ảnh webcam và hiển thị frame trống
        idle_frame = self.get_idle_frame()
        img = Image.fromarray(cv2.cvtColor(idle_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self._image_references['video_label'] = imgtk
        
        # Xóa mọi image reference khác để tránh giữ ảnh cũ
        for k in list(self._image_references.keys()):
            if k != 'video_label':
                del self._image_references[k]
        
        # Reset trạng thái nút
        self.start_btn.config(state='normal', text='Khởi động')
        self.stop_btn.config(state='disabled')
        
        if log:
            self.write_log("Hệ thống nhận diện đã được reset.")

    def retry_load_data(self):
        """Thử load dữ liệu lại nếu web app đã sẵn"""
        if self.controller.web_running:
            pass # Đã chuyển sang AttendanceDataFrame
        else:
            # Thử lại sau 2 giây nữa
            self.after(2000, self.retry_load_data)

    def stop_processes(self):
        if self.running:
            self.stop_recognition_system()
        
    def stop_recognition_system(self):
        if not self.running and self.recognition_system is None:
            return
        self.write_log("Dừng hệ thống nhận diện...")
        self.running = False
        # Dừng các luồng
        if self.webcam_thread and self.webcam_thread.is_alive():
            self.webcam_thread.join(timeout=1)
        if self.recognition_thread and self.recognition_thread.is_alive():
            self.recognition_thread.join(timeout=1)
        # Dừng recognition system
        if self.recognition_system:
            if hasattr(self.recognition_system, 'stop'):
                self.recognition_system.stop()
            del self.recognition_system
            self.recognition_system = None
            gc.collect()
        # Giải phóng PIR sensor nếu có
        if self.pir_sensor:
            try:
                self.pir_sensor.release()
            except Exception as e:
                self.write_log(f"Lỗi khi giải phóng PIR: {e}")
            self.pir_sensor = None
        self.start_btn.config(state='normal', text='Khởi động lại')
        self.stop_btn.config(state='disabled')
        self.webcam_status_var.set('Hệ thống đã dừng.')
        self.controller.release_webcam()
        idle_frame = self.get_idle_frame()
        img = Image.fromarray(cv2.cvtColor(idle_frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self._image_references['video_label'] = imgtk
        for k in list(self._image_references.keys()):
            if k != 'video_label':
                del self._image_references[k]
        time.sleep(0.5)

    def write_log(self, msg):
        self.controller.write_log(msg)

    def start_recognition_system(self):
        """This is now the main entry point to start the system via the button."""
        if self.running:
            self.write_log('(!) Hệ thống nhận diện đã chạy từ trước.')
            return
        self.write_log('=>Khởi động hệ thống nhận diện...')
        self.webcam_status_var.set('Đang khởi tạo các thành phần...')
        threading.Thread(target=self._initialize_system, daemon=True).start()

    def _initialize_system(self):
        try:
            # Nếu đã có pir_sensor cũ thì giải phóng trước khi tạo mới
            if self.pir_sensor:
                try:
                    self.pir_sensor.release()
                except Exception as e:
                    self.write_log(f"Lỗi khi giải phóng PIR cũ: {e}")
                self.pir_sensor = None
            self.recognition_system = RecognitionSystem(gui_log_func=self.write_log)
            self.recognition_system.attendance_callback = self.handle_callback
            self.pir_sensor = self.recognition_system.pir_sensor if hasattr(self.recognition_system, 'pir_sensor') else None
            self.write_log('=>Khởi tạo model và cảm biến thành công.')
            self.running = True
            self.pir_last_motion_time = time.time()
            self.pir_idle = False
            self.webcam_status_var.set('Hệ thống đang nhận diện')
            if not self.controller.initialize_webcam():
                self.write_log('=>Không thể mở webcam.')
                self.running = False
                self.webcam_status_var.set('Lỗi webcam.')
                return
            self.webcam_thread = threading.Thread(target=self.update_webcam, daemon=True)
            self.recognition_thread = threading.Thread(target=self.recognition_processing_thread, daemon=True)
            self.webcam_thread.start()
            self.recognition_thread.start()
            self.update_gui_from_queue()
            self.start_btn.config(state='disabled', text='Khởi động lại')
            self.stop_btn.config(state='normal')
            self.webcam_status_var.set('Hệ thống đang hoạt động')
            self.write_log("=>Hệ thống nhận diện đã sẵn sàng.")
        except Exception as e:
            self.write_log(f"=>Lỗi nghiêm trọng khi khởi tạo: {e}")
            self.webcam_status_var.set('Lỗi khởi tạo hệ thống.')
            self.start_btn.config(state='normal', text='Khởi động lại')
            self.stop_btn.config(state='disabled')

    def update_gui_from_queue(self):
        # Cập nhật video feed
        try:
            processed_frame = self.frame_queue.get_nowait()
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_resized = Image.fromarray(img).resize((640, 480), Image.Resampling.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img_resized)
            self._image_references['video_label'] = imgtk
            self.video_label.config(image=imgtk)
        except queue.Empty:
            pass

        # Cập nhật kết quả nhận diện
        try:
            (new_names, new_faces_imgs) = self.recognition_results_queue.get_nowait()
            if new_names:
                self.update_recognized_faces(new_names, new_faces_imgs)
        except queue.Empty:
            pass
        
        if self.running:
            self.after(30, self.update_gui_from_queue)

    def update_webcam(self):
        while self.running:
            # Luồng này chỉ đọc frame và đưa vào hàng đợi để xử lý
            now = time.time()
            is_motion = False # Mặc định là không có chuyển động
            if self.pir_sensor:
                try:
                    # Logic đọc PIR sensor trực tiếp hơn
                    is_motion = self.pir_sensor.is_motion()
                except Exception as e:
                    self.write_log(f"Lỗi đọc cảm biến: {e}")
                    is_motion = False 
            else:
                is_motion = True

            if is_motion:
                self.pir_last_motion_time = now
                if self.pir_idle:
                    self.write_log("Phát hiện chuyển động! Khởi động lại webcam...")
                    self.pir_idle = False
                    self.webcam_status_var.set('Hệ thống đang nhận diện')
                    # Reset flag khi có chuyển động
                    self.pir_wait_announced = False
                    if not self.controller.get_webcam():
                        self.controller.initialize_webcam()
            else:
                if not self.pir_idle and (now - self.pir_last_motion_time > self.pir_timeout):
                    self.pir_idle = True
                    self.webcam_status_var.set('Đang chờ tín hiệu PIR...')
                    self.controller.release_webcam()
                    self.write_log("Webcam đã tạm dừng.")
                    # Phát âm thông báo khi chuyển sang trạng thái chờ PIR, chỉ phát một lần
                    if not self.pir_wait_announced:
                        try:
                            play_name_smart("Hệ thống tạm ngưng, đang chờ chuyển động", log_func=self.write_log)
                        except Exception as e:
                            self.write_log(f"Lỗi khi phát âm thông báo PIR: {e}")
                        self.pir_wait_announced = True

            if self.pir_idle:
                idle_frame = self.get_idle_frame()
                try:
                    self.frame_queue.put_nowait(idle_frame)
                except queue.Full: pass
                time.sleep(0.5)
                continue
            
            if not self.controller.get_webcam():
                if not self.controller.initialize_webcam():
                    time.sleep(1)
                    continue

            ret, frame = self.controller.read_webcam_frame()
            if not ret: 
                time.sleep(0.1)
                continue
            
            # Cập nhật FPS và hiển thị frame ngay lập tức để giữ độ mượt
            self._fps_frame_count += 1
            elapsed = time.time() - self._fps_start_time
            if elapsed >= 1.0:
                self.fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._fps_start_time = time.time()
            
            display_frame = frame.copy()
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            try:
                self.frame_queue.put_nowait(display_frame)
            except queue.Full: pass
            
            # Đưa frame vào hàng đợi để luồng nhận diện xử lý
            try:
                # Tạo một hàng đợi riêng cho việc xử lý AI
                pass # Sẽ được xử lý ở luồng khác
            except queue.Full:
                pass
    
    def recognition_processing_thread(self):
        """Luồng riêng chỉ để xử lý AI"""
        while self.running:
            if self.pir_idle or not self.recognition_system:
                time.sleep(0.1)
                continue

            ret, frame = self.controller.read_webcam_frame()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Xử lý AI trên mỗi frame đọc được
            processed_frame, new_names, new_faces_imgs = self.process_and_draw(frame)
            
            if new_names:
                try:
                    self.recognition_results_queue.put_nowait((new_names, new_faces_imgs))
                except queue.Full:
                    pass
            
            # Giảm tải một chút để tránh 100% CPU
            time.sleep(0.01)

    def toggle_verbose_log(self):
        """Bật/tắt chế độ log chi tiết."""
        state = "BẬT" if self.verbose_logging.get() else "TẮT"
        self.write_log(f"Chế độ log chi tiết đã được {state}.")

    def write_log_verbose(self, msg):
        """Ghi log chỉ khi chế độ verbose được bật."""
        if self.verbose_logging.get():
            self.controller.write_log(f"   [VERBOSE] {msg}")

    def process_and_draw(self, frame):
        if not self.recognition_system: return frame, set(), dict()
        
        self.write_log_verbose("Gọi recognition_system.detect_and_recognize...")
        faces, qr_codes = self.recognition_system.detect_and_recognize(frame)
        self.write_log_verbose(f"Kết quả: {len(faces)} khuôn mặt, {len(qr_codes)} QR.")

        new_names, new_faces_imgs = set(), dict()

        if not faces:
            self.write_log_verbose("Không phát hiện khuôn mặt nào trong frame.")

        for face in faces:
            name = face['name']
            confidence = face.get('confidence', 0)
            
            if name == "Fake":
                self.write_log(f"Phát hiện khuôn mặt giả (Conf: {confidence:.2f})")
            elif name == "Unknown":
                self.write_log(f"Khuôn mặt không xác định (Conf: {confidence:.2f})")
            elif name not in self.recognized_names:
                self.write_log(f"Thành công: {name} (Conf: {confidence:.2f})")
                new_names.add(name)
                x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                new_faces_imgs[name] = {
                    'img': frame[y:y+h, x:x+w],
                    'confidence': confidence
                }
            else:
                self.write_log_verbose(f"Đã nhận diện '{name}' trước đó, bỏ qua log chính.")

        return frame, new_names, new_faces_imgs

    def update_recognized_faces(self, new_names, new_faces_imgs):
        has_new_face = False
        for name in new_names:
            if name not in self.recognized_names:
                has_new_face = True
                self.recognized_names.add(name)
                face_data = new_faces_imgs[name]
                original_img = face_data['img']
                confidence = face_data['confidence']
                timestamp = datetime.now().strftime('%H:%M:%S')
                self.recognized_faces.insert(0, {
                    'img': cv2.resize(original_img, (90, 90)),
                    'name': name,
                    'timestamp': timestamp,
                    'confidence': confidence
                })
                self.original_faces[name] = original_img
        if has_new_face:
            self.recognized_faces = self.recognized_faces[:20]  # Cho phép nhiều hơn để cuộn
            self.faces_canvas.delete('all')
            self.face_thumbs.clear()
            y_offset = 4
            thumb_max_w = thumb_max_h = 90  # Thumbnail vuông 90x90
            for idx, photo in enumerate(self.recognized_faces):
                face_frame = tk.Frame(self.faces_canvas, bg=DARK_PANEL, height=thumb_max_h)
                self.faces_canvas.create_window(50, y_offset, window=face_frame, anchor='nw', width=280, height=thumb_max_h)
                # Resize trực tiếp về 90x90, không padding
                img = photo['img']
                img_resized = cv2.resize(img, (thumb_max_w, thumb_max_h), interpolation=cv2.INTER_AREA)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)))
                self.face_thumbs.append(imgtk)
                img_label = tk.Label(face_frame, image=imgtk, bg=DARK_PANEL)
                img_label.pack(side='left', padx=0, pady=2)
                info_frame = tk.Frame(face_frame, bg=DARK_PANEL)
                info_frame.pack(side='left', fill='y', expand=True, padx=0)
                name_label = tk.Label(info_frame, text=photo['name'], font=('Arial', 15, 'bold'), fg=DARK_ACCENT, bg=DARK_PANEL, anchor='w')
                name_label.pack(fill='x', pady=(0, 1))
                time_label = tk.Label(info_frame, text=f"Thời gian: {photo['timestamp']}", font=('Arial', 13), fg=DARK_TEXT, bg=DARK_PANEL, anchor='w')
                time_label.pack(fill='x', pady=(0, 0))
                conf_label = tk.Label(info_frame, text=f"Độ tin cậy: {photo['confidence']*100:.2f} %", font=('Arial', 13), fg=DARK_TEXT, bg=DARK_PANEL, anchor='w')
                conf_label.pack(fill='x')
                callback = lambda e, n=photo['name'], i=self.original_faces[photo['name']]: self.show_enlarged_face(n, i)
                self.bind_all_children(face_frame, "<Button-1>", callback)
                y_offset += thumb_max_h + 2  # Các hàng sát nhau hơn
            # Cập nhật scrollregion
            total_height = max(180, y_offset)
            self.faces_canvas.config(scrollregion=(0, 0, 210, total_height))

    def show_enlarged_face(self, name, image):
        """Hiển thị cửa sổ phóng to ảnh."""
        if name in self.original_faces:
            window = EnlargedFaceWindow(self, name, image)
            self.wait_window(window)

    def bind_all_children(self, widget, event, callback):
        """Gán một event cho widget và tất cả các widget con của nó."""
        widget.bind(event, callback)
        for child in widget.winfo_children():
            self.bind_all_children(child, event, callback)

    def get_idle_frame(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def _faces_canvas_on_click(self, event):
        self._faces_scroll_start_y = event.y
        self._faces_scroll_start_view = self.faces_canvas.yview()[0]
    def _faces_canvas_on_drag(self, event):
        if hasattr(self, '_faces_scroll_start_y') and hasattr(self, '_faces_scroll_start_view'):
            delta_y = self._faces_scroll_start_y - event.y
            scroll_amount = delta_y / 40  # Giảm nhạy, cuộn chậm hơn
            self.faces_canvas.yview_scroll(int(scroll_amount), "units")
    def _faces_canvas_on_release(self, event):
        self._faces_scroll_start_y = None
        self._faces_scroll_start_view = None

    def handle_callback(self, data):
        """
        Xử lý callback từ RecognitionSystem (ví dụ: yêu cầu xác nhận QR capture)
        """
        if isinstance(data, dict) and data.get('type') == 'QR-CAPTURE-REQUEST':
            qr_data = data.get('qr_data')
            # Hiện popup xác nhận ở giữa cửa sổ nhận diện
            if self.qr_capture_in_progress:
                return  # Đang có popup, không hiện thêm
            self.qr_capture_in_progress = True
            self.show_qr_capture_popup(qr_data)

    def show_qr_capture_popup(self, qr_data):
        if self.qr_capture_popup:
            self.qr_capture_popup.destroy()
        self.qr_capture_popup = tk.Toplevel(self)
        self.qr_capture_popup.title("Xác nhận lưu ảnh QR")
        self.qr_capture_popup.geometry("400x180")
        self.qr_capture_popup.transient(self.controller)
        self.qr_capture_popup.grab_set()
        self.qr_capture_popup.configure(bg=DARK_PANEL)
        self.qr_capture_popup.resizable(False, False)
        # Đặt popup ở giữa cửa sổ nhận diện
        self.qr_capture_popup.update_idletasks()
        x = self.winfo_rootx() + self.winfo_width() // 2 - 200
        y = self.winfo_rooty() + self.winfo_height() // 2 - 90
        self.qr_capture_popup.geometry(f"400x180+{x}+{y}")
        label = tk.Label(self.qr_capture_popup, text="Bạn cần lưu ảnh khuôn mặt lên hệ thống để hoàn tất\nquá trình điểm danh bằng mã QR.", font=("Arial", 13), fg=DARK_ACCENT, bg=DARK_PANEL, wraplength=380)
        label.pack(pady=(30, 10))
        btn = ttk.Button(self.qr_capture_popup, text="Đồng ý", command=lambda: self.capture_qr_images(qr_data))
        btn.pack(pady=(0, 20), ipadx=10, ipady=5)

    def capture_qr_images(self, qr_data):
        if self.qr_capture_popup is not None:
            self.qr_capture_popup.destroy()
            self.qr_capture_popup = None
        os.makedirs('qr_captures', exist_ok=True)
        captured_paths = []
        name = qr_data.split('\n')[0].split(':')[-1].strip() if ':' in qr_data else qr_data.split('\n')[0].strip()
        name_safe = ''.join(c for c in name if c.isalnum() or c in ('-_'))
        for i in range(5):
            ret, frame = self.controller.read_webcam_frame()
            if ret:
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_path = f"qr_captures/{name_safe}_{ts}_{i+1}.jpg"
                cv2.imwrite(img_path, frame)
                captured_paths.append(img_path)
            self.update()
            time.sleep(1)
        # Sau khi lưu xong, gọi lại recognition_system để lưu QR lên Google Sheet
        if self.recognition_system is not None:
            self.recognition_system.save_qr_attendance(qr_data)
        else:
            self.write_log("recognition_system chưa được khởi tạo, không thể lưu QR.")
            messagebox.showerror("Lỗi", "Hệ thống nhận diện chưa sẵn sàng, không thể lưu QR lên Google Sheet.")
        self.qr_capture_in_progress = False
        # Hiện messagebox sau 5 giây
        self.after(5000, lambda: messagebox.showinfo("Hoàn tất", "Đã lưu ảnh thành công!"))

# =================================================================================
# === MÀN HÌNH THÊM DỮ LIỆU =======================================================
# =================================================================================
class DataEntryFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=DARK_BG)
        self.controller = controller
        self.running_webcam = False
        self.webcam_thread = None
        self.current_frame = None
        self.cropped_preview_frame = None
        self.captured_image_thumbs = []
        self.pending_images = [] # Lưu ảnh tạm thời
        self.frame_queue = queue.Queue(maxsize=2)
        self.progress_var = tk.IntVar(value=0)
        # Đảm bảo progress bar/label là con của controls_frame
        self.progress_bar = ttk.Progressbar(self, orient='horizontal', length=320, mode='determinate', variable=self.progress_var, maximum=100)
        self.progress_label = tk.Label(self, text='Tiến trình: 0%', font=('Arial', 12), fg=DARK_ACCENT, bg=DARK_BG)
        # Ban đầu ẩn
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()

        # --- Layout chính 2 cột ---
        main_frame = tk.Frame(self, bg=DARK_BG)
        main_frame.pack(side='top', fill='both', expand=True, padx=0, pady=0)
        
        # --- Cột Trái: Webcam Preview ---
        left_col = tk.Frame(main_frame, bg=DARK_BG)
        left_col.pack(side='left', fill='y', padx=0)

        webcam_container = tk.LabelFrame(
            left_col, 
            text="Webcam (Tỉ lệ 3:4)", 
            font=('Arial', 14, 'bold'), 
            bg=DARK_PANEL, 
            fg=DARK_ACCENT, 
            relief='groove', 
            bd=2, 
            labelanchor='n', 
            width=270, 
            height=360
        )
        webcam_container.pack(side='top', anchor='nw', padx=0, pady=0)
        self.video_label = tk.Label(webcam_container, bg='#11151c', width=270, height=360)
        self.video_label.pack(side='top', anchor='nw', padx=0, pady=0)

        # --- Cột Phải: Ảnh đã chụp ---
        right_col = tk.Frame(main_frame, bg=DARK_BG)
        right_col.pack(side='left', fill='both', expand=True, padx=0, pady=0)

        # Giảm chiều cao khung ảnh đã chụp
        captured_container = tk.LabelFrame(right_col, text="Ảnh đã chụp", font=('Arial', 14, 'bold'), bg=DARK_PANEL, fg=DARK_ACCENT, relief='groove', bd=2, labelanchor='n', height=387)
        captured_container.pack(side='top', fill='x', expand=False, pady=(0, 5))
        captured_container.pack_propagate(False)
        self.captured_canvas = tk.Canvas(captured_container, bg=DARK_PANEL, highlightthickness=0, height=360)
        scrollbar = ttk.Scrollbar(captured_container, orient='vertical', command=self.captured_canvas.yview)
        self.captured_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.captured_canvas.pack(side='left', fill='both', expand=True)
        self.scrollable_frame = tk.Frame(self.captured_canvas, bg=DARK_PANEL)
        self.captured_canvas.create_window((30,0), window=self.scrollable_frame, anchor='nw')
        self.scrollable_frame.bind("<Configure>", lambda e: self.captured_canvas.configure(scrollregion=self.captured_canvas.bbox("all")))

        # --- Controls (nằm dưới main_frame, full width) ---
        controls_frame = tk.Frame(self, bg=DARK_BG)
        controls_frame.pack(side='top', fill='x')
        entry_frame = tk.Frame(controls_frame, bg=DARK_BG)
        entry_frame.pack(pady=2)
        self.entry_frame = entry_frame
        tk.Label(entry_frame, text="Nhập tên:", font=('Arial', 14), fg=DARK_TEXT, bg=DARK_BG).pack(side='left', padx=5)
        self.name_entry = tk.Entry(entry_frame, font=('Arial', 14), width=25)
        self.name_entry.pack(side='left')
        self.btn_frame = tk.Frame(controls_frame, bg=DARK_BG)
        self.btn_frame.pack(pady=5)
        self.capture_btn = ttk.Button(self.btn_frame, text='Chụp 10 ảnh', command=self.capture_face)
        self.capture_btn.pack(side='left', padx=5)
        self.save_btn = ttk.Button(self.btn_frame, text='Lưu ảnh', command=self.save_captured_images)
        self.clear_btn = ttk.Button(self.btn_frame, text='Hủy bỏ', command=self.clear_pending_images)
        self.train_btn = ttk.Button(self.btn_frame, text='Huấn luyện', command=self.start_training)
        self.train_btn.pack(side='left', padx=5)
        self.back_btn = ttk.Button(self.btn_frame, text='Quay lại nhận diện', command=self.switch_to_recognition)
        self.back_btn.pack(side='left', padx=5)

        # --- Log (nằm dưới cùng, full width) ---
        log_frame = tk.Frame(self, height=100, bg=DARK_BG)
        log_frame.pack(side='bottom', fill='x', pady=(5,0))
        log_frame.pack_propagate(False)
        self.log_text = scrolledtext.ScrolledText(log_frame, font=('Consolas', 12), height=4, state='disabled', bg='#23272f', fg=DARK_TEXT)
        self.log_text.pack(side='bottom', fill='both', expand=True)
        # Thêm hỗ trợ vuốt cảm ứng cho log_text
        self.log_text.bind('<Button-1>', self._log_on_click)
        self.log_text.bind('<B1-Motion>', self._log_on_drag)
        self.log_text.bind('<ButtonRelease-1>', self._log_on_release)
        self._log_scroll_start_y = 0
        self._log_scroll_start_view = 0

    def switch_to_recognition(self):
        """Chuyển về frame nhận diện và giải phóng webcam"""
        self.stop_processes()
        # Đảm bảo webcam được giải phóng hoàn toàn
        self.controller.release_webcam()
        self.controller.show_frame(RecognitionFrame)

    def start_processes(self):
        """Overrides the base method in DataEntryFrame."""
        self.clear_pending_images() # Reset trạng thái khi frame được hiển thị
        self.name_entry.delete(0, 'end')
        self.start_webcam()

    def stop_processes(self):
        self.stop_webcam()
        # Đảm bảo webcam được giải phóng hoàn toàn
        self.controller.release_webcam()

    def start_webcam(self):
        if self.running_webcam: return
        
        # Sử dụng webcam chung từ controller
        if not self.controller.initialize_webcam():
            self.controller.write_log("Không thể khởi tạo webcam.")
            return
            
        self.running_webcam = True
        self.webcam_thread = threading.Thread(target=self.update_webcam_preview, daemon=True)
        self.webcam_thread.start()
        self.update_gui_from_queue()
        self.controller.write_log("Đã bật webcam preview.")

    def stop_webcam(self):
        if not self.running_webcam: return
        self.running_webcam = False 
        if self.webcam_thread and self.webcam_thread.is_alive():
            self.webcam_thread.join(timeout=1)
        # Không giải phóng webcam chung, chỉ dừng thread
        self.controller.write_log("Đã tắt webcam preview.")
        time.sleep(0.5)

    def update_gui_from_queue(self):
        try:
            display_frame = self.frame_queue.get_nowait()
            img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self._imgtk_ref = imgtk  # Giữ tham chiếu để không bị thu gom rác
            self.video_label.config(image=imgtk)
        except queue.Empty:
            pass

        if self.running_webcam:
            self.after(30, self.update_gui_from_queue)

    def update_webcam_preview(self):
        while self.running_webcam:
            ret, frame = self.controller.read_webcam_frame()
            if not ret: 
                time.sleep(0.1)
                continue
            
            self.current_frame = frame.copy() 
            h, w, _ = frame.shape
            
            # Lấy chiều cao tối đa, chiều rộng = 3/4 chiều cao, crop chính giữa
            target_h = h
            target_w = int(target_h * 3 / 4)
            if target_w > w:
                target_w = w
                target_h = int(w * 4 / 3)
            start_x = (w - target_w) // 2
            start_y = (h - target_h) // 2
            cropped_frame = frame[start_y:start_y+target_h, start_x:start_x+target_w]
            self.cropped_preview_frame = cropped_frame.copy()
            display_frame = cv2.resize(cropped_frame, (270, 360))
            try:
                self.frame_queue.put_nowait(display_frame)
            except queue.Full: pass

    def capture_face(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Lỗi", "Vui lòng nhập tên người cần thêm.")
            return

        self.capture_btn.config(state='disabled', text='Đang chụp...')
        self.clear_pending_images(clear_ui=False) # Xóa ảnh cũ nhưng không reset UI
        threading.Thread(target=self._capture_10_photos, args=(name,), daemon=True).start()

    def _capture_10_photos(self, name):
        self.controller.write_log(f"Bắt đầu chụp 10 ảnh cho '{name}'...")
        
        count = 0
        while count < 10:
            if not self.running_webcam or self.cropped_preview_frame is None:
                self.controller.write_log("Mất kết nối webcam hoặc không có ảnh, dừng chụp.")
                break
            
            # Lưu ảnh vào danh sách chờ
            self.pending_images.append(self.cropped_preview_frame.copy())
            self.controller.write_log(f"Đã chụp ảnh [{count + 1}/10]")
            self.after(0, self.add_captured_thumb, self.cropped_preview_frame.copy())
            
            count += 1
            time.sleep(0.2) 
        
        if self.pending_images:
            self.controller.write_log(f"Hoàn tất chụp ảnh. Vui lòng lưu hoặc hủy.")
            # Cập nhật UI sau khi chụp xong
            self.after(0, self.show_save_clear_buttons)
        else:
            # Nếu có lỗi, reset lại nút chụp
            self.after(0, lambda: self.capture_btn.config(state='normal', text='Chụp 10 ảnh'))

    def show_save_clear_buttons(self):
        self.capture_btn.pack_forget()
        self.save_btn.pack(side='left', padx=10)
        self.clear_btn.pack(side='left', padx=10)

    def hide_save_clear_buttons(self):
        self.save_btn.pack_forget()
        self.clear_btn.pack_forget()
        self.capture_btn.config(state='normal', text='Chụp 10 ảnh')
        self.capture_btn.pack(side='left', padx=10)

    def save_captured_images(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Lỗi", "Tên không được để trống khi lưu.")
            return

        if not self.pending_images:
            messagebox.showwarning("Lưu ý", "Không có ảnh nào để lưu.")
            return
            
        self.controller.write_log(f"Bắt đầu lưu {len(self.pending_images)} ảnh cho '{name}'...")
        person_dir = os.path.join('images_attendance', name)
        os.makedirs(person_dir, exist_ok=True)
        
        try:
            for i, img_data in enumerate(self.pending_images):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                filename = os.path.join(person_dir, f"{timestamp}.jpg")
                cv2.imwrite(filename, img_data)
            self.controller.write_log(f"Đã lưu thành công {len(self.pending_images)} ảnh.")
            messagebox.showinfo("Thành công", f"Đã lưu thành công {len(self.pending_images)} ảnh cho {name}.")
        except Exception as e:
            self.controller.write_log(f"[Lỗi] Không thể lưu ảnh: {e}")
            messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {e}")
        finally:
            self.clear_pending_images()

    def clear_pending_images(self, clear_ui=True):
        self.pending_images.clear()
        self.captured_image_thumbs.clear()
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        if clear_ui:
            self.hide_save_clear_buttons()

    def add_captured_thumb(self, frame):
        thumb_w, thumb_h = 126, 168  # 3:4
        img_resized = cv2.resize(frame, (thumb_w, thumb_h))
        img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.captured_image_thumbs.append(imgtk)
        thumb_label = tk.Label(self.scrollable_frame, image=imgtk, bg=DARK_PANEL)
        num_images = len(self.captured_image_thumbs) - 1
        # 1 hàng 5 ảnh, tự động xuống hàng
        col = num_images % 5
        row = num_images // 5
        # Padding trên/dưới lớn hơn padding trái/phải
        thumb_label.grid(row=row, column=col, padx=(4,4), pady=(4,4))
        # Thêm sự kiện click để phóng to ảnh
        def show_preview(event, img=frame):
            EnlargedFaceWindow(self, 'Preview', img)
        thumb_label.bind('<Button-1>', show_preview)

    def start_training(self):
        if self.pending_images:
            if not messagebox.askyesno("Xác nhận", "Bạn có ảnh chưa được lưu. Nếu tiếp tục huấn luyện, các ảnh này sẽ bị mất. Bạn có muốn tiếp tục không?"):
                return
        self.clear_pending_images()
        self.train_btn.config(state='disabled', text='Đang huấn luyện...')
        self.controller.write_log("Bắt đầu quá trình huấn luyện lại...")
        threading.Thread(target=self._run_train_script, daemon=True).start()

    def _run_train_script(self):
        try:
            process = subprocess.Popen(
                ['python3', TRAIN_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if process.stdout:
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        # Ghi trực tiếp vào khung log của DataEntryFrame
                        self.log_text.config(state='normal')
                        self.log_text.insert('end', output.strip() + '\n')
                        self.log_text.see('end')
                        self.log_text.config(state='disabled')
            rc = process.poll()
            if rc == 0:
                self.controller.write_log("Huấn luyện thành công!")
                messagebox.showinfo("Hoàn tất", "Quá trình huấn luyện lại đã hoàn tất thành công.")
            else:
                self.controller.write_log(f"Huấn luyện thất bại với mã lỗi: {rc}")
                messagebox.showerror("Lỗi", f"Quá trình huấn luyện gặp lỗi. Vui lòng kiểm tra log.")
        except Exception as e:
            self.controller.write_log(f"Lỗi nghiêm trọng khi chạy script: {e}")
            messagebox.showerror("Lỗi", f"Không thể bắt đầu quá trình huấn luyện: {e}")
        finally:
            self.train_btn.config(state='normal', text='Huấn luyện lại')

    def _log_on_click(self, event):
        self._log_scroll_start_y = event.y
        self._log_scroll_start_view = self.log_text.yview()[0]

    def _log_on_drag(self, event):
        if hasattr(self, '_log_scroll_start_y') and hasattr(self, '_log_scroll_start_view'):
            delta_y = self._log_scroll_start_y - event.y
            scroll_amount = delta_y / 40
            self.log_text.yview_scroll(int(scroll_amount), "units")

    def _log_on_release(self, event):
        self._log_scroll_start_y = None
        self._log_scroll_start_view = None

if __name__ == '__main__':
    if sys.platform.startswith('win'):
        import multiprocessing
        multiprocessing.freeze_support()
        
    app = AttendanceGUI()
    app.mainloop() 