import sqlite3
import csv
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # Xóa bảng face_profiles cũ nếu có
    cur.execute("DROP TABLE IF EXISTS face_profiles;")
    # Bảng user
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Bảng profile gương mặt (mỗi user nhiều ảnh, không còn trường status)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_profiles (
            profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            face_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def export_face_profiles_to_csv():
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    with open('face_profiles_export.csv', 'w', encoding='utf-8') as f:
        for row in cur.execute("SELECT user_id, image_path FROM face_profiles ORDER BY user_id;"):
            f.write(f"{row[0]},{row[1]}\n")
    conn.close()

def sync_users_from_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    with open('users_export.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) < 4:
                continue
            user_id, username, full_name, password = row[:4]
            # Thêm user nếu chưa có username này
            cur.execute("SELECT 1 FROM users WHERE username=?", (username,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO users (username, full_name, password) VALUES (?, ?, ?)",
                    (username, full_name, password)
                )
    conn.commit()
    conn.close()

def sync_face_profiles_from_folders():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    images_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images_attendance")
    for folder in os.listdir(images_root):
        if not folder.startswith("user_"):
            continue
        user_id = folder.split("_")[1]
        user_folder = os.path.join(images_root, folder)
        for img in os.listdir(user_folder):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(user_folder, img)
            # Kiểm tra đã có trong DB chưa
            cur.execute("SELECT 1 FROM face_profiles WHERE user_id=? AND image_path=?", (user_id, img_path))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO face_profiles (user_id, image_path) VALUES (?, ?)",
                    (user_id, img_path)
                )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    sync_users_from_csv()
    sync_face_profiles_from_folders()
    export_face_profiles_to_csv()
