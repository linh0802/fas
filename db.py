#!/usr/bin/env python3
"""
Module quản lý database cho hệ thống điểm danh khuôn mặt.
Chức năng chính: kết nối, khởi tạo, và đồng bộ dữ liệu.
"""

import sqlite3
import csv
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")

def get_db_connection():
    """Kết nối database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Khởi tạo database và tạo các bảng."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Bảng users
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
    
    # Bảng face_profiles
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
    print("✅ Đã khởi tạo database thành công")

def export_users_to_csv():
    """Xuất dữ liệu users từ database ra file CSV."""
    print("📤 XUẤT DỮ LIỆU USERS RA CSV")
    print("=" * 40)
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Lấy tất cả users từ database
    cur.execute("SELECT user_id, username, full_name, password FROM users ORDER BY user_id")
    users = cur.fetchall()
    conn.close()
    
    if not users:
        print("⚠️  Database không có users nào")
        return False
    
    # Xuất ra file CSV
    with open('users_export.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        for user in users:
            writer.writerow([
                user['user_id'],
                user['username'],
                user['full_name'] or user['username'],  # Dùng username nếu full_name rỗng
                user['password'] or '1'  # Dùng '1' nếu password rỗng
            ])
    
    print(f"✅ Đã xuất {len(users)} users ra file: users_export.csv")
    print("📋 Định dạng file: user_id|username|full_name|password")
    return True

def sync_users_from_csv():
    """Đồng bộ users từ file CSV."""
    if not os.path.exists('users_export.csv'):
        print("⚠️  File users_export.csv không tồn tại")
        print("🔄 Tự động xuất dữ liệu users từ database...")
        if export_users_to_csv():
            print("✅ Đã tạo file users_export.csv từ database hiện tại")
        else:
            print("❌ Không thể tạo file CSV")
            return
        print("💡 Bạn có thể chỉnh sửa file users_export.csv và chạy lại để đồng bộ")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    added_count = 0
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
                added_count += 1
    
    conn.commit()
    conn.close()
    print(f"✅ Đã đồng bộ {added_count} users từ CSV")

def sync_face_profiles_from_folders():
    """Đồng bộ ảnh từ thư mục vào database."""
    images_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images_attendance")
    
    if not os.path.exists(images_root):
        print("⚠️  Thư mục images_attendance không tồn tại")
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    added_count = 0
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
                added_count += 1
    
    conn.commit()
    conn.close()
    print(f"✅ Đã đồng bộ {added_count} ảnh từ thư mục")

def get_training_data_summary():
    """Lấy tổng quan về dữ liệu training."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Đếm tổng số users và ảnh
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM face_profiles")
    total_images = cur.fetchone()[0]
    
    # Lấy danh sách users với số lượng ảnh
    cur.execute("""
        SELECT u.user_id, u.username, u.full_name, COUNT(fp.profile_id) as image_count
        FROM users u
        LEFT JOIN face_profiles fp ON u.user_id = fp.user_id
        GROUP BY u.user_id, u.username, u.full_name
        ORDER BY u.user_id
    """)
    users_with_images = cur.fetchall()
    
    # Lấy users có ít hơn 5 ảnh
    cur.execute("""
        SELECT u.user_id, u.username, u.full_name, COUNT(fp.profile_id) as image_count
        FROM users u
        LEFT JOIN face_profiles fp ON u.user_id = fp.user_id
        GROUP BY u.user_id, u.username, u.full_name
        HAVING COUNT(fp.profile_id) < 5
        ORDER BY u.user_id
    """)
    users_with_few_images = cur.fetchall()
    
    conn.close()
    
    return {
        'total_users': total_users,
        'total_images': total_images,
        'users_with_images': users_with_images,
        'users_with_few_images': users_with_few_images
    }

def get_user_id_to_fullname_mapping():
    """Lấy mapping user_id -> full_name."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT user_id, full_name FROM users ORDER BY user_id")
    users = cur.fetchall()
    
    mapping = {}
    for user in users:
        mapping[str(user['user_id'])] = user['full_name']
    
    conn.close()
    return mapping

if __name__ == "__main__":
    print("🔧 KHỞI TẠO VÀ ĐỒNG BỘ DATABASE")
    print("=" * 50)
    
    # Khởi tạo database
    init_db()
    
    # Đồng bộ từ CSV (hoặc xuất CSV nếu chưa có)
    sync_users_from_csv()
    
    # Đồng bộ từ thư mục ảnh
    sync_face_profiles_from_folders()
    
    print("\n✅ Hoàn tất khởi tạo database!")
    print("\n📋 LƯU Ý:")
    print("- File users_export.csv đã được tạo từ database hiện tại")
    print("- Bạn có thể chỉnh sửa file này và chạy lại để đồng bộ")
    print("- Định dạng file: user_id|username|full_name|password")
