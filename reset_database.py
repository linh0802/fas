#!/usr/bin/env python3
"""
Script xóa toàn bộ database và tạo lại từ file CSV.
Sử dụng khi database bị lỗi nghiêm trọng và cần reset hoàn toàn.
"""

import os
import sqlite3
import csv
import shutil
from datetime import datetime

def backup_database():
    """Tạo backup database hiện tại trước khi xóa."""
    if os.path.exists('database.db'):
        backup_name = f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2('database.db', backup_name)
        print(f"✅ Đã tạo backup: {backup_name}")
        return backup_name
    return None

def delete_database():
    """Xóa database hiện tại."""
    if os.path.exists('database.db'):
        os.remove('database.db')
        print("🗑️  Đã xóa database cũ")
    else:
        print("ℹ️  Không có database để xóa")

def create_new_database():
    """Tạo database mới từ file CSV."""
    print("\n🔧 TẠO DATABASE MỚI TỪ CSV")
    print("=" * 50)
    
    # Kiểm tra file CSV
    if not os.path.exists('users_export.csv'):
        print("❌ File users_export.csv không tồn tại!")
        print("Vui lòng tạo file users_export.csv với định dạng:")
        print("user_id|username|full_name|password")
        return False
    
    # Tạo database mới
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    
    # Tạo bảng users
    cur.execute("""
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            full_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Tạo bảng face_profiles
    cur.execute("""
        CREATE TABLE face_profiles (
            profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            face_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            note TEXT
        )
    """)
    
    # Import users từ CSV
    added_users = 0
    with open('users_export.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            if len(row) < 4:
                continue
            
            user_id, username, full_name, password = row[:4]
            
            try:
                cur.execute(
                    "INSERT INTO users (username, full_name, password) VALUES (?, ?, ?)",
                    (username, full_name, password)
                )
                added_users += 1
                print(f"  ✅ Thêm user: {username} - {full_name}")
            except sqlite3.IntegrityError as e:
                print(f"  ⚠️  Bỏ qua user trùng: {username} ({e})")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Đã tạo database mới với {added_users} users")
    return True

def sync_images_from_folders():
    """Đồng bộ ảnh từ thư mục vào database mới."""
    print("\n📸 ĐỒNG BỘ ẢNH TỪ THƯ MỤC")
    print("=" * 50)
    
    images_root = "images_attendance"
    if not os.path.exists(images_root):
        print("⚠️  Thư mục images_attendance không tồn tại")
        return
    
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    
    added_images = 0
    for folder in os.listdir(images_root):
        if not folder.startswith("user_"):
            continue
        
        user_id = folder.split("_")[1]
        user_folder = os.path.join(images_root, folder)
        
        # Kiểm tra user có tồn tại trong DB không
        cur.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
        if not cur.fetchone():
            print(f"  ⚠️  Bỏ qua folder {folder}: user_id {user_id} không tồn tại trong DB")
            continue
        
        for img in os.listdir(user_folder):
            if not img.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(user_folder, img)
            
            # Kiểm tra ảnh có tồn tại không
            if not os.path.exists(img_path):
                print(f"  ⚠️  Ảnh không tồn tại: {img_path}")
                continue
            
            # Kiểm tra đã có trong DB chưa
            cur.execute("SELECT 1 FROM face_profiles WHERE user_id=? AND image_path=?", (user_id, img_path))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO face_profiles (user_id, image_path) VALUES (?, ?)",
                    (user_id, img_path)
                )
                added_images += 1
                print(f"  ✅ Thêm ảnh: {img_path}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Đã đồng bộ {added_images} ảnh")

def show_database_summary():
    """Hiển thị tổng quan database mới."""
    print("\n📊 TỔNG QUAN DATABASE MỚI")
    print("=" * 50)
    
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    
    # Thống kê users
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]
    
    # Thống kê ảnh
    cur.execute("SELECT COUNT(*) FROM face_profiles")
    total_images = cur.fetchone()[0]
    
    # Users có ảnh
    cur.execute("""
        SELECT u.user_id, u.username, u.full_name, COUNT(fp.profile_id) as image_count
        FROM users u
        LEFT JOIN face_profiles fp ON u.user_id = fp.user_id
        GROUP BY u.user_id
        HAVING COUNT(fp.profile_id) > 0
        ORDER BY u.user_id
    """)
    users_with_images = cur.fetchall()
    
    conn.close()
    
    print(f"👥 Tổng users: {total_users}")
    print(f"📸 Tổng ảnh: {total_images}")
    print(f"✅ Users có ảnh: {len(users_with_images)}")
    
    if users_with_images:
        print("\n📋 DANH SÁCH USERS CÓ ẢNH:")
        print("-" * 60)
        for user in users_with_images:
            print(f"User {user['user_id']}: {user['username']} - {user['full_name']} ({user['image_count']} ảnh)")

def main():
    print("🔄 RESET DATABASE HOÀN TOÀN")
    print("=" * 60)
    print("⚠️  CẢNH BÁO: Hành động này sẽ xóa toàn bộ database hiện tại!")
    print("   Dữ liệu sẽ được tạo lại từ file users_export.csv")
    print()
    
    # Xác nhận từ người dùng
    confirm = input("Bạn có chắc chắn muốn reset database? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Đã hủy reset database")
        return
    
    print("\n🚀 BẮT ĐẦU RESET DATABASE...")
    
    # Bước 1: Backup database cũ
    backup_file = backup_database()
    
    # Bước 2: Xóa database cũ
    delete_database()
    
    # Bước 3: Tạo database mới từ CSV
    if not create_new_database():
        print("❌ Không thể tạo database mới!")
        if backup_file:
            print(f"Bạn có thể khôi phục từ backup: {backup_file}")
        return
    
    # Bước 4: Đồng bộ ảnh từ thư mục
    sync_images_from_folders()
    
    # Bước 5: Hiển thị tổng quan
    show_database_summary()
    
    print("\n🎉 RESET DATABASE THÀNH CÔNG!")
    print("Bạn có thể chạy các lệnh sau để kiểm tra:")
    print("  python check_training_data.py")
    print("  python finish_train.py")

if __name__ == "__main__":
    main() 