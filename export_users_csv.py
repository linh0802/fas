#!/usr/bin/env python3
"""
Script xuất dữ liệu users từ database ra file CSV.
Sử dụng khi cần backup hoặc chỉnh sửa dữ liệu users.
"""

import os
import csv
import sqlite3
from datetime import datetime

def get_db_connection():
    """Kết nối database."""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def export_users_to_csv():
    """Xuất dữ liệu users từ database ra file CSV."""
    print("📤 XUẤT DỮ LIỆU USERS RA CSV")
    print("=" * 50)
    
    # Kiểm tra database có tồn tại không
    if not os.path.exists('database.db'):
        print("❌ Database không tồn tại!")
        print("Vui lòng chạy 'python db.py' để tạo database trước")
        return False
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Lấy tất cả users từ database
    cur.execute("SELECT user_id, username, full_name, password FROM users ORDER BY user_id")
    users = cur.fetchall()
    conn.close()
    
    if not users:
        print("⚠️  Database không có users nào")
        return False
    
    # Tạo backup file cũ nếu có
    if os.path.exists('users_export.csv'):
        backup_name = f"users_export_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        os.rename('users_export.csv', backup_name)
        print(f"📦 Đã backup file cũ: {backup_name}")
    
    # Xuất ra file CSV
    with open('users_export.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='|')
        
        # Hiển thị thông tin trước khi xuất
        print(f"\n📋 DỮ LIỆU SẼ XUẤT ({len(users)} users):")
        print("-" * 60)
        print(f"{'User ID':<8} | {'Username':<15} | {'Full Name':<20} | {'Password':<10}")
        print("-" * 60)
        
        for user in users:
            user_id = user['user_id']
            username = user['username']
            full_name = user['full_name'] or username
            password = user['password'] or '1'
            
            print(f"{user_id:<8} | {username:<15} | {full_name:<20} | {password:<10}")
            
            # Ghi vào file CSV
            writer.writerow([user_id, username, full_name, password])
    
    print(f"\n✅ Đã xuất {len(users)} users ra file: users_export.csv")
    print("📋 Định dạng file: user_id|username|full_name|password")
    print("💡 Bạn có thể chỉnh sửa file này và chạy 'python db.py' để đồng bộ")
    
    return True

def show_database_info():
    """Hiển thị thông tin database hiện tại."""
    print("\n🔍 THÔNG TIN DATABASE HIỆN TẠI")
    print("=" * 50)
    
    conn = get_db_connection()
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
    print("📤 XUẤT DỮ LIỆU USERS RA CSV")
    print("=" * 60)
    
    # Hiển thị thông tin database
    show_database_info()
    
    # Hỏi người dùng có muốn xuất không
    confirm = input("\nBạn có muốn xuất dữ liệu users ra CSV? (y/N): ").strip().lower()
    if confirm != 'y':
        print("❌ Đã hủy xuất dữ liệu")
        return
    
    # Xuất dữ liệu
    if export_users_to_csv():
        print("\n🎉 XUẤT DỮ LIỆU THÀNH CÔNG!")
        print("Bạn có thể:")
        print("1. Chỉnh sửa file users_export.csv")
        print("2. Chạy 'python db.py' để đồng bộ")
        print("3. Chạy 'python reset_database.py' để tạo lại database")
    else:
        print("\n❌ XUẤT DỮ LIỆU THẤT BẠI!")

if __name__ == "__main__":
    main() 