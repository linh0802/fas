#!/usr/bin/env python3
"""
Script để tắt tất cả process web app còn sót lại
"""

import psutil
import subprocess
import time

def cleanup_webapp():
    print("Đang tìm và tắt tất cả process web app...")
    
    # Tìm tất cả process Python chạy web app
    webapp_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('web_app/app.py' in arg for arg in cmdline):
                webapp_processes.append(proc)
                print(f"Tìm thấy process web app: PID {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Tắt tất cả process web app
    for proc in webapp_processes:
        try:
            print(f"Đang tắt process PID {proc.info['pid']}...")
            proc.terminate()
            proc.wait(timeout=3)
            print(f"Đã tắt process PID {proc.info['pid']}")
        except psutil.TimeoutExpired:
            print(f"Process PID {proc.info['pid']} không phản hồi, buộc dừng...")
            try:
                proc.kill()
                proc.wait(timeout=2)
                print(f"Đã kill process PID {proc.info['pid']}")
            except:
                print(f"Không thể kill process PID {proc.info['pid']}")
        except Exception as e:
            print(f"Lỗi khi tắt process PID {proc.info['pid']}: {e}")
    
    # Tắt ngrok
    try:
        print("Đang tắt ngrok...")
        subprocess.run(['ngrok', 'stop'], capture_output=True, timeout=5)
        print("Đã tắt ngrok")
    except Exception as e:
        print(f"Lỗi khi tắt ngrok: {e}")
    
    print("Hoàn tất cleanup!")

if __name__ == '__main__':
    cleanup_webapp() 