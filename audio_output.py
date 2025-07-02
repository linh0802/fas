import subprocess
import os

def play_name(name: str):
    """
    Đọc tên bằng tiếng Việt qua loa sử dụng MAX98357A (I2S).
    Chuyển tên thành file WAV rồi phát bằng aplay.
    :param name: Tên người cần đọc
    """
    try:
        # Sử dụng một file tạm thời trong /tmp để lưu âm thanh
        wav_file = f"/tmp/{name.replace(' ', '_')}.wav"

        # 1. Dùng espeak để tạo file .wav từ tên
        subprocess.run(
            ['espeak', '-v', 'vi', '-s', '140', '-w', wav_file, name],
            check=True
        )

        # 2. Dùng aplay để phát file .wav qua thiết bị I2S (card 0, device 0)
        subprocess.run(['aplay', '-D', 'plughw:0,0', wav_file], check=True)

    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy tiến trình con (espeak hoặc aplay): {e}")
    except Exception as e:
        print(f"Lỗi khi phát âm thanh: {e}")
    finally:
        # 3. Xóa file tạm sau khi phát xong để dọn dẹp
        if 'wav_file' in locals() and os.path.exists(wav_file):
            os.remove(wav_file)