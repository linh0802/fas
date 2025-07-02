from gtts import gTTS
import os
import subprocess
import logging

logging.getLogger('gtts').setLevel(logging.WARNING)

def play_name_smart(name: str, log_func=None):
    mp3_file = "/tmp/name.mp3"
    try:
        # Thử dùng gTTS (yêu cầu có mạng)
        tts = gTTS(text=name, lang='vi')
        tts.save(mp3_file)
        os.system(f"mpg123 {mp3_file}")
        os.remove(mp3_file)
        msg = "Đọc kết quả nhận diện thành công."
        if log_func:
            log_func(msg)
        else:
            logging.info(msg)
    except Exception as e:
        msg = f"Lỗi mạng, chuyển sang chế độ offline..."
        if log_func:
            log_func(msg)
        else:
            logging.error(msg)
        try:
            wav_file = "/tmp/name.wav"
            subprocess.run(['espeak', '-v', 'vi', '-s', '120', '-w', wav_file, name], check=True)
            subprocess.run(['aplay', '-D', 'plughw:0,0', wav_file], check=True)
            os.remove(wav_file)
            msg = "Đọc kết quả nhận diện thành công."
            if log_func:
                log_func(msg)
            else:
                logging.info(msg)
        except Exception as e2:
            msg = f"Lỗi khi phát âm thanh ở chế độ offline: {e2}"
            if log_func:
                log_func(msg)
            else:
                logging.error(msg)

if __name__ == "__main__":
    play_name_smart("Xin chào, Tôi là hệ thống nhận diện gương mặt!")