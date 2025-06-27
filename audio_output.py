import subprocess

def play_name(name: str):
    """
    Đọc tên bằng tiếng Việt qua loa sử dụng espeak (offline).
    :param name: Tên người cần đọc
    """
    try:
        # -v vi: chọn tiếng Việt, -s 140: tốc độ đọc
        subprocess.run(['espeak', '-v', 'vi', '-s', '140', name])
    except Exception as e:
        print(f"Lỗi khi phát âm thanh: {e}") 