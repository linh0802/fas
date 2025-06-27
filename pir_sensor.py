#!/usr/bin/env python3
import gpiod
import time
import logging
import threading

class PIRSensor:
    """
    Một lớp để quản lý cảm biến chuyển động PIR sử dụng thư viện gpiod.
    Lớp này chạy một luồng (thread) nền để liên tục theo dõi trạng thái cảm biến.
    """
    def __init__(self, pin_signal=17, chip_path='/dev/gpiochip4'):
        """
        Khởi tạo cảm biến PIR.
        Args:
            pin_signal (int): Chân GPIO nhận tín hiệu từ cảm biến.
            chip_path (str): Đường dẫn đến chip GPIO.
        """
        self.pin_signal = pin_signal
        self.chip_path = chip_path
        self.line_request = None
        self.motion_detected = False
        self.running = False
        
        try:
            logging.info(f"Khởi tạo PIR Sensor. Tín hiệu: GPIO{self.pin_signal}")
            self.line_request = gpiod.request_lines(
                self.chip_path,
                consumer="pir_sensor_app",
                config={
                    self.pin_signal: gpiod.LineSettings(
                        direction=gpiod.line.Direction.INPUT
                    )
                }
            )
            self.thread = None
            logging.info("PIR đã khởi động và sẵn sàng.")
        except Exception as e:
            logging.error(f"Lỗi khởi tạo gpiod: {e}")
            self.line_request = None
            self.motion_detected = False
            self.running = False
            self.thread = None

    def _monitor(self):
        """Phương thức chạy trong luồng nền để theo dõi cảm biến."""
        while self.running:
            try:
                if self.line_request:
                    self.motion_detected = self.line_request.get_value(self.pin_signal) == gpiod.line.Value.ACTIVE
                else:
                    self.running = False
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Lỗi nghiêm trọng trong luồng theo dõi PIR: {e}")
                self.running = False

    def start(self):
        """Bắt đầu luồng theo dõi cảm biến."""
        try:
            if not self.running and self.line_request:
                self.running = True
                self.thread = threading.Thread(target=self._monitor, daemon=True)
                self.thread.start()
                logging.info("PIR sensor monitoring thread started.")
            elif not self.line_request:
                logging.warning("PIR sensor không được khởi tạo đúng cách, không thể bắt đầu monitoring.")
        except Exception as e:
            logging.error(f"Lỗi khi bắt đầu PIR sensor: {e}")
            self.running = False

    def stop(self):
        """Dừng luồng theo dõi."""
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=1)
            logging.info("PIR sensor monitoring thread stopped.")

    def release(self):
        """Giải phóng tài nguyên GPIO."""
        self.stop()
        if self.line_request:
            self.line_request.release()
            self.line_request = None
            logging.info(f"Đã giải phóng line GPIO {self.pin_signal}.")

    def is_motion(self):
        """Kiểm tra xem có phát hiện chuyển động hay không."""
        return self.motion_detected