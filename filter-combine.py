#!/usr/bin/python3
import time
import cv2
import numpy as np
import datetime
import os
import csv
import subprocess
import re
from multiprocessing import Process, Value
from gpiozero import DigitalOutputDevice, DigitalInputDevice
from collections import deque

# ================= 配置 =================
DT_PIN = 5
SCK_PIN = 6
PIXEL_TO_MM = 0.25
SAVE_DIR = "./"
CSV_FILE = os.path.join(SAVE_DIR, "results.csv")
FILTER_SAMPLES = 10  # 滤波采样点数
WEIGHT_THRESHOLD = 0.2  # 重量变化阈值(g)

# 初始化 CSV 文件
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Diameter_mm", "Weight_g", "OCRResult"])


# ================= HX711 类（带滤波）=================
class HX711_gz:
    def __init__(self, dt_pin, sck_pin, offset=616500, scale=1000):
        self.DT = DigitalInputDevice(dt_pin)
        self.SCK = DigitalOutputDevice(sck_pin)
        self.offset = offset
        self.scale = scale
        self.readings = deque(maxlen=FILTER_SAMPLES)  # 滑动窗口滤波
        self.last_stable_weight = 0.0

    def read_raw(self):
        count = 0
        while self.DT.value == 1:
            time.sleep(0.0001)
        for _ in range(24):
            self.SCK.on()
            count = count << 1
            self.SCK.off()
            if self.DT.value:
                count += 1
        self.SCK.on()
        self.SCK.off()
        if count & 0x800000:
            count -= 0x1000000
        return count

    def read_weight_filtered(self):
        """带滤波的重量读取"""
        raw = self.read_raw()
        weight = (raw - self.offset) / self.scale
        weight = round(weight, 1)

        # 添加到读数队列
        self.readings.append(weight)

        # 计算中值滤波（抗干扰更好）
        if len(self.readings) >= 3:
            sorted_readings = sorted(self.readings)
            median_weight = sorted_readings[len(sorted_readings) // 2]
        else:
            median_weight = weight

        # 移动平均滤波
        if self.readings:
            avg_weight = sum(self.readings) / len(self.readings)
        else:
            avg_weight = weight

        # 使用中值滤波结果，结合阈值判断
        if abs(median_weight - self.last_stable_weight) > WEIGHT_THRESHOLD:
            # 变化超过阈值，更新稳定重量
            self.last_stable_weight = median_weight

        return self.last_stable_weight

    def read_weight_simple(self):
        """简单移动平均滤波"""
        raw = self.read_raw()
        weight = (raw - self.offset) / self.scale
        weight = round(weight, 1)

        self.readings.append(weight)

        if self.readings:
            # 移动平均
            avg_weight = sum(self.readings) / len(self.readings)
            return round(avg_weight, 1)
        return weight


# ================= OCR 识别函数 =================
def perform_ocr(image):
    cv2.imwrite('../../current_frame.jpg', image)

    try:
        result = subprocess.run([
            './ocr_db_crnn', 'system',
            '../../models/ch_PP-OCRv3_det_slim_opt.nb',
            '../../models/ch_PP-OCRv3_rec_slim_opt.nb',
            '../../models/ch_ppocr_mobile_v2.0_cls_slim_opt.nb',
            'arm8', 'INT8', '4', '1',
            '../../current_frame.jpg',
            '../../models/config.txt',
            '../../models/ppocr_keys_v1.txt',
            'True'
        ], capture_output=True, text=True, timeout=3)

        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if re.match(r'^\d+\s+\S+\s+[\d\.]+$', line.strip()):
                    return line.split()[1]
    except:
        pass
    return ""


# ================= HX711 进程 =================
def hx711_process(shared_weight):
    hx = HX711_gz(DT_PIN, SCK_PIN)
    while True:
        # 使用带滤波的重量读取
        weight = hx.read_weight_filtered()
        # 或者使用简单滤波：weight = hx.read_weight_simple()
        shared_weight.value = weight
        time.sleep(0.1)  # 适当增加读取间隔


# ================= 摄像头进程 =================
def camera_process(shared_weight):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    diameter_mm = 0.0
    ocr_text = ""
    last_ocr_time = 0

    while True:
        ret, img = cap.read()
        if not ret:
            break

        h, w = img.shape[:2]
        left_img = img[:, :w // 2]
        right_img = img[:, w // 2:]

        # 左半边：直径测量
        gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            _, maxVal, _, maxLoc = cv2.minMaxLoc(dist)
            r = int(maxVal)
            cx, cy = maxLoc
            if r > 5:
                diameter_mm = 2 * r * PIXEL_TO_MM
                circle_center = (cx, cy)
                radius = r
            
            cv2.circle(left_img,circle_center,radius,(0,255,0),2)

        # 右半边：OCR识别
        current_time = time.time()
        if current_time - last_ocr_time >= 1.0:
            ocr_text = perform_ocr(right_img)
            last_ocr_time = current_time

        # 显示结果
        combined = np.hstack((left_img, right_img))
        cv2.putText(combined, f"Weight: {shared_weight.value:.1f}g", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
        cv2.putText(combined, f"OCR: {ocr_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(combined, f"Diameter: {diameter_mm:.1f}mm", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Combined View", combined)

        # 保存数据
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CSV_FILE, mode="a", newline="") as f:
            csv.writer(f).writerow([now, f"{diameter_mm:.1f}", f"{shared_weight.value:.1f}", ocr_text])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ================= 主程序 =================
if __name__ == "__main__":
    shared_weight = Value('d', 0.0)
    p1 = Process(target=hx711_process, args=(shared_weight,), daemon=True)
    p1.start()
    time.sleep(0.5)
    camera_process(shared_weight)
    print("程序结束")