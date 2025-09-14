import cv2
import subprocess
import time

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
last_ocr_time = 0
ocr_interval = 0.3
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_time = time.time()

    # 计算FPS
    elapsed = current_time - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    # 定时OCR识别
    if current_time - last_ocr_time >= ocr_interval:
        cv2.imwrite('../../current_frame.jpg', frame)

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
            ], capture_output=True, text=True, timeout=2)

            if result.returncode == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {result.stdout.strip()}")
                last_ocr_time = current_time

        except Exception as e:
            print(f"识别错误: {e}")

    # 显示FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Camera OCR', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()