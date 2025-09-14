# test_camera_live.py
import cv2


def test_live_view():
    print("启动实时画面测试...")
    print("按 'q' 键退出")

    # 使用 V4L2 后端
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("V4L2 失败，尝试默认后端...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ 无法打开任何摄像头")
        return False

    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    print("✅ 摄像头已打开，开始显示画面...")

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("❌ 读取帧失败")
                break

            # 显示帧信息
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示画面
            cv2.imshow('Camera Test - Press q to quit', frame)

            # 检测按键
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 摄像头资源已释放")

    return True


if __name__ == "__main__":
    success = test_live_view()
    if success:
        print("✅ 摄像头测试成功！")
    else:
        print("❌ 摄像头测试失败")