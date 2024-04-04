import cv2
import os
from init_yolo import get_object_total_length_and_endpoints

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
COLOR = (0, 255, 0)
THICKNESS = 3
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "out")

# rtsp_url = "rtsp://192.168.1.100:8554/unicast"
# gst_str = f"rtspsrc location={rtsp_url} latency=0 buffer-mode=auto ! decodebin ! videoconvert ! appsink"
# mobile phone camera rtsp
rtsp_url = "rtsp://admin:admin@192.168.1.3:1935"  # Replace with your RTSP stream URL
gst_str = f"rtspsrc location={rtsp_url} latency=0 buffer-mode=auto width=640 height=480 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
images_to_save = []


def transform_total_length_to_cm(total_length: float) -> float:
    """Transform total length to cm."""
    return total_length


def main() -> int:
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return 1

    while True:
        ret, frame = cap.read()
        if not ret:
            print("can't read cap.")
            break
        for total_length, p1, p2 in get_object_total_length_and_endpoints(frame):
            total_length = transform_total_length_to_cm(total_length)
            middle_point = (p1 + p2) // 2
            cv2.line(frame, tuple(p1), tuple(p2), COLOR, THICKNESS)
            cv2.putText(
                frame,
                f"{total_length:.2f} cm",
                middle_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                COLOR,
                THICKNESS,
            )
        cv2.imshow("underwater fish length estimation", frame)
        if (pressed_key := cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        elif pressed_key == ord("s"):
            images_to_save.append(frame)

    for i, image in enumerate(images_to_save, start=1):
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        cv2.imwrite(
            os.path.join(OUTPUT_PATH, f"underwater fish_length_estimation{i}.jpg"),
            image,
        )
    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    exit(main())
