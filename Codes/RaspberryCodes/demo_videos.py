import time
import cv2
import subprocess
from ultralytics import YOLO

MODEL_PATH = "./best.pt"
IMG_PATH = "/tmp/frame.jpg"

W, H = 640, 480
IMGSZ = 640
CONF = 0.35
IOU = 0.5
MAX_DET = 100

CAPTURE_EVERY_N = 1   
JPEG_QUALITY = 85

model = YOLO(MODEL_PATH)

fps = 0.0
t_prev = time.time()
frame_id = 0

while True:
    if frame_id % CAPTURE_EVERY_N == 0:
        # capture one frame
        cmd = ["rpicam-still", "-n", "-o", IMG_PATH,
       "--width", str(W), "--height", str(H),
       "--quality", str(JPEG_QUALITY),
       "--timeout", "1"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frame = cv2.imread(IMG_PATH)
    if frame is None:
        print("Failed to read frame.")
        continue

    # inference
    t0 = time.time()
    r = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                      device="cpu", verbose=False)[0]
    infer_ms = (time.time() - t0) * 1000.0

    # draw
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{c:.2f}", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # FPS
    t_now = time.time()
    fps_inst = 1.0 / max(1e-6, (t_now - t_prev))
    fps = 0.9 * fps + 0.1 * fps_inst if fps > 0 else fps_inst
    t_prev = t_now

    cv2.putText(frame, f"FPS {fps:.1f} | infer {infer_ms:.1f}ms", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Pi Local YOLOv8 (libcamera)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_id += 1

cv2.destroyAllWindows()
