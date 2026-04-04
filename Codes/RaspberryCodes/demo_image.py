from ultralytics import YOLO
import cv2

MODEL_PATH = ".//best.pt"
IMG_PATH = "demo_image.jpg"
CONF = 0.4
IOU = 0.5
IMGSZ = 640

model = YOLO(MODEL_PATH)

# Run inference
results = model.predict(
    source=IMG_PATH,
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    max_det=100,
    device="cpu",
    verbose=False
)[0]

# Draw boxes on the image
img = cv2.imread(IMG_PATH)
if results.boxes is not None and len(results.boxes) > 0:
    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c in zip(xyxy, confs):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{c:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

out_path = "demo_image_result.jpg"
cv2.imwrite(out_path, img)
print(f"Saved output to {out_path}")
