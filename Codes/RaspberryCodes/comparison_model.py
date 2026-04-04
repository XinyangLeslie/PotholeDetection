import cv2
import time
import argparse
import numpy as np

# ---------------------------
# Parse arguments
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True,
                    choices=["pt", "onnx", "tflite"],
                    help="Choose model type: pt / onnx / tflite")
args = parser.parse_args()

MODEL_TYPE = args.model_type

PT_PATH = "/home/pothole/RaspberryCodes/best.pt"
ONNX_PATH = "/home/pothole/RaspberryCodes/best.onnx"
TFLITE_PATH = "/home/pothole/RaspberryCodes/best.tflite"

CONF_THRES = 0.35
IOU_THRES = 0.50
CAM_W = 640
CAM_H = 480
INFER_EVERY_N_FRAMES = 1

# ---------------------------
# Load model
# ---------------------------
if MODEL_TYPE in ["pt", "onnx"]:
    from ultralytics import YOLO
    model_path = PT_PATH if MODEL_TYPE == "pt" else ONNX_PATH
    model = YOLO(model_path)

elif MODEL_TYPE == "tflite":
    from ai_edge_litert.interpreter import Interpreter

    model_path = TFLITE_PATH
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    output_index = output_details[0]["index"]

    _, in_h, in_w, in_c = input_shape

# ---------------------------
# Helper functions for TFLite
# ---------------------------
def preprocess_tflite(frame):
    img = cv2.resize(frame, (in_w, in_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    if input_dtype == np.uint8:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(input_dtype)

    return img

def xywh_to_xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2

def scale_box_to_original(x1, y1, x2, y2, orig_w, orig_h):
    x1 = int(max(0, min(orig_w - 1, x1 * orig_w)))
    y1 = int(max(0, min(orig_h - 1, y1 * orig_h)))
    x2 = int(max(0, min(orig_w - 1, x2 * orig_w)))
    y2 = int(max(0, min(orig_h - 1, y2 * orig_h)))
    return x1, y1, x2, y2

def parse_tflite_output(output, orig_w, orig_h):
    output = np.squeeze(output)

    if output.ndim == 2 and output.shape[0] == 5:
        output = output.T

    boxes = []
    scores = []

    if output.ndim != 2 or output.shape[1] < 5:
        print("Unexpected TFLite output shape:", output.shape)
        return []

    for row in output:
        x, y, w, h, conf = row[:5]

        if conf < CONF_THRES:
            continue

        x1, y1, x2, y2 = xywh_to_xyxy(x, y, w, h)
        x1, y1, x2, y2 = scale_box_to_original(x1, y1, x2, y2, orig_w, orig_h)

        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(conf))

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRES, IOU_THRES)

    results = []
    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = boxes[idx]
            results.append((x, y, x + w, y + h, scores[idx]))

    return results

# ---------------------------
# Open USB camera
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open USB camera")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

prev_time = time.time()
frame_count = 0
last_boxes = []

fps_values = []
start_test_time = time.time()

print(f"Running model type: {MODEL_TYPE}")

# ---------------------------
# Main loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1
    h0, w0 = frame.shape[:2]

    if frame_count % INFER_EVERY_N_FRAMES == 0:

        if MODEL_TYPE in ["pt", "onnx"]:
            results = model.predict(
                source=frame,
                imgsz=640,
                conf=CONF_THRES,
                iou=IOU_THRES,
                device="cpu",
                verbose=False
            )

            result = results[0]
            last_boxes = []

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    last_boxes.append((x1, y1, x2, y2, conf))

        elif MODEL_TYPE == "tflite":
            input_tensor = preprocess_tflite(frame)
            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()
            output_tensor = interpreter.get_tensor(output_index)
            last_boxes = parse_tflite_output(output_tensor, w0, h0)

    for (x1, y1, x2, y2, conf) in last_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"pothole {conf:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    current_time = time.time()
    fps = 1.0 / max(current_time - prev_time, 1e-6)
    prev_time = current_time
    fps_values.append(fps)

    avg_fps = sum(fps_values[-30:]) / min(len(fps_values), 30)

    cv2.putText(frame, f"Format: {MODEL_TYPE.upper()}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(frame, "Device: Raspberry Pi 5 CPU", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
    cv2.putText(frame, f"Detections: {len(last_boxes)}", (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

    cv2.imshow(f"Compare Model - {MODEL_TYPE.upper()}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        filename = f"{MODEL_TYPE}_capture_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")

cap.release()
cv2.destroyAllWindows()

total_time = time.time() - start_test_time
overall_avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

print("\n========== TEST SUMMARY ==========")
print(f"Model type: {MODEL_TYPE}")
print(f"Frames processed: {len(fps_values)}")
print(f"Total runtime: {total_time:.2f} seconds")
print(f"Overall average FPS: {overall_avg_fps:.2f}")
print("==================================")