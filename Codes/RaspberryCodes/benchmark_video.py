import cv2
import time
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True, choices=["pt", "onnx", "tflite"])
parser.add_argument("--video", type=str, required=True)
args = parser.parse_args()

MODEL_TYPE = args.model_type
VIDEO_PATH = args.video

PT_PATH = "/home/pothole/RaspberryCodes/best.pt"
ONNX_PATH = "/home/pothole/RaspberryCodes/best.onnx"
TFLITE_PATH = "/home/pothole/RaspberryCodes/best.tflite"

CONF_THRES = 0.35
IOU_THRES = 0.50

if MODEL_TYPE in ["pt", "onnx"]:
    from ultralytics import YOLO
    model_path = PT_PATH if MODEL_TYPE == "pt" else ONNX_PATH
    model = YOLO(model_path)

elif MODEL_TYPE == "tflite":
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    output_index = output_details[0]["index"]

    _, in_h, in_w, in_c = input_shape


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
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


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


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Failed to open video:", VIDEO_PATH)
    raise SystemExit

frame_count = 0
processed_frames = 0
total_detections = 0
latencies = []
fps_values = []

save_frame_ids = {30, 60, 90}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h0, w0 = frame.shape[:2]

    start = time.time()
    boxes = []

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
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                boxes.append((x1, y1, x2, y2, conf))

    elif MODEL_TYPE == "tflite":
        input_tensor = preprocess_tflite(frame)
        interpreter.set_tensor(input_index, input_tensor)
        interpreter.invoke()
        output_tensor = interpreter.get_tensor(output_index)
        boxes = parse_tflite_output(output_tensor, w0, h0)

    latency = (time.time() - start) * 1000.0
    latencies.append(latency)
    fps_values.append(1000.0 / latency if latency > 0 else 0)

    total_detections += len(boxes)
    processed_frames += 1

    for (x1, y1, x2, y2, conf) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"pothole {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if frame_count in save_frame_ids:
        cv2.imwrite(f"{MODEL_TYPE}_frame_{frame_count}.png", frame)

cap.release()

avg_latency = sum(latencies) / len(latencies) if latencies else 0
avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

print("\n========== VIDEO TEST SUMMARY ==========")
print(f"Model type: {MODEL_TYPE}")
print(f"Video: {VIDEO_PATH}")
print(f"Frames processed: {processed_frames}")
print(f"Average latency (ms): {avg_latency:.2f}")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Total detections: {total_detections}")
print("========================================")

csv_path = "/home/pothole/RaspberryCodes/video_benchmark_results.csv"
write_header = False

try:
    with open(csv_path, "r"):
        pass
except FileNotFoundError:
    write_header = True

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(["model_type", "video", "frames_processed", "avg_latency_ms", "avg_fps", "total_detections"])
    writer.writerow([MODEL_TYPE, VIDEO_PATH, processed_frames, round(avg_latency, 2), round(avg_fps, 2), total_detections])

print(f"Results appended to: {csv_path}")