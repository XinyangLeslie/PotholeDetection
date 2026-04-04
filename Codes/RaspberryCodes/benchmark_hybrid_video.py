import cv2
import time
import csv
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, required=True)
parser.add_argument("--pc_ip", type=str, required=True)
args = parser.parse_args()

VIDEO_PATH = args.video
PC_IP = args.pc_ip
URL = f"http://{PC_IP}:5000/infer"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Failed to open video:", VIDEO_PATH)
    raise SystemExit

frame_count = 0
processed_frames = 0
total_detections = 0
latencies = []
fps_values = []
server_times = []

save_frame_ids = {30, 60, 90}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    start = time.time()

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        continue

    files = {
        "image": ("frame.jpg", buf.tobytes(), "image/jpeg")
    }

    try:
        resp = requests.post(URL, files=files, timeout=10)
        data = resp.json()

        boxes = data.get("boxes", [])
        server_ms = float(data.get("server_ms", 0.0))

    except Exception as e:
        print(f"Request failed on frame {frame_count}: {e}")
        boxes = []
        server_ms = 0.0

    latency = (time.time() - start) * 1000.0
    latencies.append(latency)
    fps_values.append(1000.0 / latency if latency > 0 else 0)
    server_times.append(server_ms)

    total_detections += len(boxes)
    processed_frames += 1

    for b in boxes:
        x1 = int(b["x1"])
        y1 = int(b["y1"])
        x2 = int(b["x2"])
        y2 = int(b["y2"])
        conf = float(b["conf"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"pothole {conf:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if frame_count in save_frame_ids:
        cv2.imwrite(f"hybrid_frame_{frame_count}.png", frame)

cap.release()

avg_latency = sum(latencies) / len(latencies) if latencies else 0
avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
avg_server_ms = sum(server_times) / len(server_times) if server_times else 0

print("\n========== HYBRID VIDEO TEST SUMMARY ==========")
print(f"Video: {VIDEO_PATH}")
print(f"Frames processed: {processed_frames}")
print(f"Average end-to-end latency (ms): {avg_latency:.2f}")
print(f"Average FPS: {avg_fps:.2f}")
print(f"Average server inference time (ms): {avg_server_ms:.2f}")
print(f"Total detections: {total_detections}")
print("===============================================")

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
        writer.writerow([
            "model_type", "video", "frames_processed",
            "avg_latency_ms", "avg_fps", "total_detections", "avg_server_ms"
        ])
    writer.writerow([
        "hybrid", VIDEO_PATH, processed_frames,
        round(avg_latency, 2), round(avg_fps, 2),
        total_detections, round(avg_server_ms, 2)
    ])

print(f"Results appended to: {csv_path}")