import time
import cv2
import numpy as np
import subprocess
import requests

# ====== CONFIG ======
PC_IP = "192.168.4.83"
URL = f"http://{PC_IP}:5000/infer"

CAM_W, CAM_H = 640, 480
CAM_FPS = 30

SEND_EVERY_N = 2
JPEG_QUALITY = 80
TIMEOUT = 2.0
MAX_BUF = 2_000_000
READ_CHUNK = 16384

# ====== Debug print options ======
PRINT_JSON_EVERY_SEND = True     
JSON_PREVIEW_CHARS = 600         # Print full JSON if None, or preview first N chars

# ====== Start MJPEG stream from camera ======
cmd = [
    "rpicam-vid",
    "-t", "0",
    "--width", str(CAM_W),
    "--height", str(CAM_H),
    "--framerate", str(CAM_FPS),
    "--codec", "mjpeg",
    "-o", "-"
]
print("Starting camera:", " ".join(cmd), flush=True)
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

buf = bytearray()

fps = 0.0
t_prev = time.time()
frame_id = 0

last_boxes = []
last_server_ms = 0.0
last_rtt_ms = 0.0

def decode_jpeg(jpg_bytes: bytes):
    arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

try:
    while True:
        chunk = proc.stdout.read(READ_CHUNK)
        if not chunk:
            continue
        buf.extend(chunk)

        # cap buffer to reduce latency
        if len(buf) > MAX_BUF:
            buf = buf[-MAX_BUF:]

        # take latest complete JPEG frame (drop old frames)
        end = buf.rfind(b"\xff\xd9")
        if end == -1:
            continue
        start = buf.rfind(b"\xff\xd8", 0, end)
        if start == -1:
            buf = buf[end + 2:]
            continue

        jpg = bytes(buf[start:end + 2])
        buf = buf[end + 2:]

        frame = decode_jpeg(jpg)
        if frame is None:
            continue

        # send to PC every N frames
        if frame_id % SEND_EVERY_N == 0:
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                try:
                    t0 = time.time()
                    r = requests.post(
                        URL,
                        files={"image": ("frame.jpg", enc.tobytes(), "image/jpeg")},
                        timeout=TIMEOUT
                    )
                    last_rtt_ms = (time.time() - t0) * 1000.0

                    # ---- NEW: print HTTP status + RTT ----
                    print(f"[Pi] HTTP {r.status_code} | RTT {last_rtt_ms:.1f} ms", flush=True)

                    # ---- NEW: parse and print JSON ----
                    data = r.json()  # if not JSON, this will raise and go to except
                    if PRINT_JSON_EVERY_SEND:
                        s = str(data)
                        if JSON_PREVIEW_CHARS is not None and len(s) > JSON_PREVIEW_CHARS:
                            s = s[:JSON_PREVIEW_CHARS] + " ... (truncated)"
                        print(f"[Pi] Received JSON: {s}", flush=True)

                    if data.get("ok", False):
                        last_boxes = data.get("boxes", [])
                        last_server_ms = float(data.get("server_ms", 0.0))

                        # ---- NEW: print a short summary ----
                        print(f"[Pi] ok=True | boxes={len(last_boxes)} | server_ms={last_server_ms:.1f}", flush=True)
                    else:
                        print(f"[Pi] ok=False | keys={list(data.keys())}", flush=True)

                except Exception as e:
                    # ---- NEW: do not swallow errors ----
                    print(f"[Pi] ERROR: {repr(e)}", flush=True)

        # draw last returned boxes
        for b in last_boxes:
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            conf = float(b["conf"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS
        t_now = time.time()
        fps_inst = 1.0 / max(1e-6, (t_now - t_prev))
        fps = 0.9 * fps + 0.1 * fps_inst if fps > 0 else fps_inst
        t_prev = t_now

        cv2.putText(frame,
                    f"HYBRID FPS {fps:.1f} | PC {last_server_ms:.0f}ms | RTT {last_rtt_ms:.0f}ms | send every {SEND_EVERY_N}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imshow("Pi Hybrid (Pi->PC infer->Pi display)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

finally:
    try:
        proc.terminate()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print("Stopped.", flush=True)