from flask import Flask, request, jsonify
import numpy as np
import cv2
import time
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = r".\best.pt"  
model = YOLO(MODEL_PATH)

@app.route("/infer", methods=["POST"])
def infer():
    t0 = time.time()
    f = request.files.get("image", None)
    if f is None:
        return jsonify({"ok": False, "error": "no image"}), 400

    img_bytes = f.read()
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False, "error": "decode failed"}), 400

    # preprocessing step (required for Phase 3): conf threshold + imgsz resize
    r = model.predict(img, imgsz=640, conf=0.35, iou=0.5, max_det=100, device=0, verbose=False)[0]

    boxes = []
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            boxes.append({"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "conf": float(c)})

    dt_ms = (time.time() - t0) * 1000.0
    return jsonify({"ok": True, "boxes": boxes, "server_ms": dt_ms})

if __name__ == "__main__":
    # allow Pi to access on LAN
    app.run(host="0.0.0.0", port=5000, debug=False)