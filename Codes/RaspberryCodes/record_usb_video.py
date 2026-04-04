import cv2
import time

OUTPUT_PATH = "/home/pothole/RaspberryCodes/test_video.mp4"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20
RECORD_SECONDS = 15

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Failed to open USB camera")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

print("Recording started...")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    writer.write(frame)

    elapsed = time.time() - start_time
    cv2.putText(frame, f"Recording... {elapsed:.1f}s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("USB Camera Recording", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("Recording stopped by user.")
        break

    if elapsed >= RECORD_SECONDS:
        print("Recording finished.")
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved video to: {OUTPUT_PATH}")