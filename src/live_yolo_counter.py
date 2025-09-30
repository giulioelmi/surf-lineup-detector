import cv2
from ultralytics import YOLO

# ---   Edit these to your liking ---
VIDEO_PATH   = "video/malibu_cam.mov"
WEIGHTS_PATH = "best.pt"
IMG_SIZE     = 640
CONF_THRESH  = 0.25
TARGET_FPS   = 5
WINDOW_NAME  = "YOLO Live"
# -----------------------------------

def main():
    model = YOLO(WEIGHTS_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {VIDEO_PATH}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(round(input_fps / TARGET_FPS)))
    print(f"[info] input fps={input_fps:.2f} → processing every {skip} frame(s) (~{TARGET_FPS} fps)")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[info] done (no more frames).")
            break

        frame_index += 1
        if frame_index % skip != 0:
            continue

        # Inference
        r = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH)[0]

        # Draw boxes + count
        count = 0
        for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []:
            count += 1
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        # HUD
        cv2.rectangle(frame, (10,10), (260,60), (0,0,0), -1)
        cv2.putText(frame, f"Surfers: {count}", (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            print("[info] ESC pressed; exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
