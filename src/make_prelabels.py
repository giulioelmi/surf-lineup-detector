# make_prelabels_from_list.py
import os, glob, json, shutil
from ultralytics import YOLO

# 1) Inputs
IMAGES_IN = "frames_raw"                      # your screenshots (.png)
MODEL_WTS = "weights/best.pt"
PKG_DIR   = "import_pkg"
lspath    = "/data/local-files/?d=import_pkg/images"  # must match LS Local Files storage
BATCH_SIZE = 12                                # small batches keep memory low

# 2) Collect images and copy into package/images
imgs = sorted(glob.glob(os.path.join(IMAGES_IN, "*.png")))
if not imgs:
    raise SystemExit(f"No PNGs in {IMAGES_IN}/")

os.makedirs(os.path.join(PKG_DIR, "images"), exist_ok=True)
for p in imgs:
    shutil.copy2(p, os.path.join(PKG_DIR, "images", os.path.basename(p)))

# 3) Run batched inference (Ultralytics API)
model = YOLO(MODEL_WTS)

def clip01(x):  # keep coords within 0..100 just in case
    return max(0.0, min(100.0, float(x)))

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

tasks = []
for batch in chunked(imgs, BATCH_SIZE):
    batch_paths = [os.path.join(PKG_DIR, "images", os.path.basename(p)) for p in batch]
    # Tip for Apple Silicon: add device="mps" if it works on your setup
    results = model(batch_paths, imgsz=640, conf=0.15, classes=[0])  # classes=[0] => person

    # 4) Convert results -> Label Studio tasks.json (percent coords)
    for r in results:
        h, w = r.orig_shape
        fname = os.path.basename(r.path)
        preds = []
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy()[:, :4]:
                # drop tiny boxes by relative height (more robust than raw pixels)
                if (y2 - y1) / max(1, h) < 0.01:
                    continue
                preds.append({
                    "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                    "value": {
                        "x": clip01((x1 / w) * 100), "y": clip01((y1 / h) * 100),
                        "width": clip01(((x2 - x1) / w) * 100), "height": clip01(((y2 - y1) / h) * 100),
                        "rotation": 0, "rectanglelabels": ["people"]  # <-- matches your schema
                    }
                })
        tasks.append({
            "data": {"image": f"{lspath}/{fname}"},
            "predictions": [{"result": preds}]
        })

with open(os.path.join(PKG_DIR, "tasks.json"), "w") as f:
    json.dump(tasks, f, indent=2)

print(f"Prepared {len(tasks)} tasks in {PKG_DIR}/ → import into Label Studio")
print("Remember to start LS with:")
print("export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true")
print("export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=import_pkg")
print("Then add a Local Files storage with path: images/")
