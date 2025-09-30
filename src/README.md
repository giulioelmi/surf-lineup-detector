# Surfer Detector

A simple computer vision project that detects surfers in a lineup using a YOLO model and OpenCV — and shows a live counter of how many surfers are in the frame.

The model was trained in Google Colab using 200 labeled images (100 by me, 100 prelabeled and checked by me using make_prelabels.py) of Malibu.

https://github.com/giulioelmi/surf-lineup-detector/assets/demo.gif  

---

## Features
- **YOLO-powered surfer detection** — detects people in the water frame by frame.
- **Real-time preview** — watch detections as they happen.
- **Surfer counter HUD** — live count overlay on each frame.
- **Minimal, hackable code** — ~50 lines of Python for easy learning and experimenting.
- **make_prelaberls.py** - allows to easily prelable images for Label-Studio using your own model weights. Just change best.pt

---


