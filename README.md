# Surf Lineup Detector

Counts the surfers in a lineup from surf-cam video. A YOLO model trained on Malibu footage
draws a box around each surfer and shows a live count on every frame, so you can tell how
crowded a break is before you paddle out.

## How it was built

- Took screenshots from the Malibu surf cam.
- Labeled 200 images in Label Studio: 100 by hand, plus 100 pre-labeled by an early model
  (`src/make_prelabels.py`) and then checked by hand.
- Fine-tuned a YOLO model on them in Google Colab. The trained weights are `best.pt`.

## Run it

```bash
pip install ultralytics opencv-python
python src/live_yolo_counter.py
```

It plays `video/malibu_cam.mov` with the detections and a "Surfers: N" counter. Edit the
settings at the top of the script to change the video, confidence threshold or frame rate.

## Files

| File | What it does |
|---|---|
| `src/live_yolo_counter.py` | Runs the model on a video and shows boxes and the surfer count (~50 lines). |
| `src/make_prelabels.py` | Pre-labels new screenshots with your model and packages them for import into Label Studio. |
| `best.pt` | Trained weights. |
| `video/` | Sample surf-cam clips. |

## License

MIT
