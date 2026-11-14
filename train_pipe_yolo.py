"""Fine-tune a small YOLO pipe detector on your own camera footage.

Three steps:

  1. label   - YOLO-World (zero-shot, prompt "white pole") draws rough boxes on
               frames sampled from raw recordings (record with `test.py --raw`).
  2. review  - step through the frames and fix what YOLO-World got wrong:
                   Enter/Space  accept as shown
                   d            draw the box(es) yourself (mouse; Enter after
                                each box, Esc when done)
                   n            no pipe in this frame (kept as a negative)
                   x            delete the frame (blurry, unsure, ...)
                   b            back one frame,   q  save and quit
  3. train   - fine-tune YOLO11n (default) on the reviewed frames and copy the
               best weights to pipe_detector/pipe_yolo.pt.

    python train_pipe_yolo.py label recordings/*_raw.mp4
    python train_pipe_yolo.py review
    python train_pipe_yolo.py train
    python test.py --weights pipe_detector/pipe_yolo.pt

Data lives in pipe_detector/yolo_data/ (images/, labels/ in YOLO format, one
class "pipe"). Only `label` adds frames, so you can record more, label again,
and only the new frames need reviewing. Use the *_raw.mp4 files: the normal
recording has boxes drawn on it, which the model would learn to look for.
"""
import argparse
import glob
import json
import os
import random
import shutil

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "pipe_detector", "yolo_data")
IMAGES = os.path.join(DATA, "images")
LABELS = os.path.join(DATA, "labels")
STATE = os.path.join(DATA, "reviewed.json")      # stems of frames already reviewed
WORLD_WEIGHTS = os.path.join(HERE, "pipe_detector", "yolov8x-worldv2.pt")
OUT_WEIGHTS = os.path.join(HERE, "pipe_detector", "pipe_yolo.pt")


# ---------------------------------------------------------------- helpers

def load_state():
    try:
        with open(STATE) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()


def save_state(reviewed):
    with open(STATE, "w") as f:
        json.dump(sorted(reviewed), f)


def read_boxes(stem, w, h):
    """YOLO label file -> list of pixel boxes (x0, y0, x1, y1)."""
    path = os.path.join(LABELS, stem + ".txt")
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (float(v) for v in p[1:])
                boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                              int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return boxes


def write_boxes(stem, boxes, w, h):
    """Pixel boxes -> YOLO label file (class 0). An empty file = no pipe."""
    with open(os.path.join(LABELS, stem + ".txt"), "w") as f:
        for x0, y0, x1, y1 in boxes:
            x0, x1 = sorted((max(0, x0), min(w, x1)))
            y0, y1 = sorted((max(0, y0), min(h, y1)))
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            f.write(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                    f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n")


# ---------------------------------------------------------------- 1. label

def cmd_label(args):
    from ultralytics import YOLOWorld
    os.makedirs(IMAGES, exist_ok=True)
    os.makedirs(LABELS, exist_ok=True)
    model = YOLOWorld(WORLD_WEIGHTS)
    model.set_classes(args.prompt)

    added = 0
    for video in args.videos:
        name = os.path.splitext(os.path.basename(video))[0]
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"skip {video}: cannot open")
            continue
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.every == 0:
                stem = f"{name}_{i:06d}"
                img_path = os.path.join(IMAGES, stem + ".jpg")
                if not os.path.exists(img_path):          # never overwrite reviewed work
                    r = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
                    boxes = [tuple(int(v) for v in b) for b in r.boxes.xyxy.tolist()]
                    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    h, w = frame.shape[:2]
                    write_boxes(stem, boxes, w, h)
                    added += 1
            i += 1
        print(f"{video}: {i} frames read")
    print(f"Added {added} frames to {DATA}. Next: python train_pipe_yolo.py review")


# ---------------------------------------------------------------- 2. review

def cmd_review(args):
    reviewed = load_state()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    todo = stems if args.all else [s for s in stems if s not in reviewed]
    if not todo:
        print("Nothing to review.")
        return
    print(f"{len(todo)} frames to review. Keys: Enter=accept  d=draw  n=no pipe  x=delete  b=back  q=quit")

    win = "review"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    k = 0
    while 0 <= k < len(todo):
        stem = todo[k]
        img_path = os.path.join(IMAGES, stem + ".jpg")
        if not os.path.exists(img_path):              # deleted earlier in this session
            k += 1
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        boxes = read_boxes(stem, w, h)

        view = img.copy()
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(view, f"{k + 1}/{len(todo)} {stem}  boxes={len(boxes)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(0) & 0xFF

        if key in (13, 10, 32):                       # accept
            reviewed.add(stem)
            k += 1
        elif key == ord("d"):                         # draw boxes by hand
            rois = cv2.selectROIs(win, img, showCrosshair=False)
            new = [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh in rois]
            write_boxes(stem, new, w, h)
            reviewed.add(stem)
            k += 1
        elif key == ord("n"):                         # negative frame
            write_boxes(stem, [], w, h)
            reviewed.add(stem)
            k += 1
        elif key == ord("x"):                         # drop the frame
            os.remove(img_path)
            lbl = os.path.join(LABELS, stem + ".txt")
            if os.path.exists(lbl):
                os.remove(lbl)
            reviewed.discard(stem)
            k += 1
        elif key == ord("b"):
            k = max(0, k - 1)
        elif key in (ord("q"), 27):
            break
        save_state(reviewed)

    save_state(reviewed)
    cv2.destroyAllWindows()
    left = sum(1 for s in stems if s not in reviewed and os.path.exists(os.path.join(IMAGES, s + ".jpg")))
    print(f"Reviewed {len(reviewed)} frames, {left} left.")


# ---------------------------------------------------------------- 3. train

def cmd_train(args):
    from ultralytics import YOLO
    reviewed = load_state()
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(IMAGES) if f.endswith(".jpg"))
    if not args.include_unreviewed:
        skipped = [s for s in stems if s not in reviewed]
        stems = [s for s in stems if s in reviewed]
        if skipped:
            print(f"Skipping {len(skipped)} unreviewed frames (--include-unreviewed to use them)")
    if len(stems) < 20:
        raise SystemExit(f"Only {len(stems)} usable frames; label and review more first.")

    # Split by contiguous chunks of each video, not random frames: neighbouring
    # frames are near-duplicates, so a random split would make val look too good.
    by_video = {}
    for s in stems:
        by_video.setdefault(s.rsplit("_", 1)[0], []).append(s)
    train, val = [], []
    rng = random.Random(0)
    for vid, ss in by_video.items():
        chunks = [ss[i:i + 30] for i in range(0, len(ss), 30)]
        rng.shuffle(chunks)
        n_val = max(1, round(len(chunks) * args.val))
        for j, c in enumerate(chunks):
            (val if j < n_val and len(chunks) > 1 else train).extend(c)
    if not val:                                       # single tiny video
        val = train[-max(1, len(train) // 5):]

    lists = os.path.join(DATA, "splits")
    os.makedirs(lists, exist_ok=True)
    for name, ss in (("train", train), ("val", val)):
        with open(os.path.join(lists, name + ".txt"), "w") as f:
            f.writelines(os.path.join(IMAGES, s + ".jpg") + "\n" for s in ss)
    yaml_path = os.path.join(DATA, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATA}\ntrain: {lists}/train.txt\nval: {lists}/val.txt\nnames:\n  0: pipe\n")
    n_neg = sum(1 for s in stems if os.path.getsize(os.path.join(LABELS, s + ".txt")) == 0)
    print(f"train {len(train)} / val {len(val)} frames ({n_neg} without a pipe)")

    model = YOLO(args.model)
    model.train(
        data=yaml_path, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=25, project=os.path.join(HERE, "pipe_detector", "runs"), name="pipe_yolo",
        exist_ok=True,
        # extra colour/geometry augmentation so it copes with lighting and,
        # later, water tint the recordings didn't cover
        hsv_h=0.03, hsv_s=0.7, hsv_v=0.5, degrees=10, scale=0.6, flipud=0.5, fliplr=0.5,
    )
    best = os.path.join(model.trainer.save_dir, "weights", "best.pt")
    shutil.copy(best, OUT_WEIGHTS)
    print(f"Saved {OUT_WEIGHTS}. Try it: python test.py --weights {os.path.relpath(OUT_WEIGHTS, HERE)}")


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("label", help="auto-label frames from raw videos with YOLO-World")
    p.add_argument("videos", nargs="+", help="raw recordings (test.py --raw -> *_raw.mp4)")
    p.add_argument("--every", type=int, default=5, help="keep every Nth frame (default 5 = 6/s)")
    p.add_argument("--prompt", nargs="+", default=["white pole"])
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--imgsz", type=int, default=960)
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("review", help="accept / fix / reject the auto labels")
    p.add_argument("--all", action="store_true", help="also show frames already reviewed")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("train", help="fine-tune YOLO on the reviewed frames")
    p.add_argument("--model", default="yolo11n.pt", help="base model (yolo11n.pt, yolo11s.pt, ...)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--val", type=float, default=0.2, help="fraction of each video held out")
    p.add_argument("--include-unreviewed", action="store_true")
    p.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)
