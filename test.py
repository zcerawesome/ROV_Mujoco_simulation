"""Live pipe-detector test: DepthAI camera -> detector -> Tk window with the
bounding boxes drawn on every frame.

    python test.py                      # YOLO-World (x), zero-shot, prompt "white pole"
    python test.py --prompt "white pvc pipe"
    python test.py --weights pipe_detector/pipe_yolo.pt   # fine-tuned model (train_pipe_yolo.py)
    python test.py --model cnn          # the old PipeNet ONNX model (many false positives)
    python test.py --raw                # also save the unannotated frames
    python test.py --no-record          # don't save video

The annotated feed is saved to recordings/pipe_<time>.mp4 by default.

YOLO-World takes text prompts, so no training is needed to try it. The prompt
box and confidence slider in the window change it live. Close the window (or
press Esc) to quit.
"""
import argparse
import os
import threading
import time
import tkinter as tk

import cv2
import depthai as dai

HERE = os.path.dirname(os.path.abspath(__file__))
YOLO_WEIGHTS = os.path.join(HERE, "pipe_detector", "yolov8x-worldv2.pt")  # auto-downloads if missing

FPS = 30
PREVIEW_SIZE = (960, 540)
COLORS = [(0, 255, 0), (255, 160, 0), (0, 200, 255), (255, 0, 200)]  # BGR, one per prompt


def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setPreviewSize(*PREVIEW_SIZE)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)   # getCvFrame() is then BGR
    cam.setFps(FPS)
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return pipeline


class YoloDetector:
    """YOLO detector. detect() takes a BGR frame and returns a list of
    (x0, y0, x1, y1, conf, label), best first.

    With YOLO-World weights (name contains "world") it is open-vocabulary and
    driven by text prompts. Any other .pt (e.g. the fine-tuned model from
    train_pipe_yolo.py) is used as-is with its own class names."""

    def __init__(self, prompts, conf=0.1, weights=YOLO_WEIGHTS, imgsz=960):
        self.open_vocab = "world" in os.path.basename(weights).lower()
        self.conf = conf
        self.imgsz = imgsz          # 960 > default 640: small, far-off pipes keep more pixels
        self.lock = threading.Lock()
        if self.open_vocab:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(weights)
            self.set_prompts(prompts)
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights)
            self.prompts = [self.model.names[k] for k in sorted(self.model.names)]

    def set_prompts(self, prompts):
        if not self.open_vocab:
            return
        with self.lock:
            self.prompts = list(prompts)
            self.model.set_classes(self.prompts)

    def detect(self, frame_bgr):
        with self.lock:
            r = self.model.predict(frame_bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            names = self.prompts
        out = []
        for (x0, y0, x1, y1), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(),
                                         r.boxes.cls.tolist()):
            out.append((int(x0), int(y0), int(x1), int(y1), float(c), names[int(k)]))
        return sorted(out, key=lambda d: -d[4])


class CnnDetector:
    """The original PipeNet ONNX model, wrapped to the same interface."""

    def __init__(self):
        from pipe_detector.detector import PipeDetector
        self.det = PipeDetector()
        self.conf = self.det.threshold

    def detect(self, frame_bgr):
        self.det.threshold = self.conf
        d = self.det.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        return [] if d is None else [(*d["box"], d["score"], "pipe")]


def draw(frame_bgr, dets):
    out = frame_bgr.copy()
    labels = sorted({d[5] for d in dets})
    for i, (x0, y0, x1, y1, c, label) in enumerate(dets):
        color = COLORS[labels.index(label) % len(COLORS)]
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 3 if i == 0 else 1)   # best box thicker
        cv2.putText(out, f"{label} {c:.2f}", (x0, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    h, w = out.shape[:2]
    cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    return out


class Recorder:
    """Writes frames to an .mp4 at a fixed FPS in real time: detection runs
    slower than the camera, so each frame is repeated as many times as the
    wall-clock time it covered, and playback runs at the real speed."""

    def __init__(self, path, fps=FPS):
        self.path, self.fps = path, fps
        self.writer = None
        self.start = None
        self.written = 0

    def write(self, frame_bgr):
        now = time.time()
        if self.writer is None:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            h, w = frame_bgr.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError(f"could not open video writer for {self.path}")
            self.start = now
        due = int((now - self.start) * self.fps) + 1     # frames that should exist by now
        for _ in range(max(due - self.written, 1)):
            self.writer.write(frame_bgr)
            self.written += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            print(f"Saved {self.path} ({self.written / self.fps:.1f} s)")


class Worker(threading.Thread):
    """Grabs frames, runs the detector, and keeps only the latest annotated
    result so the GUI never falls behind the camera. Every annotated frame is
    also saved to video (and the raw frame too, if raw_path is given)."""

    def __init__(self, detector, out_path=None, raw_path=None):
        super().__init__(daemon=True)
        self.detector = detector
        self.lock = threading.Lock()
        self.result = None          # (annotated BGR frame, dets, infer_ms, fps)
        self.error = None
        self.running = True
        self.recorders = [Recorder(p) if p else None for p in (out_path, raw_path)]

    def run(self):
        rec_out, rec_raw = self.recorders
        try:
            with dai.Device(build_pipeline()) as device:
                q = device.getOutputQueue("rgb", maxSize=1, blocking=False)
                last, fps = time.time(), 0.0
                while self.running:
                    msg = q.tryGet()                     # non-blocking, so close() can stop us
                    if msg is None:
                        time.sleep(0.002)
                        continue
                    frame_bgr = msg.getCvFrame()
                    t0 = time.time()
                    dets = self.detector.detect(frame_bgr)
                    infer_ms = (time.time() - t0) * 1000
                    out = draw(frame_bgr, dets)
                    if rec_out:
                        rec_out.write(out)
                    if rec_raw:
                        rec_raw.write(frame_bgr)
                    now = time.time()
                    fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
                    last = now
                    with self.lock:
                        self.result = (out, dets, infer_ms, fps)
        except Exception as e:                           # no camera, USB error, ...
            self.error = e
        finally:
            for r in self.recorders:                     # finalize the mp4 even on error
                if r:
                    r.close()


class App:
    def __init__(self, root, detector, model_name, out_path=None, raw_path=None):
        self.root = root
        self.detector = detector
        root.title(f"Pipe detector ({model_name}) - DepthAI")

        self.video = tk.Label(root, bg="black")
        self.video.pack()

        controls = tk.Frame(root)
        controls.pack(fill="x", padx=6, pady=4)
        if getattr(detector, "open_vocab", False):
            tk.Label(controls, text="Prompts (comma-separated):").pack(side="left")
            self.prompt_var = tk.StringVar(value=", ".join(detector.prompts))
            entry = tk.Entry(controls, textvariable=self.prompt_var, width=40)
            entry.pack(side="left", padx=4)
            entry.bind("<Return>", lambda e: self.apply_prompts())
            tk.Button(controls, text="Apply", command=self.apply_prompts).pack(side="left")
        tk.Label(controls, text="  conf").pack(side="left")
        self.conf = tk.Scale(controls, from_=0.001, to=0.9, resolution=0.001, orient="horizontal",
                             length=200, command=lambda v: setattr(detector, "conf", float(v)))
        self.conf.set(detector.conf)
        self.conf.pack(side="left")

        self.status = tk.Label(root, anchor="w", font=("monospace", 11), text="Connecting to camera...")
        self.status.pack(fill="x", padx=6, pady=(0, 4))

        self.worker = Worker(detector, out_path, raw_path)
        self.worker.start()
        self.photo = None
        root.bind("<Escape>", lambda e: self.close())
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    def apply_prompts(self):
        prompts = [p.strip() for p in self.prompt_var.get().split(",") if p.strip()]
        if prompts:
            self.detector.set_prompts(prompts)

    def tick(self):
        if self.worker.error is not None:
            self.status.config(text=f"Camera error: {self.worker.error}")
        else:
            with self.worker.lock:
                res = self.worker.result
            if res is not None:
                frame_bgr, dets, infer_ms, fps = res
                ok, png = cv2.imencode(".png", frame_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                if ok:
                    self.photo = tk.PhotoImage(data=png.tobytes())
                    self.video.config(image=self.photo)
                if not dets:
                    info = "no detection"
                else:
                    x0, y0, x1, y1, c, label = dets[0]
                    h, w = frame_bgr.shape[:2]
                    cx, cy = (x0 + x1) / w - 1, (y0 + y1) / h - 1      # [-1, 1], like PipeDetector
                    size = (x1 - x0) * (y1 - y0) / (w * h)
                    info = (f"{len(dets)} box(es) | best: {label} {c:.2f} "
                            f"center=({cx:+.2f},{cy:+.2f}) size={size:.3f}")
                self.status.config(text=f"{fps:5.1f} fps | infer {infer_ms:5.1f} ms | {info}")
        self.root.after(15, self.tick)

    def close(self):
        self.worker.running = False
        self.worker.join(timeout=5)     # let it finish the current frame and finalize the video
        self.root.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["yolo", "cnn"], default="yolo")
    ap.add_argument("--prompt", nargs="+", default=["white pole"], help="YOLO-World text prompts")
    ap.add_argument("--weights", default=YOLO_WEIGHTS,
                    help="YOLO weights: YOLO-World (prompted) or a fine-tuned .pt")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold")
    ap.add_argument("--out", default=None,
                    help="annotated video path (default recordings/pipe_<time>.mp4)")
    ap.add_argument("--raw", action="store_true",
                    help="also save the unannotated frames (<out>_raw.mp4), e.g. for labelling")
    ap.add_argument("--no-record", action="store_true", help="don't save any video")
    args = ap.parse_args()

    out_path = raw_path = None
    if not args.no_record:
        out_path = args.out or os.path.join(HERE, "recordings", time.strftime("pipe_%Y%m%d_%H%M%S.mp4"))
        if args.raw:
            raw_path = os.path.splitext(out_path)[0] + "_raw.mp4"
        print(f"Recording to {out_path}" + (f" and {raw_path}" if raw_path else ""))

    if args.model == "yolo":
        print(f"Loading {os.path.basename(args.weights)}...")
        detector = YoloDetector(args.prompt, args.conf, args.weights, args.imgsz)
    else:
        detector = CnnDetector()

    root = tk.Tk()
    App(root, detector, args.model, out_path, raw_path)
    root.mainloop()
