"""Runtime pipe detector: frame in, bounding box out. Needs only OpenCV and
numpy (the trained network is loaded from ONNX), so it runs anywhere
cv_processor.py does.

    from pipe_detector.detector import PipeDetector
    det = PipeDetector()
    box = det.detect(frame_rgb)      # None, or dict with box/score/center/size
"""
import os

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX = os.path.join(HERE, "pipe_net.onnx")
W, H = 256, 144          # network input, 16:9

MIN_BLOB = 6             # px at network size: smaller specks are noise
MIN_PIPE = 20            # px at network size: less than this in total is "no pipe"


def mask_to_box(mask, weight=None, min_blob=MIN_BLOB, min_pipe=MIN_PIPE, rel_mass=0.15):
    """Boolean pipe mask -> (x0, y0, x1, y1) around the pipe, or None.

    Blobs are ranked by mass (sum of `weight`, e.g. the probability map, over
    the blob; plain area if None). A blob joins the box only if it has at
    least rel_mass of the heaviest one's mass, so one stray speck cannot
    stretch the box across the frame, while a pipe split in two by an
    occluder still gets one box. The same rule makes the ground-truth boxes,
    so scores compare like with like."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    stats = stats[1:]
    mass = (stats[:, cv2.CC_STAT_AREA].astype(np.float64) if weight is None
            else np.bincount(labels.ravel(), weight.ravel(), minlength=n)[1:])
    ok = stats[:, cv2.CC_STAT_AREA] >= min_blob
    if not ok.any():
        return None
    keep = stats[ok & (mass >= rel_mass * mass[ok].max())]
    if keep[:, cv2.CC_STAT_AREA].sum() < min_pipe:
        return None
    x0, y0 = keep[:, 0].min(), keep[:, 1].min()
    x1 = (keep[:, 0] + keep[:, 2]).max()
    y1 = (keep[:, 1] + keep[:, 3]).max()
    return int(x0), int(y0), int(x1), int(y1)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PipeDetector:
    """PipeNet (ONNX) through OpenCV's DNN module, plus the mask -> box step."""

    def __init__(self, path=ONNX, threshold=0.5):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.threshold = threshold

    def prob(self, frame_rgb):
        """Per-pixel pipe probability at network size (H x W float32)."""
        small = cv2.resize(frame_rgb, (W, H), interpolation=cv2.INTER_AREA)
        self.net.setInput(small.astype(np.float32).transpose(2, 0, 1)[None])
        logits = self.net.forward()[0, 0]
        return 1 / (1 + np.exp(-logits))

    def detect(self, frame_rgb):
        """RGB frame (any size) -> None if no pipe, else a dict:
          box     (x0, y0, x1, y1) in frame pixels
          score   mean pipe probability inside the detected mask
          center  (cx, cy) of the box in [-1, 1], +x right, +y down
          size    box area as a fraction of the frame (grows as you approach)"""
        p = self.prob(frame_rgb)
        mask = p > self.threshold
        box = mask_to_box(mask, p)
        if box is None:
            return None
        fh, fw = frame_rgb.shape[:2]
        sx, sy = fw / W, fh / H
        x0, y0, x1, y1 = box
        return {
            "box": (int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)),
            "score": float(p[mask].mean()),
            "center": ((x0 + x1) / W - 1, (y0 + y1) / H - 1),
            "size": (x1 - x0) * (y1 - y0) / (W * H),
        }

    @staticmethod
    def draw(frame_rgb, det):
        out = frame_rgb.copy()
        if det is not None:
            x0, y0, x1, y1 = det["box"]
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(out, f"pipe {det['score']:.2f}", (x0, max(y0 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return out
