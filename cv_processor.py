import os
import numpy as np
os.environ.setdefault("QT_LOGGING_RULES", "default.warning=false")

import cv2

def show(name, frame, bgr=True):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, frame if bgr else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)

def hsv_process(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # show('Camera', frame, bgr=False)
    lower_white = (0, 0, 180)      # H: any, S: low, V: high
    upper_white = (180, 40, 255)

    mask = cv2.inRange(hsv_frame, lower_white, upper_white)
    return mask

def morphological_clean(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def find_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def largest_valid_contour(contours, min_area, min_aspect):
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        # if area < min_area:
        #     continue  # too small, likely noise

        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        # if aspect_ratio < min_aspect:
            # continue  # too blocky to be a pipe seen mostly side-on

        candidates.append(c)

    if len(candidates) == 0:
        best = None
    else:
        best = max(candidates, key=cv2.contourArea)
    return best

def process_pipe(frame):
    hsv_mask = hsv_process(frame)
    metamorphasis = morphological_clean(hsv_mask)
    contours = find_contours(metamorphasis)

    height, width, _ = frame.shape
    MIN_AREA = height * width * 0.0005
    MIN_ASPECT = 2.0

    best = largest_valid_contour(contours, MIN_AREA, MIN_ASPECT)
    overlay = frame.copy()
    if best is not None:
        cv2.drawContours(overlay, [best], -1, (0, 255, 0), 2)
        show('pipe', overlay, bgr=False)

    show("meta", metamorphasis)

def encode_jpeg(frame_rgb, quality=80):
    """Compress an RGB frame to JPEG bytes, for cheap accumulation in memory
    (~tens of KB vs. ~1.5MB for a raw 960x540 array) while a long capture is
    still running; decoded back only once, at save time."""
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

def save_video(jpeg_frames, output_path, fps=30.0):
    """Write a list of JPEG-encoded frames (see encode_jpeg) to a video file.
    Frames are decoded one at a time so at most one full-size array is ever
    live in memory, rather than the whole clip at once."""
    if not jpeg_frames:
        print(f"save_video: no frames to save, skipping {output_path}")
        return

    first = cv2.imdecode(np.frombuffer(jpeg_frames[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    height, width = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"save_video: failed to open writer for {output_path} "
              f"(codec/container unsupported by this OpenCV build)")
        return

    for jpeg_bytes in jpeg_frames:
        out.write(cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR))

    out.release()
    print(f"Saved video to {output_path} ({len(jpeg_frames)} frames @ {fps:.1f} fps)")