"""Turn a clean sim render into an underwater-looking frame, and its label into
one that only marks pipe the camera could actually see.

The core is the standard underwater image-formation model, per colour channel c:

    I_c = J_c * illum * t_c  +  B_c * (1 - t_c),     t_c = exp(-beta_c * dist)

J is the scene colour, t the transmission along each pixel's ray (red dies
first, so far things go blue/green), and B the veiling light scattered back
off the water itself. On top of that: an ROV spotlight with its fall-off,
shallow-water caustics, marine snow, sediment clouds, occluding fish/weed,
defocus/motion blur, white balance and gamma, sensor noise, vignetting and
JPEG artefacts (the sim streams JPEGs).

Alongside the image I, a second image D carries how strongly each pipe pixel
stands out from its surroundings; it goes through the same water, blur and
exposure, and pipe pixels whose final contrast is below the noise are marked
"ignore" in the label instead of "pipe".

All randomness comes from the passed-in Generator, so splits are reproducible.
"""
import numpy as np
import cv2

# (beta range per m, veiling colour range) per water type, RGB, linear light
WATER = {
    "clear":   ([0.35, 0.05, 0.03], [0.60, 0.12, 0.08], [0.01, 0.10, 0.18], [0.04, 0.22, 0.35]),
    "coastal": ([0.50, 0.12, 0.25], [0.90, 0.30, 0.50], [0.03, 0.15, 0.10], [0.08, 0.30, 0.20]),
    "turbid":  ([0.80, 0.60, 0.70], [1.60, 1.30, 1.60], [0.10, 0.10, 0.06], [0.25, 0.22, 0.14]),
    "pool":    ([0.15, 0.03, 0.02], [0.30, 0.06, 0.05], [0.10, 0.35, 0.45], [0.20, 0.50, 0.60]),
}
VISIBLE_CONTRAST = 0.04  # a pipe pixel must differ from its surroundings by 4% of full scale (~10 grey levels)
CLEAR_FRAMES = 0.15      # share of frames left without water, so a crisp frame is not out of distribution


def _smooth_noise(rng, h, w, scale):
    """Low-frequency noise in [-1, 1] with features ~scale px across."""
    gh, gw = max(2, int(h / scale) + 2), max(2, int(w / scale) + 2)
    n = cv2.resize(rng.standard_normal((gh, gw)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (np.abs(n).max() + 1e-6)


def geometric(rgb, dist, cov, rng, down=False):
    """Flips and zoom-crops, applied identically to image and labels."""
    if rng.random() < 0.5:
        rgb, dist, cov = rgb[:, ::-1], dist[:, ::-1], cov[:, ::-1]
    if down and rng.random() < 0.5:            # a down camera has no "up"
        rgb, dist, cov = rgb[::-1], dist[::-1], cov[::-1]
    if rng.random() < 0.3:
        h, w = cov.shape
        s = rng.uniform(0.6, 1.0)
        ch, cw = int(h * s), int(w * s)
        y, x = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
        crop = lambda a, interp: cv2.resize(np.ascontiguousarray(a[y:y + ch, x:x + cw]), (w, h), interpolation=interp)
        rgb, dist, cov = crop(rgb, cv2.INTER_LINEAR), crop(dist, cv2.INTER_LINEAR), crop(cov, cv2.INTER_LINEAR)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(dist), np.ascontiguousarray(cov)


def underwater(rgb, dist, cov, rng, frame_rng=None):
    """Returns (uint8 RGB frame, target in [0,1], per-pixel loss weight).
    Weight 0 marks pipe pixels the water/occluders made unseeable: neither
    "pipe" nor "not pipe" is a fair answer there.

    rng sets the water and light; frame_rng, if given, sets what changes from
    frame to frame (fish, snow, blur, exposure, noise, JPEG), so a sequence
    of frames can share one body of water."""
    h, w = dist.shape
    if rng.random() < CLEAR_FRAMES:
        # the sim's own crisp look (or very clear water): only the camera's noise and JPEG
        out = np.clip(rgb + rng.normal(0, rng.uniform(0, 4), rgb.shape), 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
        return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]), cov, np.ones_like(cov)
    J = (rgb.astype(np.float32) / 255) ** 2.2
    d = dist[..., None]

    # --- water body ---
    kind = rng.choice(list(WATER), p=[0.3, 0.3, 0.15, 0.25])
    blo, bhi, vlo, vhi = (np.array(a, np.float32) for a in WATER[kind])
    beta = rng.uniform(blo, bhi) * np.exp(rng.uniform(np.log(0.4), np.log(1.5)))
    veil = rng.uniform(vlo, vhi)
    t = np.exp(-beta * d)                                       # H x W x 3

    # --- lighting: ambient from the surface, plus maybe the ROV's own light ---
    ambient = np.exp(rng.uniform(np.log(0.08), np.log(1.3)))
    illum = np.full((h, w, 1), ambient, np.float32)
    if rng.random() < 0.4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.3, 0.7)
        spot = np.exp(-(((xx - cx) / (w * rng.uniform(0.25, 0.6))) ** 2 + ((yy - cy) / (h * rng.uniform(0.3, 0.8))) ** 2))
        power = rng.uniform(0.3, 2.0)
        # light reaches the surface through the water too, and spreads as 1/d^2
        illum = illum + (power * spot)[..., None] * t.mean(-1, keepdims=True) / np.maximum(d, 0.4) ** 2
    if kind in ("clear", "pool") and ambient > 0.4 and rng.random() < 0.5:
        n1, n2 = _smooth_noise(rng, h, w, rng.uniform(6, 20)), _smooth_noise(rng, h, w, rng.uniform(6, 20))
        caustic = np.exp(-np.abs(n1 - n2) * rng.uniform(8, 20))
        illum = illum * (1 + rng.uniform(0.3, 1.2) * caustic[..., None] * (dist[..., None] < 15))

    # sediment clouds make the veiling light patchy
    veil_map = veil * ambient * (1 + rng.uniform(0, 0.5) * _smooth_noise(rng, h, w, rng.uniform(20, 80))[..., None])
    I = J * illum * t + veil_map * (1 - t)
    # D tracks how much each pipe pixel differs from its surroundings once seen
    # through the water. It goes through every later step that I does, and the
    # label at the end only keeps pipe where that difference beats the noise.
    m = (cov > 0.5).astype(np.float32)
    around = cv2.GaussianBlur(J * (1 - m)[..., None], (0, 0), 3) / (cv2.GaussianBlur(1 - m, (0, 0), 3) + 1e-4)[..., None]
    D = np.abs(J - around) * illum * t

    if frame_rng is not None:
        rng = frame_rng

    # --- fish / weed passing in front of the camera ---
    if rng.random() < 0.15:
        occ = np.zeros((h, w), np.float32)
        for _ in range(rng.integers(1, 4)):
            centre = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            axes = (int(rng.uniform(4, w / 4)), int(rng.uniform(2, h / 6)))
            cv2.ellipse(occ, centre, axes, rng.uniform(0, 180), 0, 360, rng.uniform(0.6, 1.0), -1)
        occ = cv2.GaussianBlur(occ, (0, 0), 1.0)
        colour = veil * ambient * rng.uniform(0.2, 1.5)
        I = I * (1 - occ[..., None]) + colour * occ[..., None]
        D = D * (1 - occ[..., None])

    # --- marine snow: sharp near specks and big out-of-focus blobs ---
    if rng.random() < 0.7:
        snow = np.zeros((h, w), np.float32)
        for _ in range(int(rng.uniform(0, 1) ** 3 * 400)):
            r = float(np.exp(rng.uniform(np.log(0.5), np.log(2.5))))
            cv2.circle(snow, (int(rng.integers(0, w) * 4), int(rng.integers(0, h) * 4)), int(r * 4),
                       float(rng.uniform(0.1, 0.7)), -1, cv2.LINE_AA, 2)
        for _ in range(int(rng.integers(0, 8))):
            cv2.circle(snow, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(4, 12)),
                       float(rng.uniform(0.05, 0.2)), -1, cv2.LINE_AA)
        snow = cv2.GaussianBlur(snow, (0, 0), 0.6)
        flake = np.clip(veil * ambient * 2 + 0.1 * illum.mean(), 0, 1)
        I = I * (1 - snow[..., None]) + flake * snow[..., None]
        D = D * (1 - snow[..., None])

    # --- optics: a thin far pipe loses most of its contrast to blur ---
    k = rng.random()
    if k < 0.45:
        s = rng.uniform(0.3, 1.4)
        I, D = cv2.GaussianBlur(I, (0, 0), s), cv2.GaussianBlur(D, (0, 0), s)
    elif k < 0.6:
        n = int(rng.integers(3, 9))
        kern = np.zeros((n, n), np.float32)
        kern[n // 2] = 1
        kern = cv2.warpAffine(kern, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), rng.uniform(0, 180), 1), (n, n))
        kern /= max(kern.sum(), 1e-6)
        I, D = cv2.filter2D(I, -1, kern), cv2.filter2D(D, -1, kern)

    # --- camera: auto exposure, white balance, vignette, gamma, noise ---
    gain = rng.uniform(0.8, 1.25, 3).astype(np.float32) / max(np.percentile(I, 99) * rng.uniform(0.8, 1.6), 0.02)
    if rng.random() < 0.4:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        gain = gain * (1 - rng.uniform(0.1, 0.5) * r2 / 2)[..., None]
    I, D = I * gain, D * gain
    gamma = 1 / 2.2 * rng.uniform(0.8, 1.25)
    sigma = rng.uniform(0, 0.03) + 0.04 * (ambient < 0.2) * rng.random()

    # --- what the label may still claim: contrast in the output image that
    # beats both the eye's threshold and the sensor noise ---
    Ic = np.clip(I, 0, 1)
    contrast = (np.minimum(Ic + D, 1) ** gamma - Ic ** gamma).max(-1)
    visible = contrast > max(VISIBLE_CONTRAST, 2.5 * sigma)
    target = cov * visible
    weight = np.where((cov > 0.1) & ~visible, 0.0, 1.0).astype(np.float32)

    I = Ic ** gamma + rng.normal(0, sigma, I.shape).astype(np.float32)
    out = (np.clip(I, 0, 1) * 255).astype(np.uint8)
    if rng.random() < 0.8:
        ok, buf = cv2.imencode(".jpg", out[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(25, 96))])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(out), target.astype(np.float32), weight
