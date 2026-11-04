"""Render a labelled pipe dataset from the MuJoCo sim.

Each sample is a clean render from the ROV's front or down camera with the
scene randomised (pipe pose, which pipe pieces exist, colours, seabed,
lighting, distractors, camera FOV, ROV pose), plus the per-pixel ray distance
and the pipe's pixel coverage taken from MuJoCo's segmentation render. The
underwater look is NOT baked in here; underwater.py adds it at training time
from the stored distances, so every epoch sees different water.

    python -m pipe_detector.generate            # run from the repo root
"""
import os
import argparse
from multiprocessing import Pool

import numpy as np
import mujoco as mj

from simulation import MuJoCoBase

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "scene.xml")
CLEAN_SCENE = os.path.join(HERE, "..", "models", "rov.xml")
DATA = os.path.join(HERE, "data")

W, H = 256, 144          # stored / model input size (16:9, same as the sim cameras)
SUPERSAMPLE = 2          # render at 2x the stored size, then average down (anti-aliasing)
HIDDEN = 5               # geom group used to hide pieces (disabled in the scene option)
FLOOR_Z = -3.0
MAX_DIST = 30.0          # background (open water) distance, metres

# log-encode distances into uint8: 0.05 m .. MAX_DIST, ~2.5% per step
_LOG_LO, _LOG_HI = np.log(0.05), np.log(MAX_DIST)


def encode_dist(d):
    v = (np.log(np.clip(d, 0.05, MAX_DIST)) - _LOG_LO) / (_LOG_HI - _LOG_LO)
    return np.round(v * 255).astype(np.uint8)


def decode_dist(u8):
    return np.exp(_LOG_LO + u8.astype(np.float32) / 255 * (_LOG_HI - _LOG_LO))


def quat(yaw, pitch, roll):
    """Body quaternion for Rz(yaw) Ry(pitch) Rx(roll). +pitch is nose-down."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    return np.array([cy * cp * cr + sy * sp * sr,
                     cy * cp * sr - sy * sp * cr,
                     cy * sp * cr + sy * cp * sr,
                     sy * cp * cr - cy * sp * sr])


class SceneRandomizer:
    """Owns a MuJoCoBase and renders randomised, labelled frames from it."""

    def __init__(self, xml=SCENE, randomize=True):
        self.sim = MuJoCoBase(xml, width=W * SUPERSAMPLE, height=H * SUPERSAMPLE)
        self.sim.opt.geomgroup[HIDDEN] = 0
        m = self.sim.model
        self.m, self.d, self.r = m, self.sim.data, self.sim.renderer
        self.randomize = randomize

        names = [m.geom(i).name for i in range(m.ngeom)]
        self.pipe_ids = np.array([i for i, n in enumerate(names) if n.startswith("pvc_")])
        self.pipe_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "pvc_pipe_run")
        # pipe pieces as segments in the pipe body's frame, for aiming at the pipe
        self.segments = {n: (m.geom(n).pos, m.geom(n).quat, m.geom(n).size)
                         for n in ("pvc_run_x", "pvc_run_y", "pvc_riser")}
        # optional piece sets: (geoms hidden) -> the pipe's shape varies, not just its pose
        self.variants = [
            [],                                                          # full run
            ["pvc_riser", "pvc_flange"],                                 # L only
            ["pvc_run_y", "pvc_riser", "pvc_elbow_2", "pvc_coupling_3", "pvc_flange"],  # straight
            ["pvc_run_x", "pvc_coupling_1", "pvc_coupling_2", "pvc_end_cap"],          # short L + riser
        ]
        if randomize:
            self.distract = [i for i, n in enumerate(names)
                             if n.startswith(("rock_", "block_", "weed_"))]
            self.floor_mats = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, n)
                               for n in ("seabed_mat", "sand_mat", "mud_mat", "silt_mat")]
            self.mat = {n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_MATERIAL, n)
                        for n in ("pvc_mat", "pvc_fitting_mat", "rock_mat", "debris_mat", "weed_mat")}
            self.mat_rgba0 = m.mat_rgba.copy()
            # a real bed doesn't end, so mostly scene.xml's 60 m "bed" stands in
            # for rov.xml's 10 m seabed. Some frames keep the seabed so its edge
            # and far-off tile moire, both in the sim, are learned as not-pipe.
            self.beds = (names.index("seabed"), names.index("bed"))
            m.geom_group[self.beds[0]] = HIDDEN
            self.texrepeat0 = m.mat_texrepeat.copy()
        self.groups0 = m.geom_group.copy()      # visibility every sample starts from
        self.cams = {c: mj.mj_name2id(m, mj.mjtObj.mjOBJ_CAMERA, c) for c in ("front", "down")}

    # ---------- randomisation ----------
    def _pipe_points(self, rng, n=1):
        """Random world points on the visible pipe axis."""
        visible = [k for k in self.segments if self.m.geom(k).group != HIDDEN]
        bpos = self.d.xpos[self.pipe_body]
        bmat = self.d.xmat[self.pipe_body].reshape(3, 3)
        pts = []
        for _ in range(n):
            pos, q, size = self.segments[visible[rng.integers(len(visible))]]
            mat = np.zeros(9)
            mj.mju_quat2Mat(mat, q)
            axis = mat.reshape(3, 3)[:, 2]
            local = pos + axis * size[1] * rng.uniform(-1, 1)
            pts.append(bpos + bmat @ local)
        return np.array(pts)

    def _randomize_scene(self, rng):
        m = self.m
        m.geom_group[:] = self.groups0
        # pipe pose and shape
        variant = self.variants[rng.choice(4, p=[0.4, 0.25, 0.2, 0.15])]
        for n in variant:
            m.geom(n).group = HIDDEN
        m.body_pos[self.pipe_body] = [rng.uniform(-1, 1), rng.uniform(-1, 1), 0]
        m.body_quat[self.pipe_body] = quat(rng.uniform(0, 2 * np.pi), 0, 0)

        # pipe colour: mostly weathered white/grey PVC, some fouled, a few odd colours
        k = rng.random()
        if k < 0.55:
            base = np.full(3, rng.uniform(0.6, 0.98)) + rng.normal(0, 0.03, 3)
        elif k < 0.85:   # algae / sediment fouling
            foul = rng.choice([[0.45, 0.55, 0.30], [0.55, 0.48, 0.35], [0.35, 0.45, 0.35]])
            base = np.array(foul) * rng.uniform(0.7, 1.3)
        else:            # grey, black, orange, blue, yellow pipes
            base = np.array([[0.4, 0.4, 0.4], [0.12, 0.12, 0.12], [0.9, 0.45, 0.1],
                             [0.2, 0.35, 0.7], [0.85, 0.8, 0.2]][rng.integers(5)])
        base = np.clip(base, 0, 1)
        m.mat_rgba[self.mat["pvc_mat"], :3] = base
        m.mat_rgba[self.mat["pvc_fitting_mat"], :3] = np.clip(base * rng.uniform(0.75, 1.05), 0, 1)
        m.mat_specular[self.mat["pvc_mat"]] = rng.uniform(0.0, 0.5)

        # seabed: the sim's own 10 m one, or the 60 m bed (same tile size); pool tile, sand, mud or silt, tinted
        small = rng.random() < 0.25
        self.floor = self.beds[0] if small else self.beds[1]
        m.geom_group[self.beds[0]] = 0 if small else HIDDEN
        m.geom_group[self.beds[1]] = HIDDEN if small else 0
        m.mat_texrepeat[self.floor_mats] = self.texrepeat0[self.floor_mats] * (1 if small else 6)
        m.geom_matid[self.floor] = self.floor_mats[rng.integers(4)]
        for mid in self.floor_mats:
            m.mat_rgba[mid, :3] = np.clip(self.mat_rgba0[mid, :3] * rng.uniform(0.6, 1.3)
                                          * rng.uniform(0.85, 1.15, 3), 0, 1)

        # distractors: each one present w.p. 0.6, scattered over the bed
        for gid in self.distract:
            g = m.geom(gid)
            if rng.random() > 0.6:
                g.group = HIDDEN
                continue
            x, y = rng.uniform(-4, 4, 2)
            if g.name.startswith("rock_"):
                g.size = rng.uniform([0.05, 0.05, 0.03], [0.4, 0.3, 0.2])
                g.pos = [x, y, FLOOR_Z + g.size[2] * 0.5]
                g.quat = quat(rng.uniform(0, 6.3), rng.uniform(-0.3, 0.3), 0)
            elif g.name.startswith("block_"):
                g.size = rng.uniform([0.05, 0.05, 0.03], [0.4, 0.25, 0.2])
                g.pos = [x, y, FLOOR_Z + g.size[2]]
                g.quat = quat(rng.uniform(0, 6.3), 0, 0)
            else:          # weed/rope: standing or lying on the bed
                g.size = [rng.uniform(0.005, 0.03), rng.uniform(0.1, 0.6), 0]
                if rng.random() < 0.6:
                    g.pos = [x, y, FLOOR_Z + g.size[1]]
                    g.quat = quat(0, rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4))
                else:
                    g.pos = [x, y, FLOOR_Z + g.size[0]]
                    g.quat = quat(rng.uniform(0, 6.3), np.pi / 2, 0)
        m.mat_rgba[self.mat["rock_mat"], :3] = rng.uniform(0.15, 0.6) * rng.uniform(0.85, 1.15, 3)
        # debris is sometimes white like PVC - a hard negative
        m.mat_rgba[self.mat["debris_mat"], :3] = (np.full(3, rng.uniform(0.7, 0.95)) if rng.random() < 0.3
                                                  else rng.uniform(0.2, 0.7) * rng.uniform(0.85, 1.15, 3))
        m.mat_rgba[self.mat["weed_mat"], :3] = rng.uniform([0.1, 0.2, 0.05], [0.4, 0.5, 0.25])

        # lighting
        m.light_dir[0] = [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1]
        m.light_diffuse[0] = rng.uniform(0.2, 0.9)
        m.light_dir[1] = [rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), -1]
        m.light_diffuse[1] = rng.uniform(0.0, 0.4)
        m.vis.headlight.ambient[:] = rng.uniform(0.15, 0.5) * rng.uniform(0.9, 1.1, 3)
        m.vis.headlight.diffuse[:] = rng.uniform(0.0, 0.4)

        # camera lens
        m.cam_fovy[self.cams["front"]] = rng.uniform(55, 95)
        m.cam_fovy[self.cams["down"]] = rng.uniform(50, 85)

    def _place_rov(self, rng, cam):
        """Pose the ROV so that the chosen camera mostly sees the pipe."""
        if rng.random() < 0.8:
            target = self._pipe_points(rng)[0]
        else:
            target = np.array([*rng.uniform(-3, 3, 2), FLOOR_Z])
        if cam == "front":
            h = rng.uniform(0.25, 2.5)                     # camera height above the bed
            r = rng.uniform(0.4, 6.0)
            bearing = rng.uniform(0, 2 * np.pi)
            pos = target[:2] - r * np.array([np.cos(bearing), np.sin(bearing)])
            yaw = bearing + (rng.normal(0, 0.35) if rng.random() < 0.85 else rng.uniform(-np.pi, np.pi))
            # pitch toward the target (clipped), as an ROV holding depth roughly would
            pitch = np.clip(np.arctan2(h, r) + rng.normal(0, 0.15), -0.2, 0.6)
            roll = rng.normal(0, 0.08)
        else:
            h = rng.uniform(0.35, 2.8)
            pos = target[:2] + rng.normal(0, 0.6, 2)
            yaw = rng.uniform(0, 2 * np.pi)
            pitch, roll = rng.normal(0, 0.1), rng.normal(0, 0.1)
        self.d.qpos[:3] = [pos[0], pos[1], FLOOR_Z + h]
        self.d.qpos[3:7] = quat(yaw, pitch, roll)
        self.d.qvel[:] = 0

    # ---------- rendering ----------
    def render(self, cam):
        """(rgb uint8 HxWx3, ray distance float32 HxW, pipe coverage float32 HxW in [0,1])
        at the stored size, for the current state."""
        r, opt = self.r, self.sim.opt
        rgb = self.sim._frame(cam)
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=cam, scene_option=opt)
        z = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        r.update_scene(self.d, camera=cam, scene_option=opt)
        seg = r.render()
        r.disable_segmentation_rendering()

        pipe = np.isin(seg[..., 0], self.pipe_ids) & (seg[..., 1] == int(mj.mjtObj.mjOBJ_GEOM))
        # planar depth -> distance along each pixel's ray (what water attenuates over)
        h, w = z.shape
        f = (h / 2) / np.tan(np.deg2rad(self.m.cam_fovy[self.cams[cam]]) / 2)
        u = (np.arange(w) - w / 2 + 0.5) / f
        v = (np.arange(h) - h / 2 + 0.5) / f
        ray = z * np.sqrt(1 + u[None, :] ** 2 + v[:, None] ** 2)
        ray[seg[..., 0] < 0] = MAX_DIST                    # open water

        pool = lambda a: a.reshape(H, SUPERSAMPLE, W, SUPERSAMPLE, *a.shape[2:]).mean(axis=(1, 3))
        return (pool(rgb.astype(np.float32)).round().astype(np.uint8),
                pool(np.minimum(ray, MAX_DIST)).astype(np.float32),
                pool(pipe.astype(np.float32)))

    def sample(self, rng, cam=None):
        """One labelled frame: (rgb, ray distance, pipe coverage, is_down_camera)."""
        if cam is None:
            cam = "front" if rng.random() < 0.6 else "down"
        if self.randomize:
            self._randomize_scene(rng)
        mj.mj_forward(self.m, self.d)                      # pipe body pose -> world
        self._place_rov(rng, cam)
        mj.mj_forward(self.m, self.d)
        return (*self.render(cam), cam == "down")


# ---------- dataset writing ----------
def _render_chunk(args):
    """Pool worker: its own sim (one EGL context per process) renders n frames."""
    xml, randomize, seed, n = args
    rng = np.random.default_rng(seed)
    gen = SceneRandomizer(xml, randomize)
    out = [gen.sample(rng) for _ in range(n)]
    gen.sim.close()
    rgb, dist, cov, down = zip(*out)
    return np.stack(rgb), encode_dist(np.stack(dist)), (np.stack(cov) * 255).round().astype(np.uint8), np.array(down)


def build(name, n, seed, xml=SCENE, randomize=True, workers=12, chunk=100, out=DATA):
    jobs = [(xml, randomize, seed * 100003 + i, min(chunk, n - i * chunk))
            for i in range((n + chunk - 1) // chunk)]
    with Pool(workers) as pool:
        parts = []
        for k, p in enumerate(pool.imap(_render_chunk, jobs)):
            parts.append(p)
            print(f"\r{name}: {min((k + 1) * chunk, n)}/{n}", end="", flush=True)
    print()
    rgb, dist, cov, down = (np.concatenate(a) for a in zip(*parts))
    # one .npy per array so training can memory-map them instead of loading 4 GB
    os.makedirs(os.path.join(out, name), exist_ok=True)
    for key, arr in dict(rgb=rgb, dist=dist, cov=cov, down=down).items():
        np.save(os.path.join(out, name, f"{key}.npy"), arr)
    print(f"saved {name}: {len(rgb)} samples, {(cov.reshape(len(cov), -1) > 127).any(1).mean():.0%} with pipe")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--val", type=int, default=2000)
    ap.add_argument("--test", type=int, default=2000)
    ap.add_argument("--clean", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=DATA)
    a = ap.parse_args()
    # different seeds per split: no frame is shared between them
    build("train", a.train, 1, workers=a.workers, out=a.out)
    build("val", a.val, 2, workers=a.workers, out=a.out)
    build("test", a.test, 3, workers=a.workers, out=a.out)
    # the unmodified sim scene (models/rov.xml), no randomisation: what the
    # current simulation actually shows the ROV
    build("clean", a.clean, 4, xml=CLEAN_SCENE, randomize=False, workers=a.workers, out=a.out)
