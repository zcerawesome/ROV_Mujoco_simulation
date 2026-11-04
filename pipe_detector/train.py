"""Train PipeNet on the generated frames, pick the epoch that does best on the
validation split, export it to ONNX and score it on the untouched test splits.

    python -m pipe_detector.generate                 # once, renders data/
    python -m pipe_detector.train                    # trains, writes pipe_net.onnx
    python -m pipe_detector.train --resume           # carry on after a stop
    python -m pipe_detector.train --init runs/best.pt --lr 1e-3 --epochs 12   # fine-tune
    python -m pipe_detector.train --eval             # re-score the exported model only

Water effects are drawn fresh every epoch for training; validation and test
frames get a fixed seed each, so their scores are comparable between epochs.
"""
import os
import json
import time
import argparse

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pipe_detector import underwater as uw
from pipe_detector.generate import DATA, decode_dist
from pipe_detector.model import PipeNet
from pipe_detector.detector import PipeDetector, mask_to_box, box_iou, ONNX, W, H

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


class Frames(Dataset):
    """fixed_seed=None: fresh random water every time (training).
    fixed_seed=k: sample i always gets the same water (evaluation)."""

    def __init__(self, name, effects=True, fixed_seed=None, limit=None, root=DATA):
        self.dir = os.path.join(root, name)
        self.n = min(limit or 1 << 30, len(np.load(os.path.join(self.dir, "down.npy"))))
        self.effects, self.fixed_seed = effects, fixed_seed
        self.rgb = None

    def _open(self):
        # memory-mapped, and opened inside each loader worker: nothing big is
        # copied into (or pickled between) processes
        load = lambda k: np.load(os.path.join(self.dir, f"{k}.npy"), mmap_mode="r")
        self.rgb, self.dist, self.cov, self.down = load("rgb"), load("dist"), load("cov"), load("down")

    def __getstate__(self):
        return {**self.__dict__, "rgb": None, "dist": None, "cov": None, "down": None}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.rgb is None:
            self._open()
        rng = np.random.default_rng(None if self.fixed_seed is None else (self.fixed_seed, i))
        rgb, dist, cov = self.rgb[i], decode_dist(self.dist[i]), self.cov[i].astype(np.float32) / 255
        if self.effects:
            rgb, dist, cov = uw.geometric(rgb, dist, cov, rng, bool(self.down[i]))
            img, target, weight = uw.underwater(rgb, dist, cov, rng)
        else:
            img, target, weight = rgb, cov, np.ones_like(cov)
        return (torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)),
                torch.from_numpy(target[None]), torch.from_numpy(weight[None]))


def _worker_init(_):
    cv2.setNumThreads(1)


def loader(ds, shuffle, batch=32, workers=10):
    # spawn, not fork: forking after torch/OpenCV have started their thread
    # pools can deadlock the workers (it did, silently, on the first run)
    return DataLoader(ds, batch, shuffle=shuffle, num_workers=workers, worker_init_fn=_worker_init,
                      persistent_workers=True, drop_last=shuffle, multiprocessing_context="spawn")


def loss_fn(logits, target, weight):
    """Masked BCE + soft Dice. Dice keeps a thin far-off pipe (a few % of the
    pixels) from being drowned out by the background."""
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).sum() / weight.sum()
    p = torch.sigmoid(logits) * weight
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


# ---------- scoring ----------
def score(probs, targets, weights):
    """Frame-level detection metrics.

    A frame's true box comes from its visible pipe pixels via mask_to_box, the
    same rule the detector uses. "correct" = no pipe and none reported, or a
    pipe and a box with IoU >= 0.5. Frames where only unseeable/sub-threshold
    slivers of pipe exist are 'ambiguous' and left out of accuracy."""
    res = dict(pos=0, neg=0, amb=0, tp=0, fp=0, fn=0, tn=0, loc_miss=0, ious=[], centre_err=[],
               pix_inter=0, pix_union=0)
    for p, t, w in zip(probs, targets, weights):
        gt_mask, pred_mask = t > 0.5, p > 0.5
        gt, pred = mask_to_box(gt_mask, t), mask_to_box(pred_mask, p)
        res["pix_inter"] += int((gt_mask & pred_mask & (w > 0)).sum())
        res["pix_union"] += int(((gt_mask | pred_mask) & (w > 0)).sum())
        if gt is None:
            if gt_mask.any() or (w == 0).any():
                res["amb"] += 1
                continue
            res["neg"] += 1
            res["fp" if pred else "tn"] += 1
            continue
        res["pos"] += 1
        if pred is None:
            res["fn"] += 1
            res["ious"].append(0.0)
            continue
        iou = box_iou(gt, pred)
        res["ious"].append(iou)
        if iou >= 0.5:
            res["tp"] += 1
            centre = lambda b: np.array([(b[0] + b[2]) / W, (b[1] + b[3]) / H])
            res["centre_err"].append(np.abs(centre(gt) - centre(pred)).max())
        else:
            res["loc_miss"] += 1
    n = res["pos"] + res["neg"]
    ious = np.array(res["ious"])
    return {
        "frames": n, "ambiguous": res["amb"], "with_pipe": res["pos"],
        "accuracy": (res["tp"] + res["tn"]) / max(n, 1),
        "recall@0.5": res["tp"] / max(res["pos"], 1),
        "false_alarm_rate": res["fp"] / max(res["neg"], 1),
        "mean_box_iou": float(ious.mean()) if len(ious) else 0.0,
        "median_box_iou": float(np.median(ious)) if len(ious) else 0.0,
        "pixel_iou": res["pix_inter"] / max(res["pix_union"], 1),
        "centre_err_p90": float(np.percentile(res["centre_err"], 90)) if res["centre_err"] else 0.0,
    }


@torch.no_grad()
def evaluate_torch(model, dl):
    model.eval()
    probs, targets, weights, losses = [], [], [], []
    for x, t, w in dl:
        logits = model(x)
        losses.append(loss_fn(logits, t, w).item())
        probs.append(torch.sigmoid(logits)[:, 0].numpy())
        targets.append(t[:, 0].numpy())
        weights.append(w[:, 0].numpy())
    m = score(np.concatenate(probs), np.concatenate(targets), np.concatenate(weights))
    m["loss"] = float(np.mean(losses))
    return m


def evaluate_onnx(ds, path=ONNX):
    """Score the exported model through the runtime detector (OpenCV DNN):
    exactly what the ROV would run."""
    det = PipeDetector(path)
    probs, targets, weights = [], [], []
    for i in range(len(ds)):
        x, t, w = ds[i]
        probs.append(det.prob(x.numpy().transpose(1, 2, 0).astype(np.uint8)))
        targets.append(t[0].numpy())
        weights.append(w[0].numpy())
    return score(probs, targets, weights)


def export(model, path=ONNX):
    model.eval()
    torch.onnx.export(model, torch.zeros(1, 3, H, W), path, dynamo=False, opset_version=13,
                      input_names=["rgb"], output_names=["logits"])
    print(f"exported {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def fmt(m):
    return (f"acc {m['accuracy']:.3f}  rec {m['recall@0.5']:.3f}  fa {m['false_alarm_rate']:.3f}  "
            f"mIoU {m['mean_box_iou']:.3f}  pixIoU {m['pixel_iou']:.3f}")


def train(args):
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    os.makedirs(RUNS, exist_ok=True)
    train_dl = loader(Frames("train", root=args.data), True, args.batch, args.workers)
    val_dl = loader(Frames("val", fixed_seed=1, root=args.data), False, 64, 4)
    # the same scoring on a fixed slice of the training set: if this pulls
    # away from val, the network is memorising rather than generalising
    train_probe_dl = loader(Frames("train", fixed_seed=1, limit=2000, root=args.data), False, 64, 4)

    model = PipeNet(dropout=args.dropout)
    if args.init:   # fine-tune from an earlier run's weights
        model.load_state_dict(torch.load(args.init))
    print(f"PipeNet: {sum(p.numel() for p in model.parameters()) / 1e3:.0f}k parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15)

    history, best, stale, start = [], -1.0, 0, 0
    last = os.path.join(RUNS, "last.pt")
    if args.resume and os.path.exists(last):
        ck = torch.load(last, weights_only=False)
        for obj, key in ((model, "model"), (opt, "opt"), (sched, "sched")):
            obj.load_state_dict(ck[key])
        history, best, stale, start = ck["history"], ck["best"], ck["stale"], ck["epoch"] + 1
        print(f"resuming after epoch {ck['epoch']}")
    for epoch in range(start, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, t, w in train_dl:
            loss = loss_fn(model(x), t, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        val, probe = evaluate_torch(model, val_dl), evaluate_torch(model, train_probe_dl)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val, "train_probe": probe})
        val_score = val["accuracy"] + 0.1 * val["mean_box_iou"]     # box quality breaks ties
        print(f"ep {epoch:2d} {time.time() - t0:4.0f}s  loss {np.mean(losses):.3f}/{val['loss']:.3f}  "
              f"val {fmt(val)}  | train {probe['accuracy']:.3f}/{probe['mean_box_iou']:.3f}", flush=True)
        if val_score > best:
            best, stale = val_score, 0
            torch.save(model.state_dict(), os.path.join(RUNS, "best.pt"))
        else:
            stale += 1
        with open(os.path.join(RUNS, "history.json"), "w") as f:
            json.dump(history, f, indent=1)
        # everything needed to carry on with --resume after a stop
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "history": history, "best": best, "stale": stale}, last)
        if stale >= args.patience:
            print("early stop: val has not improved for", args.patience, "epochs")
            break

    model.load_state_dict(torch.load(os.path.join(RUNS, "best.pt")))
    export(model)


def final_eval(root=DATA):
    """Score the exported model on splits training never saw."""
    results = {
        "test (underwater)": evaluate_onnx(Frames("test", fixed_seed=7, root=root)),
        "test (no water effects)": evaluate_onnx(Frames("test", effects=False, root=root)),
        "clean sim (models/rov.xml)": evaluate_onnx(Frames("clean", effects=False, root=root)),
        "clean sim + underwater": evaluate_onnx(Frames("clean", fixed_seed=7, root=root)),
    }
    for k, m in results.items():
        print(f"{k:30s} {fmt(m)}  ctr90 {m['centre_err_p90']:.3f}  ({m['frames']} frames, {m['ambiguous']} ambiguous)")
    with open(os.path.join(RUNS, "test_results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--data", default=DATA, help="folder generate.py wrote")
    ap.add_argument("--resume", action="store_true", help="continue from runs/last.pt")
    ap.add_argument("--init", help="start from these weights (a .pt state dict)")
    ap.add_argument("--eval", action="store_true", help="only score the exported model")
    a = ap.parse_args()
    if not a.eval:
        train(a)
    final_eval(a.data)
