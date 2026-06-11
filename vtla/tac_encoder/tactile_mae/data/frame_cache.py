"""Decode-once frame cache for tactile MAE.

Random-access MP4 decoding (seek-to-keyframe + decode) costs ~150 ms/frame and
starves the GPU. This module dumps the *kept* (contact-filtered) frames of a
LeRobot dataset to a dense ``uint8`` memmap once, so training-time ``__getitem__``
becomes a zero-decode ``memmap[row]`` read.

Layout (per dataset, under ``<dataset_root>/<ds_id>/frames_cache/<signature>/``):
    <camera_key>.npy        # [K, S, S, 3] uint8, HWC, pre-resized to image_size
    <camera_key>.idx.npy    # [K] int64, the absolute frame index of each row
    manifest.json           # signature + per-camera counts + complete flag

The kept set is decided here (deterministically, from the contact-std cache) and
frozen into ``.idx.npy``; the dataset reads it back instead of re-rolling the
non-contact subsample. Because the kept set depends on the filter params, the
signature folder encodes them, so changing a param builds a fresh cache.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset

from vtla.datasets.lerobot_dataset import LeRobotDataset

CACHE_VERSION = 1


def cache_signature(contact_filter, contact_std_threshold, noncontact_keep_ratio,
                    contact_seed, image_size):
    """Folder name capturing every param the kept-set / pixels depend on."""
    if not contact_filter:
        return f"all_{image_size}_v{CACHE_VERSION}"
    return (f"thr{contact_std_threshold:g}_keep{noncontact_keep_ratio:g}"
            f"_seed{contact_seed}_{image_size}_v{CACHE_VERSION}")


def cache_dir(dataset_root, ds_id, signature):
    return os.path.join(dataset_root, ds_id, "frames_cache", signature)


def _kept_indices(std_cache, camera_keys, contact_filter, thr, keep_ratio, seed):
    """{camera: sorted np.ndarray of kept absolute frame indices}.

    Deterministic: contact frames (std>thr) are always kept; non-contact frames
    are kept with prob keep_ratio using a per-run-seeded RNG (rolled per camera
    in the given order, so the choice is reproducible).
    """
    rng = np.random.RandomState(seed)
    out = {}
    for cam in camera_keys:
        s = np.asarray(std_cache[cam])
        n = len(s)
        if not contact_filter:
            out[cam] = np.arange(n, dtype=np.int64)
            continue
        contact = s > thr
        roll = rng.rand(n) < keep_ratio
        keep = contact | (~contact & roll)
        out[cam] = np.nonzero(keep)[0].astype(np.int64)
    return out


def summarize_contact_filter(std_cache, camera_keys, thr, keep_ratio, seed):
    """Return per-camera counts using the same sampling rule as frame-cache filtering."""
    rng = np.random.RandomState(seed)
    per_camera = {}
    total = {
        "frames": 0,
        "contact_kept": 0,
        "noncontact_kept": 0,
        "noncontact_dropped": 0,
        "kept": 0,
    }
    for cam in camera_keys:
        scores = np.asarray(std_cache[cam])
        contact = scores > thr
        noncontact = ~contact
        roll = rng.rand(len(scores)) < keep_ratio
        kept_noncontact = noncontact & roll
        dropped_noncontact = noncontact & ~roll
        counts = {
            "frames": int(len(scores)),
            "contact_kept": int(contact.sum()),
            "noncontact_kept": int(kept_noncontact.sum()),
            "noncontact_dropped": int(dropped_noncontact.sum()),
        }
        counts["kept"] = counts["contact_kept"] + counts["noncontact_kept"]
        per_camera[cam] = counts
        for key, value in counts.items():
            total[key] += value
    return {"per_camera": per_camera, "total": total}


def print_contact_filter_summary(dataset_id, summary, thr, keep_ratio):
    total = summary["total"]
    dropped = total["noncontact_dropped"]
    frames = total["frames"]
    dropped_pct = 100.0 * dropped / frames if frames else 0.0
    print(
        f"[contact_filter] {dataset_id}: kept {total['kept']}/{frames} tactile images; "
        f"filtered non-contact={dropped} ({dropped_pct:.2f}%) | "
        f"contact kept={total['contact_kept']}, "
        f"non-contact kept={total['noncontact_kept']} | "
        f"thr={thr}, keep_ratio={keep_ratio}"
    )
    for cam, counts in summary["per_camera"].items():
        cam_frames = counts["frames"]
        cam_dropped = counts["noncontact_dropped"]
        cam_pct = 100.0 * cam_dropped / cam_frames if cam_frames else 0.0
        print(
            f"[contact_filter]   {cam}: kept {counts['kept']}/{cam_frames}; "
            f"filtered non-contact={cam_dropped} ({cam_pct:.2f}%)"
        )


class _DecodeProbe(Dataset):
    """Decode each frame once and emit it for every camera that kept it.

    ``ds[base]`` decodes *all* requested finger cameras in one shot, so we iterate
    the union of kept frames and yield ``(abs_idx, {cam: uint8 HWC})`` only for the
    cameras that actually kept that frame — avoiding a second decode pass per cam.
    """

    def __init__(self, ds, camera_keys, abs2base, union_abs, kept_sets, image_size):
        self.ds = ds
        self.camera_keys = camera_keys
        self.abs2base = abs2base
        self.union_abs = union_abs
        self.kept_sets = kept_sets  # {cam: set(abs idx)}
        self.image_size = image_size

    def __len__(self):
        return len(self.union_abs)

    def __getitem__(self, i):
        a = int(self.union_abs[i])
        item = self.ds[self.abs2base[a]]
        out = {}
        for cam in self.camera_keys:
            if a not in self.kept_sets[cam]:
                continue
            img = item[cam]  # CHW float [0,1], native res
            img = F.interpolate(img[None], size=(self.image_size, self.image_size),
                                mode="bilinear", align_corners=False)[0]
            out[cam] = (img.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).contiguous()
        return a, out


def _manifest_ok(cdir, camera_keys, signature):
    fp = os.path.join(cdir, "manifest.json")
    if not os.path.exists(fp):
        return False
    try:
        with open(fp, encoding="utf-8") as f:
            man = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return (man.get("complete") and man.get("signature") == signature
            and all(c in man.get("counts", {}) for c in camera_keys))


def build_frame_cache(dataset_root, ds_id, camera_keys, image_size=224,
                      contact_filter=True, contact_std_threshold=0.5,
                      noncontact_keep_ratio=0.05, contact_seed=0,
                      tolerance_s=0.1, video_backend="pyav", num_workers=16,
                      force=False):
    """Build (or reuse) the frame cache for one dataset. Returns its directory."""
    sig = cache_signature(contact_filter, contact_std_threshold,
                          noncontact_keep_ratio, contact_seed, image_size)
    cdir = cache_dir(dataset_root, ds_id, sig)
    if not force and _manifest_ok(cdir, camera_keys, sig):
        return cdir

    # kept set comes from the contact-std cache (compute it if missing)
    if contact_filter:
        from .contact import load_or_compute_contact_std
        std_cache = load_or_compute_contact_std(
            dataset_root, ds_id, camera_keys, tolerance_s=tolerance_s,
            video_backend=video_backend, num_workers=num_workers)
    else:
        std_cache = None

    ds = LeRobotDataset(repo_id=ds_id, root=os.path.join(dataset_root, ds_id),
                        video_backend=video_backend, tolerance_s=tolerance_s)
    for k in [k for k in ds.meta.video_keys if k not in camera_keys]:
        ds.meta.features.pop(k, None)
    reader = ds._ensure_reader()
    if reader.hf_dataset is None:
        reader.load_and_activate()
    abs_index = np.asarray(reader.hf_dataset["index"])
    abs2base = {int(a): b for b, a in enumerate(abs_index)}

    if std_cache is None:  # cache-all path: every abs index present in this split
        std_cache = {c: np.zeros(int(abs_index.max()) + 1, np.float32) for c in camera_keys}
        # nonexistent indices stay 0; kept=all handled via contact_filter=False below
    kept = _kept_indices(std_cache, camera_keys, contact_filter,
                         contact_std_threshold, noncontact_keep_ratio, contact_seed)
    if not contact_filter:
        kept = {c: abs_index.astype(np.int64) for c in camera_keys}

    os.makedirs(cdir, exist_ok=True)
    print(f"[frame_cache] building {ds_id} sig={sig}: "
          + ", ".join(f"{c.split('.')[-1]}={len(kept[c])}" for c in camera_keys))
    counts = {c: int(len(kept[c])) for c in camera_keys}
    # one memmap per camera + abs->row lookup; decode the union once and scatter
    arrs, row_of, kept_sets = {}, {}, {}
    data_tmps = {}
    for cam in camera_keys:
        data_tmps[cam] = os.path.join(cdir, f"{cam}.npy.tmp")
        arrs[cam] = open_memmap(data_tmps[cam], mode="w+", dtype=np.uint8,
                                shape=(len(kept[cam]), image_size, image_size, 3))
        row_of[cam] = {int(a): r for r, a in enumerate(kept[cam])}
        kept_sets[cam] = set(int(a) for a in kept[cam])
    union_abs = np.array(sorted(set().union(*kept_sets.values())), dtype=np.int64)

    probe = _DecodeProbe(ds, camera_keys, abs2base, union_abs, kept_sets, image_size)
    loader = DataLoader(probe, batch_size=64, num_workers=num_workers,
                        collate_fn=lambda b: b, prefetch_factor=4)
    done = 0
    for batch in loader:
        for a, frames in batch:
            for cam, img in frames.items():
                arrs[cam][row_of[cam][a]] = img.numpy()
        done += len(batch)
        if done % 5000 < 64:
            print(f"[frame_cache]   {ds_id}: {done}/{len(union_abs)} frames decoded")

    for cam in camera_keys:
        arrs[cam].flush()
        del arrs[cam]
        os.replace(data_tmps[cam], os.path.join(cdir, f"{cam}.npy"))
        idx_tmp = os.path.join(cdir, f"{cam}.idx.npy.tmp")
        with open(idx_tmp, "wb") as f:  # file handle => np.save won't append .npy
            np.save(f, kept[cam])
        os.replace(idx_tmp, os.path.join(cdir, f"{cam}.idx.npy"))

    manifest = {
        "signature": sig, "version": CACHE_VERSION, "image_size": image_size,
        "cameras": list(camera_keys), "counts": counts,
        "contact_filter": bool(contact_filter),
        "contact_std_threshold": contact_std_threshold,
        "noncontact_keep_ratio": noncontact_keep_ratio,
        "contact_seed": contact_seed, "complete": True,
    }
    man_tmp = os.path.join(cdir, "manifest.json.tmp")
    with open(man_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(man_tmp, os.path.join(cdir, "manifest.json"))
    print(f"[frame_cache] saved {cdir}")
    return cdir


def load_frame_cache(dataset_root, ds_id, camera_keys, signature):
    """Return {camera: (memmap [K,S,S,3] uint8, abs_index_array [K])}."""
    cdir = cache_dir(dataset_root, ds_id, signature)
    out = {}
    for cam in camera_keys:
        arr = np.load(os.path.join(cdir, f"{cam}.npy"), mmap_mode="r")
        idx = np.load(os.path.join(cdir, f"{cam}.idx.npy"))
        out[cam] = (arr, idx)
    return out
