"""Convert a flat directory of tactile PNGs into a tactile-MAE frame cache.

This bypasses LeRobot entirely: it writes the exact decode-once cache layout that
``data/frame_cache.py`` produces (a dense ``uint8`` memmap of pre-resized frames +
an absolute-index sidecar + a manifest), so training can read it directly via the
``--raw_frame_cache`` path with no parquet / mp4 / LeRobot metadata.

Source layout (e.g. AnyTouch ``data_tac2_s``):
    <src_dir>/000000000.png, 000000001.png, ...   # one continuous stream, HxW RGB

Output layout (matches data/frame_cache.py):
    <dataset_root>/<dataset_id>/frames_cache/<signature>/
        <camera_key>.npy        # [N, S, S, 3] uint8, HWC, resized to image_size
        <camera_key>.idx.npy    # [N] int64, here just arange(N) (no episodes)
        manifest.json           # signature + counts + complete flag

Example:
    python -m vtla.tac_encoder.tactile_mae.tools.png_to_frame_cache \
        --src_dir lerobot_tactile_ws/AnyTouch/zxd/data/data_tac2_s \
        --dataset_root playground/data --dataset_id pretrained_data \
        --camera_key observation.images.cam_finger0 --image_size 224 --num_workers 16
"""
import argparse
import glob
import json
import os

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..data.frame_cache import CACHE_VERSION, cache_dir, cache_signature


class _PngResize(Dataset):
    """Decode + resize one PNG to (S, S, 3) uint8 HWC. Returns (row, array)."""

    def __init__(self, paths, image_size):
        self.paths = paths
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        # bilinear matches the F.interpolate(mode="bilinear") used by the live
        # frame-cache builder; training Resizes again so exact parity isn't critical.
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return i, np.asarray(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser("PNG stream -> tactile-MAE frame cache")
    ap.add_argument("--src_dir", required=True, help="Flat dir of *.png frames")
    ap.add_argument("--dataset_root", default="playground/data")
    ap.add_argument("--dataset_id", required=True,
                    help="Output dataset folder name under dataset_root")
    ap.add_argument("--camera_key", default="observation.images.cam_finger0")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--glob", default="*.png", help="Glob pattern for source frames")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if a complete cache already exists")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.src_dir, args.glob)))
    if not paths:
        raise FileNotFoundError(f"No frames matched {args.glob} in {args.src_dir}")
    n = len(paths)
    S = args.image_size

    # contact_filter=False -> "all_<S>_v<ver>" signature (the no-filter convention)
    sig = cache_signature(False, 0.5, 0.05, 0, S)
    cdir = cache_dir(args.dataset_root, args.dataset_id, sig)
    os.makedirs(cdir, exist_ok=True)

    man_path = os.path.join(cdir, "manifest.json")
    if not args.force and os.path.exists(man_path):
        with open(man_path, encoding="utf-8") as f:
            man = json.load(f)
        if man.get("complete") and man.get("counts", {}).get(args.camera_key) == n:
            print(f"[png->cache] already complete: {cdir} ({n} frames)")
            return

    print(f"[png->cache] {n} frames {args.src_dir} -> {cdir}")
    print(f"[png->cache]   camera={args.camera_key} size={S}x{S} sig={sig}")

    data_tmp = os.path.join(cdir, f"{args.camera_key}.npy.tmp")
    arr = open_memmap(data_tmp, mode="w+", dtype=np.uint8, shape=(n, S, S, 3))

    loader = DataLoader(_PngResize(paths, S), batch_size=64,
                        num_workers=args.num_workers, collate_fn=lambda b: b,
                        prefetch_factor=4)
    done = 0
    for batch in loader:
        for row, img in batch:
            arr[row] = img
        done += len(batch)
        if done % 5000 < 64:
            print(f"[png->cache]   {done}/{n} frames")
    arr.flush()
    del arr
    os.replace(data_tmp, os.path.join(cdir, f"{args.camera_key}.npy"))

    # no episodes in the source stream -> absolute index is just 0..n-1
    idx_tmp = os.path.join(cdir, f"{args.camera_key}.idx.npy.tmp")
    with open(idx_tmp, "wb") as f:  # file handle => np.save won't append .npy
        np.save(f, np.arange(n, dtype=np.int64))
    os.replace(idx_tmp, os.path.join(cdir, f"{args.camera_key}.idx.npy"))

    manifest = {
        "signature": sig, "version": CACHE_VERSION, "image_size": S,
        "cameras": [args.camera_key], "counts": {args.camera_key: n},
        "contact_filter": False, "contact_std_threshold": 0.5,
        "noncontact_keep_ratio": 0.05, "contact_seed": 0, "complete": True,
        "source": os.path.abspath(args.src_dir),
    }
    man_tmp = man_path + ".tmp"
    with open(man_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(man_tmp, man_path)
    print(f"[png->cache] done: {cdir}")


if __name__ == "__main__":
    main()
