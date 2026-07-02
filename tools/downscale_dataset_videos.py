#!/usr/bin/env python
"""Create a downscaled copy of a LeRobot (v3.0) dataset to speed up training data loading.

Why: training decodes camera frames on the CPU every step. The RGB cameras here are large
(e.g. 1920x1080 / 1440x1080), but the policies resize every frame to ~224x224 anyway
(`resize_imgs_to`). Decoding full-res frames just to throw the pixels away makes the data
loader the bottleneck: a fast model (e.g. 250M diffusion, ~0.1s/step) starves the GPU and
stalls 3-4s whenever the prefetch buffer drains. Pre-scaling the RGB videos to ~256px cuts the
per-frame decode cost several-fold and removes the stall. (256 keeps headroom above the model's
224 crop; native->256->224 differs from native->224 by only ~2/255 mean pixels, PSNR ~33dB.)

What it does (non-destructive — writes a new dataset copy):
  - copies meta/ and data/ verbatim (parquet, stats, episodes, tasks, etc.)
  - re-encodes only the large 8-bit RGB camera videos, scaled to SIZE x SIZE, preserving the
    exact frame count / fps / timestamps (only spatial resolution changes), with a dense keyframe
    interval (small GOP) so random-access seeks during training stay cheap
  - COPIES other camera videos verbatim — in particular the tactile finger cams, which are stored
    lossless 16-bit (ffv1 / gbrp16le, .mkv); re-encoding those to lossy 8-bit would corrupt them
  - patches meta/info.json so each re-encoded video feature's shape / height / width / codec match
  - skips frames_cache/ (a derived cache keyed to the old resolution; it is regenerated)

A camera is a downscale target when it uses the global (.mp4) video path, is 8-bit (pix_fmt has
no '16'), is not a lossless ffv1 stream, and its short side is larger than SIZE. Override the
auto-selection with --cameras.

Square scaling (SIZE x SIZE) matches what the policies already do — they squash every camera to a
square 224x224 — so it introduces no extra distortion vs. current training.

Usage:
    python tools/downscale_dataset_videos.py \
        --src playground/data/rm_umi_dual_pen_open \
        --dst playground/data/rm_umi_dual_pen_open_256 \
        --size 256

Then train against --dataset.root=<dst> (same repo_id works).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ffmpeg encoder name -> the codec tag stored in info.json (matches LeRobot's convention).
_CODEC_TAG = {
    "libx264": "h264",
    "libx264rgb": "h264",
    "libx265": "hevc",
    "libsvtav1": "av1",
    "h264_nvenc": "h264",
    "hevc_nvenc": "hevc",
}
_LOSSLESS_CODECS = {"ffv1", "huffyuv", "rawvideo"}

# Source codec (as ffprobe reports it) -> NVDEC (cuvid) decoder for GPU-offloaded decode.
# Only decode is offloaded; scale + libx264 encode stay on CPU (the A100 has no NVENC unit).
_CUVID = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1": "av1_cuvid",
    "vp9": "vp9_cuvid",
    "vp8": "vp8_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "mpeg2video": "mpeg2_cuvid",
}


def _probe_codec(path: str) -> str:
    """First video stream codec name (e.g. 'hevc', 'h264'); '' on failure."""
    try:
        return subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=nokey=1:noprint_wrappers=1", path],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _nvdec_functional(sample_src: str, dec_name: str) -> bool:
    """One-shot check that NVDEC can actually decode this codec on this box (a build listing the
    decoder does not guarantee a capable device; e.g. driver/GPU mismatch)."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-c:v", dec_name,
             "-i", sample_src, "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"],
            capture_output=True,
        )
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _scaled_encode(args: tuple) -> tuple[str, bool, str]:
    """Re-encode one video scaled to size x size. Returns (src_path, ok, message).

    gpu_decode offloads decode to NVDEC (cuvid) — a big win for large HEVC cams (e.g. 1440x1080);
    the scale (swscale) and encode (libx264) stay on CPU. Falls back to CPU decode transparently
    when the source codec has no NVDEC decoder.
    """
    src, dst, size, gop, crf, codec, scale_flags, gpu_decode = args
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    dec_name = _CUVID.get(_probe_codec(src)) if gpu_decode else None
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if dec_name:
        cmd += ["-hwaccel", "cuda", "-c:v", dec_name]
    cmd += [
        "-i", src,
        "-map", "0:v:0",                                  # video stream only
        "-vf", f"scale={size}:{size}:flags={scale_flags}",
        "-c:v", codec,
        "-crf", str(crf),
        "-g", str(gop),                                   # dense keyframes -> cheap seeks
        "-pix_fmt", "yuv420p",
        "-an",                                            # drop audio
        "-vsync", "0",                                    # passthrough: preserve frame count / pts
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or str(e)).strip()
        return src, False, msg.splitlines()[-1] if msg else str(e)
    return src, True, ""


def _copy_verbatim(args: tuple[str, str]) -> tuple[str, bool, str]:
    """Copy one video file unchanged. Returns (src_path, ok, message)."""
    src, dst = args
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    except Exception as e:  # noqa: BLE001
        return src, False, str(e)
    return src, True, ""


def _frame_count(path: Path) -> int | None:
    """Video frame count via ffprobe. Uses -count_packets (demux only, no decode): for these CFR
    videos the packet count equals the frame count, and it is ~500x faster than -count_frames,
    which decodes every frame single-threaded (minutes per HEVC file). Returns None on failure."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        return int(out)
    except Exception:  # noqa: BLE001
        return None


def select_downscale_targets(info: dict, size: int, override: list[str] | None) -> set[str]:
    """Camera (video feature) keys to re-encode. Tactile/lossless/16-bit cams are excluded."""
    feats = info.get("features", {})
    if override is not None:
        return set(override)
    targets: set[str] = set()
    for key, ft in feats.items():
        if ft.get("dtype") != "video":
            continue
        vinfo = ft.get("info", {})
        # Per-feature video_path override marks the special tactile .mkv streams -> leave alone.
        if "video_path" in ft:
            continue
        if vinfo.get("video.codec") in _LOSSLESS_CODECS:
            continue
        if "16" in str(vinfo.get("video.pix_fmt", "")):  # e.g. gbrp16le -> 16-bit, not RGB8
            continue
        shape = ft.get("shape", [])  # [H, W, C]
        if len(shape) == 3 and min(shape[0], shape[1]) <= size:
            continue  # already small enough; don't upscale
        targets.add(key)
    return targets


def copy_non_video(src: Path, dst: Path) -> None:
    """Copy everything except videos/ and frames_cache/ verbatim."""
    skip = {"videos", "frames_cache"}
    dst.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src.iterdir()):
        if entry.name in skip:
            continue
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)


def patch_info_json(dst: Path, targets: set[str], size: int, codec: str) -> int:
    """Update the re-encoded video features' shape/height/width/codec in meta/info.json."""
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    tag = _CODEC_TAG.get(codec, codec)
    n = 0
    for key, ft in info.get("features", {}).items():
        if key not in targets or ft.get("dtype") != "video":
            continue
        ch = ft["shape"][2] if len(ft.get("shape", [])) == 3 else 3
        ft["shape"] = [size, size, ch]
        vinfo = ft.setdefault("info", {})
        vinfo["video.height"] = size
        vinfo["video.width"] = size
        vinfo["video.codec"] = tag
        n += 1
    info_path.write_text(json.dumps(info, indent=4))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="Source dataset root (unchanged).")
    ap.add_argument("--dst", type=Path, required=True, help="Destination dataset root (created).")
    ap.add_argument("--size", type=int, default=256, help="Output square resolution (default 256).")
    ap.add_argument("--gop", type=int, default=4, help="Keyframe interval; small = fast seeks (default 4).")
    ap.add_argument("--crf", type=int, default=18, help="x264 quality, lower = better/larger (default 18).")
    ap.add_argument("--codec", default="libx264", help="ffmpeg video encoder for RGB (default libx264).")
    ap.add_argument("--scale-flags", default="lanczos",
                    help="ffmpeg swscale kernel for downscaling (default lanczos = best detail).")
    ap.add_argument("--gpu-decode", choices=["auto", "on", "off"], default="auto",
                    help="Offload video decode to NVDEC/cuvid ('auto' = use it when a capable GPU "
                         "and NVDEC decoder exist; scale/encode always stay on CPU).")
    ap.add_argument("--jobs", type=int, default=8, help="Parallel ffmpeg/copy workers (default 8).")
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="Force this exact set of camera keys to downscale (overrides auto-detect).")
    ap.add_argument("--overwrite", action="store_true", help="Redo files that already exist in dst.")
    ap.add_argument("--verify", action="store_true",
                    help="ffprobe-check that each re-encoded frame count matches the source "
                         "(fast: demux-only packet count).")
    args = ap.parse_args()

    src, dst = args.src.resolve(), args.dst.resolve()
    if not (src / "meta" / "info.json").is_file():
        ap.error(f"{src} does not look like a LeRobot dataset (missing meta/info.json).")
    if src == dst:
        ap.error("--src and --dst must differ.")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        ap.error("ffmpeg/ffprobe not found on PATH.")
    video_root = src / "videos"
    if not video_root.is_dir():
        ap.error(f"No videos/ directory under {src}.")

    info = json.loads((src / "meta" / "info.json").read_text())
    targets = select_downscale_targets(info, args.size, args.cameras)

    print(f"Source: {src}\nDest:   {dst}")
    print(f"Downscale targets ({len(targets)}): {sorted(targets)}")
    print(f"  -> {args.size}x{args.size}, codec={args.codec}, gop={args.gop}, crf={args.crf}, flags={args.scale_flags}")
    print("Copying meta/ and data/ verbatim (skipping videos/, frames_cache/) ...")
    copy_non_video(src, dst)
    n_feats = patch_info_json(dst, targets, args.size, args.codec)
    print(f"Patched {n_feats} video feature(s) in meta/info.json.")

    # Plan per-file work: re-encode targets, copy everything else verbatim.
    enc_jobs: list[tuple] = []
    copy_jobs: list[tuple] = []
    for vid in sorted(video_root.rglob("*")):
        if not vid.is_file():
            continue
        cam = vid.relative_to(video_root).parts[0]
        if cam in targets:
            out = (dst / vid.relative_to(src)).with_suffix(".mp4")  # RGB targets -> .mp4 (global path)
            if out.exists() and not args.overwrite:
                continue
            enc_jobs.append([str(vid), str(out), args.size, args.gop, args.crf, args.codec, args.scale_flags])
        else:
            out = dst / vid.relative_to(src)  # preserve extension (e.g. tactile .mkv)
            if out.exists() and not args.overwrite:
                continue
            copy_jobs.append((str(vid), str(out)))

    # Resolve GPU-decode capability once (per-file fallback still applies inside the worker for
    # codecs without an NVDEC decoder). Scale + encode always run on CPU.
    gpu_decode = False
    if args.gpu_decode != "off" and enc_jobs:
        sample_codec = _probe_codec(enc_jobs[0][0])
        dec_name = _CUVID.get(sample_codec)
        if dec_name and _nvdec_functional(enc_jobs[0][0], dec_name):
            gpu_decode = True
            print(f"GPU decode: on (NVDEC for '{sample_codec}'; scale/encode on CPU).")
        elif args.gpu_decode == "on":
            ap.error(f"--gpu-decode=on but NVDEC unavailable for codec '{sample_codec}'.")
        else:
            print(f"GPU decode: off (no working NVDEC decoder for codec '{sample_codec}').")
    for j in enc_jobs:
        j.append(gpu_decode)

    print(f"Re-encoding {len(enc_jobs)} RGB video(s); copying {len(copy_jobs)} other video(s) verbatim "
          f"with {args.jobs} worker(s) ...")
    failures: list[tuple[str, str]] = []
    total = len(enc_jobs) + len(copy_jobs)
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_scaled_encode, j): ("enc", j) for j in enc_jobs}
        futs.update({pool.submit(_copy_verbatim, j): ("cp", j) for j in copy_jobs})
        for fut in as_completed(futs):
            srcf, ok, msg = fut.result()
            done += 1
            kind = futs[fut][0]
            rel = Path(srcf).relative_to(video_root)
            if ok:
                print(f"  [{done}/{total}] {kind:3s} ok  {rel}")
            else:
                failures.append((srcf, msg))
                print(f"  [{done}/{total}] {kind:3s} FAIL {rel}: {msg}")

    if args.verify and not failures:
        print("Verifying re-encoded frame counts (ffprobe) ...")
        for j in enc_jobs:
            cs, cd = _frame_count(Path(j[0])), _frame_count(Path(j[1]))
            if cs is not None and cs != cd:
                failures.append((j[0], f"frame count {cd} != source {cs}"))
                print(f"  MISMATCH {Path(j[1]).name}: {cd} != {cs}")

    if failures:
        print(f"\nDONE WITH {len(failures)} FAILURE(S):", file=sys.stderr)
        for f, m in failures:
            print(f"  {f}: {m}", file=sys.stderr)
        return 1

    try:
        old = sum(p.stat().st_size for p in video_root.rglob("*") if p.is_file())
        new = sum(p.stat().st_size for p in (dst / "videos").rglob("*") if p.is_file())
        print(f"\nVideo size: {old/1e9:.2f} GB -> {new/1e9:.2f} GB ({new/max(old,1)*100:.0f}%).")
    except Exception:  # noqa: BLE001
        pass
    print(f"Done. Train with --dataset.root={dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
