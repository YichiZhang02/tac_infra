"""Tactile MAE evaluation & visualization.

Three modes (any combination via flags):
  --recon_metric : average masked-patch MSE over a held-out split
  --recon_vis    : save a [original | masked | reconstruction | pasted] grid
  --tsne         : t-SNE of CLS features, colored by dataset / camera

Example:
  python -m vtla.tac_encoder.tactile_mae.eval --arch vit_l \
     --checkpoint playground/pretrained_models/checkpoint.pth \
     --dataset_ids rm_nist_260320_strawberry --val_ratio 0.1 \
     --recon_metric --recon_vis --tsne --output_dir playground/results/tac_mae_eval
"""
import os
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from .models import build_model, load_pretrained
from .data import LeRobotTactileDataset, build_transform, IMAGENET_MEAN, IMAGENET_STD


def denorm(x):
    """CHW normalized tensor -> HWC uint8-ish float in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(3, 1, 1)
    x = x * std + mean
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _sample_contact_std(dataset, idx):
    """Max per-channel std on the 0-255 scale for one dataset sample.

    Mirrors data.contact.contact_score but operates on the (already normalized)
    image returned by the dataset, undoing the ImageNet normalization first.
    """
    img = torch.as_tensor(dataset[idx][0])
    x = denorm(img) * 255.0  # HWC in [0, 255]
    return float(max(x[..., c].std() for c in range(x.shape[2])))


def select_vis_indices(dataset, n_per_level=3, max_candidates=400, seed=0):
    """Pick low / mid / high contact-std samples for the reconstruction grid.

    Random sampling makes the grid hard to read (mostly idle frames). Instead we
    score a random subset by contact std and return three groups spanning the
    range: near-idle, moderate contact, and strong contact.

    Returns a list of (label, [(idx, std), ...]) groups.
    """
    n = len(dataset)
    g = torch.Generator().manual_seed(seed)
    cand = torch.randperm(n, generator=g)[:min(max_candidates, n)].tolist()
    scored = sorted((_sample_contact_std(dataset, i), i) for i in cand)
    k = max(1, n_per_level)

    def take(items):
        return [(i, s) for s, i in items]

    lo = take(scored[:k])
    hi = take(scored[-k:])
    mid_start = max(0, len(scored) // 2 - k // 2)
    mid = take(scored[mid_start:mid_start + k])
    return [
        ("low-std\n(≈no contact)", lo),
        ("mid-std\n(light contact)", mid),
        ("high-std\n(strong contact)", hi),
    ]


@torch.no_grad()
def reconstruction_metric(model, loader, device, max_batches=None, amp_dtype=torch.float16):
    model.eval()
    total, n = 0.0, 0
    n_batches = min(max_batches, len(loader)) if max_batches else len(loader)
    for i, (imgs, sensors) in enumerate(tqdm(loader, total=n_batches, desc="recon-eval",
                                             dynamic_ncols=True, leave=False)):
        if max_batches and i >= max_batches:
            break
        imgs = imgs.to(device)
        sensors = sensors.to(device).long()
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss, _, _ = model(imgs, sensor_type=sensors)
        total += loss.item() * imgs.size(0)
        n += imgs.size(0)
    return total / max(n, 1)


@torch.no_grad()
def reconstruction_vis(model, dataset, device, out_path, groups=None, n=8, seed=0):
    """Save a [original | masked | reconstruction | error] grid.

    ``groups`` is a list of (label, [(idx, std), ...]); when omitted we fall back
    to ``n`` random samples (single unlabeled group).
    """
    model.eval()
    if groups is None:
        g = torch.Generator().manual_seed(seed)
        idxs = torch.randperm(len(dataset), generator=g)[:n].tolist()
        groups = [("", [(i, None) for i in idxs])]

    # flatten rows; remember a per-row label (only the first row of a group)
    flat_idxs, row_labels = [], []
    for label, items in groups:
        for j, (i, std) in enumerate(items):
            flat_idxs.append(i)
            tag = label if j == 0 else ""
            if std is not None:
                tag = (tag + "\n" if tag else "") + f"std={std:.1f}"
            row_labels.append(tag)

    imgs = torch.stack([torch.as_tensor(dataset[i][0]) for i in flat_idxs]).to(device)
    sensors = torch.tensor([dataset[i][1] for i in flat_idxs], device=device).long()

    with torch.amp.autocast("cuda"):
        _, pred, mask = model(imgs, sensor_type=sensors)
    pred = pred.float()

    target = model.patchify(imgs)
    # full reconstruction: model's prediction for *every* patch (visible ones too)
    recon_img = model.unpatchify(pred)
    # masked input: zero out (gray) the masked patches of the original
    masked_patches = target * (1 - mask.unsqueeze(-1))
    masked_img = model.unpatchify(masked_patches)

    rows = len(flat_idxs)
    fig, axes = plt.subplots(rows, 4, figsize=(8.5, 2 * rows))
    col_titles = ["original", "masked", "reconstruction", "error"]
    err_im = None
    for r in range(rows):
        orig = denorm(imgs[r])
        recon = denorm(recon_img[r])
        # per-pixel reconstruction error: mean abs diff across channels in [0,1]
        err = np.abs(orig - recon).mean(axis=2)
        for c, im in enumerate([imgs[r], masked_img[r], recon_img[r]]):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.imshow(denorm(im))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c])
            if c == 0 and row_labels[r]:
                ax.set_ylabel(row_labels[r], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=24)
        ax = axes[r, 3] if rows > 1 else axes[3]
        err_im = ax.imshow(err, cmap="inferno", vmin=0.0, vmax=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        if r == 0:
            ax.set_title(col_titles[3])
    fig.colorbar(err_im, ax=axes.ravel().tolist() if rows > 1 else axes,
                 fraction=0.02, pad=0.01, label="|orig - recon| (mean over RGB)")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[recon_vis] saved {out_path}")


@torch.no_grad()
def tsne_vis(model, dataset, device, out_path, max_samples=1000, color_by="camera",
             batch_size=64, seed=0):
    from sklearn.manifold import TSNE
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(dataset), generator=g)[:max_samples].tolist()

    feats, labels = [], []
    for start in range(0, len(idxs), batch_size):
        batch_idx = idxs[start:start + batch_size]
        imgs = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        sensors = torch.tensor([dataset[i][1] for i in batch_idx], device=device).long()
        meta = [dataset[i][2] for i in batch_idx]
        with torch.amp.autocast("cuda"):
            f = model.extract_features(imgs, sensors)
        feats.append(f.float().cpu().numpy())
        labels += [m[color_by] for m in meta]
    feats = np.concatenate(feats, 0)

    emb = TSNE(n_components=2, init="pca", perplexity=min(30, len(feats) - 1),
               random_state=seed).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for j, u in enumerate(uniq):
        m = np.array([l == u for l in labels])
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(j % 10), label=str(u).split(".")[-1])
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(f"t-SNE of CLS features (colored by {color_by}, n={len(feats)})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[tsne] saved {out_path}")


def main():
    ap = argparse.ArgumentParser("Tactile MAE eval")
    ap.add_argument("--arch", default="vit_l", choices=["vit_b", "vit_l"])
    ap.add_argument("--checkpoint", required=True,
                    help="Trained MAE ckpt (.pth) or any pretrained_path source")
    ap.add_argument("--dataset_root", default="playground/data")
    ap.add_argument("--dataset_ids", nargs="+", required=True)
    ap.add_argument("--camera_keys", nargs="+",
                    default=["observation.images.cam_finger0", "observation.images.cam_finger1"])
    ap.add_argument("--sensor_id", type=int, default=-1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--use_sensor_token", action="store_true", default=True)
    ap.add_argument("--no_sensor_token", dest="use_sensor_token", action="store_false")
    ap.add_argument("--use_same_patchemb", action="store_true", default=True)
    ap.add_argument("--no_same_patchemb", dest="use_same_patchemb", action="store_false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="playground/results/tac_mae_eval")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=None)
    ap.add_argument("--tsne_samples", type=int, default=1000)
    ap.add_argument("--vis_per_level", type=int, default=3,
                    help="reconstruction grid rows per contact level (low/mid/high std)")
    ap.add_argument("--recon_metric", action="store_true")
    ap.add_argument("--recon_vis", action="store_true")
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    if not (args.recon_metric or args.recon_vis or args.tsne):
        args.recon_metric = args.recon_vis = args.tsne = True
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(arch=args.arch, mask_ratio=args.mask_ratio,
                        use_sensor_token=args.use_sensor_token,
                        use_same_patchemb=args.use_same_patchemb)
    load_pretrained(model, args.checkpoint)
    model.to(device)

    # eval-time transform (no augmentation)
    tf = build_transform(train=False)
    dataset = LeRobotTactileDataset(
        dataset_root=args.dataset_root, dataset_ids=args.dataset_ids,
        camera_keys=args.camera_keys, sensor_id=args.sensor_id, split=args.split,
        val_ratio=args.val_ratio, transform=tf, return_meta=True, frame_cache=False)
    print(f"Eval dataset ({args.split}): {len(dataset)} samples")

    if args.recon_metric:
        loader = torch.utils.data.DataLoader(
            _DropMeta(dataset), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers)
        mse = reconstruction_metric(model, loader, device, max_batches=args.max_batches)
        print(f"[recon_metric] masked-patch MSE (mask_ratio={args.mask_ratio}): {mse:.6f}")
        with open(os.path.join(args.output_dir, "metrics.txt"), "a") as f:
            f.write(f"masked_mse mask_ratio={args.mask_ratio} {mse:.6f}\n")

    if args.recon_vis:
        groups = select_vis_indices(dataset, n_per_level=args.vis_per_level)
        reconstruction_vis(model, dataset, device,
                           os.path.join(args.output_dir, "reconstruction.png"),
                           groups=groups)

    if args.tsne:
        color_by = "dataset" if len(args.dataset_ids) > 1 else "camera"
        tsne_vis(model, dataset, device, os.path.join(args.output_dir, "tsne.png"),
                 max_samples=args.tsne_samples, color_by=color_by)


class _DropMeta(torch.utils.data.Dataset):
    """Wrap a return_meta dataset to yield (img, sensor) for batched loaders."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, sensor, _ = self.ds[i]
        return img, sensor


if __name__ == "__main__":
    main()
