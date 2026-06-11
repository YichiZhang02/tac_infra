"""Tactile-image dataset on top of LeRobot v3.0.

Reads tactile frames directly from one or more LeRobot datasets (no format
conversion). Each requested finger camera stream contributes every frame as an
independent MAE sample. Only the requested tactile camera keys are decoded
(other cameras such as the 896x896 top view are skipped for speed).
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from vtla.datasets.lerobot_dataset import LeRobotDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CAMERA_KEYS = ("observation.images.cam_finger0", "observation.images.cam_finger1")


def build_transform(train=True, image_size=224):
    """Frame-level transform. Input/Output: CHW float tensor.

    Matches AnyTouch stage1 augmentation (Resize -> H/V flip -> ColorJitter ->
    ImageNet normalize). Input frames are already float tensors in [0, 1], so no
    ToTensor step is needed.
    """
    ops = [transforms.Resize((image_size, image_size), antialias=True)]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.3),
        ]
    ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


class LeRobotTactileDataset(Dataset):
    def __init__(self, dataset_root, dataset_ids, camera_keys=DEFAULT_CAMERA_KEYS,
                 sensor_id=-1, split="train", val_ratio=0.0, transform=None,
                 tolerance_s=0.1, video_backend="pyav", return_meta=False,
                 contact_filter=False, contact_std_threshold=0.5,
                 noncontact_keep_ratio=0.05, contact_seed=0, contact_num_workers=8,
                 contact_stride=1, frame_cache=True, image_size=224):
        if isinstance(dataset_ids, str):
            dataset_ids = [dataset_ids]
        if isinstance(camera_keys, str):
            camera_keys = [camera_keys]
        self.dataset_ids = list(dataset_ids)
        self.camera_keys = list(camera_keys)
        self.sensor_id = int(sensor_id)
        self.return_meta = return_meta
        self.transform = transform if transform is not None else build_transform(split == "train")
        self.frame_cache = frame_cache

        if frame_cache:
            self._init_frame_cache(dataset_root, dataset_ids, split, val_ratio,
                                   tolerance_s, video_backend, contact_filter,
                                   contact_std_threshold, noncontact_keep_ratio,
                                   contact_seed, image_size, contact_num_workers)
            return

        rng = np.random.RandomState(contact_seed)
        self.subs = []
        self.index = []  # (sub_idx, base_idx, cam_key)
        n_contact = n_kept_noncontact = n_dropped = 0
        for sub_idx, ds_id in enumerate(dataset_ids):
            root = os.path.join(dataset_root, ds_id)
            meta_only = LeRobotDataset(repo_id=ds_id, root=root, video_backend=video_backend,
                                       tolerance_s=tolerance_s)
            total_ep = meta_only.meta.total_episodes
            del meta_only

            episodes = self._split_episodes(total_ep, split, val_ratio)
            if not episodes:
                continue
            ds = LeRobotDataset(repo_id=ds_id, root=root, episodes=episodes,
                                video_backend=video_backend, tolerance_s=tolerance_s)
            # decode only the requested tactile cameras
            missing = [c for c in self.camera_keys if c not in ds.meta.video_keys]
            if missing:
                raise KeyError(f"{ds_id}: camera keys {missing} not in dataset "
                               f"(available: {ds.meta.video_keys})")
            for k in [k for k in ds.meta.video_keys if k not in self.camera_keys]:
                ds.meta.features.pop(k, None)

            # per-frame std lookup (only when filtering); maps base_idx -> abs index
            std_cache = abs_index = None
            if contact_filter:
                from .contact import load_or_compute_contact_std
                std_cache = load_or_compute_contact_std(
                    dataset_root, ds_id, self.camera_keys,
                    tolerance_s=tolerance_s, video_backend=video_backend,
                    num_workers=contact_num_workers, stride=contact_stride)
                reader = ds._ensure_reader()
                if reader.hf_dataset is None:
                    reader.load_and_activate()
                abs_index = np.asarray(reader.hf_dataset["index"])

            self.subs.append((ds_id, ds))
            for base in range(len(ds)):
                for cam in self.camera_keys:
                    if contact_filter:
                        std = float(std_cache[cam][abs_index[base]])
                        if std > contact_std_threshold:
                            n_contact += 1
                        elif rng.rand() < noncontact_keep_ratio:
                            n_kept_noncontact += 1
                        else:
                            n_dropped += 1
                            continue
                    self.index.append((sub_idx, base, cam))

        if not self.index:
            raise RuntimeError("Empty dataset (no episodes selected). Check dataset_ids / val_ratio / split.")

        if contact_filter:
            print(f"[contact] {split}: kept {len(self.index)} samples "
                  f"(contact={n_contact}, non-contact kept={n_kept_noncontact}, "
                  f"dropped={n_dropped}) | perchannel-std thr={contact_std_threshold} "
                  f"keep_ratio={noncontact_keep_ratio}")

    def _init_frame_cache(self, dataset_root, dataset_ids, split, val_ratio,
                          tolerance_s, video_backend, contact_filter,
                          contact_std_threshold, noncontact_keep_ratio,
                          contact_seed, image_size, num_workers):
        from .frame_cache import build_frame_cache, load_frame_cache, cache_signature
        sig = cache_signature(contact_filter, contact_std_threshold,
                              noncontact_keep_ratio, contact_seed, image_size)
        self.frames = {}   # (sub_idx, cam) -> uint8 memmap [K, S, S, 3]
        self.index = []    # (sub_idx, cam, row)
        for sub_idx, ds_id in enumerate(dataset_ids):
            root = os.path.join(dataset_root, ds_id)
            # build on demand (instant skip if already complete); training pre-builds
            # on the main process so non-main ranks just hit the finished cache here.
            build_frame_cache(
                dataset_root, ds_id, self.camera_keys, image_size=image_size,
                contact_filter=contact_filter, contact_std_threshold=contact_std_threshold,
                noncontact_keep_ratio=noncontact_keep_ratio, contact_seed=contact_seed,
                tolerance_s=tolerance_s, video_backend=video_backend, num_workers=num_workers)
            cache = load_frame_cache(dataset_root, ds_id, self.camera_keys, sig)

            # absolute frame indices that fall in this split's episodes (metadata only)
            meta_only = LeRobotDataset(repo_id=ds_id, root=root,
                                       video_backend=video_backend, tolerance_s=tolerance_s)
            total_ep = meta_only.meta.total_episodes
            del meta_only
            episodes = self._split_episodes(total_ep, split, val_ratio)
            if not episodes:
                continue
            ds = LeRobotDataset(repo_id=ds_id, root=root, episodes=episodes,
                                video_backend=video_backend, tolerance_s=tolerance_s)
            reader = ds._ensure_reader()
            if reader.hf_dataset is None:
                reader.load_and_activate()
            split_abs = set(int(a) for a in np.asarray(reader.hf_dataset["index"]))
            del ds

            for cam in self.camera_keys:
                arr, kept_abs = cache[cam]
                self.frames[(sub_idx, cam)] = arr
                for row, a in enumerate(kept_abs):
                    if int(a) in split_abs:
                        self.index.append((sub_idx, cam, row))

        if not self.index:
            raise RuntimeError("Empty dataset (frame_cache). Check dataset_ids / val_ratio / split.")
        print(f"[frame_cache] {split}: {len(self.index)} samples from {dataset_ids} (sig={sig})")

    @staticmethod
    def _split_episodes(total_ep, split, val_ratio):
        if val_ratio <= 0.0:
            return list(range(total_ep)) if split == "train" else []
        n_val = max(1, round(total_ep * val_ratio))
        n_val = min(n_val, total_ep)
        val_eps = list(range(total_ep - n_val, total_ep))
        train_eps = list(range(0, total_ep - n_val))
        return train_eps if split == "train" else val_eps

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.frame_cache:
            sub_idx, cam, row = self.index[idx]
            arr = self.frames[(sub_idx, cam)]
            img = torch.from_numpy(np.array(arr[row]))     # writable copy (memmap is read-only)
            img = img.permute(2, 0, 1).float().div_(255.0)  # HWC uint8 -> CHW [0,1]
            img = self.transform(img)
            if self.return_meta:
                return img, self.sensor_id, {"dataset": self.dataset_ids[sub_idx], "camera": cam}
            return img, self.sensor_id

        sub_idx, base, cam = self.index[idx]
        ds_id, ds = self.subs[sub_idx]
        img = ds[base][cam]
        img = self.transform(img)
        if self.return_meta:
            return img, self.sensor_id, {"dataset": ds_id, "camera": cam}
        return img, self.sensor_id
