"""Tactile MAE pretraining entry point.

Run (single GPU):
    python -m vtla.tac_encoder.tactile_mae.train --arch vit_l \
        --pretrained_path playground/pretrained_models/CLIP-ViT-L-14-DataComp.XL-s13B-b90K \
        --dataset_ids rm_nist_260320_strawberry --output_dir playground/results/tac_mae

Distributed:
    torchrun --nproc_per_node=4 -m vtla.tac_encoder.tactile_mae.train ...
"""
import os
import sys
import json
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory

from .config import get_args_parser
from .models import build_model, load_pretrained
from .data import LeRobotTactileDataset, build_transform
from .engine import misc, train_one_epoch
from .engine.misc import NativeScalerWithGradNormCount as NativeScaler
from .eval import reconstruction_metric, reconstruction_vis, select_vis_indices


def main(args):
    misc.init_distributed_mode(args)
    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ----- data -----
    # Precompute the contact-std + decode-once frame caches on the main process
    # only (avoids a DDP race where every rank would decode & write concurrently).
    if misc.is_main_process():
        if args.contact_filter:
            from .data.contact import load_or_compute_contact_std
            for ds_id in args.dataset_ids:
                load_or_compute_contact_std(
                    args.dataset_root, ds_id, args.camera_keys,
                    tolerance_s=args.tolerance_s, video_backend=args.video_backend,
                    num_workers=args.num_workers, stride=args.contact_stride)
        if args.frame_cache:
            from .data.frame_cache import build_frame_cache
            for ds_id in args.dataset_ids:
                build_frame_cache(
                    args.dataset_root, ds_id, args.camera_keys, image_size=args.image_size,
                    contact_filter=args.contact_filter,
                    contact_std_threshold=args.contact_std_threshold,
                    noncontact_keep_ratio=args.noncontact_keep_ratio,
                    contact_seed=args.contact_seed, tolerance_s=args.tolerance_s,
                    video_backend=args.video_backend, num_workers=args.num_workers)
    if args.distributed:
        torch.distributed.barrier()

    dataset_train = LeRobotTactileDataset(
        dataset_root=args.dataset_root, dataset_ids=args.dataset_ids,
        camera_keys=args.camera_keys, sensor_id=args.sensor_id, split="train",
        val_ratio=args.val_ratio, tolerance_s=args.tolerance_s, video_backend=args.video_backend,
        contact_filter=args.contact_filter,
        contact_std_threshold=args.contact_std_threshold,
        noncontact_keep_ratio=args.noncontact_keep_ratio, contact_seed=args.contact_seed,
        contact_num_workers=args.num_workers, contact_stride=args.contact_stride,
        frame_cache=args.frame_cache, image_size=args.image_size)
    print(f"Train dataset: {len(dataset_train)} samples from {args.dataset_ids}")

    # Held-out val set (main process only; used for periodic eval + reconstruction viz).
    dataset_val = None
    if args.val_ratio > 0 and misc.is_main_process():
        dataset_val = LeRobotTactileDataset(
            dataset_root=args.dataset_root, dataset_ids=args.dataset_ids,
            camera_keys=args.camera_keys, sensor_id=args.sensor_id, split="val",
            val_ratio=args.val_ratio, tolerance_s=args.tolerance_s, video_backend=args.video_backend,
            transform=build_transform(train=False),
            contact_filter=args.contact_filter,
            contact_std_threshold=args.contact_std_threshold,
            noncontact_keep_ratio=args.noncontact_keep_ratio, contact_seed=args.contact_seed,
            contact_num_workers=args.num_workers, contact_stride=args.contact_stride,
            return_meta=False, frame_cache=args.frame_cache, image_size=args.image_size)
        print(f"Val dataset: {len(dataset_val)} samples")

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    # ----- model -----
    model = build_model(arch=args.arch, mask_ratio=args.mask_ratio,
                        use_sensor_token=args.use_sensor_token,
                        use_same_patchemb=args.use_same_patchemb,
                        norm_pix_loss=args.norm_pix_loss,
                        visible_loss_weight=args.visible_loss_weight)
    load_pretrained(model, args.pretrained_path)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.99))
    # GradScaler only applies to fp16; bfloat16 has fp32's exponent range and
    # needs no loss scaling.
    loss_scaler = NativeScaler(enabled=(args.amp_dtype == "float16"))
    print(optimizer)

    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    val_loader = None
    if dataset_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dataset_val, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    vis_dir = os.path.join(args.output_dir, "recon_vis") if args.output_dir else None
    vis_groups = None
    if vis_dir and dataset_val is not None and misc.is_main_process():
        os.makedirs(vis_dir, exist_ok=True)
        # Pick a fixed set of low / mid / high contact-std frames once, so every
        # checkpoint's reconstruction grid shows the same comparable samples
        # (and we only pay the frame-decoding cost for selection a single time).
        vis_groups = select_vis_indices(dataset_val, n_per_level=args.vis_per_level,
                                         seed=args.seed)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_val_loss = float("inf")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            args=args, start_time=start_time)

        is_save = epoch % args.save_freq == 0 or epoch + 1 == args.epochs
        if args.output_dir and is_save:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        # ----- periodic eval + per-checkpoint reconstruction viz (main process) -----
        val_loss = None
        if val_loader is not None and misc.is_main_process() \
                and args.eval_freq > 0 and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs):
            val_amp = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
            val_loss = reconstruction_metric(model_without_ddp, val_loader, device,
                                              max_batches=args.eval_max_batches, amp_dtype=val_amp)
            print(f"[eval] epoch {epoch}: val masked-MSE = {val_loss:.6f}")
            if args.output_dir and val_loss < best_val_loss:
                best_val_loss = val_loss
                misc.save_best_model(args=args, epoch=epoch, model_without_ddp=model_without_ddp,
                                     optimizer=optimizer, loss_scaler=loss_scaler, val_loss=val_loss)
                print(f"[eval] new best val masked-MSE = {val_loss:.6f} -> saved best.pth")
        if vis_groups is not None and misc.is_main_process() and is_save:
            reconstruction_vis(model_without_ddp, dataset_val, device,
                               os.path.join(vis_dir, f"recon_epoch{epoch:04d}.png"),
                               groups=vis_groups)
        model.train(True)  # restore train mode after eval/viz set eval()
        if args.distributed:
            torch.distributed.barrier()

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if val_loss is not None:
            log_stats["val_loss"] = val_loss
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
