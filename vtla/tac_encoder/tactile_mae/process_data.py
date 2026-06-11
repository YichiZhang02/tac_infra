"""Warm up tactile MAE dataset caches before launching training.

This entry point intentionally reuses the existing cache builders instead of
owning a second preprocessing implementation. It is meant to run once in a
single process before the multi-GPU training job starts.
"""
import datetime
import time

from .config import get_args_parser
from .data.contact import load_or_compute_contact_std
from .data.frame_cache import build_frame_cache


def main(args):
    start_time = time.time()
    print("Warm up tactile MAE dataset caches")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset(s): {args.dataset_ids}")
    print(f"Camera key(s): {args.camera_keys}")
    print(f"Contact filter: {args.contact_filter}")
    print(f"Frame cache: {args.frame_cache}")

    for ds_id in args.dataset_ids:
        print(f"[process_data] dataset={ds_id}")
        if args.contact_filter:
            load_or_compute_contact_std(
                args.dataset_root,
                ds_id,
                args.camera_keys,
                tolerance_s=args.tolerance_s,
                video_backend=args.video_backend,
                num_workers=args.num_workers,
                stride=args.contact_stride,
            )
        if args.frame_cache:
            build_frame_cache(
                args.dataset_root,
                ds_id,
                args.camera_keys,
                image_size=args.image_size,
                contact_filter=args.contact_filter,
                contact_std_threshold=args.contact_std_threshold,
                noncontact_keep_ratio=args.noncontact_keep_ratio,
                contact_seed=args.contact_seed,
                tolerance_s=args.tolerance_s,
                video_backend=args.video_backend,
                num_workers=args.num_workers,
            )

    total_time = time.time() - start_time
    print("Data processing time {}".format(
        str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
