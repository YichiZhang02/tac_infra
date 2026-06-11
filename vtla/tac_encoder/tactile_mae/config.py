import argparse

from .data.lerobot_tactile_dataset import DEFAULT_CAMERA_KEYS


def get_args_parser():
    parser = argparse.ArgumentParser("Tactile MAE pretraining", add_help=True)

    # ----- model -----
    parser.add_argument("--arch", default="vit_l", choices=["vit_b", "vit_l"])
    parser.add_argument("--pretrained_path", default="",
                        help="Unified init source: empty=scratch | HF CLIP dir | AnyTouch ckpt(.pth)/converted dir")
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--visible_loss_weight", type=float, default=0.0,
                        help="lambda for visible-patch loss: loss = loss_masked + lambda*loss_visible "
                             "(0 = standard MAE, masked-only)")
    parser.add_argument("--use_sensor_token", action="store_true", default=True)
    parser.add_argument("--no_sensor_token", dest="use_sensor_token", action="store_false")
    parser.add_argument("--use_same_patchemb", action="store_true", default=True,
                        help="Route images through the 3D video patch-embed (stage1 default)")
    parser.add_argument("--no_same_patchemb", dest="use_same_patchemb", action="store_false")
    parser.add_argument("--sensor_token_for_all", action="store_true",
                        help="Progressively replace sensor id with -1 (agnostic) during training")
    parser.add_argument("--beta_start", type=float, default=0.0)
    parser.add_argument("--beta_end", type=float, default=0.75)

    # ----- data -----
    parser.add_argument("--dataset_root", default="playground/data")
    parser.add_argument("--dataset_ids", nargs="+", required=True,
                        help="One or more LeRobot dataset folder names under dataset_root")
    parser.add_argument("--camera_keys", nargs="+", default=list(DEFAULT_CAMERA_KEYS))
    parser.add_argument("--sensor_id", type=int, default=-1,
                        help="Sensor-token slot for our tactile frames (-1 = agnostic; 3 = gelsight)")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Fraction of episodes (per dataset) held out for eval")
    parser.add_argument("--tolerance_s", type=float, default=0.1)
    parser.add_argument("--video_backend", default="pyav")

    # ----- contact-frame filtering (per-channel std on 0-255 scale) -----
    parser.add_argument("--contact_filter", action="store_true",
                        help="Train only on contact frames; subsample non-contact frames")
    parser.add_argument("--contact_std_threshold", type=float, default=0.5,
                        help="per-channel std > threshold => contact frame")
    parser.add_argument("--noncontact_keep_ratio", type=float, default=0.05,
                        help="Fraction of non-contact frames to keep (drop the rest)")
    parser.add_argument("--contact_seed", type=int, default=0)
    parser.add_argument("--contact_stride", type=int, default=1,
                        help="When building the contact-std cache, decode/score every "
                             "N-th frame per episode and nearest-fill the gaps (faster build)")

    # ----- decode-once frame cache (uint8 memmap of kept, pre-resized frames) -----
    parser.add_argument("--frame_cache", action="store_true", default=True,
                        help="Dump kept frames to a uint8 memmap once; train reads it decode-free")
    parser.add_argument("--no_frame_cache", dest="frame_cache", action="store_false")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Cached/training image size (frames are pre-resized to this)")

    # ----- optim -----
    parser.add_argument("--batch_size", type=int, default=64, help="per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--accum_iter", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=None, help="absolute lr (overrides blr)")
    parser.add_argument("--blr", type=float, default=1e-3, help="base lr; lr = blr*eff_bs/256")
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="autocast dtype; bfloat16 avoids fp16 overflow->NaN on A100 "
                             "(GradScaler is auto-disabled for bfloat16)")

    # ----- runtime -----
    parser.add_argument("--output_dir", default="playground/results/tactile_mae")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader batches prefetched per worker")
    parser.add_argument("--resume", default="", help="resume training from a .pth checkpoint")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=5, help="save checkpoint every N epochs")
    parser.add_argument("--eval_freq", type=int, default=5,
                        help="run val reconstruction eval every N epochs (0=off)")
    parser.add_argument("--eval_max_batches", type=int, default=50,
                        help="cap number of val batches per eval (None=full val)")
    parser.add_argument("--vis_per_level", type=int, default=3,
                        help="reconstruction grid rows per contact level (low/mid/high std)")

    # ----- distributed (consumed by engine.misc) -----
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser
