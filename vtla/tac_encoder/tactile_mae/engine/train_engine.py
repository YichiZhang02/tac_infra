"""One-epoch MAE training. Image-only path, faithful to AnyTouch stage1_engine."""
import datetime
import math
import sys
import time

import torch
from tqdm import tqdm

from . import misc
from . import lr_sched


def train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                    args=None, start_time=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    n_iter = len(data_loader)
    total_epochs = args.epochs
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "float16") == "bfloat16" else torch.float16
    if start_time is None:
        start_time = time.time()

    # Single tqdm bar per epoch (main process only). The bar's own ETA is for the
    # current epoch; we also surface a whole-run ETA via the postfix.
    pbar = None
    if misc.is_main_process():
        pbar = tqdm(total=n_iter, desc=f"Epoch {epoch + 1}/{total_epochs}",
                    dynamic_ncols=True, leave=True)

    for data_iter_step, (samples, sensors) in enumerate(data_loader):

        # per-iteration (not per-epoch) lr schedule
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / n_iter + epoch, args)

        samples = samples.to(device, non_blocking=True)
        sensors = sensors.to(device, non_blocking=True).long()

        if args.sensor_token_for_all:
            now = data_iter_step / n_iter + epoch
            sensor_p = args.beta_start + (args.beta_end - args.beta_start) * (now / (args.epochs * 1.0))
            bern = torch.bernoulli(torch.full(sensors.shape, sensor_p, device=device))
            sensors = (sensors * (1 - bern) - bern).long()

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            loss, _, _ = model(samples, sensor_type=sensors)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if pbar is not None:
            # whole-run ETA: progress across *all* epochs, not just this one
            done = epoch * n_iter + data_iter_step + 1
            total = total_epochs * n_iter
            elapsed = time.time() - start_time
            eta_total = elapsed / done * (total - done)
            pbar.set_postfix(loss=f"{metric_logger.meters['loss'].avg:.4f}",
                             lr=f"{lr:.2e}",
                             eta_all=str(datetime.timedelta(seconds=int(eta_total))))
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    metric_logger.synchronize_between_processes()
    print(f"[epoch {epoch}] averaged loss={metric_logger.meters['loss'].global_avg:.4f} "
          f"lr={metric_logger.meters['lr'].value:.2e}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
