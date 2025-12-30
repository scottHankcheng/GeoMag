import argparse
import json
import os
import time
from collections import OrderedDict
import math
import numpy as np
from typing import Optional
from glob import glob

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data as data
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast
from pytorch_msssim import SSIM
from PIL import Image
import lpips 
# import wandb

from utils.data_loader_augmentation import ImageFromFolder
from utils.avgMeter import AverageMeter
from models.model import GeoMag


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model

def setup_device_and_distributed(args):
    env_rank = os.environ.get('RANK')
    env_world_size = os.environ.get('WORLD_SIZE')
    env_local_rank = os.environ.get('LOCAL_RANK')

    if env_rank is not None:
        args.rank = int(env_rank)
    if env_world_size is not None:
        args.world_size = int(env_world_size)
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)

    args.distributed = (
        (args.world_size is not None and args.world_size > 1)
        or (args.rank is not None and args.rank not in (-1, None))
        or (args.local_rank is not None and args.local_rank not in (-1, None))
    )

    if not args.distributed:
        args.rank = 0
        args.world_size = 1
        args.local_rank = -1
        if args.device == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(args.device)
        print(f'Using device: {device}')
        return device

    if args.device == 'cpu':
        device = torch.device('cpu')
        backend = 'gloo'
    else:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA not available for distributed training.')
        if args.local_rank in (-1, None):
            args.local_rank = 0
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
        backend = args.dist_backend

    init_kwargs = dict(backend=backend, init_method=args.dist_url)
    if args.rank not in (-1, None):
        init_kwargs['rank'] = args.rank
    if args.world_size not in (-1, None):
        init_kwargs['world_size'] = args.world_size
    dist.init_process_group(**init_kwargs)
    if is_main_process():
        print(f"DDP initialized: rank={get_rank()} world_size={dist.get_world_size()} backend={backend}")
    return device

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()

def reduce_scalar(value, device):
    if not is_dist_avail_and_initialized():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor.item()

def reduce_meter(meter, device):
    if not hasattr(meter, 'sum') or not hasattr(meter, 'count'):
        return reduce_scalar(meter.avg, device)
    if not is_dist_avail_and_initialized():
        return meter.avg
    stats = torch.tensor([meter.sum, float(meter.count)], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    total, count = stats.tolist()
    if count == 0:
        return 0.0
    return total / count

def main(args):
    device = setup_device_and_distributed(args)

    # if args.use_wandb and is_main_process():
    #     wandb.init(
    #         project=args.wandb_project,
    #         name=args.wandb_name,
    #         config=vars(args),
    #         resume="allow" if args.resume else None
    #     )
    #     print(f"Wandb initialized: {wandb.run.url}")

    if is_main_process():
        print("=> Creating GeoMag model...")
    

    model = GeoMag(
        img_size=384,
        in_chans=3,
        embed_dim=192,
        depth=12,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        d_state=16,
        d_conv=4,
        expand=2,
        manipulator_num_resblk=1,
        use_checkpoint=False,  
        img_range=1.0,
    ).to(device)


    use_bf16 = args.use_amp and torch.cuda.is_bf16_supported()
    if is_main_process() and args.use_amp:
        print(f"=> Mixed Precision: Enabled ({'BFloat16' if use_bf16 else 'Float16'})")

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank] if device.type == 'cuda' else None,
            output_device=args.local_rank if device.type == 'cuda' else None,
            find_unused_parameters=args.find_unused_params,
        )
    base_model = unwrap_model(model)

    criterion_l1 = nn.L1Loss(reduction='mean').to(device)
    criterion_lpips = lpips.LPIPS(net='alex').to(device).eval()
    for param in criterion_lpips.parameters():
        param.requires_grad = False

    sample_eval_ssim = None
    if args.sample_eval_dir:
        sample_eval_ssim = SSIM(data_range=255, size_average=True, channel=3).to(device)

    start_epoch = 0
    losses_recon, losses_reg1, losses_lpips_list = [], [], []
    losses_total_dynamic, losses_total_fixed = [], []
    
    if is_main_process():
        if not os.path.exists(args.ckpt):
            os.makedirs(args.ckpt)
        print(f"Checkpoints will be saved to: {args.ckpt}")


    dataset_mag = ImageFromFolder(args.dataset, num_data=args.num_data, preprocessing=True, augmentation=True)
    sampler = None
    if args.distributed:
        sampler = DistributedSampler(dataset_mag, num_replicas=dist.get_world_size(), rank=get_rank(), shuffle=True, drop_last=False)
    
    data_loader = data.DataLoader(
        dataset_mag,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        sampler=sampler,
        drop_last=False,
        persistent_workers=True
    )

    eval_loader = None
    eval_ssim_module = None
    eval_dataset_path = args.eval_dataset if args.eval_dataset else args.dataset
    eval_num_data = args.eval_num_data if args.eval_num_data > 0 else args.num_data
    if args.eval_freq > 0 and args.eval_num_samples > 0:
        if is_main_process():
            eval_dataset = ImageFromFolder(
                eval_dataset_path,
                num_data=eval_num_data,
                preprocessing=True,
                augmentation=False
            )
            eval_loader = data.DataLoader(
                eval_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                num_workers=min(args.eval_workers, args.workers),
                pin_memory=(device.type == 'cuda'),
                drop_last=False
            )
            eval_ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
            save_dir = os.path.join(args.ckpt, args.eval_output_dir)
            os.makedirs(save_dir, exist_ok=True)
            print(f"=> Evaluation enabled (freq={args.eval_freq} epochs). Samples will be saved to {save_dir}")

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    
    def get_lr_lambda(warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return lr_lambda
    
    warmup_epochs = max(5, int(args.epochs * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(warmup_epochs, args.epochs))


    if args.resume and os.path.isfile(args.resume):
        if is_main_process():
            print("=> Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        first_key = list(checkpoint['state_dict'].keys())[0]
        state_dict = checkpoint['state_dict']
        if first_key.startswith('module.'):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict)
        if 'optimizer' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint: start_epoch = checkpoint['epoch']

    if is_main_process():
        print('===================================================================')
        print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        warmup_epochs_lpips = 10
        if epoch < warmup_epochs_lpips:
            current_lpips_weight = args.lpips_start_weight
        else:
            start_weight = args.lpips_start_weight
            end_weight = args.weight_lpips
            progress = (epoch - warmup_epochs_lpips) / max(1, args.epochs - warmup_epochs_lpips)
            current_lpips_weight = start_weight + 0.5 * (1.0 - math.cos(math.pi * progress)) * (end_weight - start_weight)
        
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process():
            print(f"Epoch {epoch}: LR = {current_lr:.6f}, LPIPS weight = {current_lpips_weight:.4f}")

        metrics = train(data_loader, model, criterion_l1, criterion_lpips, optimizer, epoch, device, args, current_lpips_weight, use_bf16)
        
        # if args.use_wandb and is_main_process():
        #     if epoch % 5 == 0:
        #         sample_images = get_sample_images(data_loader, model, device, args, num_samples=4, use_bf16=use_bf16)
        #         if sample_images: wandb.log({"sample_images": sample_images, "epoch": epoch}, commit=False)
        #     wandb.log({"epoch": epoch, "learning_rate": current_lr})

        scheduler.step()

        if args.sample_eval_dir and is_main_process():

            try:
                run_periodic_sample_eval(
                    base_model,
                    device,
                    args,
                    epoch,
                    criterion_lpips,
                    sample_eval_ssim,
                    torch.bfloat16 if use_bf16 else torch.float16,
                )
            except Exception as e:
                print(f"[Warning] Sample eval failed at epoch {epoch}: {e}")

        if eval_loader is not None and is_main_process():
            try:
                run_periodic_ckpt_eval(
                    base_model,
                    eval_loader,
                    device,
                    args,
                    epoch,
                    criterion_lpips,
                    eval_ssim_module,
                    args.use_amp,
                    torch.bfloat16 if use_bf16 else torch.float16
                )
            except Exception as e:
                print(f"[Warning] Ckpt eval failed at epoch {epoch}: {e}")
        
        if is_main_process():
            state_dict_to_save = {k: v.cpu() for k, v in base_model.state_dict().items()}
            dict_checkpoint = {
                "epoch": epoch + 1,
                "state_dict": state_dict_to_save,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            fpath = os.path.join(args.ckpt, 'ckpt_e%03d.pth.tar' % (epoch + 1))
            torch.save(dict_checkpoint, fpath)
            print(f"Checkpoint saved to {fpath}")
            
            # if args.use_wandb and (epoch + 1) % args.wandb_save_freq == 0:
            #     wandb.save(fpath)

def train(loader, model, criterion_l1, criterion_lpips, optimizer, epoch, device, args, current_lpips_weight, use_bf16):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    
    model.train()
    end = time.time()
    

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    for i, (y, xa, xb, xc, mag_factor) in enumerate(loader):
        y, xa, xb, xc = y.to(device), xa.to(device), xb.to(device), xc.to(device)
        mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)


        if getattr(args, "max_mag_factor", None) is not None and args.max_mag_factor > 0:
            mag_factor = torch.clamp(mag_factor, min=0.0, max=args.max_mag_factor)
        data_time.update(time.time() - end)

        with autocast(enabled=args.use_amp, dtype=amp_dtype):
            y_hat, _, res_b, res_c = model(xa, xb, mag_factor, xc)
            loss_recon = criterion_l1(y_hat, y)
            loss_lpips_val = criterion_lpips(y_hat, y).mean()
            
            loss_reg1 = torch.tensor(0.0, device=device)
            if res_c is not None:
                loss_reg1 = args.weight_reg1 * torch.abs(res_b - res_c).mean()
            
            loss = loss_recon + current_lpips_weight * loss_lpips_val + loss_reg1


            if torch.isnan(loss) or torch.isinf(loss):

                max_y = y.detach().abs().max()
                max_y_hat = y_hat.detach().abs().max()
                max_res_b = res_b.detach().abs().max() if res_b is not None else torch.tensor(0.0, device=device)
                max_res_c = res_c.detach().abs().max() if res_c is not None else torch.tensor(0.0, device=device)
                mf_view = mag_factor.detach().view(mag_factor.size(0), -1)
                mf_head = mf_view[:, 0].float()

                print(f"[NaN DETECTED] epoch={epoch}, iter={i}, "
                      f"max|y|={max_y.item():.6f}, "
                      f"max|y_hat|={max_y_hat.item():.6f}, "
                      f"max|res_b|={max_res_b.item():.6f}, "
                      f"max|res_c|={max_res_c.item():.6f}, "
                      f"mag_factor_head={mf_head[:8].tolist()}")
                raise RuntimeError("Loss became NaN/Inf – see printed stats for this batch.")


        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if is_main_process() and i % args.print_freq == 0:
            val_loss = loss.item()
            val_recon = loss_recon.item()
            val_lpips = loss_lpips_val.item()
            val_reg1 = loss_reg1.item()
            
            print(f'Epoch: [{epoch}][{i}/{len(loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Total {val_loss:.4f}\t'
                  f'L1 {val_recon:.4f}\t'
                  f'LPIPS {val_lpips:.4f}\t'
                  f'Reg {val_reg1:.4f}')

            # if args.use_wandb:
            #     batch_total_loss_fixed = val_recon + args.fixed_lpips_weight * val_lpips + args.fixed_reg1_weight * val_reg1
            #     wandb.log({
            #         "batch_total_loss": val_loss,
            #         "batch_loss_recon": val_recon,
            #         "batch_loss_lpips": val_lpips,
            #         "batch_loss_reg1": val_reg1,
            #         "batch_total_loss_fixed": batch_total_loss_fixed,
            #         "batch_lpips_weight": current_lpips_weight
            #     })

    return 0, 0, 0 
def get_sample_images(data_loader, model, device, args, num_samples=4, use_bf16=False):
    try:
        model.eval()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        with torch.no_grad():
            for i, (y, xa, xb, xc, mag_factor) in enumerate(data_loader):
                if i >= 1: break
                y, xa, xb, xc = y.to(device), xa.to(device), xb.to(device), xc.to(device)
                mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
                
                with autocast(enabled=args.use_amp, dtype=amp_dtype):
                    y_hat, _, _, _ = model(xa, xb, mag_factor, xc)
                
                batch_size = min(num_samples, y.size(0))
                images = []
                for j in range(batch_size):
                    y_img = np.clip(y[j].float().cpu().permute(1, 2, 0).numpy(), 0, 1)
                    y_hat_img = np.clip(y_hat[j].float().cpu().permute(1, 2, 0).numpy(), 0, 1)
                    xa_img = np.clip(xa[j].float().cpu().permute(1, 2, 0).numpy(), 0, 1)
                    xb_img = np.clip(xb[j].float().cpu().permute(1, 2, 0).numpy(), 0, 1)
                    xc_img = np.clip(xc[j].float().cpu().permute(1, 2, 0).numpy(), 0, 1)
                    
                    row1 = np.concatenate([xa_img, xb_img, xc_img], axis=1)
                    row2 = np.concatenate([y_img, y_hat_img, np.abs(y_img - y_hat_img)], axis=1)
                    combined_img = np.concatenate([row1, row2], axis=0)
                    # images.append(wandb.Image(combined_img, caption=f"Sample {j}"))
                    images.append(combined_img)
                model.train()
                return images
    except Exception as e:
        print(f"Warning: Image gen failed: {e}")
        model.train()
        return None

def load_eval_image(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert('RGB')
    arr = np.array(img).astype(np.float32)
    arr = arr / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor

def tensor_to_uint255(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.clamp((tensor + 1.0) * 0.5, 0.0, 1.0)
    return (tensor * 255.0).to(torch.float32)

def run_periodic_sample_eval(
    model,
    device: torch.device,
    args,
    epoch: int,
    lpips_module,
    ssim_module: Optional[SSIM],
    amp_dtype: torch.dtype,
):
    if (epoch + 1) % args.sample_eval_freq != 0:
        return

    sample_root = args.sample_eval_dir
    if not os.path.isdir(sample_root):
        print(f"[SampleEval] 跳过，未找到目录: {sample_root}")
        return

    frameA_dir = os.path.join(sample_root, 'frameA')
    frameB_dir = os.path.join(sample_root, 'frameB')
    amplified_dir = os.path.join(sample_root, 'amplified')
    mf_path = os.path.join(sample_root, 'train_mf.txt')

    if not all(os.path.isdir(p) for p in (frameA_dir, frameB_dir, amplified_dir)) or not os.path.isfile(mf_path):
        print(f"[SampleEval] 数据结构不完整，请检查 {sample_root}")
        return

    frameA_paths = sorted(glob(os.path.join(frameA_dir, "*.png")))
    frameB_paths = sorted(glob(os.path.join(frameB_dir, "*.png")))
    amplified_paths = sorted(glob(os.path.join(amplified_dir, "*.png")))

    try:
        mf_values = np.loadtxt(mf_path)
        if np.ndim(mf_values) == 0:
            mf_values = [float(mf_values)]
        else:
            mf_values = mf_values.tolist()
    except Exception as exc:
        print(f"[SampleEval] 读取 train_mf.txt 失败: {exc}")
        return

    limit = min(
        args.sample_eval_num,
        len(frameA_paths),
        len(frameB_paths),
        len(amplified_paths),
        len(mf_values),
    )
    if limit == 0:
        print("[SampleEval] 没有足够的样本可评估")
        return

    save_root = os.path.join(args.ckpt, args.sample_eval_output_dir)
    os.makedirs(save_root, exist_ok=True)
    log_path = os.path.join(args.ckpt, 'sample_eval_log.txt')

    model_was_training = model.training
    model.eval()

    ssim_scores, lpips_scores, psnr_scores, rmse_scores = [], [], [], []

    with torch.no_grad():
        for idx in range(limit):
            xa = load_eval_image(frameA_paths[idx], device)
            xb = load_eval_image(frameB_paths[idx], device)
            gt = load_eval_image(amplified_paths[idx], device)
            mag = torch.tensor(mf_values[idx], dtype=torch.float32, device=device).view(1, 1, 1, 1)

            with autocast(enabled=args.use_amp, dtype=amp_dtype):
                pred, _, _, _ = model(xa, xb, mag, xb)

            pred = torch.clamp(pred, -1.0, 1.0)
            gt_clamped = torch.clamp(gt, -1.0, 1.0)

            output_uint = tensor_to_uint255(pred)
            gt_uint = tensor_to_uint255(gt_clamped)
            input_uint = tensor_to_uint255(xb)

            if ssim_module is not None:
                ssim_scores.append(ssim_module(output_uint, gt_uint).item())

            lpips_scores.append(lpips_module(pred, gt_clamped).item())

            mse_val = torch.mean((output_uint - gt_uint) ** 2)
            rmse_val = torch.sqrt(mse_val + 1e-8).item()
            rmse_scores.append(rmse_val)
            psnr_val = 20 * math.log10(255.0 / max(rmse_val, 1e-8))
            psnr_scores.append(psnr_val)

            concat = torch.cat([input_uint, output_uint, gt_uint], dim=3)
            array = concat.squeeze(0).permute(1, 2, 0).cpu().numpy()
            array = np.clip(array, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(
                os.path.join(save_root, f"epoch{epoch + 1:03d}_sample_{idx:02d}.png")
            )

    if model_was_training:
        model.train()

    def safe_mean(values):
        return float(np.mean(values)) if values else float('nan')

    summary = (
        f"Epoch {epoch + 1}: "
        f"SSIM={safe_mean(ssim_scores):.4f}, "
        f"LPIPS={safe_mean(lpips_scores):.4f}, "
        f"PSNR={safe_mean(psnr_scores):.2f}, "
        f"RMSE={safe_mean(rmse_scores):.4f}, "
        f"Samples={limit}"
    )
    print(f"[SampleEval] {summary}")
    with open(log_path, 'a') as f:
        f.write(summary + "\n")

def run_periodic_ckpt_eval(
    model,
    eval_loader,
    device,
    args,
    epoch,
    lpips_module,
    ssim_module,
    amp_enabled,
    amp_dtype,
):
    if eval_loader is None or (epoch + 1) % args.eval_freq != 0:
        return

    save_root = os.path.join(
        args.ckpt,
        args.eval_output_dir,
        f"epoch{epoch + 1:03d}"
    )
    os.makedirs(save_root, exist_ok=True)

    model_was_training = model.training
    model.eval()

    l1_scores, lpips_scores, ssim_scores, psnr_scores = [], [], [], []
    saved_images = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (y, xa, xb, xc, mag_factor) in enumerate(eval_loader):
            y = y.to(device)
            xa = xa.to(device)
            xb = xb.to(device)
            xc = xc.to(device)
            mag_factor = mag_factor.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                pred, _, _, _ = model(xa, xb, mag_factor, xc)

            pred = torch.clamp(pred, 0.0, 1.0)
            target = torch.clamp(y, 0.0, 1.0)

            pred = pred.float()
            target = target.float()

            l1_val = torch.mean(torch.abs(pred - target)).item()
            l1_scores.append(l1_val)

            lpips_val = lpips_module(pred, target).mean().item()
            lpips_scores.append(lpips_val)

            if ssim_module is not None:
                ssim_scores.append(ssim_module(pred, target).item())

            mse_val = torch.mean((pred - target) ** 2).item()
            mse_val = max(mse_val, 1e-8)
            psnr_scores.append(20.0 * math.log10(1.0 / math.sqrt(mse_val)))

            batch_size = target.size(0)
            total_samples += batch_size

            if saved_images < args.eval_num_samples:
                num_to_save = min(batch_size, args.eval_num_samples - saved_images)
                save_eval_grids(
                    xa[:num_to_save],
                    xb[:num_to_save],
                    target[:num_to_save],
                    pred[:num_to_save],
                    save_root,
                    start_index=saved_images
                )
                saved_images += num_to_save

            if args.eval_max_batches > 0 and (batch_idx + 1) >= args.eval_max_batches:
                break

    if model_was_training:
        model.train()

    def safe_mean(vals):
        return float(np.mean(vals)) if vals else float('nan')

    summary = {
        "epoch": epoch + 1,
        "samples": total_samples,
        "l1": safe_mean(l1_scores),
        "lpips": safe_mean(lpips_scores),
        "ssim": safe_mean(ssim_scores),
        "psnr": safe_mean(psnr_scores)
    }

    metrics_path = os.path.join(save_root, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[Eval@Epoch {epoch + 1}] L1={summary['l1']:.4f} LPIPS={summary['lpips']:.4f} "
          f"SSIM={summary['ssim']:.4f} PSNR={summary['psnr']:.2f} Saved to {save_root}")

def save_eval_grids(xa, xb, target, pred, save_root, start_index=0):
    xa = xa.detach().cpu()
    xb = xb.detach().cpu()
    target = target.detach().cpu()
    pred = pred.detach().cpu()

    batch = target.size(0)
    for idx in range(batch):
        xa_img = tensor_to_uint_image(xa[idx])
        xb_img = tensor_to_uint_image(xb[idx])
        target_img = tensor_to_uint_image(target[idx])
        pred_img = tensor_to_uint_image(pred[idx])
        diff_img = np.abs(target_img.astype(np.float32) - pred_img.astype(np.float32)).astype(np.uint8)

        blank_img = np.zeros_like(xa_img)
        row1 = np.concatenate([xa_img, xb_img, blank_img], axis=1)
        row2 = np.concatenate([target_img, pred_img, diff_img], axis=1)
        combined = np.concatenate([row1, row2], axis=0)

        Image.fromarray(combined).save(
            os.path.join(save_root, f"sample_{start_index + idx:02d}.png")
        )

def tensor_to_uint_image(tensor):
    array = tensor.clamp(0, 1).float().permute(1, 2, 0).numpy()
    return (array * 255.0).astype(np.uint8)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vision Mamba Optimized Training')
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('-lr', '--learning_rate', default=1e-5, type=float)
    parser.add_argument('-wd', '--weight_decay', default=0.0, type=float)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-n', '--num_data', type=int, required=True)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--ckpt', default='ckpt_optimized', type=str)
    parser.add_argument('-p', '--print_freq', default=100, type=int)
    parser.add_argument('--device', default='auto', type=str, choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--dist-url', default='env://', type=str)
    parser.add_argument('--world_size', default=-1, type=int)
    parser.add_argument('--rank', default=-1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--weight_reg1', default=0.15, type=float)
    parser.add_argument('--weight_lpips', default=0.35, type=float)
    parser.add_argument('--lpips_start_weight', default=0.1, type=float)
    parser.add_argument('--fixed_lpips_weight', default=1.0, type=float)
    parser.add_argument('--fixed_reg1_weight', default=0.15, type=float)
    parser.add_argument('--max_mag_factor', default=40.0, type=float,
                        help='Clamp upper bound for motion magnification factor during training (<=0 disables clamping)')
    
    # parser.add_argument('--use_wandb', action='store_true')
    # parser.add_argument('--wandb_project', default='vim-vmm-training', type=str)
    # parser.add_argument('--wandb_name', default=None, type=str)
    # parser.add_argument('--wandb_save_freq', default=10, type=int)
    
    parser.add_argument('--use_amp', action='store_true')

    parser.add_argument('--sample_eval_dir', default='', type=str, help='Path to sample dataset with frameA/frameB/amplified/train_mf.txt')
    parser.add_argument('--sample_eval_freq', default=5, type=int, help='Epoch interval for periodic evaluation')
    parser.add_argument('--sample_eval_num', default=10, type=int, help='Number of samples per evaluation run')
    parser.add_argument('--sample_eval_output_dir', default='sample_eval_viz', type=str, help='Directory (under ckpt) to store evaluation visualizations')

    # Lightweight periodic eval on training/eval dataset
    parser.add_argument('--eval_dataset', default='', type=str, help='Optional dataset path for periodic checkpoint evaluation (defaults to training dataset)')
    parser.add_argument('--eval_num_data', default=64, type=int, help='Number of samples to load from eval dataset (<=0 uses training num_data)')
    parser.add_argument('--eval_batch_size', default=8, type=int, help='Batch size for periodic evaluation loader')
    parser.add_argument('--eval_workers', default=4, type=int, help='Number of workers for eval dataloader')
    parser.add_argument('--eval_num_samples', default=4, type=int, help='How many samples to save as images every eval run')
    parser.add_argument('--eval_freq', default=5, type=int, help='Run evaluation every N epochs (0 disables)')
    parser.add_argument('--eval_max_batches', default=2, type=int, help='Max eval batches per run (<=0 means all)')
    parser.add_argument('--eval_output_dir', default='eval_samples', type=str, help='Subdirectory under ckpt to store eval images/metrics')

    args = parser.parse_args()
    try:
        main(args)
    finally:
        cleanup_distributed()