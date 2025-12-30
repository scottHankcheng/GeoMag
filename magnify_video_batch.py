#!/usr/bin/env python3


import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

try:
    from models.model import GeoMag
except ImportError as e:
    print(f"Error importing GeoMag: {e}")
    print("Please ensure models/model.py is accessible.")
    sys.exit(1)

try:
    from utils.data_loader_augmentation import ImageFromFolderTest
except ImportError as e:
    print(f"Error importing ImageFromFolderTest: {e}")
    print("Please ensure utils/data_loader_augmentation.py is accessible.")
    sys.exit(1)

def auto_pad(img, d=64, fill_colour=(0, 0, 0)):
    """Pad image to make it divisible by d"""
    from PIL import Image
    x, y = img.size
    if x % d != 0:
        x = (x // d + 1) * d
    if y % d != 0:
        y = (y // d + 1) * d
    
    new_img = Image.new('RGB', (x, y), fill_colour)
    new_img.paste(img, ((x - img.size[0]) // 2, (y - img.size[1]) // 2))
    return new_img


def detect_video_fps(video_path: Path) -> int:
    """使用ffprobe检测视频帧率"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        detected_fps = result.stdout.strip()
        if detected_fps and detected_fps != "0/0":
            if "/" in detected_fps:
                numerator, denominator = detected_fps.split("/")
                if denominator and float(denominator) != 0:
                    fps = float(numerator) / float(denominator)
                    return int(round(fps))
            else:
                return int(float(detected_fps))
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
        print(f"[Warning] Failed to detect FPS from video: {e}", file=sys.stderr)
    
    print(f"[Warning] Using default FPS: 60", file=sys.stderr)
    return 60


def extract_frames(video_path: Path, output_dir: Path) -> Tuple[int, Tuple[int, int]]:
    """从视频中提取帧"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用ffmpeg提取帧
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        f"{output_dir}/frame_%06d.png",
        "-hide_banner",
        "-loglevel", "error"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to extract frames: {result.stderr}")
    
    # 计算帧数
    frame_files = sorted(output_dir.glob("frame_*.png"))
    num_frames = len(frame_files)
    
    if num_frames == 0:
        raise RuntimeError("No frames extracted from video")
    
    # 获取第一帧的尺寸
    first_frame = Image.open(frame_files[0])
    size = (first_frame.width, first_frame.height)
    
    return num_frames, size


def pad_frames(input_dir: Path, output_dir: Path, divisor: int = 8, max_workers: int = 16):
    """对帧进行补边处理"""
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted(input_dir.glob("frame_*.png"))
    
    def pad_single_frame(frame_path: Path):
        try:
            img = Image.open(frame_path)
            padded_img = auto_pad(img, d=divisor, fill_colour=(0, 0, 0))
            output_path = output_dir / frame_path.name
            padded_img.save(output_path)
            return True
        except Exception as e:
            print(f"Error padding frame {frame_path}: {e}", file=sys.stderr)
            return False
    
    # 使用多进程处理
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(pad_single_frame, frame_files))
    
    success_count = sum(results)
    if success_count != len(frame_files):
        raise RuntimeError(f"Failed to pad {len(frame_files) - success_count} frames")
    
    return len(frame_files)


def run_model_inference(
    frame_dir: Path,
    model_path: Path,
    output_dir: Path,
    num_frames: int,
    mag_factor: float,
    mode: str,
    device: str,
    batch_size: int = 1,
    workers: int = 16,
    print_freq: int = 100
) -> None:
    """运行模型推理"""
    device_obj = torch.device('cuda' if (device == 'cuda' or device == 'auto') and torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device_obj}')

    # 加载模型
    model = GeoMag(
        img_size=384, patch_size=1, in_chans=3,
        embed_dim=192, depth=12, mlp_ratio=2.0,
        d_state=16, d_conv=4, expand=2,
        manipulator_num_resblk=1, use_checkpoint=False, img_range=1.
    ).to(device_obj)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"=> 加载检查点 '{model_path}'")
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    
    incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
    if incompatible_keys.missing_keys:
        print(f"警告: 缺少的键: {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"警告: 意外的键: {incompatible_keys.unexpected_keys}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"结果将保存至: {output_dir}")

    # 创建数据集和数据加载器
    dataset_mag = ImageFromFolderTest(
        str(frame_dir / "frame"), 
        mag=mag_factor, 
        mode=mode, 
        num_data=num_frames - 1, 
        preprocessing=False
    )
    data_loader = data.DataLoader(
        dataset_mag,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False
    )

    model.eval()

    with torch.no_grad():
        for i, (xa, xb, mag_factor_tensor) in enumerate(data_loader):
            if i % print_freq == 0:
                print(f'处理样本: {i}/{num_frames - 1}')

            mag_factor_tensor = mag_factor_tensor.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            xa = xa.to(device_obj)
            xb = xb.to(device_obj)
            mag_factor_tensor = mag_factor_tensor.to(device_obj)

            y_hat, _, _, _ = model(xa, xb, mag_factor_tensor)

            # 保存第一帧（xa）
            if i == 0:
                tmp = xa.permute(0, 2, 3, 1).cpu().detach().numpy()
                tmp = np.clip(tmp, -1.0, 1.0)
                tmp = ((tmp + 1.0) * 127.5).astype(np.uint8)
                
                fn = output_dir / f'VimVMM_{mode}_000000.png'
                im = Image.fromarray(np.concatenate(tmp, 0))
                im.save(fn)

            # 保存放大后的帧
            y_hat_np = y_hat.permute(0, 2, 3, 1).cpu().detach().numpy()
            y_hat_np = np.clip(y_hat_np, -1.0, 1.0)
            y_hat_np = ((y_hat_np + 1.0) * 127.5).astype(np.uint8)
            
            fn = output_dir / f'VimVMM_{mode}_{i+1:06d}.png'
            im = Image.fromarray(np.concatenate(y_hat_np, 0))
            im.save(fn)


def combine_frames_to_video(
    frame_dir: Path,
    output_video_path: Path,
    fps: int,
    mode: str
) -> None:
    """将帧合成为视频"""
    # 查找模式匹配的帧文件
    frame_pattern = frame_dir / f"VimVMM_{mode}_%06d.png"
    cmd = [
        "ffmpeg",
        "-framerate", str(fps),
        "-i", str(frame_pattern),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_video_path),
        "-hide_banner",
        "-loglevel", "error"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to combine frames: {result.stderr}")


def process_single_video(
    input_video: Path,
    model_path: Path,
    output_dir: Path,
    output_prefix: str,
    mag_factor: float,
    mode: str,
    fps: Optional[int],
    device: str,
    batch_size: int,
    workers: int,
    pad_divisor: int,
    cleanup: bool = True
) -> Tuple[bool, Optional[float]]:
    """处理单个视频的完整流程"""
    print(f"\n{'='*60}")
    print(f"处理视频: {input_video.name}")
    print(f"放大倍数: {mag_factor}x, 模式: {mode}")
    print(f"{'='*60}")
    
    # 检测FPS
    if fps is None:
        fps = detect_video_fps(input_video)
        print(f"检测到的FPS: {fps}")
    else:
        print(f"使用指定的FPS: {fps}")
    
    # 创建临时目录
    temp_base = Path(tempfile.gettempdir()) / f"magnify_{output_prefix}_x{int(mag_factor)}"
    original_frames_dir = temp_base / "original_frames"
    padded_frames_dir = temp_base / "padded_frames"
    magnified_frames_dir = temp_base / "magnified_frames"
    
    inference_start_time = None
    inference_end_time = None
    
    try:
        # 步骤1: 提取帧
        print("\n[步骤 1/4] 提取视频帧...")
        num_frames, size = extract_frames(input_video, original_frames_dir)
        print(f"提取了 {num_frames} 帧，尺寸: {size}")
        
        # 步骤2: 补边
        print("\n[步骤 2/4] 对帧进行补边...")
        shutil.copytree(original_frames_dir, padded_frames_dir, dirs_exist_ok=True)
        pad_frames(padded_frames_dir, padded_frames_dir, divisor=pad_divisor)
        print("补边完成")
        
        # 步骤3: 模型推理
        print(f"\n[步骤 3/4] 运行模型推理 ({num_frames - 1} 帧)...")
        inference_start_time = time.time()
        run_model_inference(
            padded_frames_dir,
            model_path,
            magnified_frames_dir,
            num_frames - 1,
            mag_factor,
            mode,
            device,
            batch_size,
            workers
        )
        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        print(f"模型推理完成，耗时: {inference_duration:.2f} 秒")
        print(f"[TIMESTAMP_INFERENCE_TOTAL] {inference_duration:.6f}")
        
        # 步骤4: 合成视频
        print("\n[步骤 4/4] 合成最终视频...")
        output_filename = f"{output_prefix}_x{int(mag_factor)}_{mode}_output.mp4"
        output_video_path = output_dir / output_filename
        combine_frames_to_video(magnified_frames_dir, output_video_path, fps, mode)
        print(f"视频已保存至: {output_video_path}")
        
        return True, inference_duration
        
    except Exception as e:
        print(f"\n错误: 处理视频时发生异常: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False, None
        
    finally:
        # 清理临时文件
        if cleanup and temp_base.exists():
            print("\n清理临时文件...")
            shutil.rmtree(temp_base, ignore_errors=True)
            print("清理完成")


CONFIG_DEFAULTS = {
    "MODEL": "checkpoints/model.pth",
    "OUTPUT_DIR": "./output",
    "MODE": "static",
    "MAGNIFICATIONS": [10.0],
    "DEVICE": "auto",
    "BATCH_SIZE": 1,
    "WORKERS": 16,
    "PAD_DIVISOR": 8,
    "CONTINUE_ON_ERROR": False,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量视频运动放大处理（合并版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单个视频，多个放大倍数
  python magnify_video_batch.py -i video.avi -o output_prefix -M 10 20 30
  
  # 处理多个视频
  python magnify_video_batch.py -i video1.avi video2.avi -o prefix1 prefix2 -M 10
  
  # 自动检测FPS
  python magnify_video_batch.py -i video.avi -o output -M 10
  
  # 指定FPS
  python magnify_video_batch.py -i video.avi -o output -M 10 -f 30
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="输入视频文件路径（可指定多个）"
    )
    parser.add_argument(
        "-m", "--model",
        default=CONFIG_DEFAULTS["MODEL"],
        help=f"模型检查点路径 (默认: {CONFIG_DEFAULTS['MODEL']})"
    )
    parser.add_argument(
        "-o", "--output-prefix",
        nargs="+",
        required=True,
        help="输出文件前缀（必须与输入视频数量匹配）"
    )
    parser.add_argument(
        "-s", "--output-dir",
        default=CONFIG_DEFAULTS["OUTPUT_DIR"],
        help=f"输出目录 (默认: {CONFIG_DEFAULTS['OUTPUT_DIR']})"
    )
    parser.add_argument(
        "-f", "--fps",
        type=int,
        nargs='?',
        default=None,
        help="视频帧率（不指定则自动检测）"
    )
    parser.add_argument(
        "--mode",
        choices=["static", "dynamic"],
        nargs="+",
        default=[CONFIG_DEFAULTS["MODE"]],
        help="放大模式: static 或 dynamic（可指定多个，必须与输入视频数量匹配）"
    )
    parser.add_argument(
        "-M", "--magnifications",
        nargs="+",
        type=float,
        default=CONFIG_DEFAULTS["MAGNIFICATIONS"],
        help="放大倍数列表，例如: -M 5 10 15"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default=CONFIG_DEFAULTS["DEVICE"],
        help="计算设备 (默认: auto)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=CONFIG_DEFAULTS["BATCH_SIZE"],
        help=f"批处理大小 (默认: {CONFIG_DEFAULTS['BATCH_SIZE']})"
    )
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=CONFIG_DEFAULTS["WORKERS"],
        help=f"数据加载工作线程数 (默认: {CONFIG_DEFAULTS['WORKERS']})"
    )
    parser.add_argument(
        "--pad-divisor",
        type=int,
        default=CONFIG_DEFAULTS["PAD_DIVISOR"],
        help=f"补边除数 (默认: {CONFIG_DEFAULTS['PAD_DIVISOR']})"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=CONFIG_DEFAULTS["CONTINUE_ON_ERROR"],
        help="遇到错误时继续处理其他视频"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="保留临时文件（用于调试）"
    )
    
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """验证参数"""
    if len(args.input) != len(args.output_prefix):
        sys.exit(
            f"错误: 输入视频数量 ({len(args.input)}) 必须与输出前缀数量 ({len(args.output_prefix)}) 匹配\n"
            f"输入: {args.input}\n"
            f"输出前缀: {args.output_prefix}"
        )
    
    if len(args.mode) == 1:
        args.mode_list = args.mode * len(args.input)
    elif len(args.mode) == len(args.input):
        args.mode_list = args.mode
    else:
        sys.exit(
            f"错误: 模式数量 ({len(args.mode)}) 必须为 1（所有视频共用）或与输入视频数量 ({len(args.input)}) 匹配\n"
            f"模式: {args.mode}\n"
            f"输入: {args.input}"
        )
    
    # 验证输入文件存在
    args.input_paths = []
    for input_file in args.input:
        input_path = Path(input_file)
        if not input_path.exists():
            sys.exit(f"输入文件不存在: {input_path}")
        args.input_paths.append(input_path.resolve())
    
    # 验证模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"模型文件不存在: {model_path}")
    args.model_path = model_path
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir_path = output_dir


def main():
    args = parse_args()
    validate_args(args)
    
    print("\n" + "="*60)
    print("批量视频运动放大处理")
    print("="*60)
    print(f"模型: {args.model_path}")
    print(f"输入视频数: {len(args.input_paths)}")
    print(f"放大倍数: {args.magnifications}")
    print(f"输出目录: {args.output_dir_path}")
    print("="*60)
    
    failures: List[Tuple[str, float, str]] = []
    timing_info: List[Tuple[str, float, float]] = []
    
    for input_idx, (input_path, output_prefix, mode) in enumerate(
        zip(args.input_paths, args.output_prefix, args.mode_list)
    ):
        input_name = input_path.name
        print(f"\n处理输入 {input_idx + 1}/{len(args.input_paths)}: {input_name} (模式: {mode})")
        
        for mag in args.magnifications:
            success, inference_time = process_single_video(
                input_path,
                args.model_path,
                args.output_dir_path,
                output_prefix,
                mag,
                mode,
                args.fps,
                args.device,
                args.batch_size,
                args.workers,
                args.pad_divisor,
                cleanup=not args.no_cleanup
            )
            
            if success:
                if inference_time is not None:
                    timing_info.append((input_name, mag, inference_time))
                    print(f"[统计] 模型推理时间 ({input_name}, mag {mag}): {inference_time:.2f} 秒")
            else:
                failures.append((input_name, mag, "处理失败"))
                if not args.continue_on_error:
                    print("\n处理失败，退出。", file=sys.stderr)
                    sys.exit(1)
    
    # 打印统计信息
    if timing_info:
        print("\n" + "="*60)
        print("模型推理时间统计:")
        print("="*60)
        by_input = {}
        for input_name, mag, duration in timing_info:
            if input_name not in by_input:
                by_input[input_name] = []
            by_input[input_name].append((mag, duration))
        
        total_time = 0.0
        for input_name, timings in by_input.items():
            print(f"\n  {input_name}:")
            input_total = 0.0
            for mag, duration in timings:
                print(f"    magnification {mag}: {duration:.2f} 秒")
                input_total += duration
                total_time += duration
            if len(timings) > 1:
                print(f"    小计: {input_total:.2f} 秒")
        
        if len(by_input) > 1 or len(timing_info) > 1:
            print(f"\n  总计: {total_time:.2f} 秒")
            avg_time = total_time / len(timing_info)
            print(f"  平均: {avg_time:.2f} 秒")
    
    if failures:
        print("\n" + "="*60)
        print("处理失败的视频:")
        print("="*60)
        for input_name, mag, reason in failures:
            print(f"  {input_name} (magnification {mag}): {reason}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("所有处理完成！")
    print("="*60)


if __name__ == "__main__":
    main()

