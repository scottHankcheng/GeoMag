#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import datetime
import cv2
import json
import math
import argparse
import random
import numpy as np
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def _odd_ksize_from_val(v: float) -> int:
    k = int(max(1, round(float(v))))
    if k % 2 == 0:
        k += 1
    return k
def _srgb_to_linear(img_float_0_1: np.ndarray) -> np.ndarray:
    a = 0.055
    threshold = 0.04045
    below = img_float_0_1 <= threshold
    above = ~below
    out = np.empty_like(img_float_0_1, dtype=np.float32)
    out[below] = img_float_0_1[below] / 12.92
    out[above] = ((img_float_0_1[above] + a) / (1 + a)) ** 2.4
    return out
def _linear_to_srgb(img_float_0_1: np.ndarray) -> np.ndarray:
    a = 0.055
    threshold = 0.0031308
    below = img_float_0_1 <= threshold
    above = ~below
    out = np.empty_like(img_float_0_1, dtype=np.float32)
    out[below] = img_float_0_1[below] * 12.92
    out[above] = (1 + a) * (img_float_0_1[above] ** (1 / 2.4)) - a
    return out
def _gaussian_blur_linear_srgb_uint8(img_uint8: np.ndarray, sigma_px: float) -> np.ndarray:
    
    if sigma_px is None or float(sigma_px) <= 0:
        return img_uint8
    k = _odd_ksize_from_val(max(3, int(math.ceil(float(sigma_px) * 4))))
    img_lin = _srgb_to_linear(np.clip(img_uint8.astype(np.float32) / 255.0, 0.0, 1.0))
    img_lin = cv2.GaussianBlur(img_lin, (k, k), sigmaX=float(sigma_px), borderType=cv2.BORDER_REFLECT_101)
    img_srgb = _linear_to_srgb(np.clip(img_lin, 0.0, 1.0))
    return np.clip(np.rint(img_srgb * 255.0), 0, 255).astype(np.uint8)
def _gaussian_blur_mask_uint8(mask_uint8: np.ndarray, sigma_px: float) -> np.ndarray:
    
    if sigma_px is None or float(sigma_px) <= 0:
        if mask_uint8.dtype != np.uint8:
            return mask_uint8.astype(np.float32)
        return (mask_uint8.astype(np.float32) / 255.0)
    k = _odd_ksize_from_val(max(3, int(math.ceil(float(sigma_px) * 4))))
    m = np.clip((mask_uint8.astype(np.float32) / 255.0) if mask_uint8.dtype == np.uint8 else mask_uint8.astype(np.float32), 0.0, 1.0)
    m = cv2.GaussianBlur(m, (k, k), sigmaX=float(sigma_px), borderType=cv2.BORDER_REFLECT_101)
    return np.clip(m, 0.0, 1.0).astype(np.float32)
def _gaussian_blur_linear_float(img_lin_float: np.ndarray, sigma_px: float) -> np.ndarray:
    
    if sigma_px is None or float(sigma_px) <= 0:
        return img_lin_float
    k = _odd_ksize_from_val(max(3, int(math.ceil(float(sigma_px) * 4))))
    out = cv2.GaussianBlur(img_lin_float, (k, k), sigmaX=float(sigma_px), borderType=cv2.BORDER_REFLECT_101)
    return np.clip(out.astype(np.float32), 0.0, 1.0)
def _get_obj_srgb(obj_path: str, fg_blur_sigma: float):
    
    obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    if obj_img is None:
        return None, None
    if obj_img.ndim == 3 and obj_img.shape[-1] == 4:
        img = obj_img[:, :, :3].astype(np.float32) / 255.0
        alpha = obj_img[:, :, 3].astype(np.float32) / 255.0
        if fg_blur_sigma and float(fg_blur_sigma) > 0:
            img = cv2.GaussianBlur(img, (0, 0), float(fg_blur_sigma))
            alpha = cv2.GaussianBlur(alpha, (0, 0), float(fg_blur_sigma))
            alpha = np.clip(alpha, 0.0, 1.0)
    else:
        img = obj_img.astype(np.float32) / 255.0
        alpha = np.ones(img.shape[:2], dtype=np.float32)
    return img, alpha
def _make_soft_mask_from_binary(binary_mask_uint8: np.ndarray, soften_px: float) -> np.ndarray:
    
    if soften_px <= 0:
        return (binary_mask_uint8.astype(np.float32) / 255.0)
    bin01 = (binary_mask_uint8 > 127).astype(np.uint8)
    dist_in = cv2.distanceTransform(bin01, cv2.DIST_L2, 3)
    bin01_inv = (bin01 == 0).astype(np.uint8)
    dist_out = cv2.distanceTransform(bin01_inv, cv2.DIST_L2, 3)
    signed = dist_in.astype(np.float32) - dist_out.astype(np.float32)
    mask = np.clip(0.5 + signed / max(1e-6, soften_px), 0.0, 1.0).astype(np.float32)
    return mask
def _add_poisson_noise_linear(img_linear_float: np.ndarray, strength: float) -> np.ndarray:
    
    if strength <= 0:
        return img_linear_float
    scale = 255.0 * strength
    scaled = img_linear_float * scale
    noisy = np.random.poisson(np.clip(scaled, 0, None)).astype(np.float32)
    return np.clip(noisy / scale, 0.0, 1.0)
def _add_quantization_noise(img_uint8: np.ndarray) -> np.ndarray:
    
    noise = np.random.uniform(-0.5, 0.5, img_uint8.shape).astype(np.float32)
    return np.clip(img_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
def _sample_smooth_curve(length: int, num_ctrl: int = 16) -> np.ndarray:
    
    if length <= 1:
        return np.zeros(length, dtype=np.float32)
    num_ctrl = max(2, int(num_ctrl))
    xs = np.linspace(0, length - 1, num_ctrl)
    ys = np.random.randn(num_ctrl).astype(np.float32)
    curve = np.interp(np.arange(length, dtype=np.float32), xs, ys)
    curve -= np.mean(curve)
    max_abs = np.max(np.abs(curve))
    if max_abs < 1e-6:
        return np.zeros(length, dtype=np.float32)
    return (curve / max_abs).astype(np.float32)
def _sample_smooth_field(height: int, width: int, grid: int = 32) -> np.ndarray:
    
    gh = max(2, int(round(height / max(2, grid))))
    gw = max(2, int(round(width / max(2, grid))))
    base = np.random.randn(gh, gw).astype(np.float32)
    field = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    field -= np.mean(field)
    max_abs = np.max(np.abs(field))
    if max_abs < 1e-6:
        return np.zeros_like(field, dtype=np.float32)
    return (field / max_abs).astype(np.float32)
def _sample_temporal_curve(num_frames: int, jitter: float) -> np.ndarray:
    
    if num_frames <= 0 or jitter <= 0:
        return np.zeros(max(0, num_frames), dtype=np.float32)
    steps = np.random.randn(num_frames).astype(np.float32) * float(jitter)
    curve = np.cumsum(steps)
    curve -= np.mean(curve)
    max_abs = np.max(np.abs(curve))
    if max_abs < 1e-6:
        return np.zeros(num_frames, dtype=np.float32)
    return (curve / max_abs).astype(np.float32)
def _apply_input_flicker_artifacts(frames_linear: list,
                                   row_strength: float,
                                   col_strength: float,
                                   lowfreq_strength: float,
                                   global_gain_strength: float,
                                   global_bias_strength: float,
                                   temporal_jitter: float,
                                   grid_scale: float) -> list:
    
    if not frames_linear:
        return frames_linear
    frames = [np.clip(f.astype(np.float32), 0.0, 1.0) for f in frames_linear]
    H, W = frames[0].shape[:2]
    num_frames = len(frames)
    ctrl = max(4, int(min(H, W) / 48))
    row_pattern = _sample_smooth_curve(H, ctrl).reshape(H, 1, 1)
    col_pattern = _sample_smooth_curve(W, ctrl).reshape(1, W, 1)
    lowfreq_grid = max(8, int(min(H, W) / max(2.0, float(grid_scale))))
    lowfreq_field = _sample_smooth_field(H, W, lowfreq_grid)[..., None]
    row_curve = _sample_temporal_curve(num_frames, temporal_jitter)
    col_curve = _sample_temporal_curve(num_frames, temporal_jitter)
    gain_curve = _sample_temporal_curve(num_frames, temporal_jitter)
    bias_curve = _sample_temporal_curve(num_frames, temporal_jitter)
    lf_curve = _sample_temporal_curve(num_frames, temporal_jitter)
    for idx in range(num_frames):
        frame = frames[idx]
        if row_strength > 0:
            frame += row_pattern * (row_curve[idx] * row_strength)
        if col_strength > 0:
            frame += col_pattern * (col_curve[idx] * col_strength)
        if lowfreq_strength > 0:
            frame += lowfreq_field * (lf_curve[idx] * lowfreq_strength)
        if global_gain_strength > 0:
            frame *= (1.0 + gain_curve[idx] * global_gain_strength)
        if global_bias_strength > 0:
            frame += bias_curve[idx] * global_bias_strength
        frames[idx] = np.clip(frame.astype(np.float32), 0.0, 1.0)
    return frames
def paste_object_linear_blend(canvas_lin_float: np.ndarray,
                              obj_img_lin_float: np.ndarray,
                              obj_mask_float_0_1: np.ndarray,
                              position: tuple,
                              feather_gauss_px: float,
                              use_dt_feather: bool,
                              dt_soften_px: float) -> np.ndarray:
    
    h, w = obj_mask_float_0_1.shape
    x, y = position
    H, W = canvas_lin_float.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return canvas_lin_float.copy()
    ox0, oy0 = max(0, -x), max(0, -y)
    ox1, oy1 = w - max(0, (x + w) - W), h - max(0, (y + h) - H)
    obj_img_crop = obj_img_lin_float[oy0:oy1, ox0:ox1]
    obj_mask_crop = obj_mask_float_0_1[oy0:oy1, ox0:ox1]
    if use_dt_feather and dt_soften_px > 0:
        bin_m = np.clip(np.rint(obj_mask_crop * 255.0), 0, 255).astype(np.uint8)
        mask = _make_soft_mask_from_binary(bin_m, dt_soften_px)
    else:
        mask = np.clip(obj_mask_crop.astype(np.float32), 0.0, 1.0)
    if feather_gauss_px and feather_gauss_px > 0:
        k = _odd_ksize_from_val(max(1, feather_gauss_px * 4))
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=float(feather_gauss_px), borderType=cv2.BORDER_REFLECT_101)
    canvas_result = canvas_lin_float.copy()
    roi = canvas_result[y0:y1, x0:x1]
    mask_3 = np.stack([mask, mask, mask], axis=-1)
    blended_lin = roi * (1.0 - mask_3) + obj_img_crop * mask_3
    roi[:] = np.clip(blended_lin, 0.0, 1.0).astype(np.float32)
    return canvas_result
def paste_object_linear_blend_subpixel(canvas_lin_float: np.ndarray,
                                       obj_img_lin_float: np.ndarray,
                                       obj_mask_float_0_1: np.ndarray,
                                       position_xy_float: tuple,
                                       feather_gauss_px: float,
                                       use_dt_feather: bool,
                                       dt_soften_px: float) -> np.ndarray:
    
    H, W = canvas_lin_float.shape[:2]
    tx, ty = float(position_xy_float[0]), float(position_xy_float[1])
    if use_dt_feather and dt_soften_px > 0:
        bin_m = np.clip(np.rint(obj_mask_float_0_1 * 255.0), 0, 255).astype(np.uint8)
        local_mask = _make_soft_mask_from_binary(bin_m, dt_soften_px)
    else:
        local_mask = np.clip(obj_mask_float_0_1.astype(np.float32), 0.0, 1.0)
    if feather_gauss_px and feather_gauss_px > 0:
        k = _odd_ksize_from_val(max(1, feather_gauss_px * 4))
        local_mask = cv2.GaussianBlur(local_mask, (k, k), sigmaX=float(feather_gauss_px), borderType=cv2.BORDER_REFLECT_101)
    M = np.float32([[1.0, 0.0, tx], [0.0, 1.0, ty]])
    warped_img = cv2.warpAffine(obj_img_lin_float, M, (W, H), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0, 0.0))
    warped_mask = cv2.warpAffine(local_mask, M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    warped_mask = np.clip(warped_mask.astype(np.float32), 0.0, 1.0)
    canvas_result = canvas_lin_float.copy()
    mask_3 = np.stack([warped_mask, warped_mask, warped_mask], axis=-1)
    blended_lin = canvas_result * (1.0 - mask_3) + warped_img * mask_3
    canvas_result[:] = np.clip(blended_lin, 0.0, 1.0).astype(np.float32)
    return canvas_result
def rotate_center(im: np.ndarray, msk: np.ndarray, angle_deg: float, center: tuple = None):
    if abs(angle_deg) < 1e-6:
        return im.copy(), msk.copy()
    h, w = im.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rim = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    rmask = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rim, rmask
def _apply_phase_jitter_hr(img_hr_lin: np.ndarray, jx_hr: float, jy_hr: float) -> np.ndarray:
    
    if img_hr_lin is None:
        return img_hr_lin
    H, W = img_hr_lin.shape[:2]
    if abs(float(jx_hr)) < 1e-9 and abs(float(jy_hr)) < 1e-9:
        return img_hr_lin
    M = np.float32([[1.0, 0.0, float(jx_hr)], [0.0, 1.0, float(jy_hr)]])
    out = cv2.warpAffine(
        img_hr_lin,
        M,
        (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return np.clip(out.astype(np.float32), 0.0, 1.0)
def _apply_global_affine_jitter_hr(img_hr_lin: np.ndarray,
                                   rot_deg: float = 0.0,
                                   scale_x: float = 1.0,
                                   scale_y: float = 1.0) -> np.ndarray:
    
    if img_hr_lin is None:
        return img_hr_lin
    H, W = img_hr_lin.shape[:2]
    cx, cy = W * 0.5, H * 0.5
    M_rot = cv2.getRotationMatrix2D((cx, cy), float(rot_deg), 1.0)
    img_r = cv2.warpAffine(img_hr_lin, M_rot, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    M_s = np.float32([[scale_x, 0, cx * (1 - scale_x)], [0, scale_y, cy * (1 - scale_y)]])
    out = cv2.warpAffine(img_r, M_s, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(out.astype(np.float32), 0.0, 1.0)
def _apply_prequant_dither_srgb_float(img_srgb_float: np.ndarray,
                                      amp_lsb: float,
                                      prob: float) -> np.ndarray:
    
    if amp_lsb <= 0 or prob <= 0:
        return img_srgb_float
    if random.random() > max(0.0, min(1.0, float(prob))):
        return img_srgb_float
    noise = np.random.uniform(-amp_lsb, amp_lsb, img_srgb_float.shape).astype(np.float32) / 255.0
    out = np.clip(img_srgb_float + noise, 0.0, 1.0)
    return out
def finalize_frame_from_linear(high_res_lin_float: np.ndarray,
                               target_size: tuple,
                               scale_factor: int,
                               sigma_mult: float,
                               down_method: str,
                               poisson_strength: float = 0.0,
                               aa_dst_sigma_px: float = None,
                               enable_prequant_dither: bool = False,
                               prequant_amp_lsb: float = 0.5,
                               prequant_prob: float = 1.0) -> np.ndarray:
    
    img_lin = np.clip(high_res_lin_float.astype(np.float32), 0.0, 1.0)
    if aa_dst_sigma_px is not None and float(aa_dst_sigma_px) > 0:
        sigma = max(0.3, scale_factor * float(aa_dst_sigma_px))
    else:
        sigma = max(0.3, scale_factor * 0.5 * max(0.1, float(sigma_mult)))
    ksize = _odd_ksize_from_val(max(3, int(math.ceil(sigma * 4))))
    img_lin = cv2.GaussianBlur(img_lin, (ksize, ksize), sigmaX=float(sigma), borderType=cv2.BORDER_REFLECT_101)
    if poisson_strength > 0:
        img_lin = _add_poisson_noise_linear(img_lin, poisson_strength)
    interp = cv2.INTER_AREA if down_method == 'area' else cv2.INTER_LANCZOS4
    img_lin_ds = cv2.resize(img_lin, target_size, interpolation=interp)
    img_srgb = _linear_to_srgb(np.clip(img_lin_ds, 0.0, 1.0))
    if enable_prequant_dither:
        img_srgb = _apply_prequant_dither_srgb_float(
            img_srgb,
            amp_lsb=float(prequant_amp_lsb),
            prob=float(prequant_prob),
        )
    return np.clip(np.rint(img_srgb * 255.0), 0, 255).astype(np.uint8)
def _compute_adaptive_amplified_limit_px(img_size, max_px_opt, frac_opt):
    if max_px_opt is not None and max_px_opt > 0:
        return float(max_px_opt)
    frac = 0.08 if frac_opt is None else max(0.0, min(0.5, float(frac_opt)))
    return max(8.0, img_size * frac)
def _alpha_adjusted_probs(alpha, base_probs, mix_strength=0.5):
    p_trans, p_rot, p_both = base_probs
    s = max(0.0, min(1.0, float(mix_strength)))
    alpha_norm = np.tanh((alpha - 1.0) / 6.0)
    w_trans = 1.0 + s * alpha_norm
    w_rot = max(0.0, 1.0 - s * alpha_norm)
    w_both = 1.0 + 0.5 * s * (alpha_norm - 0.5)
    weighted = np.array([p_trans * w_trans, p_rot * w_rot, p_both * w_both], dtype=np.float32)
    if weighted.sum() <= 0:
        return np.array([0.3, 0.3, 0.4], dtype=np.float32)
    return weighted / weighted.sum()
def _clamp_vector_norm(dx: float, dy: float, max_norm: float):
    n = float(np.hypot(dx, dy))
    if n <= 1e-12 or n <= max_norm:
        return dx, dy, False
    scale = max_norm / n
    return dx * scale, dy * scale, True
def _transform_corners_aabb(w: int, h: int, center: tuple, angle_deg: float, translate_xy: tuple):
    # Compute rotated corners' AABB after rotation about center and translation
    cx, cy = center
    tx, ty = translate_xy
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    xs, ys = [], []
    for (x, y) in corners:
        x0 = x - cx
        y0 = y - cy
        xr = x0 * cos_a - y0 * sin_a + cx + tx
        yr = x0 * sin_a + y0 * cos_a + cy + ty
        xs.append(xr)
        ys.append(yr)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max
def generate_sample_data(sample_idx, args, bg_files, fg_files, mask_files):
    try:
        seed = (int(args.base_seed) ^ (sample_idx * 10007)) & 0x7FFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        try:
            cv2.setRNGSeed(seed)
        except Exception:
            pass
        target_size = (args.img_size, args.img_size)
        hi_size = (args.img_size * 3, args.img_size * 3)
        try:
            ratios = [float(x) for x in args.sample_type_ratios.split(':')]
            if len(ratios) != 5:
                ratios = [6.0, 1.0, 1.0, 1.0, 1.0]
            total_ratio = sum(ratios)
            ratios = [r / total_ratio for r in ratios]
        except Exception:
            ratios = [0.6, 0.1, 0.1, 0.1, 0.1]
        sample_type_rand = random.random()
        cumsum = 0.0
        sample_types = ['standard', 'blur_bg', 'bg_only', 'static', 'fg_only']
        sample_type = 'standard'
        for i, ratio in enumerate(ratios):
            cumsum += ratio
            if sample_type_rand < cumsum:
                sample_type = sample_types[i]
                break
        is_static_sample = (sample_type == 'static')
        if not fg_files:
            return None
        bg_path = random.choice(bg_files)
        bg_img = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if bg_img is None:
            return None
        bg_srgb = np.clip(bg_img.astype(np.float32) / 255.0, 0.0, 1.0)
        bg_lin = _srgb_to_linear(bg_srgb)
        bg_hi_lin = cv2.resize(bg_lin, hi_size, interpolation=cv2.INTER_LANCZOS4)
        if sample_type == 'blur_bg':
            try:
                extra_prob = max(0.0, min(1.0, float(getattr(args, 'blur_bg_extra_prob', 1.0))))
                extra_sigma = float(getattr(args, 'blur_bg_extra_sigma', 0.0))
            except Exception:
                extra_prob, extra_sigma = 1.0, 0.0
            if extra_sigma > 0 and random.random() < extra_prob:
                bg_hi_lin = _gaussian_blur_linear_float(bg_hi_lin, extra_sigma)
        bg_sigma = args.bg_noise_sigma
        bg_noise_std = args.bg_noise_std
        bg_blur_prob = getattr(args, "bg_blur_prob", 0.0)
        bg_noise_std_prob = getattr(args, "bg_noise_std_prob", 0.0)
        try:
            bg_blur_prob = float(bg_blur_prob)
        except Exception:
            bg_blur_prob = 0.0
        try:
            bg_noise_std_prob = float(bg_noise_std_prob)
        except Exception:
            bg_noise_std_prob = 0.0
        bg_blur_prob = max(0.0, min(1.0, bg_blur_prob))
        bg_noise_std_prob = max(0.0, min(1.0, bg_noise_std_prob))
        bg_blur_applied = False
        bg_noise_applied = False
        if bg_sigma > 0 and random.random() < bg_blur_prob:
            bg_hi_lin = _gaussian_blur_linear_float(bg_hi_lin, bg_sigma)
            bg_blur_applied = True
        if bg_noise_std > 0 and random.random() < bg_noise_std_prob:
            std_lin = float(bg_noise_std) / 255.0
            noise = np.random.normal(0.0, std_lin, bg_hi_lin.shape).astype(np.float32)
            bg_hi_lin = np.clip(bg_hi_lin + noise, 0.0, 1.0)
            bg_noise_applied = True
        num_foregrounds = random.randint(min(args.min_fg, len(fg_files)), min(args.max_fg, len(fg_files)))
        selected_indices = random.sample(range(len(fg_files)), num_foregrounds)
        layers = []
        for idx in selected_indices:
            fg_path, mask_path = fg_files[idx], mask_files[idx]
            fg_srgb_float, alpha_png = _get_obj_srgb(fg_path, args.fg_blur_sigma)
            m_ext = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if fg_srgb_float is None or m_ext is None:
                continue
            m_ext = (m_ext.astype(np.float32) / 255.0)
            if alpha_png is None:
                alpha_combined = m_ext
            else:
                alpha_combined = np.maximum(alpha_png.astype(np.float32), m_ext)
            try:
                m_uint8 = np.clip(np.rint(alpha_combined * 255.0), 0, 255).astype(np.uint8)
                _, bin_m = cv2.threshold(m_uint8, 10, 255, cv2.THRESH_BINARY)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_m, connectivity=8)
                if num_labels <= 1:
                    continue
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                obj_x = int(stats[largest_label, cv2.CC_STAT_LEFT])
                obj_y = int(stats[largest_label, cv2.CC_STAT_TOP])
                obj_w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
                obj_h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
                if obj_w <= 0 or obj_h <= 0:
                    continue
                if args.bbox_crop_margin and args.bbox_crop_margin > 0:
                    margin_px = float(args.bbox_crop_margin)
                    if margin_px < 1.0:
                        margin_px = min(obj_w, obj_h) * margin_px
                else:
                    margin_px = 0.0
                mx = int(max(0, math.floor(obj_x - margin_px)))
                my = int(max(0, math.floor(obj_y - margin_px)))
                mx2 = int(min(fg_srgb_float.shape[1], math.ceil(obj_x + obj_w + margin_px)))
                my2 = int(min(fg_srgb_float.shape[0], math.ceil(obj_y + obj_h + margin_px)))
                if mx2 <= mx or my2 <= my:
                    continue
                fg_srgb_float = fg_srgb_float[my:my2, mx:mx2]
                alpha_combined = alpha_combined[my:my2, mx:mx2]
                obj_x, obj_y = 0, 0
                obj_w = fg_srgb_float.shape[1]
                obj_h = fg_srgb_float.shape[0]
            except Exception:
                continue
            min_hi_size = 96
            max_hi_size = hi_size[0] // 2
            min_scale = max(
                (min_hi_size / max(1e-6, float(obj_h))) ,
                (min_hi_size / max(1e-6, float(obj_w))) 
            )
            max_scale = min(
                (max_hi_size / max(1e-6, float(obj_h))) ,
                (max_hi_size / max(1e-6, float(obj_w))) 
            )
            if min_scale > max_scale:
                continue
            max_scale = min(max_scale, 2.0)
            if min_scale < 1.0:
                s = random.uniform(1.0, min(max_scale, 2.0))
            else:
                s = random.uniform(min_scale, min(max_scale, 2.0))
            nh, nw = int(fg_srgb_float.shape[0] * s), int(fg_srgb_float.shape[1] * s)
            if nh <= 0 or nw <= 0:
                continue
            interp = cv2.INTER_LANCZOS4 if s >= 1.0 else cv2.INTER_AREA
            fg_srgb_float = cv2.resize(fg_srgb_float, (nw, nh), interpolation=interp)
            alpha_resized = cv2.resize(alpha_combined, (nw, nh), interpolation=cv2.INTER_LINEAR)
            alpha_resized = np.clip(alpha_resized, 0.0, 1.0)
            fg_lin = _srgb_to_linear(np.clip(fg_srgb_float, 0.0, 1.0))
            mean_bg, std_bg = np.mean(bg_hi_lin), np.std(bg_hi_lin)
            mean_fg, std_fg = np.mean(fg_lin), np.std(fg_lin)
            if std_fg < 1e-3:
                std_fg = 1.0
            fg_lin = (fg_lin - mean_fg) / std_fg * std_bg + mean_bg
            fg_lin = np.clip(fg_lin.astype(np.float32), 0.0, 1.0)
            fg_lin = _gaussian_blur_linear_float(fg_lin, args.fg_blur_sigma)
            if nh >= hi_size[0] or nw >= hi_size[1]:
                scale_fit_h = (hi_size[0] - 1) / max(1e-6, float(nh))
                scale_fit_w = (hi_size[1] - 1) / max(1e-6, float(nw))
                fit_s = max(1e-3, min(scale_fit_h, scale_fit_w))
                if fit_s < 0.999:
                    nw2, nh2 = max(1, int(round(nw * fit_s))), max(1, int(round(nh * fit_s)))
                    fg_lin = cv2.resize(fg_lin, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
                    alpha_resized = cv2.resize(alpha_resized, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
                    nw, nh = nw2, nh2
            if random.random() > 0.5:
                fg_lin = cv2.flip(fg_lin, 1)
                alpha_resized = cv2.flip(alpha_resized, 1)
            obj_x_s = int(round(obj_x * s))
            obj_y_s = int(round(obj_y * s))
            obj_w_s = max(1, int(round(obj_w * s)))
            obj_h_s = max(1, int(round(obj_h * s)))
            max_x = hi_size[1] - obj_w_s
            max_y = hi_size[0] - obj_h_s
            if max_x < 0 or max_y < 0:
                continue
            bbox_x = random.uniform(0.0, float(max_x))
            bbox_y = random.uniform(0.0, float(max_y))
            img_pos_x = float(bbox_x - obj_x_s)
            img_pos_y = float(bbox_y - obj_y_s)
            obj_h_scaled = float(obj_h_s)
            obj_w_scaled = float(obj_w_s)
            radius_target_px = 0.5 * float(np.hypot(obj_w_scaled, obj_h_scaled)) / 3.0
            layers.append({
                'img_lin': fg_lin,
                'mask': (np.clip(alpha_resized, 0.0, 1.0)).astype(np.float32),
                'pos': (img_pos_x, img_pos_y),
                'bbox_offset': (obj_x_s, obj_y_s),
                'bbox_size': (obj_w_s, obj_h_s),
                'radius_target_px': radius_target_px
            })
        if not layers:
            return None
        if is_static_sample or sample_type == 'bg_only':
            global_alpha = 0.0
            choice = 'trans'
            base_p = np.array([args.p_trans, args.p_rot, args.p_both], dtype=np.float32)
            if base_p.sum() <= 0:
                base_p = np.array([0.3, 0.3, 0.4], dtype=np.float32)
            adj_p = _alpha_adjusted_probs(global_alpha, base_p, mix_strength=args.alpha_prob_mix_strength)
        else:
            bucket = random.randint(0, 9)
            min_alpha_seg = 1.0 + bucket * 10.0
            max_alpha_seg = min_alpha_seg + 9.0
            global_alpha = random.uniform(min_alpha_seg, max_alpha_seg)
            base_p = np.array([args.p_trans, args.p_rot, args.p_both], dtype=np.float32)
            if base_p.sum() <= 0:
                base_p = np.array([0.3, 0.3, 0.4], dtype=np.float32)
            adj_p = _alpha_adjusted_probs(global_alpha, base_p, mix_strength=args.alpha_prob_mix_strength)
            choice = np.random.choice(['trans', 'rot', 'both'], p=adj_p)
        d_bg_hr = [0.0, 0.0]
        if sample_type == 'bg_only' or sample_type == 'standard':
            if random.random() < args.p_bg_motion:
                dx = random.uniform(-args.bg_drift_max_px, args.bg_drift_max_px)
                dy = random.uniform(-args.bg_drift_max_px, args.bg_drift_max_px)
                if abs(dx) < args.bg_drift_min_px:
                    dx = math.copysign(args.bg_drift_min_px, dx if dx != 0 else 1.0)
                if abs(dy) < args.bg_drift_min_px:                                           
                    dy = math.copysign(args.bg_drift_min_px, dy if dy != 0 else 1.0)
                d_bg_hr = [dx * 3.0, dy * 3.0]
        elif sample_type == 'fg_only' or sample_type == 'static':
            d_bg_hr = [0.0, 0.0]
        elif sample_type == 'blur_bg':
            if random.random() < args.p_bg_motion:
                dx = random.uniform(-args.bg_drift_max_px, args.bg_drift_max_px)
                dy = random.uniform(-args.bg_drift_max_px, args.bg_drift_max_px)
                if abs(dx) < args.bg_drift_min_px:
                    dx = math.copysign(args.bg_drift_min_px, dx if dx != 0 else 1.0)
                if abs(dy) < args.bg_drift_min_px:                                           
                    dy = math.copysign(args.bg_drift_min_px, dy if dy != 0 else 1.0)
                d_bg_hr = [dx * 3.0, dy * 3.0]
            else:
                d_bg_hr = [0.0, 0.0]
        I1_hr_lin = bg_hi_lin.copy()
        for layer in layers:
                src_img_a, src_mask_a = layer['img_lin'], layer['mask']
                fg_sigma_a = None
                if bool(args.fg_preblur_before_warp):
                    try:
                        if args.fg_preblur_sigma_px is not None:
                            fg_sigma_a = float(args.fg_preblur_sigma_px)
                    except Exception:
                        fg_sigma_a = None
                if fg_sigma_a is not None and fg_sigma_a > 0:
                    fg_sigma_hr_a = float(fg_sigma_a) * 3.0
                    src_img_a = _gaussian_blur_linear_float(src_img_a, fg_sigma_hr_a)
                    src_mask_a = _gaussian_blur_mask_uint8(src_mask_a, fg_sigma_hr_a)
                if bool(args.enable_subpixel_paste):
                    I1_hr_lin = paste_object_linear_blend_subpixel(
                        I1_hr_lin, src_img_a, src_mask_a, (float(layer['pos'][0]), float(layer['pos'][1])),
                        feather_gauss_px=args.feather, use_dt_feather=args.use_mask_distance_feather,
                        dt_soften_px=args.mask_dt_soften_px)
                else:
                    pos1_int = (int(layer['pos'][0]), int(layer['pos'][1]))
                    I1_hr_lin = paste_object_linear_blend(
                        I1_hr_lin, src_img_a, src_mask_a, pos1_int,
                        feather_gauss_px=args.feather, use_dt_feather=args.use_mask_distance_feather,
                        dt_soften_px=args.mask_dt_soften_px)
        bg_b_lin = cv2.warpAffine(bg_hi_lin, np.float32([[1, 0, d_bg_hr[0]], [0, 1, d_bg_hr[1]]]), hi_size, borderMode=cv2.BORDER_REFLECT_101)
        bg_c_lin = cv2.warpAffine(bg_hi_lin, np.float32([[1, 0, d_bg_hr[0] * (1 + args.step_factor)], [0, 1, d_bg_hr[1] * (1 + args.step_factor)]]), hi_size, borderMode=cv2.BORDER_REFLECT_101)
        I2_hr_lin = bg_b_lin.copy()
        I3_hr_lin = bg_c_lin.copy()
        if sample_type == 'static' or sample_type == 'bg_only':
            d_bg_hr_amplified = d_bg_hr
        else:
            d_bg_hr_amplified = [d * global_alpha for d in d_bg_hr]
        I_mag_hr_lin = cv2.warpAffine(bg_hi_lin, np.float32([[1, 0, d_bg_hr_amplified[0]], [0, 1, d_bg_hr_amplified[1]]]), hi_size, borderMode=cv2.BORDER_REFLECT_101)
        input_trans_px_list, amp_trans_px_list, rot_small_list, rot_mag_list = [], [], [], []
        fg_out_of_bounds = 0
        for layer in layers:
            img, mask, pos = layer['img_lin'], layer['mask'], layer['pos']
            if sample_type == 'static' or sample_type == 'bg_only':
                d_hr = [0.0, 0.0]
                rot_small = 0.0
                d_mag_hr = [0.0, 0.0]
                rot_mag = 0.0
                clamped_input = False
                clamped_amplified = False
            else:
                local_choice = choice
                if args.per_fg_motion:
                    local_choice = np.random.choice(['trans', 'rot', 'both'], p=adj_p)
                amplified_limit_px = _compute_adaptive_amplified_limit_px(args.img_size, args.max_amplified_motion_px, args.max_amplified_motion_frac)
                max_input_trans_target = min(args.base_translation, amplified_limit_px / max(1e-6, global_alpha))
                max_input_trans_hr = max_input_trans_target * 3
                d_hr = [random.uniform(-max_input_trans_hr, max_input_trans_hr), random.uniform(-max_input_trans_hr, max_input_trans_hr)]
                max_input_norm_hr = args.base_translation * 3.0
                d_hr[0], d_hr[1], clamped_input = _clamp_vector_norm(d_hr[0], d_hr[1], max_norm=max_input_norm_hr)
                radius_target_px = max(1e-6, float(layer.get('radius_target_px', 1.0)))
                rot_deg_limit_by_pixel = (amplified_limit_px / max(1e-6, global_alpha)) / (radius_target_px * (np.pi / 180.0))
                rot_deg_limit_amplified_cap = args.max_rot_amplified / max(1e-6, global_alpha)
                max_input_rotation = min(args.max_rot_small, rot_deg_limit_amplified_cap, rot_deg_limit_by_pixel)
                rot_small = random.uniform(-max_input_rotation, max_input_rotation)
                if local_choice == 'trans':
                    rot_small = 0.0
                elif local_choice == 'rot':
                    d_hr = [0.0, 0.0]
                d_mag_hr = [d * global_alpha for d in d_hr]
                max_amplified_norm_hr = amplified_limit_px * 3.0
                d_mag_hr[0], d_mag_hr[1], clamped_amplified = _clamp_vector_norm(d_mag_hr[0], d_mag_hr[1], max_norm=max_amplified_norm_hr)
                rot_mag = rot_small * global_alpha
                input_trans_px_list.append(np.hypot(d_hr[0], d_hr[1]) / 3.0)
                amp_trans_px_list.append(np.hypot(d_mag_hr[0], d_mag_hr[1]) / 3.0)
                rot_small_list.append(abs(rot_small))
                rot_mag_list.append(abs(rot_mag))
            try:
                m_bin = (mask > 10).astype(np.uint8)
                moments = cv2.moments(m_bin)
                if moments['m00'] > 0:
                    cx = moments['m10'] / moments['m00']
                    cy = moments['m01'] / moments['m00']
                    rot_center = (cx, cy)
                else:
                    rot_center = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
            except Exception:
                rot_center = (mask.shape[1] / 2.0, mask.shape[0] / 2.0)
            src_img, src_mask = img, mask
            fg_sigma = None
            if bool(args.fg_preblur_before_warp):
                try:
                    if args.fg_preblur_sigma_px is not None:
                        fg_sigma = float(args.fg_preblur_sigma_px)
                except Exception:
                    fg_sigma = None
            if fg_sigma is not None and fg_sigma > 0:
                fg_sigma_hr = float(fg_sigma) * 3.0
                src_img = _gaussian_blur_linear_float(src_img, fg_sigma_hr)
                src_mask = _gaussian_blur_mask_uint8(src_mask, fg_sigma_hr)
            img_r_b, m_r_b = rotate_center(src_img, src_mask, rot_small, center=rot_center)
            if bool(args.enable_subpixel_paste):
                pos2f = (float(pos[0] + d_bg_hr[0] + d_hr[0]), float(pos[1] + d_bg_hr[1] + d_hr[1]))
                I2_hr_lin = paste_object_linear_blend_subpixel(I2_hr_lin, img_r_b, m_r_b, pos2f, args.feather, args.use_mask_distance_feather, args.mask_dt_soften_px)
            else:
                pos2 = (int(pos[0] + d_bg_hr[0] + d_hr[0]), int(pos[1] + d_bg_hr[1] + d_hr[1]))
                I2_hr_lin = paste_object_linear_blend(I2_hr_lin, img_r_b, m_r_b, pos2, args.feather, args.use_mask_distance_feather, args.mask_dt_soften_px)
            img_r_c, m_r_c = rotate_center(src_img, src_mask, rot_small * (1 + args.step_factor), center=rot_center)
            if bool(args.enable_subpixel_paste):
                pos_cf = (
                    float(pos[0] + d_bg_hr[0] * (1 + args.step_factor) + d_hr[0] * (1 + args.step_factor)),
                    float(pos[1] + d_bg_hr[1] * (1 + args.step_factor) + d_hr[1] * (1 + args.step_factor))
                )
                I3_hr_lin = paste_object_linear_blend_subpixel(I3_hr_lin, img_r_c, m_r_c, pos_cf, args.feather, args.use_mask_distance_feather, args.mask_dt_soften_px)
            else:
                pos_c = (int(pos[0] + d_bg_hr[0] * (1 + args.step_factor) + d_hr[0] * (1 + args.step_factor)), int(pos[1] + d_bg_hr[1] * (1 + args.step_factor) + d_hr[1] * (1 + args.step_factor)))
                I3_hr_lin = paste_object_linear_blend(I3_hr_lin, img_r_c, m_r_c, pos_c, args.feather, args.use_mask_distance_feather, args.mask_dt_soften_px)
            img_r_m, m_r_m = rotate_center(src_img, src_mask, rot_mag, center=rot_center)
            if bool(args.enable_subpixel_paste):
                pos_mf = (float(pos[0] + d_bg_hr_amplified[0] + d_mag_hr[0]), float(pos[1] + d_bg_hr_amplified[1] + d_mag_hr[1]))
                I_mag_hr_lin = paste_object_linear_blend_subpixel(I_mag_hr_lin, img_r_m, m_r_m, pos_mf, args.feather, args.use_mask_distance_feather, args.mask_dt_soften_px)
                pos_m = (int(round(pos_mf[0])), int(round(pos_mf[1])))
            else:
                pos_m = (int(pos[0] + d_bg_hr_amplified[0] + d_mag_hr[0]), int(pos[1] + d_bg_hr_amplified[1] + d_mag_hr[1]))
                I_mag_hr_lin = paste_object_linear_blend(I_mag_hr_lin, img_r_m, m_r_m, pos_m, args.feather, args.use_mask_distance_feather, args.mask_dt_soften_px)
            h_fg, w_fg = mask.shape[:2]
            bbox_w, bbox_h = layer.get('bbox_size', (w_fg, h_fg))
            bbox_pos_m_x = pos_m[0] + layer.get('bbox_offset', (0, 0))[0]
            bbox_pos_m_y = pos_m[1] + layer.get('bbox_offset', (0, 0))[1]
            if (bbox_pos_m_x < 0 or bbox_pos_m_y < 0 or bbox_pos_m_x + bbox_w > hi_size[1] or bbox_pos_m_y + bbox_h > hi_size[0]):
                fg_out_of_bounds += 1
        jx_hr, jy_hr = 0.0, 0.0
        try:
            phase_max = getattr(args, 'phase_jitter_hr_max', 0.5)
        except Exception:
            phase_max = 0.5
        if phase_max is not None and float(phase_max) > 0.0:
            jx_hr = random.uniform(-float(phase_max), float(phase_max))
            jy_hr = random.uniform(-float(phase_max), float(phase_max))
            I1_hr_lin = _apply_phase_jitter_hr(I1_hr_lin, jx_hr, jy_hr)
            I2_hr_lin = _apply_phase_jitter_hr(I2_hr_lin, jx_hr, jy_hr)
            I3_hr_lin = _apply_phase_jitter_hr(I3_hr_lin, jx_hr, jy_hr)
            I_mag_hr_lin = _apply_phase_jitter_hr(I_mag_hr_lin, jx_hr, jy_hr)
        try:
            _deg = 0.2 * (1 if random.random() < 0.5 else -1)
            _sx = 1.0 + random.uniform(-0.005, 0.005)
            _sy = 1.0 + random.uniform(-0.005, 0.005)
        except Exception:
            _deg, _sx, _sy = 0.2, 1.0, 1.0
        I1_hr_lin = _apply_global_affine_jitter_hr(I1_hr_lin, _deg, _sx, _sy)
        I2_hr_lin = _apply_global_affine_jitter_hr(I2_hr_lin, _deg, _sx, _sy)
        I3_hr_lin = _apply_global_affine_jitter_hr(I3_hr_lin, _deg, _sx, _sy)
        I_mag_hr_lin = _apply_global_affine_jitter_hr(I_mag_hr_lin, _deg, _sx, _sy)
        flicker_debug = None
        if bool(args.enable_input_flicker):
            flicker_frames = _apply_input_flicker_artifacts(
                [I1_hr_lin, I2_hr_lin, I3_hr_lin],
                row_strength=max(0.0, float(args.flicker_row_strength)),
                col_strength=max(0.0, float(args.flicker_col_strength)),
                lowfreq_strength=max(0.0, float(args.flicker_lowfreq_strength)),
                global_gain_strength=max(0.0, float(args.flicker_global_gain_strength)),
                global_bias_strength=max(0.0, float(args.flicker_global_bias_strength)),
                temporal_jitter=max(0.0, float(args.flicker_temporal_jitter)),
                grid_scale=max(2.0, float(args.flicker_lowfreq_grid_scale))
            )
            if flicker_frames:
                I1_hr_lin, I2_hr_lin, I3_hr_lin = flicker_frames
                flicker_debug = {
                    'row_strength': float(args.flicker_row_strength),
                    'col_strength': float(args.flicker_col_strength),
                    'lowfreq_strength': float(args.flicker_lowfreq_strength),
                    'global_gain_strength': float(args.flicker_global_gain_strength),
                    'global_bias_strength': float(args.flicker_global_bias_strength),
                    'temporal_jitter': float(args.flicker_temporal_jitter),
                    'lowfreq_grid_scale': float(args.flicker_lowfreq_grid_scale)
                }
        try:
            if args.poisson_strength is not None and float(args.poisson_strength) > 0.0:
                poisson_strength_sample = float(args.poisson_strength)
            else:
                ps_min = getattr(args, 'poisson_strength_min', None)
                ps_max = getattr(args, 'poisson_strength_max', None)
                if ps_min is not None and ps_max is not None and float(ps_max) >= max(0.0, float(ps_min)):
                    poisson_strength_sample = float(np.random.uniform(float(ps_min), float(ps_max)))
                else:
                    poisson_strength_sample = 0.0
        except Exception:
            poisson_strength_sample = 0.0
        I1 = finalize_frame_from_linear(
            I1_hr_lin, target_size, 3, args.antialias_sigma_mult, args.down_method, poisson_strength_sample,
            args.aa_dst_sigma_px,
            args.enable_prequant_dither,
            args.prequant_amp_lsb,
            1.0
        )
        I2 = finalize_frame_from_linear(
            I2_hr_lin, target_size, 3, args.antialias_sigma_mult, args.down_method, poisson_strength_sample,
            args.aa_dst_sigma_px,
            args.enable_prequant_dither,
            args.prequant_amp_lsb,
            1.0
        )
        I3 = finalize_frame_from_linear(
            I3_hr_lin, target_size, 3, args.antialias_sigma_mult, args.down_method, poisson_strength_sample,
            args.aa_dst_sigma_px,
            args.enable_prequant_dither,
            args.prequant_amp_lsb,
            1.0
        )
        I_mag = finalize_frame_from_linear(
            I_mag_hr_lin, target_size, 3, args.antialias_sigma_mult, args.down_method, 0.0,
            args.aa_dst_sigma_px,
            False,
            0.5,
            1.0
        )
        if False:
            _qnoise = np.random.uniform(-0.5, 0.5, I1.shape).astype(np.float32)
            def _apply_qnoise(img_uint8, noise):
                return np.clip(img_uint8.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            I1 = _apply_qnoise(I1, _qnoise)
            I2 = _apply_qnoise(I2, _qnoise)
            I3 = _apply_qnoise(I3, _qnoise)
        quality = {
            'mean_input_trans_px': float(np.mean(input_trans_px_list)) if input_trans_px_list else 0.0,
            'mean_amplified_trans_px': float(np.mean(amp_trans_px_list)) if amp_trans_px_list else 0.0,
            'mean_rot_small_deg': float(np.mean(rot_small_list)) if rot_small_list else 0.0,
            'mean_rot_amplified_deg': float(np.mean(rot_mag_list)) if rot_mag_list else 0.0,
            'bg_motion_px': float(np.hypot(d_bg_hr[0], d_bg_hr[1]) / 3.0),
            'fg_out_of_bounds_ratio': float(fg_out_of_bounds / max(1, len(layers)))
        }
        if (not is_static_sample) and (sample_type != 'bg_only'):
            if quality['fg_out_of_bounds_ratio'] > args.max_out_of_bounds_ratio:
                return None
            if quality['mean_amplified_trans_px'] < args.min_mean_amplified_trans_px:
                return None
            if args.max_mean_amplified_trans_px is not None and quality['mean_amplified_trans_px'] > args.max_mean_amplified_trans_px:
                return None
            if (quality['mean_amplified_trans_px'] < 1e-6) and (quality['mean_rot_amplified_deg'] < 1e-6):
                return None
        try:
            mean_input = float(np.mean(input_trans_px_list)) if input_trans_px_list else 0.0
            mean_amp = float(np.mean(amp_trans_px_list)) if amp_trans_px_list else 0.0
        except Exception:
            mean_input, mean_amp = 0.0, 0.0
        return {
            'id': f"{sample_idx:06d}",
            'frameA': I1,
            'frameB': I2,
            'frameC': I3,
            'amplified': I_mag,
            'meta': {
                'sample_id': sample_idx,
                'global_alpha': global_alpha,
                'motion_choice': choice,
                'probs_used': adj_p.tolist(),
                'quality': quality,
                'sample_type': sample_type,
                'debug': {
                    'enable_subpixel_paste': bool(args.enable_subpixel_paste),
                    'per_fg_motion': bool(args.per_fg_motion),
                    'bbox_crop_margin': float(args.bbox_crop_margin),
                    'mean_input_trans_px': mean_input,
                    'mean_amplified_trans_px': mean_amp,
                    'is_static_sample': is_static_sample,
                    'sample_type': sample_type,
                    'poisson_strength': float(poisson_strength_sample),
                    'quantization_noise': bool(args.enable_quantization_noise),
                    'aa_dst_sigma_px': (float(args.aa_dst_sigma_px) if args.aa_dst_sigma_px is not None else None),
                    'prequant_dither': bool(args.enable_prequant_dither),
                    'prequant_amp_lsb': float(args.prequant_amp_lsb),
                    'prequant_prob': float(args.prequant_prob),
                    'phase_jitter_hr': [float(jx_hr), float(jy_hr)],
                    'global_affine_jitter': {
                        'deg': float(_deg), 'sx': float(_sx), 'sy': float(_sy)
                    },
                    'bg_blur_applied': bg_blur_applied,
                    'bg_blur_prob': float(bg_blur_prob),
                    'bg_blur_sigma': float(bg_sigma) if bg_sigma > 0 else None,
                    'bg_noise_applied': bg_noise_applied,
                    'bg_noise_std_prob': float(bg_noise_std_prob),
                    'bg_noise_std': float(bg_noise_std) if bg_noise_std > 0 else None,
                    'input_flicker': flicker_debug,
                }
            },
            'global_alpha': global_alpha
        }
    except Exception as e:
        print(f"[ERROR] generate_sample_data({sample_idx}) failed: {e}")
        return None
def main():
    parser = argparse.ArgumentParser(description="Stable VMM synthetic data generator v11_g1 (sRGB mask version)")
    parser.add_argument('--bg_txt', type=str, required=False)
    parser.add_argument('--fg_txt', type=str, required=False)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--min_fg', type=int, default=7)
    parser.add_argument('--max_fg', type=int, default=15)
    parser.add_argument('--base_translation', type=float, default=10.0)
    parser.add_argument('--max_rot_small', type=float, default=3.0)
    parser.add_argument('--max_rot_amplified', type=float, default=10.0)
    parser.add_argument('--antialias_sigma_mult', type=float, default=0.0, help='Pre-filter strength multiplier 0.4~0.9')
    parser.add_argument('--down_method', type=str, default='area', choices=['area', 'lanczos'])
    parser.add_argument('--aa_dst_sigma_px', type=float, default=0.0, help='Destination domain AA sigma (in target pixels), takes priority over antialias_sigma_mult')
    parser.add_argument('--max_amplified_motion_px', type=float, default=30)
    parser.add_argument('--max_amplified_motion_frac', type=float, default=0.08)
    parser.add_argument('--alpha_prob_mix_strength', type=float, default=0.5)
    parser.add_argument('--p_bg_motion', type=float, default=0.2)
    parser.add_argument('--bg_drift_min_px', type=float, default=0.1)
    parser.add_argument('--bg_drift_max_px', type=float, default=0.3)
    parser.add_argument('--p_trans', type=float, default=0.3)
    parser.add_argument('--p_rot', type=float, default=0.3)
    parser.add_argument('--p_both', type=float, default=0.4)
    parser.add_argument('--feather', type=float, default=0.5, help='Additional Gaussian feathering pixels for mask')
    parser.add_argument('--use_mask_distance_feather', action='store_true', help='Enable distance transform feathering soft mask')
    parser.add_argument('--mask_dt_soften_px', type=float, default=1.5, help='Distance feathering width (pixels)')
    parser.add_argument('--base_seed', type=int, default=20251015)
    parser.add_argument('--step_factor', type=float, default=1.0)
    parser.add_argument('--max_out_of_bounds_ratio', type=float, default=0.15)
    parser.add_argument('--min_mean_amplified_trans_px', type=float, default=0.0, help='Minimum mean amplified translation pixels, set to 0.0 to avoid filtering pure rotation samples')
    parser.add_argument('--max_mean_amplified_trans_px', type=float, default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--attempt_multiplier', type=float, default=10.0)
    parser.add_argument('--unlimited_attempts', action='store_true')
    parser.add_argument('--per_fg_motion', action='store_true', help='Sample motion type and translation/rotation independently for each foreground (shared alpha)')
    parser.add_argument('--enable_subpixel_paste', action='store_true', help='Enable subpixel-level affine pasting to avoid jitter from integer rounding')
    parser.add_argument('--bbox_crop_margin', type=float, default=0.0, help='Additional margin pixels or relative edge length ratio outside max connected component bbox (>0 and <1 treated as ratio)')
    parser.add_argument('--fg_preblur_before_warp', action='store_true', help='Apply same sigma linear domain Gaussian blur to foreground and mask before affine/composition')
    parser.add_argument('--fg_preblur_sigma_px', type=float, default=None, help='Foreground pre-blur sigma (in source foreground pixels); not enabled if unspecified')
    parser.add_argument('--poisson_strength', type=float, default=0.0, help='Poisson noise strength, 0 to disable, typical values 0.5-2.0')
    parser.add_argument('--poisson_strength_min', type=float, default=0.5, help='Poisson noise strength lower bound (random per sample), default 0.5')
    parser.add_argument('--poisson_strength_max', type=float, default=2.5, help='Poisson noise strength upper bound (random per sample), default 2.5')
    parser.add_argument('--enable_quantization_noise', action='store_true', help='Enable uniform quantization noise (±0.5 pixels)')
    parser.add_argument('--enable_prequant_dither', action='store_true', help='Add dithering to sRGB float domain before quantizing to 8bit')
    parser.add_argument('--prequant_amp_lsb', type=float, default=0.35, help='Pre-quantization dither amplitude (LSB), 0.15~0.5')
    parser.add_argument('--prequant_prob', type=float, default=1.0, help='Pre-quantization dither probability [0,1]')
    parser.add_argument('--sample_type_ratios', type=str, default='6:1:1:1:1', help='Sample type ratios: standard:blur_bg:bg_only:static:fg_only, format like 6:1:1:1:1')
    parser.add_argument('--flush_every', type=int, default=200, help='Batch flush/refresh every N samples')
    parser.add_argument('--finalize_shuffle', action='store_true', help='Perform global shuffle and rename on existing output directory, only run after all samples generated')
    parser.add_argument('--shuffle_seed', type=int, default=None, help='Shuffle random seed (reproducible)')
    parser.add_argument('--bg_noise_sigma', type=float, default=0.9, 
        help='Background Gaussian blur sigma, enhance background naturalness (paper suggests 0.9~1.2, too large loses details)')
    parser.add_argument('--bg_noise_std', type=float, default=3.0, 
        help='Background Gaussian noise std (papers commonly use 2~5, too large affects micro-motion, too small distribution too clean)')
    parser.add_argument('--bg_noise_std_prob', type=float, default=0.0, 
        help='Background Gaussian noise std probability')
    parser.add_argument('--bg_blur_prob', type=float, default=0.0, help='Background Gaussian blur probability [0,1]')
    parser.add_argument('--blur_bg_extra_sigma', type=float, default=0.0, help='Additional Gaussian blur sigma for blur_bg samples (pixels)')
    parser.add_argument('--blur_bg_extra_prob', type=float, default=1.0, help='Additional blur probability for blur_bg samples [0,1]')
    parser.add_argument('--fg_blur_sigma', type=float, default=0.0, help='Gaussian blur sigma for foreground sharpness normalization (suggested 0.7~1.2)')
    parser.add_argument('--enable_input_flicker', action='store_true', help='Add stripe/exposure drift only to input frames, keep labels clean')
    parser.add_argument('--flicker_row_strength', type=float, default=0.02, help='Input frame row-direction stripe amplitude (linear domain)')
    parser.add_argument('--flicker_col_strength', type=float, default=0.02, help='Input frame column-direction stripe amplitude')
    parser.add_argument('--flicker_lowfreq_strength', type=float, default=0.015, help='Input frame low-frequency 2D noise amplitude')
    parser.add_argument('--flicker_global_gain_strength', type=float, default=0.03, help='Input frame global gain jitter amplitude')
    parser.add_argument('--flicker_global_bias_strength', type=float, default=0.02, help='Input frame global brightness bias amplitude')
    parser.add_argument('--flicker_temporal_jitter', type=float, default=1.0, help='Temporal random walk strength for input frame stripe/exposure drift')
    parser.add_argument('--flicker_lowfreq_grid_scale', type=float, default=48.0, help='Grid scale for low-frequency noise generation (larger is lower frequency)')
    args = parser.parse_args()
    def _print_and_save_run_config(mode: str):
        try:
            ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pid = os.getpid()
            print("\n====== Run Configuration Summary ======")
            print(f"Time: {ts} | PID: {pid}")
            print(f"Mode: {mode} | Output directory: {args.output_dir}")
            if mode == 'generate':
                print(f"Number of samples: {args.num_samples} | Resolution: {args.img_size} | Foregrounds: {args.min_fg}-{args.max_fg}")
                print(f"AA: mult={args.antialias_sigma_mult} | aa_dst_sigma_px={args.aa_dst_sigma_px} | Downsampling: {args.down_method}")
                print(f"Motion: base_trans={args.base_translation} | rot_small={args.max_rot_small} | rot_amp={args.max_rot_amplified}")
                print(f"Background noise: blur_sigma={args.bg_noise_sigma} | blur_prob={args.bg_blur_prob} | noise_std={args.bg_noise_std} | noise_std_prob={args.bg_noise_std_prob}")
                print(f"Blur background samples: extra_sigma={args.blur_bg_extra_sigma} | extra_prob={args.blur_bg_extra_prob}")
                print(f"Background motion: motion_prob={args.p_bg_motion} | min={args.bg_drift_min_px} | max={args.bg_drift_max_px}")
                print(f"Subpixel paste: enable_subpixel_paste={args.enable_subpixel_paste} ")
                print(f"Foreground: fg_blur_sigma={args.fg_blur_sigma} | preblur={bool(args.fg_preblur_before_warp)} sigma={args.fg_preblur_sigma_px}")
                print(f"Poisson noise: fix={args.poisson_strength} | range=[{args.poisson_strength_min}, {args.poisson_strength_max}]")
                print(f"Quantization noise: enable={bool(args.enable_quantization_noise)} | prequant dither: enable={bool(args.enable_prequant_dither)} amp={args.prequant_amp_lsb} prob={args.prequant_prob}")
                print(f"Input flicker: enable={bool(args.enable_input_flicker)} row={args.flicker_row_strength} col={args.flicker_col_strength} lowfreq={args.flicker_lowfreq_strength} gain={args.flicker_global_gain_strength} bias={args.flicker_global_bias_strength}")
                print(f"worker={args.num_workers} | attempt_multiplier={args.attempt_multiplier} | seed={args.base_seed}")
            else:
                print(f"shuffle_seed={args.shuffle_seed}")
            print("==============================\n")
        except Exception as e:
            print(f"[WARN] Failed to print run configuration: {e}")
    print("=" * 60)
    print("Mask processing method: sRGB domain mask processing (v11_g1)")
    print("Note: Masks are blurred and blended in sRGB domain, images are blended in linear domain")
    if args.use_mask_distance_feather:
        print(f"Distance feathering: Enabled (softening width: {args.mask_dt_soften_px}px)")
    else:
        print("Distance feathering: Disabled")
    print(f"Gaussian feathering: {args.feather}px")
    print("=" * 60)
    if args.finalize_shuffle:
        _print_and_save_run_config(mode='finalize_shuffle')
        out_dir = args.output_dir
        subdirs = ['frameA', 'frameB', 'frameC', 'amplified', 'meta']
        for sd in subdirs:
            if not os.path.isdir(os.path.join(out_dir, sd)):
                raise RuntimeError(f"Missing subdirectory: {sd}")
        def _scan_ids(dir_path: str, suffix: str):
            ids = []
            try:
                for name in os.listdir(dir_path):
                    if name.endswith(suffix):
                        stem = name[:-len(suffix)]
                        if len(stem) == 6 and stem.isdigit():
                            ids.append(stem)
            except Exception:
                pass
            return sorted(ids)
        ids_png = _scan_ids(os.path.join(out_dir, 'frameA'), '.png')
        ids_json = _scan_ids(os.path.join(out_dir, 'meta'), '.json')
        ids_all = sorted(list(set(ids_png).intersection(set(ids_json))))
        if not ids_all:
            raise RuntimeError("No sample files found in output directory for shuffling")
        for sd in ['frameB', 'frameC', 'amplified']:
            ids_chk = _scan_ids(os.path.join(out_dir, sd), '.png')
            if set(ids_chk) != set(ids_all):
                raise RuntimeError(f"Subdirectory {sd} sample ID set inconsistent with frameA")
        n = len(ids_all)
        old_ids_sorted = sorted(ids_all)
        rng = random.Random(args.shuffle_seed)
        new_order = old_ids_sorted.copy()
        rng.shuffle(new_order)
        target_ids = [f"{i:06d}" for i in range(1, n + 1)]
        mapping = {old_id: new_id for old_id, new_id in zip(new_order, target_ids)}
        for sd, ext in [('frameA', '.png'), ('frameB', '.png'), ('frameC', '.png'), ('amplified', '.png'), ('meta', '.json')]:
            src_dir = os.path.join(out_dir, sd)
            tmp_dir = os.path.join(out_dir, f"._shuffle_tmp_{sd}")
            os.makedirs(tmp_dir, exist_ok=True)
            for old_id in old_ids_sorted:
                new_id = mapping[old_id]
                src = os.path.join(src_dir, old_id + ext)
                dst = os.path.join(tmp_dir, new_id + ext)
                if not os.path.exists(src):
                    raise RuntimeError(f"Missing file: {src}")
                os.replace(src, dst)
            for name in os.listdir(tmp_dir):
                os.replace(os.path.join(tmp_dir, name), os.path.join(src_dir, name))
            try:
                os.rmdir(tmp_dir)
            except Exception:
                pass
        train_mf_path = os.path.join(out_dir, 'train_mf.txt')
        try:
            with open(train_mf_path, 'w') as fmf:
                for i in range(1, n + 1):
                    sid = f"{i:06d}"
                    meta_p = os.path.join(out_dir, 'meta', sid + '.json')
                    try:
                        with open(meta_p, 'r') as fj:
                            meta = json.load(fj)
                        alpha = float(meta.get('global_alpha', 0.0))
                    except Exception:
                        alpha = 0.0
                    fmf.write(f"{alpha:.3f}\n")
        except Exception as e:
            print(f"[WARN] Failed to rebuild train_mf.txt: {e}")
        manifest_path = os.path.join(out_dir, 'shuffle_manifest.json')
        try:
            with open(manifest_path, 'w') as fm:
                json.dump({ 'mapping': mapping, 'count': n, 'seed': args.shuffle_seed }, fm, indent=2)
        except Exception:
            pass
        print(f"Completed global shuffle and rename, total {n} samples; Output directory: {out_dir}")
        return
    _print_and_save_run_config(mode='generate')
    if not args.bg_txt or not args.fg_txt:
        raise RuntimeError('--bg_txt and --fg_txt are required (unless using --finalize_shuffle)')
    os.makedirs(args.output_dir, exist_ok=True)
    for sd in ['frameA', 'frameB', 'frameC', 'amplified', 'meta']:
        os.makedirs(os.path.join(args.output_dir, sd), exist_ok=True)
    train_mf_path = os.path.join(args.output_dir, 'train_mf.txt')
    if os.path.exists(train_mf_path):
        os.remove(train_mf_path)
    mf_parts_dir = os.path.join(args.output_dir, '_train_mf_parts')
    os.makedirs(mf_parts_dir, exist_ok=True)
    try:
        for name in os.listdir(mf_parts_dir):
            p = os.path.join(mf_parts_dir, name)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
    except Exception:
        pass
    with open(args.bg_txt, 'r') as f:
        bg_files = [line.strip() for line in f if line.strip()]
    fg_files, mask_files = [], []
    with open(args.fg_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                fg, mk = parts
                if os.path.exists(fg) and os.path.exists(mk):
                    fg_files.append(fg)
                    mask_files.append(mk)
    worker = partial(generate_sample_data, args=args, bg_files=bg_files, fg_files=fg_files, mask_files=mask_files)
    target_samples = args.num_samples if not args.limit else min(args.num_samples, args.limit)
    next_id = 1
    flush_every = max(1, int(args.flush_every))
    next_attempt_idx = 0
    attempts_made = 0
    max_attempts = int(target_samples * max(1.0, float(args.attempt_multiplier)))
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        pbar_total = None if args.unlimited_attempts else max_attempts
        _disable_pb = False
        pbar = tqdm(total=pbar_total, desc='Generate samples', disable=_disable_pb)
        written = 0
        try:
            while written < target_samples:
                if (not args.unlimited_attempts) and (attempts_made >= max_attempts):
                    break
                remaining_needed = target_samples - written
                batch_size = min(max(remaining_needed * 4, 256), 4096)
                if not args.unlimited_attempts:
                    batch_size = min(batch_size, max(0, max_attempts - attempts_made))
                    if batch_size == 0:
                        break
                attempt_indices = list(range(next_attempt_idx, next_attempt_idx + batch_size))
                next_attempt_idx += batch_size
                attempts_made += batch_size
                for res in ex.map(worker, attempt_indices, chunksize=10):
                    pbar.update(1)
                    if res is None:
                        continue
                    if written >= target_samples:
                        break
                    sid = f"{next_id:06d}"
                    next_id += 1
                    try:
                        cv2.imwrite(os.path.join(args.output_dir, 'frameA', f"{sid}.png"), res['frameA'])
                        cv2.imwrite(os.path.join(args.output_dir, 'frameB', f"{sid}.png"), res['frameB'])
                        cv2.imwrite(os.path.join(args.output_dir, 'frameC', f"{sid}.png"), res['frameC'])
                        cv2.imwrite(os.path.join(args.output_dir, 'amplified', f"{sid}.png"), res['amplified'])
                        try:
                            if isinstance(res.get('meta'), dict):
                                res['meta']['sample_id'] = int(sid)
                        except Exception:
                            pass
                        with open(os.path.join(args.output_dir, 'meta', f"{sid}.json"), 'w') as fjson:
                            json.dump(res['meta'], fjson, indent=2)
                        try:
                            with open(os.path.join(mf_parts_dir, f"{sid}.txt"), 'w') as fpart:
                                fpart.write(f"{res['global_alpha']:.3f}\n")
                        except Exception as e:
                            print(f"[WARN] Failed to write alpha fragment {sid}: {e}")
                        written += 1
                    except Exception as e:
                        print(f"[WARN] Failed to write sample {sid}: {e}")
                        next_id -= 1
                    if written >= target_samples:
                        break
        finally:
            pbar.close()
    try:
        with open(train_mf_path, 'w') as fmf:
            for i in range(1, written + 1):
                sid = f"{i:06d}"
                part_path = os.path.join(mf_parts_dir, f"{sid}.txt")
                if os.path.exists(part_path):
                    try:
                        with open(part_path, 'r') as fp:
                            fmf.write(fp.read())
                    except Exception as e:
                        print(f"[WARN] Failed to read alpha fragment {sid}: {e}")
                        fmf.write("0.000\n")
                else:
                    fmf.write("0.000\n")
    except Exception as e:
        print(f"[WARN] Failed to merge train_mf fragments: {e}")
    try:
        for name in os.listdir(mf_parts_dir):
            p = os.path.join(mf_parts_dir, name)
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        os.rmdir(mf_parts_dir)
    except Exception:
        pass
    if written < target_samples:
        print(f"Target {target_samples} samples, actually collected only {written} samples. (may have reached max attempts)")
    else:
        print(f"Saved {written} samples, output directory: {args.output_dir}")
        print("Magnification factors written to train_mf.txt (aligned by ID order)")
if __name__ == '__main__':
    main()
