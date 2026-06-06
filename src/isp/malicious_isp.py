import os
import cv2
import numpy as np
import rawpy
from typing import Dict, Optional, Any
from src.isp.simple_isp import SimpleISP, ISPConfig  # 导入你的基类


class MaliciousISP(SimpleISP):
    """
    恶意 ISP 模拟器：继承了极速的 RAW 域裁剪，
    但在演化过程中会随机注入反物理规律的扰动，用于生成困难负样本。
    """

    def __init__(self, config: Optional[ISPConfig] = None, perturbation_prob: float = 0.8):
        super().__init__(config)
        self.perturbation_prob = perturbation_prob

    def run(self, raw_path: str, config_override: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        # ======= 【复用你完美的裁剪与对齐逻辑】 =======
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"RAW 文件不存在: {raw_path}")

        cfg = ISPConfig(**self.base_config.__dict__)
        if config_override:
            for k, v in config_override.items():
                if hasattr(cfg, k): setattr(cfg, k, v)

        stages = {}

        with rawpy.imread(raw_path) as raw:
            rgb_linear_full = raw.postprocess(
                gamma=(1, 1), no_auto_bright=True, output_bps=16, user_flip=0
            ).astype(np.float32) / 65535.0

            raw_visible = raw.raw_image_visible.astype(np.float32)
            H_raw, W_raw = raw_visible.shape[:2]
            H_rgb, W_rgb = rgb_linear_full.shape[:2]
            valid_H, valid_W = min(H_raw, H_rgb), min(W_raw, W_rgb)

            if cfg.crop_size is not None and valid_H > cfg.crop_size and valid_W > cfg.crop_size:
                if cfg.center_crop:
                    top, left = (valid_H - cfg.crop_size) // 2, (valid_W - cfg.crop_size) // 2
                else:
                    top, left = np.random.randint(0, valid_H - cfg.crop_size), np.random.randint(0,
                                                                                                 valid_W - cfg.crop_size)
                crop_h, crop_w = cfg.crop_size, cfg.crop_size
            else:
                top, left, crop_h, crop_w = 0, 0, valid_H, valid_W

            top, left = top & ~1, left & ~1
            crop_h, crop_w = crop_h & ~1, crop_w & ~1

            # --- Stage 1: RAW ---
            raw_data = raw_visible[top:top + crop_h, left:left + crop_w]
            black_level = np.mean(raw.black_level_per_channel)
            raw_data = np.clip((raw_data - black_level) / (raw.white_level - black_level), 0, 1)
            if raw_data.ndim == 2:
                raw_data = np.expand_dims(raw_data, axis=2)
            stages["stage_raw"] = raw_data

            # --- Stage 2: Demosaic ---
            x_demosaic = rgb_linear_full[top:top + crop_h, left:left + crop_w].copy()

        # 🔥 【注入扰动 1：空间通道错位】(破坏 CFA 拜耳阵列微观相关性)
        if np.random.rand() < self.perturbation_prob:
            # R 通道向右移 1 像素，B 通道向下移 1 像素
            r = np.pad(x_demosaic[:, :, 0], ((0, 0), (1, 0)), mode='edge')[:, :-1]
            b = np.pad(x_demosaic[:, :, 2], ((1, 0), (0, 0)), mode='edge')[:-1, :]
            x_demosaic[:, :, 0] = r
            x_demosaic[:, :, 2] = b

        stages["stage_demosaic"] = x_demosaic

        # --- Stage 3: Denoise ---
        img_uint8 = np.clip(x_demosaic * 255, 0, 255).astype(np.uint8)
        k_size = int(cfg.denoise_strength * 10) | 1
        if k_size > 1:
            x_denoise = cv2.GaussianBlur(img_uint8, (k_size, k_size), 0).astype(np.float32) / 255.0
        else:
            x_denoise = x_demosaic.copy()

        # 🔥 【注入扰动 2：非物理频域截断/异常噪声】(破坏泊松-高斯噪声模型)
        if np.random.rand() < self.perturbation_prob:
            # 注入与信号强度无关的强行高频均匀噪声
            noise = (np.random.rand(*x_denoise.shape).astype(np.float32) - 0.5) * 0.15
            x_denoise = np.clip(x_denoise + noise, 0, 1)

        stages["stage_denoise"] = x_denoise

        # --- Stage 4: Color Correction ---
        x_color = x_denoise.copy()
        if cfg.saturation != 1.0:
            hsv = cv2.cvtColor(x_color, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg.saturation, 0, 1)
            x_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if cfg.contrast != 1.0:
            x_color = np.clip((x_color - 0.5) * cfg.contrast + 0.5, 0, 1)

        # 🔥 【注入扰动 3：物理顺序颠倒】(在 Gamma 前提前非线性扭曲)
        if np.random.rand() < self.perturbation_prob:
            distortion_gamma = np.random.uniform(0.3, 0.7)
            x_color = np.power(np.maximum(x_color, 1e-6), distortion_gamma)

        stages["stage_color"] = np.clip(x_color, 0, 1)

        # --- Stage 5: RGB ---
        img_rgb = np.power(np.maximum(stages["stage_color"], 1e-6), 1.0 / cfg.gamma)
        stages["rgb"] = np.clip(img_rgb, 0, 1)

        return stages
