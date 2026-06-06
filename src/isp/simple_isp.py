import os
import cv2
import numpy as np
import rawpy
from dataclasses import dataclass
from typing import Dict, Optional, Any


@dataclass
class ISPConfig:
    """默认的 ISP 参数配置"""
    gamma: float = 2.2
    saturation: float = 1.0
    contrast: float = 1.0
    denoise_strength: float = 0.5
    # 新增黑科技：RAW 域直裁尺寸 (设置 None 为全图)
    crop_size: Optional[int] = 512
    center_crop: bool = False  # <--- 新增：特征提取专用的中心裁剪开关


def apply_rgb_degradation(
    rgb_image: np.ndarray,
    degradation_config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Apply a benign post-ISP degradation to the final RGB image."""
    if degradation_config is None:
        return rgb_image.copy()

    degradation_type = degradation_config.get("type")
    if degradation_type in (None, "", "none"):
        return rgb_image.copy()

    if degradation_type != "jpeg_compression":
        raise ValueError(f"Unsupported degradation type: {degradation_type}")

    quality = int(np.clip(degradation_config.get("quality", 95), 1, 100))
    rgb_uint8 = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    ok, encoded = cv2.imencode(
        ".jpg",
        bgr_uint8,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("Failed to encode RGB image for JPEG degradation.")

    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("Failed to decode JPEG-degraded RGB image.")

    degraded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.clip(degraded_rgb, 0.0, 1.0)


class SimpleISP:
    """
    极简版软件 ISP 模拟器。
    加入 RAW Domain Cropping 技术，在 Demosaic 阶段前截取 ROI，速度飙升百倍！
    """

    def __init__(self, config: Optional[ISPConfig] = None):
        self.base_config = config or ISPConfig()

    def run(
        self,
        raw_path: str,
        config_override: Optional[Dict[str, Any]] = None,
        degradation_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"RAW 文件不存在: {raw_path}")

        # 合并参数
        cfg = ISPConfig(**self.base_config.__dict__)
        if config_override:
            for k, v in config_override.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

        stages = {}

        # 1. 读取 RAW 文件
        with rawpy.imread(raw_path) as raw:
            # 【修复点 1】: 先做全图 Demosaic，并强制禁止自动旋转 (user_flip=0)
            # 这样才能保证输出的 RGB 图和底层 RAW 数组的空间坐标 100% 对齐！
            rgb_linear_full = raw.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                output_bps=16,
                user_flip=0  # <--- 核心修复：禁止 EXIF 自动旋转
            ).astype(np.float32) / 65535.0

            raw_visible = raw.raw_image_visible.astype(np.float32)
            H_raw, W_raw = raw_visible.shape[:2]
            H_rgb, W_rgb = rgb_linear_full.shape[:2]

            # 【修复点 2】: 取 RAW 和 RGB 尺寸的最小值，防止边缘越界
            valid_H = min(H_raw, H_rgb)
            valid_W = min(W_raw, W_rgb)

            # === 计算裁剪坐标 (支持随机裁剪与中心裁剪) ===
            if cfg.crop_size is not None and valid_H > cfg.crop_size and valid_W > cfg.crop_size:
                if cfg.center_crop:
                    # ★ 特征提取模式：绝对居中裁剪
                    top = (valid_H - cfg.crop_size) // 2
                    left = (valid_W - cfg.crop_size) // 2
                else:
                    # ★ 训练模式：随机裁剪
                    top = np.random.randint(0, valid_H - cfg.crop_size)
                    left = np.random.randint(0, valid_W - cfg.crop_size)
                crop_h, crop_w = cfg.crop_size, cfg.crop_size
            else:
                top, left = 0, 0
                crop_h, crop_w = valid_H, valid_W

            # 【修复点 3】: 保证坐标和尺寸永远是偶数，完美对齐 RGGB 拜耳阵列
            top = top & ~1
            left = left & ~1
            crop_h = crop_h & ~1
            crop_w = crop_w & ~1

            # ==========================================
            # Stage 1: Raw (仅截取 ROI 区域)
            # ==========================================
            raw_data = raw_visible[top:top + crop_h, left:left + crop_w]
            black_level = np.mean(raw.black_level_per_channel)
            raw_data = np.maximum(raw_data - black_level, 0)
            white_level = raw.white_level - black_level
            raw_data = raw_data / white_level

            if raw_data.ndim == 2:
                raw_data = np.expand_dims(raw_data, axis=2)
            stages["stage_raw"] = raw_data

            # ==========================================
            # Stage 2: Demosaic (完美对应的裁剪区域)
            # ==========================================
            stages["stage_demosaic"] = rgb_linear_full[top:top + crop_h, left:left + crop_w]

        # ==========================================
        # Stage 3: Denoise (现在只处理 512x512 的小图了，速度极快)
        # ==========================================
        if cfg.denoise_strength > 0:
            img_uint8 = np.clip(stages["stage_demosaic"] * 255, 0, 255).astype(np.uint8)
            k_size = int(cfg.denoise_strength * 10) | 1
            if k_size > 1:
                denoised = cv2.GaussianBlur(img_uint8, (k_size, k_size), 0)
                stages["stage_denoise"] = denoised.astype(np.float32) / 255.0
            else:
                stages["stage_denoise"] = stages["stage_demosaic"].copy()
        else:
            stages["stage_denoise"] = stages["stage_demosaic"].copy()

        # ==========================================
        # Stage 4: Color Correction (只处理 512x512)
        # ==========================================
        img_color = stages["stage_denoise"].copy()
        if cfg.saturation != 1.0:
            hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * cfg.saturation, 0, 1)
            img_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if cfg.contrast != 1.0:
            img_color = (img_color - 0.5) * cfg.contrast + 0.5

        stages["stage_color"] = np.clip(img_color, 0, 1)

        # ==========================================
        # Stage 5: RGB (只处理 512x512)
        # ==========================================
        img_rgb = np.power(np.maximum(stages["stage_color"], 1e-6), 1.0 / cfg.gamma)
        stages["rgb"] = np.clip(img_rgb, 0, 1)
        stages["rgb"] = apply_rgb_degradation(stages["rgb"], degradation_override)

        return stages
