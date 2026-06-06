import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ChannelAdapter(nn.Module):
    """
    动态通道适配器：根据输入张量的实际通道数，动态映射为 3 通道。
    目的不是“学一个新颜色空间”，而是尽量保守地把输入喂给 ImageNet 骨干。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        if c == 3:
            return x
        elif c == 1:
            return x.repeat(1, 3, 1, 1)
        elif c == 4:
            # 假定 4 通道是 RGGB 展平
            r = x[:, 0:1]
            g = 0.5 * (x[:, 1:2] + x[:, 2:3])
            b = x[:, 3:4]
            return torch.cat([r, g, b], dim=1)
        else:
            raise ValueError(f"Unsupported input channels: {c}")


class FixedHighPassFilterBank(nn.Module):
    """
    固定高通滤波器组。
    相比“单个 Laplacian”，多滤波器组更不容易把痕迹建模过度绑定到某一种局部模式。
    这里输出一个 3 通道 trace map，作为输入图像的高频痕迹增强项。
    """
    def __init__(self):
        super().__init__()

        # 5 组轻量固定滤波器（兼顾拉普拉斯 / 方向差分 / 二阶差分）
        kernels = torch.tensor([
            [[0,  0,  0],
             [0,  1, -1],
             [0,  0,  0]],

            [[0,  0,  0],
             [0,  1,  0],
             [0, -1,  0]],

            [[0,  0,  0],
             [-1, 2, -1],
             [0,  0,  0]],

            [[0, -1,  0],
             [0,  2,  0],
             [0, -1,  0]],

            [[-1, -1, -1],
             [-1,  8, -1],
             [-1, -1, -1]],
        ], dtype=torch.float32)  # (K, 3, 3)

        self.num_kernels = kernels.shape[0]

        # depthwise 方式：每个 RGB 通道各自做同一组滤波
        # 最终权重形状：(3*K, 1, 3, 3)
        bank = kernels.unsqueeze(1).repeat(3, 1, 1, 1)  # (3*K, 1, 3, 3)
        self.register_buffer("bank", bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        return: trace_map (B, 3, H, W)
        """
        # reflect pad，避免边界伪高频
        x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")

        # groups=3 做 depthwise，高通响应输出形状：(B, 3*K, H, W)
        resp = F.conv2d(x_pad, self.bank, stride=1, padding=0, groups=3)

        b, ck, h, w = resp.shape
        k = self.num_kernels
        resp = resp.view(b, 3, k, h, w)

        # 用 abs + mean 聚合不同滤波器响应，得到每个通道一个高频痕迹图
        trace_map = resp.abs().mean(dim=2)  # (B, 3, H, W)
        return trace_map


class FeatureBackbone(nn.Module):
    r"""
    统一特征提取骨干网络。

    输出:
        f: (B, d_f)   -> 用于 Stage 1 路径表征学习
        F: (B, C_f, H', W') -> 用于后续拓扑/空间分支

    设计目标：
    1. 保留你当前工程接口不变；
    2. 抑制语义 shortcut，增强低层成像痕迹；
    3. 比“纯 GAP”多保留一点二阶统计信息，但不把实现复杂度拉爆。
    """
    def __init__(
        self,
        d_f: int = 512,
        C_f: int = 256,
        freeze_bn: bool = True,
        max_trace_alpha: float = 0.15,
        imagenet_weights: bool = True,
    ):
        super().__init__()
        self.d_f = d_f
        self.C_f = C_f
        self.freeze_bn = freeze_bn
        self.max_trace_alpha = max_trace_alpha

        # 1. 通道适配
        self.adapter = ChannelAdapter()

        # 2. 高频滤波器组
        self.trace_bank = FixedHighPassFilterBank()

        # 3. ResNet50 主干
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if imagenet_weights else None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        backbone_out_dim = 2048

        # 4. 给空间特征图降维，供后续拓扑/空间分析
        self.map_proj = nn.Sequential(
            nn.Conv2d(backbone_out_dim, C_f, kernel_size=1, bias=False),
            nn.BatchNorm2d(C_f),
            nn.ReLU(inplace=True),
        )

        # 5. 轻量“二阶统计”全局描述：
        #    用 mean + std 代替单纯 GAP，比 full covariance pooling 更稳、更轻。
        pooled_dim = backbone_out_dim * 2  # mean + std

        self.global_proj = nn.Sequential(
            nn.Linear(pooled_dim, d_f),
            nn.LayerNorm(d_f),
            nn.GELU(),
            nn.Linear(d_f, d_f),
        )

        # 6. 可学习的高频融合强度，约束在 (0, max_trace_alpha)
        self.trace_alpha_raw = nn.Parameter(torch.tensor(0.0))

        if self.freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bn:
            self._freeze_batchnorm()
        return self

    def _fuse_trace(self, x: torch.Tensor) -> torch.Tensor:
        """
        在输入侧做“柔和的”高频痕迹融合。
        不是直接替换原图，而是在原图基础上注入少量 trace cue。
        """
        trace = self.trace_bank(x)  # (B, 3, H, W)

        # 每张图按自身统计做归一化，避免 trace 量级失控
        trace_mean = trace.mean(dim=(2, 3), keepdim=True)
        trace_std = trace.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        trace = (trace - trace_mean) / trace_std

        alpha = self.max_trace_alpha * torch.sigmoid(self.trace_alpha_raw)
        x = x + alpha * trace
        return x

    def _global_descriptor(self, feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W)
        输出 mean + std 拼接后的全局描述
        """
        mean = feat.mean(dim=(2, 3))
        std = feat.std(dim=(2, 3), unbiased=False)
        desc = torch.cat([mean, std], dim=1)  # (B, 2C)
        return desc

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            f: (B, d_f)
            F: (B, C_f, H', W')
        """
        # 1. 统一成 3 通道
        x = self.adapter(x)

        # 2. 注入高频痕迹
        x = self._fuse_trace(x)

        # 3. 主干提特征
        feat = self.backbone(x)          # (B, 2048, H/32, W/32)

        # 4. 空间图分支
        F_map = self.map_proj(feat)      # (B, C_f, H/32, W/32)

        # 5. 全局描述分支（mean + std）
        desc = self._global_descriptor(feat)   # (B, 4096)
        f = self.global_proj(desc)             # (B, d_f)

        return f, F_map

    def extract_global(self, x: torch.Tensor) -> torch.Tensor:
        f, _ = self.forward(x)
        return f

    def extract_map(self, x: torch.Tensor) -> torch.Tensor:
        _, F_map = self.forward(x)
        return F_map
