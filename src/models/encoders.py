import torch
import torch.nn as nn


class PathEncoder(nn.Module):
    def __init__(self, d_f=512, d_z=256, num_stages=5):
        super().__init__()
        self.d_f = d_f
        self.num_stages = num_stages

        # 🔥 核心修改 1: 引入显式的阶段位置编码 (Stage Embedding)
        # 让模型知道哪个特征是 raw，哪个是 rgb
        self.stage_embed = nn.Parameter(torch.randn(1, num_stages, d_f) * 0.02)

        # 🔥 核心修改 2: 引入 [CLS] Token 用于全局路径特征聚合
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_f) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_f, nhead=8, dim_feedforward=d_f * 4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 保留 LayerNorm：比 BatchNorm 更适合多阶段/多分布特征
        self.proj = nn.Sequential(
            nn.Linear(d_f, d_f),
            nn.LayerNorm(d_f),
            nn.GELU(),
            nn.Linear(d_f, d_z)
        )

    def forward(self, f_seq):
        """
        输入: f_seq 形状 (B, K, d_f)  (K=5)
        输出: z 形状 (B, d_z)
        """
        B = f_seq.size(0)

        # 1. 给输入的 5 个阶段特征注入位置信息
        f_seq = f_seq + self.stage_embed

        # 2. 拼接 [CLS] Token 到序列最前面 (B, 6, d_f)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, f_seq), dim=1)

        # 3. 进 Transformer 进行序列交互
        out_seq = self.transformer(x)

        # 4. 🔥 核心修改 3: 放弃暴力的 mean pooling，只取 [CLS] Token 作为路径表征
        path_feat = out_seq[:, 0, :]

        # 5. 投影到对比空间
        z = self.proj(path_feat)
        return z