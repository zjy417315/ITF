import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


class MarkovRandomWalkEncoder(nn.Module):
    """
    Phase 2: 基于马尔可夫随机游走的拓扑状态转移网络 (左塔/Teacher)

    适配真实数据版:
    输入阶段特征 p_seq 形状为 (Batch, 5, 20, 20)，即持久图像 PI。
    网络会自动将其展平为 (Batch, 5, 400)，然后执行图上的随机游走。
    """

    def __init__(self, d_p: int = 400, d_A: int = 256, num_stages: int = 5, walk_steps: int = 3, tau: float = 0.1):
        # 注意：这里的 d_p 默认改为了 400 (20x20)
        super().__init__()
        self.d_p = d_p
        self.d_A = d_A
        self.num_stages = num_stages
        self.walk_steps = walk_steps
        self.tau = tau

        self.node_proj = nn.Sequential(
            nn.Linear(d_p, d_p),
            nn.LayerNorm(d_p),
            nn.GELU()
        )

        self.readout = nn.Sequential(
            nn.Linear(d_p, d_A),
            nn.LayerNorm(d_A),
            nn.GELU(),
            nn.Linear(d_A, d_A)
        )

    def forward(self, p_seq: torch.Tensor) -> torch.Tensor:
        """
        输入: p_seq (Batch, 5, 20, 20) 或 (Batch, 5, 400)
        输出: A_traj (Batch, 256)
        """
        # 如果输入是 20x20 的图像格式，先把它展平成 400 维的向量
        if len(p_seq.shape) == 4:
            B, S, H, W = p_seq.shape
            p_seq = p_seq.view(B, S, H * W)

        B, S, D = p_seq.shape
        if S != self.num_stages:
            raise ValueError(f"预期 {self.num_stages} 个物理阶段，得到 {S} 个阶段。")
        if D != self.d_p:
            raise ValueError(f"预期特征维度 {self.d_p}，但得到 {D}。")

        # 1. 预处理 -> (Batch, 5, 400)
        X = self.node_proj(p_seq)

        # 2. 构建转移概率矩阵 M -> (Batch, 5, 5)
        sim_matrix = torch.bmm(X, X.transpose(1, 2)) / self.tau
        M = F.softmax(sim_matrix, dim=-1)

        # 3. 模拟 K 步随机游走
        M_k = M.clone()
        for _ in range(self.walk_steps - 1):
            M_k = torch.bmm(M_k, M)

        # 4. 特征扩散 -> (Batch, 5, 400)
        walked_X = torch.bmm(M_k, X)

        # 5. Readout 汇聚与 L2 归一化 -> (Batch, 256)
        global_repr = walked_X.mean(dim=1)
        A_traj = self.readout(global_repr)
        A_traj = F.normalize(A_traj, p=2, dim=-1)

        return A_traj


# ==========================================
# 真实 .pt 文件测试模块
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 启动 Phase 2 真实 .pt 数据打通测试...")
    print("=" * 50)

    # 指向你刚刚代码里保存特征的目录
    FEATURES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "features"

    if not FEATURES_DIR.exists() or not any(FEATURES_DIR.iterdir()):
        print(f"❌ 找不到特征目录或目录为空: {FEATURES_DIR}")
        exit(1)

    # 找到所有的 .pt 文件
    pt_files = list(FEATURES_DIR.glob("*.pt"))
    print(f"📦 成功找到 {len(pt_files)} 个真实的 .pt 特征文件！")

    # 抽取前 4 个文件作为一个 Batch
    batch_size = min(4, len(pt_files))
    test_files = pt_files[:batch_size]

    p_seq_list = []

    for pt_file in test_files:
        # 读取 .pt 文件
        data = torch.load(pt_file)
        p_seq = data['p_seq']  # 你的代码里写了: feature_pack["p_seq"]

        # 确保它在 CPU 上并且转为 Float
        p_seq_list.append(p_seq.cpu().float())

    # 组合成最终的 Batch -> shape (Batch, 5, 20, 20)
    real_input_tensor = torch.stack(p_seq_list)
    print(f"🎯 成功构建真实输入 Tensor，形状: {real_input_tensor.shape}")

    # 实例化模型 (d_p=400)
    model = MarkovRandomWalkEncoder(d_p=400, d_A=256, num_stages=5)
    model.eval()

    with torch.no_grad():
        real_output_signature = model(real_input_tensor)

    print(f"✅ 网络前向传播成功！")
    print(f"🧬 输出物理轨迹签名 (A_traj) 形状: {real_output_signature.shape}")
    print("=" * 50)
    print("🎉 恭喜！你的真实 .pt 数据已经完美打通了左塔！")
    print("=" * 50)