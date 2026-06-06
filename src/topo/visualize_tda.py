import os
import sys
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path

# 将项目根目录加入环境变量
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def visualize_tda_evolution(num_samples=3, save_filename="tda_evolution.png"):
    """
    随机抽取样本，可视化其拓扑特征在 ISP 各阶段的演变
    """
    feature_dir = PROJECT_ROOT / "data" / "features"
    feature_files = list(feature_dir.glob("*.pt"))

    if not feature_files:
        print("❌ 找不到特征文件，请检查 data/features/ 目录！")
        return

    print(f"🔍 找到 {len(feature_files)} 个特征文件，准备随机抽取 {num_samples} 个进行可视化...")
    sampled_files = random.sample(feature_files, min(num_samples, len(feature_files)))

    # 你的 ISP 5 个阶段
    stage_names = ["RAW", "Demosaic", "Denoise", "Color", "RGB"]

    # 创建画布 (行数 = 样本数, 列数 = 5 个阶段)
    fig, axes = plt.subplots(len(sampled_files), 5, figsize=(18, 3.5 * len(sampled_files)))

    # 统一降维处理（防止只抽1个样本时 axes 是一维数组报错）
    if len(sampled_files) == 1:
        axes = [axes]

    for i, file_path in enumerate(sampled_files):
        img_id = file_path.stem
        # 加载离线特征 (weights_only=True 保持安全好习惯)
        feature_pack = torch.load(file_path, weights_only=True)
        p_seq = feature_pack["p_seq"].numpy()  # 提取拓扑特征，shape: (5, 20, 20)

        for j in range(5):
            ax = axes[i][j]
            pi_matrix = p_seq[j]

            # 绘制 20x20 的热力图
            # cmap='magma' 或 'viridis' 是学术界非常喜欢的拓扑可视化配色
            im = ax.imshow(pi_matrix, cmap='magma', origin='lower')

            # 设置标题和排版
            if i == 0:
                ax.set_title(f"Stage: {stage_names[j]}", fontsize=14, fontweight='bold', pad=10)
            if j == 0:
                ax.set_ylabel(f"Sample: {img_id}", fontsize=12, fontweight='bold', labelpad=10)

            ax.set_xticks([])
            ax.set_yticks([])

            # 为每个子图添加颜色条，方便观察数值范围的微观变化
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # 保存高清图供论文使用
    save_path = PROJECT_ROOT / "data" / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化完成！高清热力图已保存至: {save_path}")

    # 弹出展示窗口
    plt.show()


if __name__ == "__main__":
    # 你可以修改 num_samples 来决定一次看几个样本
    visualize_tda_evolution(num_samples=3)