import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gudhi
import cv2
from persim import PersistenceImager

# 将项目根目录加入环境变量
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.raw_index import RawIndex
from src.isp.simple_isp import SimpleISP, ISPConfig
from src.models.backbone import FeatureBackbone


class TopoFeatureExtractor:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 初始化拓扑特征提取器，设备: {self.device}")

        # 0. 将该类型加入安全白名单
        torch.serialization.add_safe_globals([argparse.Namespace])

        # 1. 挂载满级神装 (ResNet-50)
        self.backbone = FeatureBackbone(d_f=512, C_f=256).to(self.device)

        # 2. 读取并加载权重 (保持 weights_only=True)
        # 删去了上面冗余的 state_dict 读取，统一放到这里
        state_dict = torch.load(checkpoint_path, map_location=self.device)['backbone']
        self.backbone.load_state_dict(state_dict)
        self.backbone.eval()  # 极其重要：切断 Dropout 和 BatchNorm 的更新

        # 2. 初始化 ISP (开启中心裁剪，保证提取指纹的唯一性和确定性)
        self.isp = SimpleISP(config=ISPConfig(center_crop=True))
        self.stage_order = ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]

        # 3. 初始化持久同调图像生成器 (Persistence Imager)
        # 将拓扑散点图(PD)转化为 20x20 的高维矩阵(PI)供网络学习
        self.pimager = PersistenceImager(pixel_size=0.05, birth_range=(0, 1), pers_range=(0, 1))
        self.pimager.kernel_params = {'sigma': 0.05}

        # 4. 创建保存目录
        self.save_dir = PROJECT_ROOT / "data" / "features"
        os.makedirs(self.save_dir, exist_ok=True)

    def extract_tda_from_feature_map(self, F_k: torch.Tensor) -> np.ndarray:
        """
        核心数学逻辑：将 256x7x7 的特征图转化为标量场，计算 Cubical Homology (拓扑空洞)
        """
        # 将通道维度取平均，得到 7x7 的 2D 标量场 (激活强度分布)
        scalar_field = F_k.mean(dim=0).cpu().numpy()  # shape: (7, 7)

        # 为了防止全 0 或常数导致无法计算拓扑，加入极小的微扰
        scalar_field += np.random.uniform(0, 1e-5, scalar_field.shape)

        # 归一化到 [0, 1] 区间，统一拓扑阈值范围
        scalar_field = (scalar_field - scalar_field.min()) / (scalar_field.max() - scalar_field.min() + 1e-8)

        # 构建立方复形 (Cubical Complex)
        cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=scalar_field)
        cubical_complex.compute_persistence()
        pd = cubical_complex.persistence()

        # 提取一维拓扑空洞 (H1 Homology - 代表连通分量间的环状结构，这是 ISP 指纹最密集的地方)
        h1_intervals = [p[1] for p in pd if p[0] == 1]

        # 转化为持久图像 PI (Persistence Image)
        if len(h1_intervals) == 0:
            pi = np.zeros(self.pimager.resolution)  # (20, 20)
        else:
            pi = self.pimager.transform(np.array(h1_intervals))

        return pi.astype(np.float32)

    @torch.no_grad()
    def process_dataset(self, data_dir: str):
        raw_index = RawIndex.from_folder(data_dir)
        # 1. 加上下划线获取列表
        all_samples = raw_index._samples

        print(f"📦 找到 {len(all_samples)} 个样本，准备离线提取特征...")

        for sample in tqdm(all_samples, desc="提取拓扑 & 语义特征"):
            # 2. 改用对象属性的点号调用，并使用准确的属性名 sample_id
            raw_path = sample.raw_path
            img_id = sample.sample_id

            save_path = self.save_dir / f"{img_id}.pt"

            # 如果已经提取过，就跳过 (支持断点续传)
            if save_path.exists():
                continue

            # 1. 运行 ISP 拿到 5 个阶段图像
            stages_dict = self.isp.run(raw_path)

            feature_pack = {
                "f_seq": [],  # 存储全局语义特征 f_k
                "p_seq": []  # 存储拓扑特征 p_k (PI)
            }

            for stage_name in self.stage_order:
                # 图像转换为 Tensor (1, C, 224, 224)
                # 图像转换为 Tensor
                img_np = stages_dict[stage_name]

                # 【新增】：解决 OpenCV 不支持 float64 (CV_64F) 的问题
                if img_np.dtype == np.float64:
                    img_np = img_np.astype(np.float32)

                img_np = cv2.resize(img_np, (224, 224))

                # 【修改这里】：如果是单通道 (例如 raw 阶段)，转为 3 通道再送给骨干网络
                if len(img_np.shape) == 2 or img_np.shape[-1] == 1:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                # 此时 img_np 一定是 (224, 224, 3)，变成 Tensor 为 (1, 3, 224, 224)
                img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

                # 2. 传入 ResNet-50 骨干网络
                # 修改 Backbone 让它同时返回 f_k 和 F_k
                features = self.backbone.backbone(img_tensor)  # (1, 2048, 7, 7)
                F_k = self.backbone.map_proj(features)  # (1, 256, 7, 7)

                # 池化并降维得到 f_k (1, 512)
                # 【修改这里】：应该对 2048 通道的 features 进行池化，而不是 256 通道的 F_k
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                f_k = self.backbone.global_proj(pooled).squeeze(0)  # 2048 -> 512

                # 3. 计算拓扑特征 PI
                pi = self.extract_tda_from_feature_map(F_k.squeeze(0))  # 传入 (256, 7, 7)

                feature_pack["f_seq"].append(f_k.cpu().clone())
                feature_pack["p_seq"].append(torch.from_numpy(pi))

            # 4. 堆叠并保存为离线 Tensor 字典
            feature_pack["f_seq"] = torch.stack(feature_pack["f_seq"])  # (5, 512)
            feature_pack["p_seq"] = torch.stack(feature_pack["p_seq"])  # (5, 20, 20)

            torch.save(feature_pack, save_path)

        print("✅ 所有拓扑与语义特征离线提取完毕！保存至: data/features/")


if __name__ == "__main__":
    # 请确保 checkpoint 名字和你跑出来的一致
    CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints" / "stage1" / "stage1_best.pt")
    DATA_DIR = str(PROJECT_ROOT / "data")

    extractor = TopoFeatureExtractor(CHECKPOINT_PATH)
    extractor.process_dataset(DATA_DIR)