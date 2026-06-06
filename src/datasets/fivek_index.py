import os
from pathlib import Path
import json


class FiveKIndex:
    def __init__(self, root_dir):
        """
        专为 MIT-Adobe FiveK 官方目录结构设计的数据索引器
        root_dir: 应该是 temp_download/MITAboveFiveK 的路径
        """
        self.root_dir = Path(root_dir)
        self.raw_dir = self.root_dir / "raw"
        self.expert_a_dir = self.root_dir / "processed" / "tiff16_a"
        self.expert_b_dir = self.root_dir / "processed" / "tiff16_b"

        self.samples = []
        self._build_index()

    def _build_index(self):
        print("🔍 正在扫描 FiveK 数据集进行严格配对...")

        # 1. 递归扫描所有的 DNG 文件 (穿透相机子文件夹)
        raw_files = list(self.raw_dir.rglob("*.dng"))

        valid_count = 0
        for raw_path in raw_files:
            basename = raw_path.stem  # 例如: a0001-jmac_DSC1459

            # 2. 寻找对应的 Expert A 和 Expert B
            exp_a_path = self.expert_a_dir / f"{basename}.tif"
            exp_b_path = self.expert_b_dir / f"{basename}.tif"

            # 3. 严格校验三者是否同时存在
            if exp_a_path.exists() and exp_b_path.exists():
                self.samples.append({
                    "basename": basename,
                    "raw_path": str(raw_path),
                    "exp_a_path": str(exp_a_path),
                    "exp_b_path": str(exp_b_path)
                })
                valid_count += 1

        print(f"✅ 扫描完毕！成功配对 {valid_count} 组 (RAW + Expert A + Expert B) 完整数据。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# 测试代码
if __name__ == "__main__":
    # 请根据你的实际路径修改
    data_root = r"<artifact-local-path-redacted>"
    index = FiveKIndex(data_root)
    if len(index) > 0:
        print("\n✨ 抽查第一个样本:")
        print(json.dumps(index[0], indent=4, ensure_ascii=False))