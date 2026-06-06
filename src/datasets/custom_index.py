import json
from pathlib import Path


class CustomForensicsIndex:
    """
    专门为我们自己制造的“轻量化/多裂变”溯源数据集编写的读取索引。
    直接通过 dataset_meta.json 账本进行极速挂载！
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.meta_path = self.dataset_dir / "dataset_meta.json"

        self.samples = []
        self._build_index()

    def _build_index(self):
        print(f"正在读取专属数据集元数据: {self.meta_path}")

        if not self.meta_path.exists():
            print("未找到 dataset_meta.json，请确认数据构建脚本已运行完毕。")
            return

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        valid_count = 0
        for version_id, info in meta_data.items():
            raw_filename = info["raw_anchor"]

            # 拼接绝对路径
            raw_path = self.dataset_dir / "raw" / raw_filename
            png_path = self.dataset_dir / "rgb_clean_png" / f"{version_id}.png"
            jpg_path = self.dataset_dir / "rgb_web_jpg" / f"{version_id}.jpg"

            # 快速校验文件是否存在
            if raw_path.exists() and jpg_path.exists():
                self.samples.append({
                    "version_id": version_id,
                    "raw_path": str(raw_path),
                    "png_path": str(png_path),  # 干净正样本
                    "jpg_path": str(jpg_path),  # 网络传播正样本 (用作模型训练的 View A/B)
                    "meta": info
                })
                valid_count += 1

        print(f"挂载完成，可用样本版本数: {valid_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
