# 给训练用的 Dataset 提供“raw + 对应 rgb 路径”，这一步不做 ISP

#class RawRgbDataset(torch.utils.data.Dataset):

#raw_index: RawIndex
#split: str —— 'train', 'val' 或 'test'
#__getitem__(self, idx: int) -> Dict：输入：样本索引 idx,输出：字典,例如：{
 # "raw_path": str,    # raw 文件路径
 # "rgb_path": str,    # 原始 rgb 文件路径（如果有）
 # "sample_id": int    # 全局唯一 ID
#}

"""
raw-rgb Dataset 模块
===================

为训练提供 Dataset，返回 raw + 对应 rgb 的路径信息。
这一层不做 ISP 处理，只负责索引管理。

Classes:
    RawRgbDataset: PyTorch Dataset，返回样本路径和元信息
"""

import os
import logging
from typing import Dict, Optional, Callable, Any, List, Tuple
from pathlib import Path
from ..utils import env  # noqa: F401  # 确保环境变量先被设置
import torch
from torch.utils.data import Dataset, DataLoader

from .raw_index import RawIndex, RawSample, SplitConfig, SplitType

logger = logging.getLogger(__name__)


class RawRgbDataset(Dataset):
    """
    raw-rgb 路径级 Dataset

    提供 raw 和对应 rgb 的路径信息，不做实际的图像处理。
    图像处理由下游的 PathDataset 调用 ISP 完成。

    Attributes:
        raw_index: 数据索引
        split: 数据集划分类型
        transform: 可选的元数据变换（不是图像变换）

    Example:
        >>> index = RawIndex.from_folder("./data")
        >>> dataset = RawRgbDataset(index, split="train")
        >>> sample = dataset[0]
        >>> print(sample["raw_path"], sample["sample_id"])
    """

    def __init__(
            self,
            raw_index: RawIndex,
            split: str = "all",
            split_config: Optional[SplitConfig] = None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            require_rgb: bool = False
    ):
        """
        初始化 Dataset

        Args:
            raw_index: RawIndex 数据索引对象
            split: 数据集划分 ("train", "val", "test", "all")
            split_config: 划分配置，默认 80/10/10
            transform: 可选的元数据变换函数
            require_rgb: 是否要求样本必须有对应RGB
        """
        super().__init__()

        self.split = split
        self.transform = transform
        self.require_rgb = require_rgb

        # 根据 split 获取对应的索引
        if split == "all" or split == SplitType.ALL:
            self._index = raw_index
        else:
            if split_config is None:
                split_config = SplitConfig()

            train_idx, val_idx, test_idx = raw_index.train_val_test_split(split_config)

            if split == "train" or split == SplitType.TRAIN:
                self._index = train_idx
            elif split == "val" or split == SplitType.VAL:
                self._index = val_idx
            elif split == "test" or split == SplitType.TEST:
                self._index = test_idx
            else:
                raise ValueError(f"Unknown split: {split}")

        # 如果要求RGB，过滤掉没有RGB的样本
        if require_rgb:
            self._index = self._index.filter_by_has_rgb(True)

        logger.info(f"RawRgbDataset [{split}]: {len(self._index)} samples")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            字典，包含:
                - raw_path: str, raw 文件路径
                - rgb_path: Optional[str], rgb 文件路径
                - sample_id: int, 全局唯一 ID
                - camera_model: Optional[str], 相机型号
                - meta: Dict, 其他元信息
        """
        sample: RawSample = self._index.get_sample(idx)

        result = {
            "raw_path": sample.raw_path,
            "rgb_path": sample.rgb_path,
            "sample_id": sample.sample_id,
            "camera_model": sample.camera_model,
            "scene_id": sample.scene_id,
            "meta": sample.meta,
            "index": idx,  # 在当前 split 中的索引
        }

        # 应用变换
        if self.transform is not None:
            result = self.transform(result)

        return result

    def get_sample_by_id(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        按 sample_id 获取样本

        Args:
            sample_id: 样本ID

        Returns:
            样本字典，如果不存在返回 None
        """
        sample = self._index.get_sample_by_id(sample_id)
        if sample is None:
            return None

        # 找到对应的 idx
        for idx in range(len(self._index)):
            if self._index[idx].sample_id == sample_id:
                return self[idx]

        return None

    def get_all_paths(self) -> List[Tuple[str, Optional[str]]]:
        """
        获取所有样本的路径对

        Returns:
            [(raw_path, rgb_path), ...] 列表
        """
        return [(s.raw_path, s.rgb_path) for s in self._index]

    def get_camera_distribution(self) -> Dict[str, int]:
        """获取相机分布统计"""
        stats = self._index.get_statistics()
        return stats.get("camera_models", {})

    @property
    def index(self) -> RawIndex:
        """获取底层的 RawIndex"""
        return self._index

    def subset(self, indices: List[int]) -> "RawRgbDataset":
        """
        获取子集

        Args:
            indices: 索引列表

        Returns:
            新的 RawRgbDataset 子集
        """
        sub_index = self._index.subset(indices)

        # 创建新的 dataset
        new_dataset = RawRgbDataset.__new__(RawRgbDataset)
        new_dataset._index = sub_index
        new_dataset.split = self.split
        new_dataset.transform = self.transform
        new_dataset.require_rgb = self.require_rgb

        return new_dataset

    @classmethod
    def from_folder(
            cls,
            root_dir: str,
            split: str = "all",
            split_config: Optional[SplitConfig] = None,
            **kwargs
    ) -> "RawRgbDataset":
        """
        便捷方法：直接从文件夹创建 Dataset

        Args:
            root_dir: 数据根目录
            split: 数据集划分
            split_config: 划分配置
            **kwargs: 传递给构造函数的其他参数

        Returns:
            RawRgbDataset 对象
        """
        index = RawIndex.from_folder(root_dir)
        return cls(index, split=split, split_config=split_config, **kwargs)


def create_dataloaders(
        raw_index: RawIndex,
        split_config: Optional[SplitConfig] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        require_rgb: bool = False,
        **dataloader_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    便捷函数：创建训练/验证/测试 DataLoader

    Args:
        raw_index: 数据索引
        split_config: 划分配置
        batch_size: 批大小
        num_workers: 工作进程数
        require_rgb: 是否要求有RGB
        **dataloader_kwargs: 传递给 DataLoader 的其他参数

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = RawRgbDataset(
        raw_index, split="train", split_config=split_config, require_rgb=require_rgb
    )
    val_dataset = RawRgbDataset(
        raw_index, split="val", split_config=split_config, require_rgb=require_rgb
    )
    test_dataset = RawRgbDataset(
        raw_index, split="test", split_config=split_config, require_rgb=require_rgb
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **dataloader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **dataloader_kwargs
    )

    return train_loader, val_loader, test_loader


# 用于 DataLoader 的自定义 collate 函数
def raw_rgb_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    自定义 collate 函数

    将多个样本字典合并为批次字典。
    路径和元信息保持为列表。

    Args:
        batch: 样本字典列表

    Returns:
        批次字典
    """
    return {
        "raw_path": [item["raw_path"] for item in batch],
        "rgb_path": [item["rgb_path"] for item in batch],
        "sample_id": torch.tensor([item["sample_id"] for item in batch]),
        "camera_model": [item["camera_model"] for item in batch],
        "scene_id": [item["scene_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch]),
    }


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./data"

    print(f"Creating dataset from: {data_dir}")

    try:
        # 从文件夹创建
        dataset = RawRgbDataset.from_folder(data_dir, split="all")
        print(f"\nDataset size: {len(dataset)}")

        if len(dataset) > 0:
            # 获取第一个样本
            sample = dataset[0]
            print(f"\nFirst sample:")
            print(f"  raw_path: {sample['raw_path']}")
            print(f"  rgb_path: {sample['rgb_path']}")
            print(f"  sample_id: {sample['sample_id']}")
            print(f"  camera_model: {sample['camera_model']}")

            # 测试 DataLoader
            print("\nTesting DataLoader...")
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                collate_fn=raw_rgb_collate_fn
            )

            for batch in loader:
                print(f"Batch size: {len(batch['raw_path'])}")
                print(f"Sample IDs: {batch['sample_id']}")
                break

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please download dataset first using: python scripts/download_dataset.py")
