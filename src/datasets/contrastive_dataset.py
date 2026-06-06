import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.tools.data_roots import resolve_dataset_root, resolve_experiment_root, resolve_meta_path


class CrossModalContrastiveDataset(Dataset):
    """
    Stage-3 cross-modal dataset.

    Teacher side:
        precomputed topology-aware teacher cache (teacher_seq / topo_vec_seq)

    Student side:
        RGB web JPEG image corresponding to the same version_id
    """

    def __init__(
        self,
        teacher_cache_dir: str,
        meta_path: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        teacher_key: str = "teacher_seq",
        transform=None,
    ):
        super().__init__()
        self.teacher_cache_dir = Path(teacher_cache_dir)
        self.meta_path = Path(meta_path) if meta_path is not None else resolve_meta_path()
        self.rgb_dir = Path(rgb_dir) if rgb_dir is not None else resolve_dataset_root() / "rgb_web_jpg"
        self.teacher_key = teacher_key

        if not self.teacher_cache_dir.exists():
            raise FileNotFoundError(f"Teacher cache directory not found: {self.teacher_cache_dir}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"dataset_meta.json not found: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Dict] = json.load(f)

        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        teacher_files = sorted(self.teacher_cache_dir.glob("*.pt"))
        if not teacher_files:
            raise FileNotFoundError(f"No teacher-cache .pt files found in {self.teacher_cache_dir}")

        self.records: List[Dict] = []
        missing_rgb = 0
        missing_meta = 0

        for teacher_path in teacher_files:
            version_id = teacher_path.stem
            info = self.meta.get(version_id)
            if info is None:
                missing_meta += 1
                continue

            rgb_path = self._resolve_rgb_path(version_id, info)
            if rgb_path is None:
                missing_rgb += 1
                continue

            self.records.append(
                {
                    "version_id": version_id,
                    "raw_anchor": info["raw_anchor"],
                    "version": int(info.get("version", 0)),
                    "teacher_path": teacher_path,
                    "rgb_path": rgb_path,
                }
            )

        if missing_meta > 0:
            print(f"Warning: skipped {missing_meta} teacher-cache files without metadata entries.")
        if missing_rgb > 0:
            print(f"Warning: skipped {missing_rgb} teacher-cache files without RGB matches.")
        if not self.records:
            raise RuntimeError(
                f"No valid teacher/RGB pairs found. teacher_cache_dir={self.teacher_cache_dir}, rgb_dir={self.rgb_dir}"
            )

        self.group_ids = [record["raw_anchor"] for record in self.records]

    def _resolve_rgb_path(self, version_id: str, info: Dict) -> Optional[Path]:
        jpg_path = info.get("paths", {}).get("jpg")
        if jpg_path:
            path = Path(jpg_path)
            if path.exists():
                return path

        candidate = self.rgb_dir / f"{version_id}.jpg"
        if candidate.exists():
            return candidate

        # fallback to a loose scan for safety
        matches = list(self.rgb_dir.glob(f"{version_id}.*"))
        return matches[0] if matches else None

    def _load_teacher_seq(self, teacher_path: Path) -> torch.Tensor:
        pack = torch.load(teacher_path, map_location="cpu")
        if self.teacher_key in pack:
            seq = pack[self.teacher_key]
        elif self.teacher_key == "teacher_seq" and "topo_vec_seq" in pack:
            seq = pack["topo_vec_seq"]
        else:
            available = ", ".join(sorted(str(k) for k in pack.keys()))
            raise KeyError(f"Teacher key '{self.teacher_key}' not found in {teacher_path.name}. Available: {available}")

        seq = torch.as_tensor(seq, dtype=torch.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected teacher sequence with shape (K, D), got {tuple(seq.shape)} from {teacher_path}")
        return seq

    def _load_rgb_tensor(self, rgb_path: Path) -> torch.Tensor:
        img_data = np.fromfile(str(rgb_path), dtype=np.uint8)
        img_np = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError(f"Image decode failed: {rgb_path}")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(img_np))

    @property
    def teacher_dim(self) -> int:
        return int(self._load_teacher_seq(self.records[0]["teacher_path"]).shape[-1])

    @property
    def num_stages(self) -> int:
        return int(self._load_teacher_seq(self.records[0]["teacher_path"]).shape[0])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        teacher_seq = self._load_teacher_seq(record["teacher_path"])
        rgb_tensor = self._load_rgb_tensor(record["rgb_path"])
        return {
            "sample_id": record["version_id"],
            "raw_anchor": record["raw_anchor"],
            "teacher_seq": teacher_seq,
            "p_seq": teacher_seq,  # backward-compatible alias for legacy code
            "rgb_image": rgb_tensor,
        }


if __name__ == "__main__":
    dataset = CrossModalContrastiveDataset(
        teacher_cache_dir=str(resolve_experiment_root() / "stage3_teacher_cache"),
    )
    print(f"Loaded {len(dataset)} valid teacher/RGB pairs.")
    print(f"Teacher sequence shape: ({dataset.num_stages}, {dataset.teacher_dim})")
