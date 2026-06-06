import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.tools.data_roots import resolve_dataset_root, resolve_experiment_root, resolve_meta_path


class PrototypeDistillationDataset(Dataset):
    def __init__(
        self,
        prototype_cache_dir: str,
        meta_path: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        teacher_cache_dir: Optional[str] = None,
        teacher_key: str = "teacher_seq",
        include_versions: Optional[Iterable[int]] = None,
        transform=None,
    ):
        super().__init__()
        self.prototype_cache_dir = Path(prototype_cache_dir)
        self.meta_path = Path(meta_path) if meta_path is not None else resolve_meta_path()
        self.rgb_dir = Path(rgb_dir) if rgb_dir is not None else resolve_dataset_root() / "rgb_web_jpg"
        self.teacher_cache_dir = (
            Path(teacher_cache_dir)
            if teacher_cache_dir is not None
            else resolve_experiment_root() / "stage3_teacher_cache_full"
        )
        self.teacher_key = teacher_key

        if not self.prototype_cache_dir.exists():
            raise FileNotFoundError(f"Prototype cache directory not found: {self.prototype_cache_dir}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"dataset_meta.json not found: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Dict] = json.load(f)
        self.include_versions = None if include_versions is None else sorted({int(v) for v in include_versions})

        index_path = self.prototype_cache_dir / "prototype_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"prototype_index.json not found: {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            self.prototype_index: Dict[str, str] = json.load(f)

        self.prototype_bank: Dict[str, torch.Tensor] = {}
        self.prototype_seq_bank: Dict[str, torch.Tensor] = {}
        for raw_anchor, file_name in self.prototype_index.items():
            pack = torch.load(self.prototype_cache_dir / file_name, map_location="cpu")
            self.prototype_bank[raw_anchor] = torch.as_tensor(pack["prototype_vec"], dtype=torch.float32)
            if "prototype_seq" in pack:
                self.prototype_seq_bank[raw_anchor] = torch.as_tensor(pack["prototype_seq"], dtype=torch.float32)

        self.transform = transform or T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.records: List[Dict] = []
        for version_id, info in sorted(self.meta.items()):
            raw_anchor = info["raw_anchor"]
            if raw_anchor not in self.prototype_bank:
                continue
            version_num = int(info.get("version", 0))
            if self.include_versions is not None and version_num not in self.include_versions:
                continue
            rgb_path = self._resolve_rgb_path(version_id, info)
            if rgb_path is None:
                continue
            self.records.append(
                {
                    "version_id": version_id,
                    "raw_anchor": raw_anchor,
                    "version": version_num,
                    "rgb_path": rgb_path,
                }
            )

        if not self.records:
            raise RuntimeError("No valid RGB/prototype records found for prototype distillation.")

        self.teacher_vec_bank: Dict[str, torch.Tensor] = {}
        self.teacher_seq_bank: Dict[str, torch.Tensor] = {}
        if self.teacher_cache_dir.exists():
            for record in self.records:
                version_id = record["version_id"]
                teacher_path = self.teacher_cache_dir / f"{version_id}.pt"
                if not teacher_path.exists():
                    continue
                pack = torch.load(teacher_path, map_location="cpu")
                seq = torch.as_tensor(pack[self.teacher_key], dtype=torch.float32)
                self.teacher_seq_bank[version_id] = seq
                self.teacher_vec_bank[version_id] = F.normalize(seq.mean(dim=0), dim=0)

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
        matches = list(self.rgb_dir.glob(f"{version_id}.*"))
        return matches[0] if matches else None

    def _load_rgb_tensor(self, rgb_path: Path) -> torch.Tensor:
        img_data = np.fromfile(str(rgb_path), dtype=np.uint8)
        img_np = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img_np is None:
            raise ValueError(f"Image decode failed: {rgb_path}")
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(img_np))

    @property
    def prototype_dim(self) -> int:
        return int(next(iter(self.prototype_bank.values())).numel())

    @property
    def raw_anchors(self):
        return sorted(self.prototype_bank.keys())

    def get_prototype_tensor(self, raw_anchor: str) -> torch.Tensor:
        return self.prototype_bank[raw_anchor]

    def get_prototype_sequence(self, raw_anchor: str) -> torch.Tensor:
        if raw_anchor not in self.prototype_seq_bank:
            raise KeyError(f"Prototype sequence not found for raw anchor: {raw_anchor}")
        return self.prototype_seq_bank[raw_anchor]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "sample_id": record["version_id"],
            "raw_anchor": record["raw_anchor"],
            "version": record["version"],
            "prototype_vec": self.prototype_bank[record["raw_anchor"]].clone(),
            "prototype_seq": self.prototype_seq_bank.get(record["raw_anchor"], self.prototype_bank[record["raw_anchor"]].unsqueeze(0)).clone(),
            "teacher_vec": self.teacher_vec_bank.get(record["version_id"], self.prototype_bank[record["raw_anchor"]]).clone(),
            "teacher_seq": self.teacher_seq_bank.get(record["version_id"], self.prototype_bank[record["raw_anchor"]].unsqueeze(0)).clone(),
            "rgb_image": self._load_rgb_tensor(record["rgb_path"]),
        }
