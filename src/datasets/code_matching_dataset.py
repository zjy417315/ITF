import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.tools.data_roots import resolve_dataset_root, resolve_meta_path


class CodeMatchingDataset(Dataset):
    def __init__(
        self,
        prototype_cache_dir: str,
        meta_path: Optional[str] = None,
        rgb_dir: Optional[str] = None,
        include_versions: Optional[Iterable[int]] = None,
        transform=None,
    ):
        super().__init__()
        self.prototype_cache_dir = Path(prototype_cache_dir)
        self.meta_path = Path(meta_path) if meta_path is not None else resolve_meta_path()
        self.rgb_dir = Path(rgb_dir) if rgb_dir is not None else resolve_dataset_root() / "rgb_web_jpg"

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

        self.code_bank: Dict[str, torch.Tensor] = {}
        self.code_seq_bank: Dict[str, torch.Tensor] = {}
        for raw_anchor, file_name in self.prototype_index.items():
            pack = torch.load(self.prototype_cache_dir / file_name, map_location="cpu")
            self.code_bank[raw_anchor] = torch.as_tensor(pack["prototype_vec"], dtype=torch.float32)
            if "prototype_seq" in pack:
                self.code_seq_bank[raw_anchor] = torch.as_tensor(pack["prototype_seq"], dtype=torch.float32)

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
            if raw_anchor not in self.code_bank:
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
            raise RuntimeError("No valid RGB/code records found for stage3 code matching.")

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
    def code_dim(self) -> int:
        return int(next(iter(self.code_bank.values())).numel())

    @property
    def raw_anchors(self):
        return sorted(self.code_bank.keys())

    def get_code_tensor(self, raw_anchor: str) -> torch.Tensor:
        return self.code_bank[raw_anchor]

    def get_code_sequence(self, raw_anchor: str) -> torch.Tensor:
        if raw_anchor not in self.code_seq_bank:
            raise KeyError(f"Code sequence not found for raw anchor: {raw_anchor}")
        return self.code_seq_bank[raw_anchor]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        code_vec = self.code_bank[record["raw_anchor"]]
        code_seq = self.code_seq_bank.get(record["raw_anchor"], code_vec.unsqueeze(0))
        return {
            "sample_id": record["version_id"],
            "raw_anchor": record["raw_anchor"],
            "version": record["version"],
            "claim_code": code_vec.clone(),
            "claim_code_seq": code_seq.clone(),
            "rgb_image": self._load_rgb_tensor(record["rgb_path"]),
        }
