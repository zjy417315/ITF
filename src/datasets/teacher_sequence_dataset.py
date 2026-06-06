import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset

from src.tools.data_roots import resolve_experiment_root, resolve_meta_path


class TeacherSequenceDataset(Dataset):
    def __init__(
        self,
        prototype_cache_dir: Optional[str] = None,
        teacher_cache_dir: Optional[str] = None,
        meta_path: Optional[str] = None,
        include_versions: Optional[Iterable[int]] = None,
        teacher_key: str = "teacher_seq",
        prototype_key: str = "prototype_seq",
    ):
        super().__init__()
        self.prototype_cache_dir = (
            Path(prototype_cache_dir)
            if prototype_cache_dir is not None
            else resolve_experiment_root() / "stage3_prototype_cache_anchor12"
        )
        self.teacher_cache_dir = (
            Path(teacher_cache_dir)
            if teacher_cache_dir is not None
            else resolve_experiment_root() / "stage3_teacher_cache_full"
        )
        self.meta_path = Path(meta_path) if meta_path is not None else resolve_meta_path()
        self.teacher_key = teacher_key
        self.prototype_key = prototype_key
        self.include_versions = None if include_versions is None else sorted({int(v) for v in include_versions})

        if not self.prototype_cache_dir.exists():
            raise FileNotFoundError(f"Prototype cache directory not found: {self.prototype_cache_dir}")
        if not self.teacher_cache_dir.exists():
            raise FileNotFoundError(f"Teacher cache directory not found: {self.teacher_cache_dir}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"dataset_meta.json not found: {self.meta_path}")

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta: Dict[str, Dict] = json.load(f)
        with open(self.prototype_cache_dir / "prototype_index.json", "r", encoding="utf-8") as f:
            prototype_index: Dict[str, str] = json.load(f)

        self.prototype_seq_bank: Dict[str, torch.Tensor] = {}
        for raw_anchor, file_name in prototype_index.items():
            pack = torch.load(self.prototype_cache_dir / file_name, map_location="cpu")
            self.prototype_seq_bank[raw_anchor] = torch.as_tensor(pack[self.prototype_key], dtype=torch.float32)

        self.records: List[Dict] = []
        for teacher_path in sorted(self.teacher_cache_dir.glob("*.pt")):
            pack = torch.load(teacher_path, map_location="cpu")
            version_id = str(pack["version_id"])
            meta_info = self.meta.get(version_id, {})
            raw_anchor = str(pack["raw_anchor"])
            if raw_anchor not in self.prototype_seq_bank:
                continue
            version_num = int(meta_info.get("version", 0))
            if self.include_versions is not None and version_num not in self.include_versions:
                continue
            teacher_seq = torch.as_tensor(pack[self.teacher_key], dtype=torch.float32)
            self.records.append(
                {
                    "version_id": version_id,
                    "raw_anchor": raw_anchor,
                    "version": version_num,
                    "teacher_seq": teacher_seq,
                }
            )

        if not self.records:
            raise RuntimeError("No valid teacher/prototype sequence pairs found.")

        self.group_ids = [record["raw_anchor"] for record in self.records]

    @property
    def sequence_dim(self) -> int:
        return int(self.records[0]["teacher_seq"].shape[-1])

    @property
    def stage_count(self) -> int:
        return int(self.records[0]["teacher_seq"].shape[0])

    def get_prototype_sequence(self, raw_anchor: str) -> torch.Tensor:
        return self.prototype_seq_bank[raw_anchor]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        return {
            "sample_id": record["version_id"],
            "raw_anchor": record["raw_anchor"],
            "version": record["version"],
            "teacher_seq": record["teacher_seq"].clone(),
            "prototype_seq": self.prototype_seq_bank[record["raw_anchor"]].clone(),
        }
