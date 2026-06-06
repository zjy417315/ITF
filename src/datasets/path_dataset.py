import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import h5py
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class PathSample:
    stage_images: Dict[str, torch.Tensor]
    sample_id: str
    raw_path: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def stage_names(self) -> List[str]:
        return list(self.stage_images.keys())


class PathDataset(Dataset):
    """
    Dataset for stage-wise imaging-path learning.

    Each item is a pair of versions that originate from the same RAW anchor:
    view A is the anchor version, and view B is another cached variant.
    """

    def __init__(
        self,
        fivek_index,
        mode: str = "dual_view",
        cache_dir: str = "data/my_forensics_dataset/stage_cache",
        is_val: bool = False,
        deterministic_val: bool = True,
    ):
        self.index = fivek_index
        self.mode = mode
        self.is_val = is_val
        self.deterministic_val = deterministic_val

        cache_path = Path(cache_dir)
        if cache_path.suffix.lower() == ".h5":
            self.h5_path = cache_path
            self.pt_cache_dir = cache_path.with_suffix("")
        else:
            self.pt_cache_dir = cache_path
            self.h5_path = cache_path.parent / "stage_cache.h5"

        self.h5_file = None
        self._h5_disabled = False
        self._h5_disable_reason: Optional[str] = None
        self._h5_stage_ids: Set[str] = set()
        self._pt_stage_ids: Set[str] = set()

        self.stage_order = ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]
        self.available_stage_ids = self._discover_available_stage_ids()

        self.raw_groups = defaultdict(list)
        skipped_missing_cache = 0
        for sample in self.index.samples:
            version_id = self._get_sample_version_id(sample)
            if version_id not in self.available_stage_ids:
                skipped_missing_cache += 1
                continue

            raw_anchor = sample["meta"]["raw_anchor"]
            self.raw_groups[raw_anchor].append(sample)

        skipped_insufficient_views = 0
        invalid_raws = [raw for raw, versions in self.raw_groups.items() if len(versions) < 2]
        for raw_anchor in invalid_raws:
            skipped_insufficient_views += len(self.raw_groups[raw_anchor])
            del self.raw_groups[raw_anchor]

        self.unique_raws = sorted(self.raw_groups.keys())
        if not self.unique_raws:
            raise RuntimeError("No RAW anchors with at least two cached versions were found.")

        self.filtered_samples = [sample for versions in self.raw_groups.values() for sample in versions]
        self.raw_to_instance_label = {raw: i for i, raw in enumerate(self.unique_raws)}

        path_type_keys = [self._get_path_type_key(sample.get("meta", {})) for sample in self.filtered_samples]
        self.path_type_vocab = {key: i for i, key in enumerate(sorted(set(path_type_keys)))}
        self.id_to_path_type = {idx: key for key, idx in self.path_type_vocab.items()}

        backend_parts = []
        if self._pt_stage_ids:
            backend_parts.append(f"pt={len(self._pt_stage_ids)}")
        if self._h5_stage_ids:
            backend_parts.append(f"h5={len(self._h5_stage_ids)}")
        backend_summary = ", ".join(backend_parts) if backend_parts else "none"

        logger.info(
            "PathDataset initialized | raw scenes=%d | cached versions=%d | path types=%d | backends=%s | val=%s",
            len(self.unique_raws),
            len(self.filtered_samples),
            len(self.path_type_vocab),
            backend_summary,
            self.is_val,
        )
        logger.info("Path vocab: %s", self.path_type_vocab)
        if skipped_missing_cache > 0:
            logger.warning("Skipped %d versions without stage cache.", skipped_missing_cache)
        if skipped_insufficient_views > 0:
            logger.warning("Skipped %d cached versions because fewer than 2 views remained.", skipped_insufficient_views)
        if self._h5_disable_reason is not None:
            logger.warning("HDF5 cache disabled, fallback to PT cache only. Reason: %s", self._h5_disable_reason)

    def __len__(self) -> int:
        return len(self.unique_raws)

    def __del__(self):
        if getattr(self, "h5_file", None) is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass

    # =====================================================
    # Cache discovery / loading
    # =====================================================
    def _get_sample_version_id(self, sample: Dict[str, Any]) -> str:
        return sample.get("version_id", sample.get("basename", "unknown"))

    def _disable_h5(self, reason: str):
        self._h5_disabled = True
        self._h5_disable_reason = reason
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass
            self.h5_file = None

    def _discover_available_stage_ids(self) -> Set[str]:
        self._pt_stage_ids = self._discover_pt_stage_ids()
        self._h5_stage_ids = self._discover_h5_stage_ids()
        available = self._pt_stage_ids | self._h5_stage_ids

        if not available:
            raise FileNotFoundError(
                f"No usable stage cache found under PT dir {self.pt_cache_dir} "
                f"or HDF5 file {self.h5_path}"
            )
        return available

    def _discover_pt_stage_ids(self) -> Set[str]:
        if not self.pt_cache_dir.exists() or not self.pt_cache_dir.is_dir():
            return set()
        return {path.stem for path in self.pt_cache_dir.glob("*.pt")}

    def _discover_h5_stage_ids(self) -> Set[str]:
        if self._h5_disabled or not self.h5_path.exists():
            return set()

        try:
            with h5py.File(self.h5_path, "r") as h5f:
                return {str(key) for key in h5f.keys()}
        except Exception as exc:
            self._disable_h5(str(exc))
            return set()

    def _read_stages_from_h5(self, version_id: str) -> Dict[str, torch.Tensor]:
        if self._h5_disabled:
            raise RuntimeError(self._h5_disable_reason or "HDF5 cache is disabled")

        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.h5_path, "r")
            except Exception as exc:
                self._disable_h5(str(exc))
                raise RuntimeError(self._h5_disable_reason) from exc

        stages: Dict[str, torch.Tensor] = {}
        try:
            for stage_name in self.stage_order:
                stages[stage_name] = torch.from_numpy(self.h5_file[f"{version_id}/{stage_name}"][:]).float()
        except Exception as exc:
            self._disable_h5(str(exc))
            raise RuntimeError(self._h5_disable_reason) from exc

        return stages

    def _read_stages_from_pt(self, version_id: str) -> Dict[str, torch.Tensor]:
        pt_path = self.pt_cache_dir / f"{version_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Stage cache file not found: {pt_path}")

        pack = torch.load(pt_path, map_location="cpu")
        stages: Dict[str, torch.Tensor] = {}
        for stage_name in self.stage_order:
            if stage_name not in pack:
                raise KeyError(f"{pt_path.name} is missing stage {stage_name}")

            tensor = pack[stage_name]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            stages[stage_name] = tensor.float()

        return stages

    def _read_stages(self, version_id: str) -> Dict[str, torch.Tensor]:
        if not self._h5_disabled and self.h5_path.exists():
            try:
                return self._read_stages_from_h5(version_id)
            except Exception as exc:
                logger.warning("Failed to read %s from HDF5, falling back to PT cache: %s", version_id, exc)

        return self._read_stages_from_pt(version_id)

    # =====================================================
    # Metadata parsing
    # =====================================================
    def _extract_version_id(self, version_id: str) -> int:
        match = re.search(r"_v(\d+)$", version_id)
        return int(match.group(1)) if match else 0

    def _get_path_type_key(self, meta: Dict[str, Any]) -> str:
        if meta.get("isp_mode") is not None:
            return str(meta["isp_mode"])

        if meta.get("variant_name") is not None:
            return str(meta["variant_name"])

        difficulty = meta.get("difficulty", "unknown")
        version = meta.get("version", None)
        if version is not None:
            return f"{difficulty}_v{version}"

        version_id = meta.get("version_id", "")
        match = re.search(r"_v(\d+)$", version_id)
        if match:
            return f"version_{match.group(1)}"

        return "unknown"

    def _get_path_type_id(self, meta: Dict[str, Any]) -> int:
        key = self._get_path_type_key(meta)
        return self.path_type_vocab[key]

    def _get_jpeg_quality(self, meta: Dict[str, Any]) -> float:
        degradation = meta.get("degradation", {})
        if isinstance(degradation, dict) and "quality" in degradation:
            return float(degradation["quality"])
        if "jpeg_quality" in meta:
            return float(meta["jpeg_quality"])
        return 95.0

    def _get_isp_params(self, meta: Dict[str, Any]) -> Dict[str, float]:
        isp_params = meta.get("isp_parameters", {})
        return {
            "gamma": float(isp_params.get("gamma", 2.20)),
            "saturation": float(isp_params.get("saturation", 1.00)),
            "contrast": float(isp_params.get("contrast", 1.00)),
            "denoise_strength": float(isp_params.get("denoise_strength", 0.12)),
            "jpeg_quality": self._get_jpeg_quality(meta),
        }

    def _make_path_signature(self, meta: Dict[str, Any]) -> torch.Tensor:
        params = self._get_isp_params(meta)
        return torch.tensor(
            [
                (params["gamma"] - 2.20) / 0.20,
                (params["saturation"] - 1.00) / 0.20,
                (params["contrast"] - 1.00) / 0.20,
                (params["denoise_strength"] - 0.12) / 0.12,
                (params["jpeg_quality"] - 90.0) / 10.0,
            ],
            dtype=torch.float32,
        )

    def _get_path_severity(self, meta: Dict[str, Any]) -> int:
        key = self._get_path_type_key(meta)
        if "anchor" in key:
            return 0
        if "easy" in key:
            return 1
        if "medium" in key:
            return 2
        return 3

    def _pack_meta(self, sample: Dict[str, Any], raw_anchor: str) -> Dict[str, Any]:
        meta = sample.get("meta", {})
        version_id = self._get_sample_version_id(sample)
        return {
            "raw_anchor": raw_anchor,
            "version_id": version_id,
            "instance_label": self.raw_to_instance_label[raw_anchor],
            "path_type_key": self._get_path_type_key(meta),
            "path_type_id": self._get_path_type_id(meta),
            "path_signature": self._make_path_signature(meta),
            "path_severity": self._get_path_severity(meta),
            "path_params": self._get_isp_params(meta),
        }

    # =====================================================
    # Ranking supervision
    # =====================================================
    def _rank_target(self, a: float, b: float) -> float:
        if abs(a - b) < 1e-12:
            return 0.5
        return 1.0 if a > b else 0.0

    def _build_rank_targets(self, meta_a: Dict[str, Any], meta_b: Dict[str, Any]) -> Dict[str, float]:
        params_a = meta_a["path_params"]
        params_b = meta_b["path_params"]

        return {
            "gamma": self._rank_target(params_a["gamma"], params_b["gamma"]),
            "saturation": self._rank_target(params_a["saturation"], params_b["saturation"]),
            "contrast": self._rank_target(params_a["contrast"], params_b["contrast"]),
            "denoise_strength": self._rank_target(params_a["denoise_strength"], params_b["denoise_strength"]),
            "jpeg_quality": self._rank_target(params_a["jpeg_quality"], params_b["jpeg_quality"]),
        }

    # =====================================================
    # Sampling
    # =====================================================
    def _build_item(self, idx: int) -> Dict[str, Any]:
        raw_anchor = self.unique_raws[idx]
        versions = sorted(
            self.raw_groups[raw_anchor],
            key=lambda sample: self._extract_version_id(self._get_sample_version_id(sample)),
        )

        if len(versions) < 2:
            raise ValueError(f"Scene {raw_anchor} has fewer than 2 cached versions.")

        sample_a = versions[0]
        version_id_a = self._get_sample_version_id(sample_a)
        stages_a = self._read_stages(version_id_a)
        meta_a = self._pack_meta(sample_a, raw_anchor)

        if self.is_val and self.deterministic_val:
            sample_b = versions[1]
        else:
            sample_b = random.choice(versions[1:])

        version_id_b = self._get_sample_version_id(sample_b)
        stages_b = self._read_stages(version_id_b)
        meta_b = self._pack_meta(sample_b, raw_anchor)

        rank_targets = self._build_rank_targets(meta_a, meta_b)

        return {
            "view_a": PathSample(stages_a, version_id_a, sample_a["raw_path"], meta=meta_a),
            "view_b": PathSample(stages_b, version_id_b, sample_b["raw_path"], meta=meta_b),
            "labels_a": meta_a["path_type_id"],
            "labels_b": meta_b["path_type_id"],
            "instance_label": meta_a["instance_label"],
            "path_signature_a": meta_a["path_signature"],
            "path_signature_b": meta_b["path_signature"],
            "rank_targets": rank_targets,
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        max_attempts = min(8, len(self.unique_raws))
        current_idx = idx
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                return self._build_item(current_idx)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Stage sample read failed (idx=%d, attempt=%d/%d): %s",
                    current_idx,
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                current_idx = random.randint(0, len(self.unique_raws) - 1)

        raise RuntimeError(
            f"Failed to load a valid path sample after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )


def path_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    view_a = [item["view_a"] for item in batch]
    view_b = [item["view_b"] for item in batch]

    collated_a = _collate_samples(view_a)
    collated_b = _collate_samples(view_b)

    labels_a = torch.tensor([item["labels_a"] for item in batch], dtype=torch.long)
    labels_b = torch.tensor([item["labels_b"] for item in batch], dtype=torch.long)
    instance_labels = torch.tensor([item["instance_label"] for item in batch], dtype=torch.long)

    path_signature_a = torch.stack([item["path_signature_a"] for item in batch], dim=0)
    path_signature_b = torch.stack([item["path_signature_b"] for item in batch], dim=0)

    rank_targets: Dict[str, torch.Tensor] = {}
    rank_names = batch[0]["rank_targets"].keys()
    for name in rank_names:
        rank_targets[name] = torch.tensor([item["rank_targets"][name] for item in batch], dtype=torch.float32)

    return {
        "view_a": collated_a,
        "view_b": collated_b,
        "labels_a": labels_a,
        "labels_b": labels_b,
        "instance_labels": instance_labels,
        "path_signature_a": path_signature_a,
        "path_signature_b": path_signature_b,
        "rank_targets": rank_targets,
    }


def _collate_samples(samples: List[PathSample]) -> Dict[str, Any]:
    stage_names = samples[0].stage_names
    stage_tensors = {}
    for stage_name in stage_names:
        stage_tensors[stage_name] = torch.stack([sample.stage_images[stage_name] for sample in samples], dim=0)

    return {
        "stages": stage_tensors,
        "sample_ids": [sample.sample_id for sample in samples],
        "raw_paths": [sample.raw_path for sample in samples],
        "metas": [sample.meta for sample in samples],
    }
