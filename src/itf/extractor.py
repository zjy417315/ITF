from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn.functional as F

from src.models.backbone import FeatureBackbone


DEFAULT_STAGE_ORDER = ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]


@dataclass
class ITFStageResult:
    feature_map: torch.Tensor
    energy_map: torch.Tensor
    itf_map: torch.Tensor
    norm_mean: torch.Tensor
    norm_std: torch.Tensor


class ImagingTraceFieldExtractor:
    """
    Stage-wise ITF extractor.

    For each stage image I^(k), we compute:
        F_k = phi(I^(k))
        E_k(x) = ||F_k(x)||_2
        ITF_k(x) = (E_k(x) - mean(E_k)) / (std(E_k) + eps)
    """

    def __init__(
        self,
        backbone: FeatureBackbone,
        stage_order: Optional[Iterable[str]] = None,
        device: Optional[str] = None,
        eps: float = 1e-6,
        feature_source: str = "layer4",
        scalarization: str = "l2",
        normalization: str = "zscore",
    ):
        self.backbone = backbone
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.backbone.to(self.device)
        self.backbone.eval()
        self.stage_order = list(stage_order or DEFAULT_STAGE_ORDER)
        self.eps = eps
        self.feature_source = feature_source
        self.scalarization = scalarization
        self.normalization = normalization

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[str] = None,
        stage_order: Optional[Iterable[str]] = None,
        eps: float = 1e-6,
        feature_source: str = "layer4",
        scalarization: str = "l2",
        normalization: str = "zscore",
    ) -> "ImagingTraceFieldExtractor":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        args = checkpoint.get("args", {})
        d_f = int(args.get("d_f", 512))
        c_f = int(args.get("C_f", 256))

        backbone = FeatureBackbone(d_f=d_f, C_f=c_f, freeze_bn=True, imagenet_weights=False)
        backbone.load_state_dict(checkpoint["backbone"])
        return cls(
            backbone=backbone,
            stage_order=stage_order,
            device=device,
            eps=eps,
            feature_source=feature_source,
            scalarization=scalarization,
            normalization=normalization,
        )

    def _ensure_bchw(self, stage_tensor: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(stage_tensor, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3, 4) and tensor.shape[1] not in (1, 3, 4):
                tensor = tensor.permute(0, 3, 1, 2)
        if tensor.ndim != 4:
            raise ValueError(f"Expected stage tensor with shape CHW or BCHW, got {tuple(tensor.shape)}")
        return tensor

    def _extract_backbone_maps(self, stage_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not all(hasattr(self.backbone, attr) for attr in ["adapter", "backbone"]):
            if not hasattr(self.backbone, "extract_map"):
                raise AttributeError(
                    "Backbone must either provide adapter/backbone modules or an extract_map method."
                )
            return {"fallback": self.backbone.extract_map(stage_tensor)}

        x = self.backbone.adapter(stage_tensor)
        x = self.backbone._fuse_trace(x)

        feature_maps: Dict[str, torch.Tensor] = {}
        current = x
        for idx, module in enumerate(self.backbone.backbone):
            current = module(current)
            if idx == 6:
                feature_maps["layer3"] = current
            elif idx == 7:
                feature_maps["layer4"] = current

        feature_maps["map_proj"] = self.backbone.map_proj(feature_maps["layer4"])
        return feature_maps

    def _select_feature_map(self, stage_tensor: torch.Tensor) -> torch.Tensor:
        feature_maps = self._extract_backbone_maps(stage_tensor)

        if "fallback" in feature_maps:
            return feature_maps["fallback"]

        if self.feature_source == "map_proj":
            return feature_maps["map_proj"]
        if self.feature_source == "layer4":
            return feature_maps["layer4"]
        if self.feature_source == "layer3":
            return feature_maps["layer3"]
        if self.feature_source == "multiscale_l34":
            layer3 = feature_maps["layer3"]
            layer4 = F.interpolate(
                feature_maps["layer4"],
                size=layer3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            layer3_energy = torch.linalg.vector_norm(layer3, ord=2, dim=1, keepdim=True)
            layer4_energy = torch.linalg.vector_norm(layer4, ord=2, dim=1, keepdim=True)
            return torch.cat([layer3_energy, layer4_energy], dim=1)
        raise ValueError(f"Unsupported feature_source: {self.feature_source}")

    def _scalarize_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        if self.scalarization == "l2":
            return torch.linalg.vector_norm(feature_map, ord=2, dim=1)
        if self.scalarization == "l1":
            return feature_map.abs().sum(dim=1)
        if self.scalarization == "mean_abs":
            return feature_map.abs().mean(dim=1)
        if self.scalarization == "max_abs":
            return feature_map.abs().max(dim=1).values
        raise ValueError(f"Unsupported scalarization: {self.scalarization}")

    def _normalize_energy_map(self, energy_map: torch.Tensor):
        if self.normalization == "zscore":
            mean = energy_map.mean(dim=(-2, -1), keepdim=True)
            std = energy_map.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(self.eps)
            itf_map = (energy_map - mean) / std
            return itf_map, mean, std
        if self.normalization == "robust_zscore":
            median = energy_map.median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
            mad = (energy_map - median).abs().median(dim=-1, keepdim=True).values.median(dim=-2, keepdim=True).values
            scale = (1.4826 * mad).clamp_min(self.eps)
            itf_map = (energy_map - median) / scale
            return itf_map, median, scale
        if self.normalization == "minmax":
            min_val = energy_map.amin(dim=(-2, -1), keepdim=True)
            max_val = energy_map.amax(dim=(-2, -1), keepdim=True)
            scale = (max_val - min_val).clamp_min(self.eps)
            itf_map = (energy_map - min_val) / scale
            return itf_map, min_val, scale
        raise ValueError(f"Unsupported normalization: {self.normalization}")

    def _record_normalization_metadata(self, norm_mean: torch.Tensor, norm_std: torch.Tensor):
        return norm_mean.reshape(-1), norm_std.reshape(-1)

    def _describe_variant(self) -> str:
        return f"{self.feature_source}|{self.scalarization}|{self.normalization}"

    @torch.no_grad()
    def extract_stage(self, stage_tensor: torch.Tensor) -> ITFStageResult:
        stage_tensor = self._ensure_bchw(stage_tensor).to(self.device, non_blocking=True)
        feature_map = self._select_feature_map(stage_tensor)
        energy_map = self._scalarize_feature_map(feature_map)
        itf_map, norm_mean, norm_std = self._normalize_energy_map(energy_map)

        return ITFStageResult(
            feature_map=feature_map.detach(),
            energy_map=energy_map.detach(),
            itf_map=itf_map.detach(),
            norm_mean=norm_mean.detach(),
            norm_std=norm_std.detach(),
        )

    @torch.no_grad()
    def extract_sequence(
        self,
        stage_dict: Dict[str, torch.Tensor],
        return_feature_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feature_seq: List[torch.Tensor] = []
        energy_seq: List[torch.Tensor] = []
        itf_seq: List[torch.Tensor] = []
        norm_mean_seq: List[torch.Tensor] = []
        norm_std_seq: List[torch.Tensor] = []

        for stage_name in self.stage_order:
            if stage_name not in stage_dict:
                raise KeyError(f"Missing stage '{stage_name}' in stage_dict.")

            result = self.extract_stage(stage_dict[stage_name])
            if return_feature_maps:
                feature_seq.append(result.feature_map.squeeze(0).cpu())
            energy_seq.append(result.energy_map.squeeze(0).cpu())
            itf_seq.append(result.itf_map.squeeze(0).cpu())
            norm_mean, norm_std = self._record_normalization_metadata(result.norm_mean, result.norm_std)
            norm_mean_seq.append(norm_mean.cpu())
            norm_std_seq.append(norm_std.cpu())

        pack: Dict[str, torch.Tensor] = {
            "stage_order": self.stage_order,
            "energy_seq": torch.stack(energy_seq, dim=0),
            "itf_seq": torch.stack(itf_seq, dim=0),
            "norm_mean_seq": torch.cat(norm_mean_seq, dim=0).float(),
            "norm_std_seq": torch.cat(norm_std_seq, dim=0).float(),
            "variant": self._describe_variant(),
        }
        if return_feature_maps:
            pack["feature_seq"] = torch.stack(feature_seq, dim=0)
        return pack

    @torch.no_grad()
    def extract_from_stage_cache_file(
        self,
        stage_cache_path: str,
        return_feature_maps: bool = False,
    ) -> Dict[str, torch.Tensor]:
        cache = torch.load(Path(stage_cache_path), map_location="cpu")
        return self.extract_sequence(cache, return_feature_maps=return_feature_maps)
