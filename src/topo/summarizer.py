from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import gudhi
import numpy as np
import torch
from persim import PersistenceImager


DEFAULT_STAGE_ORDER = ["stage_raw", "stage_demosaic", "stage_denoise", "stage_color", "rgb"]


@dataclass
class TopologyStageResult:
    diagram_h0: np.ndarray
    diagram_h1: np.ndarray
    image_h0: torch.Tensor
    image_h1: torch.Tensor
    vector: torch.Tensor


class ITFTopologySummarizer:
    """
    Summarize stage-wise ITF maps into topology-aware signatures.

    Input:
        ITF_k in R^(H x W)

    Output per stage:
        D_k^(0), D_k^(1) -> persistence images -> fixed-length topology vector
    """

    def __init__(
        self,
        stage_order: Optional[Iterable[str]] = None,
        pixel_size: float = 0.25,
        birth_range: Tuple[float, float] = (-3.0, 3.0),
        pers_range: Tuple[float, float] = (0.0, 6.0),
        kernel_sigma: float = 0.2,
        tie_break_eps: float = 1e-6,
    ):
        self.stage_order = list(stage_order or DEFAULT_STAGE_ORDER)
        self.pixel_size = float(pixel_size)
        self.birth_range = tuple(float(x) for x in birth_range)
        self.pers_range = tuple(float(x) for x in pers_range)
        self.kernel_sigma = float(kernel_sigma)
        self.tie_break_eps = float(tie_break_eps)

        self.pimager = PersistenceImager(
            pixel_size=self.pixel_size,
            birth_range=self.birth_range,
            pers_range=self.pers_range,
        )
        self.pimager.kernel_params = {"sigma": self.kernel_sigma}

    def describe_variant(self) -> str:
        return (
            f"pi[{self.birth_range[0]},{self.birth_range[1]}|"
            f"{self.pers_range[0]},{self.pers_range[1]}|"
            f"px={self.pixel_size}|sigma={self.kernel_sigma}]"
        )

    def _ensure_hw(self, itf_map: torch.Tensor) -> np.ndarray:
        tensor = torch.as_tensor(itf_map, dtype=torch.float32)
        if tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Expected a 2D ITF map, got shape {tuple(tensor.shape)}")
        return tensor.cpu().numpy().astype(np.float32, copy=False)

    def _stabilize_field(self, scalar_field: np.ndarray) -> np.ndarray:
        if self.tie_break_eps <= 0:
            return scalar_field
        yy, xx = np.indices(scalar_field.shape, dtype=np.float32)
        tie_break = yy + (xx / max(float(scalar_field.shape[1]), 1.0))
        return scalar_field + self.tie_break_eps * tie_break

    def _finite_diagram(self, cubical_complex: gudhi.CubicalComplex, dimension: int) -> np.ndarray:
        diagram = cubical_complex.persistence_intervals_in_dimension(dimension)
        if diagram.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        diagram = np.asarray(diagram, dtype=np.float32).reshape(-1, 2)
        finite_mask = np.isfinite(diagram).all(axis=1)
        return diagram[finite_mask]

    def _diagram_to_image(self, diagram: np.ndarray) -> np.ndarray:
        resolution = tuple(int(x) for x in self.pimager.resolution)
        if diagram.size == 0:
            return np.zeros(resolution, dtype=np.float32)

        image = self.pimager.transform(diagram, skew=True)
        if isinstance(image, list):
            image = image[0]
        return np.asarray(image, dtype=np.float32)

    def summarize_stage(self, itf_map: torch.Tensor) -> TopologyStageResult:
        scalar_field = self._ensure_hw(itf_map)
        scalar_field = self._stabilize_field(scalar_field)

        cubical_complex = gudhi.CubicalComplex(top_dimensional_cells=scalar_field)
        cubical_complex.persistence()

        diagram_h0 = self._finite_diagram(cubical_complex, dimension=0)
        diagram_h1 = self._finite_diagram(cubical_complex, dimension=1)

        image_h0 = torch.from_numpy(self._diagram_to_image(diagram_h0))
        image_h1 = torch.from_numpy(self._diagram_to_image(diagram_h1))
        vector = torch.cat([image_h0.reshape(-1), image_h1.reshape(-1)], dim=0)

        return TopologyStageResult(
            diagram_h0=diagram_h0,
            diagram_h1=diagram_h1,
            image_h0=image_h0,
            image_h1=image_h1,
            vector=vector.float(),
        )

    def summarize_sequence(
        self,
        itf_seq: torch.Tensor,
        stage_order: Optional[Iterable[str]] = None,
        include_diagrams: bool = False,
    ) -> Dict[str, torch.Tensor]:
        itf_seq = torch.as_tensor(itf_seq, dtype=torch.float32)
        if itf_seq.ndim != 3:
            raise ValueError(f"Expected ITF sequence with shape (K, H, W), got {tuple(itf_seq.shape)}")

        if stage_order is None:
            if len(self.stage_order) == int(itf_seq.shape[0]):
                current_stage_order = list(self.stage_order)
            else:
                current_stage_order = [f"stage_{idx}" for idx in range(int(itf_seq.shape[0]))]
        else:
            current_stage_order = list(stage_order)
            if len(current_stage_order) != int(itf_seq.shape[0]):
                raise ValueError(
                    f"Stage-order length {len(current_stage_order)} does not match ITF sequence length {int(itf_seq.shape[0])}."
                )

        topo_h0_seq: List[torch.Tensor] = []
        topo_h1_seq: List[torch.Tensor] = []
        topo_vec_seq: List[torch.Tensor] = []
        diagram_count_h0: List[int] = []
        diagram_count_h1: List[int] = []
        persistence_mass_h0: List[float] = []
        persistence_mass_h1: List[float] = []
        diagrams_h0: List[np.ndarray] = []
        diagrams_h1: List[np.ndarray] = []

        for stage_map in itf_seq:
            result = self.summarize_stage(stage_map)
            topo_h0_seq.append(result.image_h0)
            topo_h1_seq.append(result.image_h1)
            topo_vec_seq.append(result.vector)
            diagram_count_h0.append(int(result.diagram_h0.shape[0]))
            diagram_count_h1.append(int(result.diagram_h1.shape[0]))
            persistence_mass_h0.append(
                float((result.diagram_h0[:, 1] - result.diagram_h0[:, 0]).sum()) if result.diagram_h0.size else 0.0
            )
            persistence_mass_h1.append(
                float((result.diagram_h1[:, 1] - result.diagram_h1[:, 0]).sum()) if result.diagram_h1.size else 0.0
            )
            if include_diagrams:
                diagrams_h0.append(result.diagram_h0)
                diagrams_h1.append(result.diagram_h1)

        pack: Dict[str, torch.Tensor] = {
            "stage_order": current_stage_order,
            "topo_h0_seq": torch.stack(topo_h0_seq, dim=0),
            "topo_h1_seq": torch.stack(topo_h1_seq, dim=0),
            "topo_vec_seq": torch.stack(topo_vec_seq, dim=0),
            "diagram_count_h0": torch.tensor(diagram_count_h0, dtype=torch.int64),
            "diagram_count_h1": torch.tensor(diagram_count_h1, dtype=torch.int64),
            "persistence_mass_h0": torch.tensor(persistence_mass_h0, dtype=torch.float32),
            "persistence_mass_h1": torch.tensor(persistence_mass_h1, dtype=torch.float32),
            "variant": self.describe_variant(),
        }

        if include_diagrams:
            pack["diagram_h0_seq"] = diagrams_h0
            pack["diagram_h1_seq"] = diagrams_h1

        return pack
