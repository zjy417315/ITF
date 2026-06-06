import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.data_roots import resolve_experiment_root, resolve_meta_path


def load_meta(meta_path: Path) -> Dict[str, Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_stage3_prototype_cache(
    teacher_cache_dir: Path,
    output_dir: Path,
    meta_path: Path,
    teacher_key: str = "teacher_seq",
    prototype_versions: Optional[Iterable[int]] = None,
    overwrite: bool = False,
):
    teacher_cache_dir = Path(teacher_cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = load_meta(meta_path)
    prototype_versions = None if prototype_versions is None else sorted({int(v) for v in prototype_versions})

    groups = defaultdict(list)
    teacher_files = sorted(teacher_cache_dir.glob("*.pt"))
    if not teacher_files:
        raise FileNotFoundError(f"No teacher-cache files found in {teacher_cache_dir}")

    for teacher_path in teacher_files:
        pack = torch.load(teacher_path, map_location="cpu")
        raw_anchor = pack["raw_anchor"]
        groups[raw_anchor].append(
            {
                "version_id": pack["version_id"],
                "teacher_path": teacher_path,
                "pack": pack,
            }
        )

    index = {}
    print("=" * 72)
    print("Build Stage-3 Prototype Cache")
    print(f"Teacher cache : {teacher_cache_dir}")
    print(f"Meta path     : {meta_path}")
    print(f"Output dir    : {output_dir}")
    print(f"Raw groups    : {len(groups)}")
    print(f"Proto vers    : {prototype_versions if prototype_versions is not None else 'all'}")
    print("=" * 72)

    for raw_anchor, entries in tqdm(sorted(groups.items()), desc="Prototype groups"):
        safe_name = f"{raw_anchor}.pt"
        save_path = output_dir / safe_name
        if save_path.exists() and not overwrite:
            index[raw_anchor] = safe_name
            continue

        seqs = []
        version_ids = []
        source_versions = []
        for entry in entries:
            pack = entry["pack"]
            version_id = pack["version_id"]
            meta_info = meta.get(version_id, {})
            version_num = int(meta_info.get("version", 0))
            if prototype_versions is not None and version_num not in prototype_versions:
                continue
            if teacher_key in pack:
                seq = pack[teacher_key]
            elif teacher_key == "teacher_seq" and "topo_vec_seq" in pack:
                seq = pack["topo_vec_seq"]
            else:
                available = ", ".join(sorted(str(k) for k in pack.keys()))
                raise KeyError(f"Teacher key '{teacher_key}' not found in {entry['teacher_path'].name}. Available: {available}")

            seqs.append(torch.as_tensor(seq, dtype=torch.float32))
            version_ids.append(version_id)
            source_versions.append(version_num)

        if not seqs:
            continue

        seqs = torch.stack(seqs, dim=0)  # (V, K, D)
        prototype_seq = seqs.mean(dim=0)
        prototype_vec = F.normalize(prototype_seq.mean(dim=0), dim=0)

        within_proto_cos = torch.matmul(
            F.normalize(seqs.view(seqs.shape[0], -1), dim=1),
            F.normalize(prototype_seq.view(1, -1), dim=1).T,
        ).squeeze(1)

        proto_pack = {
            "raw_anchor": raw_anchor,
            "teacher_key": teacher_key,
            "prototype_seq": prototype_seq.float(),
            "prototype_vec": prototype_vec.float(),
            "version_ids": version_ids,
            "source_versions": source_versions,
            "num_versions": len(version_ids),
            "prototype_stage_count": int(prototype_seq.shape[0]),
            "prototype_dim": int(prototype_seq.shape[-1]),
            "mean_within_proto_cos": float(within_proto_cos.mean().item()),
            "min_within_proto_cos": float(within_proto_cos.min().item()),
        }
        torch.save(proto_pack, save_path)
        index[raw_anchor] = safe_name

    with open(output_dir / "prototype_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print(f"Saved prototype groups: {len(index)}")
    print(f"Output dir            : {output_dir}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Build raw-level prototype cache from per-version teacher cache")
    parser.add_argument(
        "--teacher_cache_dir",
        type=str,
        default=str(resolve_experiment_root() / "stage3_teacher_cache_full"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(resolve_experiment_root() / "stage3_prototype_cache_anchor12"),
    )
    parser.add_argument("--meta_path", type=str, default=str(resolve_meta_path()))
    parser.add_argument("--teacher_key", type=str, default="teacher_seq")
    parser.add_argument("--prototype_versions", type=int, nargs="*", default=[1, 2])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_stage3_prototype_cache(
        teacher_cache_dir=Path(args.teacher_cache_dir),
        output_dir=Path(args.output_dir),
        meta_path=Path(args.meta_path),
        teacher_key=args.teacher_key,
        prototype_versions=args.prototype_versions,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
