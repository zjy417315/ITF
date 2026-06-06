import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


DEMO_ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = DEMO_ROOT / "reference"
CANDIDATE_DIR = DEMO_ROOT / "candidate"
OUTPUT_DIR = DEMO_ROOT / "outputs"
MANIFEST_PATH = DEMO_ROOT / "demo_manifest.json"
PAYLOAD_PATH = DEMO_ROOT / "tiny_source_payloads.npz"
CHECKPOINT_PATH = DEMO_ROOT / "tiny_checkpoint.pt"
EXPECTED_PATH = DEMO_ROOT / "expected_outputs.json"

NUM_SOURCES = 16
IMAGE_SIZE = 64
NUM_ACCEPT = 12
THRESHOLD = 0.985


def source_pattern(source_idx: int, variant: str) -> np.ndarray:
    y, x = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE].astype(np.float32)
    phase = float(source_idx + 1)
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    base[..., 0] = 0.5 + 0.32 * np.sin((x + phase * 3.0) / (5.0 + (source_idx % 3)))
    base[..., 1] = 0.5 + 0.30 * np.cos((y + phase * 2.0) / (6.0 + (source_idx % 4)))
    base[..., 2] = 0.5 + 0.24 * np.sin((x + y + phase) / (7.0 + (source_idx % 5)))
    base += (source_idx % 7) * 0.012

    if variant == "candidate_accept":
        base = 0.985 * base + 0.015 * np.roll(base, shift=1, axis=1)
    elif variant == "candidate_reject":
        base = 1.0 - base
        base[:, IMAGE_SIZE // 3 : IMAGE_SIZE // 3 + 10, :] *= 0.35

    return np.clip(base * 255.0, 0, 255).astype(np.uint8)


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def extract_feature(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    rgb_mean = arr.mean(axis=(0, 1))
    rgb_std = arr.std(axis=(0, 1))
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).reshape(-1)
    feature = np.concatenate([rgb_mean, rgb_std, luma], axis=0)
    feature = feature - feature.mean()
    norm = float(np.linalg.norm(feature))
    if norm > 0:
        feature = feature / norm
    return feature.astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def save_checkpoint(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "demo_only": True,
        "feature": "rgb8_luma_normalized",
        "threshold": THRESHOLD,
        "note": "Synthetic checkpoint for compact artifact functionality checks only.",
    }
    try:
        import torch

        torch.save(payload, path)
    except Exception:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> dict:
    try:
        import torch

        return dict(torch.load(path, map_location="cpu"))
    except Exception:
        return json.loads(path.read_text(encoding="utf-8"))


def generate_demo() -> None:
    for directory in (REFERENCE_DIR, CANDIDATE_DIR, OUTPUT_DIR):
        if directory.exists():
            shutil.rmtree(directory)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    source_ids = []
    payloads = []
    for idx in range(NUM_SOURCES):
        source_id = f"S{idx:02d}"
        source_ids.append(source_id)
        ref_path = REFERENCE_DIR / f"{source_id}_ref.png"
        save_image(ref_path, source_pattern(idx, "reference"))
        payloads.append(extract_feature(ref_path))

        should_accept = idx < NUM_ACCEPT
        candidate_variant = "candidate_accept" if should_accept else "candidate_reject"
        candidate_source_idx = idx if should_accept else (idx + 5) % NUM_SOURCES
        cand_path = CANDIDATE_DIR / f"{source_id}_candidate.png"
        save_image(cand_path, source_pattern(candidate_source_idx, candidate_variant))
        records.append(
            {
                "sample_id": f"demo_{idx:02d}",
                "source_id": source_id,
                "reference_image": str(ref_path.relative_to(DEMO_ROOT)).replace("\\", "/"),
                "candidate_image": str(cand_path.relative_to(DEMO_ROOT)).replace("\\", "/"),
                "claim_source_id": source_id,
                "expected_decision": "accept" if should_accept else "reject",
            }
        )

    manifest = {
        "schema_version": 1,
        "description": "Synthetic compact enrollment-verification demo.",
        "num_samples": NUM_SOURCES,
        "score_name": "r(I,c_s)",
        "records": records,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    np.savez(PAYLOAD_PATH, source_ids=np.asarray(source_ids, dtype=object), payloads=np.stack(payloads, axis=0))
    save_checkpoint(CHECKPOINT_PATH)

    expected = run_verification(write_outputs=False)
    EXPECTED_PATH.write_text(json.dumps(expected["summary"], indent=2), encoding="utf-8")


def load_payloads() -> dict:
    pack = np.load(PAYLOAD_PATH, allow_pickle=True)
    return {str(source_id): payload.astype(np.float32) for source_id, payload in zip(pack["source_ids"], pack["payloads"])}


def run_verification(write_outputs: bool = True) -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    threshold = float(checkpoint["threshold"])
    payloads = load_payloads()
    rows = []
    for record in manifest["records"]:
        candidate_feature = extract_feature(DEMO_ROOT / record["candidate_image"])
        source_feature = payloads[record["claim_source_id"]]
        score = cosine(candidate_feature, source_feature)
        decision = "accept" if score >= threshold else "reject"
        rows.append(
            {
                "sample_id": record["sample_id"],
                "claim_source_id": record["claim_source_id"],
                "score": score,
                "threshold": threshold,
                "decision": decision,
                "expected_decision": record["expected_decision"],
                "correct": decision == record["expected_decision"],
            }
        )

    correct = sum(1 for row in rows if row["correct"])
    accepts = sum(1 for row in rows if row["decision"] == "accept")
    rejects = len(rows) - accepts
    summary = {
        "schema_version": 1,
        "num_samples": len(rows),
        "accepts": accepts,
        "rejects": rejects,
        "accuracy": round(correct / max(len(rows), 1), 6),
        "threshold": threshold,
        "min_accept_score": round(min(row["score"] for row in rows if row["expected_decision"] == "accept"), 6),
        "max_reject_score": round(max(row["score"] for row in rows if row["expected_decision"] == "reject"), 6),
    }

    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with (OUTPUT_DIR / "demo_scores.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        (OUTPUT_DIR / "demo_results.json").write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2),
            encoding="utf-8",
        )

    return {"summary": summary, "rows": rows}


def ensure_prepared(regenerate: bool) -> None:
    required = [MANIFEST_PATH, PAYLOAD_PATH, CHECKPOINT_PATH, EXPECTED_PATH]
    if regenerate or not all(path.exists() for path in required):
        generate_demo()


def verify_expected() -> None:
    result = run_verification(write_outputs=True)
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    for key, value in expected.items():
        observed = result["summary"].get(key)
        if isinstance(value, float):
            if not math.isclose(float(observed), float(value), rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"Mismatch for {key}: expected {value}, observed {observed}")
        elif observed != value:
            raise AssertionError(f"Mismatch for {key}: expected {value}, observed {observed}")
    print(json.dumps(result["summary"], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact synthetic VTrace demo.")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate synthetic demo assets.")
    parser.add_argument("--prepare-only", action="store_true", help="Generate assets and exit without verification.")
    parser.add_argument("--verify", action="store_true", help="Verify outputs against expected_outputs.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_prepared(regenerate=args.regenerate)
    if args.prepare_only:
        print(f"prepared compact demo at {DEMO_ROOT}")
        return
    verify_expected()


if __name__ == "__main__":
    main()
