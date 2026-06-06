import json
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_ROOT = PROJECT_ROOT / "Paper"
GENERATED_ROOT = PAPER_ROOT / "generated"
MAIN_RESULTS_SUMMARY_PATH = GENERATED_ROOT / "main_results_summary.json"
EXTERNAL_BASELINE_ROOT = Path(r"<artifact-local-path-redacted>")
OFFICIAL_RESULT_PATH = Path(
    r"<artifact-local-path-redacted>"
)
OFFICIAL_STRONGEST_VARIANT_PATH = Path(
    r"<artifact-local-path-redacted>"
)
STUDENT_ONLY_COSINE_PATH = Path(r"<artifact-local-path-redacted>")
OFFICIAL_NO_VERIFIER_PATH = Path(r"<artifact-local-path-redacted>")
OFFICIAL_LINEAR_COMBINER_PATH = Path(r"<artifact-local-path-redacted>")


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def format_percent(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "--"
    return f"{100.0 * float(value):.{digits}f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def build_internal_rows() -> List[Dict]:
    itf = load_json(PROJECT_ROOT / "data" / "itf_eval_live32_v4_stage12.json")
    topo = load_json(PROJECT_ROOT / "data" / "topology_eval_live32_v4_tuned_stage12.json")
    joint = load_json(PROJECT_ROOT / "data" / "joint_scaling_summary.json")
    proto_teacher = load_json(Path(r"<artifact-local-path-redacted>"))
    code_teacher = load_json(Path(r"<artifact-local-path-redacted>"))
    official = load_json(OFFICIAL_RESULT_PATH)
    official_variant = load_json(OFFICIAL_STRONGEST_VARIANT_PATH)
    student_only = load_json(STUDENT_ONLY_COSINE_PATH)

    return [
        {
            "stage": "P1",
            "name": "ITF field verifier",
            "input": "stage-wise ITF fields",
            "simple_assumption": "No",
            "oracle": "No",
            "primary_metric": "Ordered-triplet acc.",
            "value_a": itf["ordering_checks"]["ordered_triplet_accuracy"],
            "value_b_name": "Cross-to-shift ratio",
            "value_b": itf["derived_scores"]["cross_over_isp_ratio"],
        },
        {
            "stage": "P2",
            "name": "Topology verifier",
            "input": "persistent topology vectors",
            "simple_assumption": "No",
            "oracle": "No",
            "primary_metric": "Ordered-triplet acc.",
            "value_a": topo["topology"]["ordering_checks"]["ordered_triplet_accuracy"],
            "value_b_name": "Cross-to-shift ratio",
            "value_b": topo["topology"]["derived_scores"]["cross_over_isp_ratio"],
        },
        {
            "stage": "P3",
            "name": "Joint process verifier",
            "input": "joint ITF + topology distance",
            "simple_assumption": "Yes",
            "oracle": "No",
            "primary_metric": "Accuracy",
            "value_a": None,
            "value_b_name": "v4 acc @32/64/128",
            "value_b": {
                "32": joint["weighted_32"]["v4_acc"],
                "64": joint["weighted_64"]["v4_acc"],
                "128": joint["weighted_128"]["v4_acc"],
            },
        },
        {
            "stage": "P4",
            "name": "Prototype oracle",
            "input": "registered process prototype",
            "simple_assumption": "Yes",
            "oracle": "Yes",
            "primary_metric": "AUC",
            "value_a": proto_teacher["pairwise_auc"],
            "value_b_name": "Gallery acc.",
            "value_b": proto_teacher["full_gallery_acc"],
        },
        {
            "stage": "P5",
            "name": "Code oracle",
            "input": "registered process code",
            "simple_assumption": "Yes",
            "oracle": "Yes",
            "primary_metric": "AUC",
            "value_a": code_teacher["pairwise_auc"],
            "value_b_name": "EER",
            "value_b": code_teacher["eer"],
            "value_c_name": "Top-1 acc.",
            "value_c": code_teacher["top1_acc"],
        },
        {
            "stage": "P6a",
            "name": "RGB student baseline",
            "input": "RGB-only student representation",
            "simple_assumption": "No",
            "oracle": "No",
            "primary_metric": "AUC",
            "value_a": student_only["claimed_pairwise_auc"],
            "value_b_name": "EER",
            "value_b": student_only["claimed_eer"],
        },
        {
            "stage": "P6b",
            "name": "Official active verifier",
            "input": "RGB + process certificate",
            "simple_assumption": "No",
            "oracle": "No",
            "primary_metric": "AUC",
            "value_a": official["claimed_pairwise_auc"],
            "value_b_name": "EER",
            "value_b": official["claimed_eer"],
            "value_c_name": "TAR (FAR=$10^{-2}$)",
            "value_c": official["claimed_tar_at_far_1e2"],
        },
        {
            "stage": "P6c",
            "name": "Strongest official variant",
            "input": "RGB + process certificate",
            "simple_assumption": "No",
            "oracle": "No",
            "primary_metric": "AUC",
            "value_a": official_variant["claimed_pairwise_auc"],
            "value_b_name": "EER",
            "value_b": official_variant["claimed_eer"],
            "value_c_name": "Hard AUC",
            "value_c": official_variant["claimed_hard_auc"],
        },
    ]


def build_external_comparison_rows() -> List[Dict]:
    return [
        {
            "method": "TruFor",
            "problem_type": "passive forgery detection",
            "input": "RGB-only",
            "camera_change": "No",
            "watermark_or_signature": "No",
            "benign_postproc": "Partial",
            "claimed_source_verification": "No (needs protocol adaptation)",
            "code_or_weight": "Code available; no final local checkpoint",
            "transferable_to_ours": "Yes",
            "usage": "screened proxy candidate",
            "status": "screened",
        },
        {
            "method": "Universal Fake Detectors",
            "problem_type": "passive fake-image detection",
            "input": "RGB-only",
            "camera_change": "No",
            "watermark_or_signature": "No",
            "benign_postproc": "Partial",
            "claimed_source_verification": "No (needs protocol adaptation)",
            "code_or_weight": "Code + weight available",
            "transferable_to_ours": "Yes",
            "usage": "proxy quantitative candidate",
            "status": "ran",
        },
        {
            "method": "FatFormer",
            "problem_type": "passive fake-image detection",
            "input": "RGB-only",
            "camera_change": "No",
            "watermark_or_signature": "No",
            "benign_postproc": "Partial",
            "claimed_source_verification": "No (needs protocol adaptation)",
            "code_or_weight": "Code + checkpoint available",
            "transferable_to_ours": "Yes",
            "usage": "proxy quantitative candidate",
            "status": "ran",
        },
        {
            "method": "DIRE",
            "problem_type": "passive diffusion detection",
            "input": "RGB-only",
            "camera_change": "No",
            "watermark_or_signature": "No",
            "benign_postproc": "Yes",
            "claimed_source_verification": "No (needs protocol adaptation)",
            "code_or_weight": "Code available; no final local checkpoint or reconstruction cache",
            "transferable_to_ours": "Yes",
            "usage": "screened proxy candidate",
            "status": "screened",
        },
        {
            "method": "Content Authentication for Neural Imaging Pipelines",
            "problem_type": "active content authentication",
            "input": "RGB + imaging credential",
            "camera_change": "Yes",
            "watermark_or_signature": "Implicit learned credential",
            "benign_postproc": "Yes",
            "claimed_source_verification": "Partial",
            "code_or_weight": "Paper-level reference",
            "transferable_to_ours": "Narrative only",
            "usage": "narrative reference",
            "status": "screened",
        },
        {
            "method": "RAWIW",
            "problem_type": "active RAW-to-RGB watermarking",
            "input": "RAW/RGB + watermark",
            "camera_change": "Yes",
            "watermark_or_signature": "Yes",
            "benign_postproc": "Yes",
            "claimed_source_verification": "Partial",
            "code_or_weight": "Paper-level reference",
            "transferable_to_ours": "Narrative only",
            "usage": "narrative reference",
            "status": "screened",
        },
    ]


def build_external_metric_rows() -> List[Dict]:
    rows = []
    specs = [
        ("RN50", "Generic", EXTERNAL_BASELINE_ROOT / "rn50_claim_eval_sameimage_1024_v1.json"),
        ("CLIP-L/14", "Generic", EXTERNAL_BASELINE_ROOT / "clip_claim_eval_sameimage_1024_v1.json"),
        ("DINOv2-B", "Generic", EXTERNAL_BASELINE_ROOT / "dino_claim_eval_sameimage_1024_v1.json"),
        ("SigLIP-B/16", "Generic", EXTERNAL_BASELINE_ROOT / "siglip_claim_eval_sameimage_1024_v1.json"),
        ("UniFD", "Forensic", EXTERNAL_BASELINE_ROOT / "unifd_claim_eval_sameimage_1024_v1.json"),
        ("FatFormer", "Forensic", EXTERNAL_BASELINE_ROOT / "fatformer_claim_eval_sameimage_1024_v1.json"),
        ("Ours-GR", "Ours", EXTERNAL_BASELINE_ROOT / "ours_student_rgbref_globalrepr_sameimage_1024_v1.json"),
        ("Ours-FV", "Ours", EXTERNAL_BASELINE_ROOT / "ours_student_rgbref_featurevec_sameimage_1024_v1.json"),
    ]
    for method, family, path in specs:
        if not path.exists():
            continue
        result = load_json(path)
        rows.append(
            {
                "protocol": "RGB-ref proxy",
                "method": method,
                "family": family,
                "input": "RGB + RGB ref",
                "pairwise_auc": result["pairwise_auc"],
                "eer": result["eer"],
                "hard_auc": result["hard_auc"],
                "tar_at_far_1e2": result.get("tar_at_far_1e2"),
                "anchor_full_gallery_acc": result["anchor_full_gallery_acc"],
                "shift_full_gallery_acc": result["shift_full_gallery_acc"],
                "source_json": str(path),
            }
        )
    return rows


def build_credential_binding_stress_rows() -> List[Dict]:
    official_variant = load_json(OFFICIAL_STRONGEST_VARIANT_PATH)
    return [
        {
            "protocol": "Credential-swap stress",
            "method": "RGB-only input limit",
            "family": "External",
            "input": "RGB only",
            "pairwise_auc": 0.5,
            "eer": 0.5,
            "hard_auc": 0.5,
            "tar_at_far_1e2": None,
            "anchor_full_gallery_acc": None,
            "shift_full_gallery_acc": None,
            "source_json": "analytic: identical RGB input under swapped process credentials",
        },
        {
            "protocol": "Credential-swap stress",
            "method": "Ours-Active",
            "family": "Ours",
            "input": "RGB + ITF credential",
            "pairwise_auc": official_variant["claimed_pairwise_auc"],
            "eer": official_variant["claimed_eer"],
            "hard_auc": official_variant["claimed_hard_auc"],
            "tar_at_far_1e2": official_variant.get("claimed_tar_at_far_1e2"),
            "anchor_full_gallery_acc": None,
            "shift_full_gallery_acc": None,
            "source_json": str(OFFICIAL_STRONGEST_VARIANT_PATH),
        },
    ]


def load_official_protocol_analysis() -> Dict | None:
    path = GENERATED_ROOT / "official_protocol_analysis.json"
    if not path.exists():
        return None
    return load_json(path)


def load_official_ablation_rows() -> List[Dict]:
    full = load_json(OFFICIAL_RESULT_PATH)
    linear = load_json(OFFICIAL_LINEAR_COMBINER_PATH)
    no_verifier = load_json(OFFICIAL_NO_VERIFIER_PATH)
    student_only = load_json(STUDENT_ONLY_COSINE_PATH)
    return [
        {
            "variant": "Full scorer",
            "meaning": "student + verifier + protocol + learned fusion",
            "auc": full["claimed_pairwise_auc"],
            "eer": full["claimed_eer"],
            "hard_auc": full["claimed_hard_auc"],
            "tar": full["claimed_tar_at_far_1e2"],
        },
        {
            "variant": "Without protocol branch",
            "meaning": "main branch only (student claim + verifier)",
            "auc": full["claimed_main_pairwise_auc"],
            "eer": full["claimed_main_eer"],
            "hard_auc": full["claimed_main_hard_auc"],
            "tar": full["claimed_main_tar_at_far_1e2"],
        },
        {
            "variant": "Without learned fusion",
            "meaning": "linear main/protocol combiner",
            "auc": linear["claimed_pairwise_auc"],
            "eer": linear["claimed_eer"],
            "hard_auc": linear["claimed_hard_auc"],
            "tar": linear["claimed_tar_at_far_1e2"],
        },
        {
            "variant": "Without claim verifier",
            "meaning": "student + protocol + fusion",
            "auc": no_verifier["claimed_pairwise_auc"],
            "eer": no_verifier["claimed_eer"],
            "hard_auc": no_verifier["claimed_hard_auc"],
            "tar": no_verifier["claimed_tar_at_far_1e2"],
        },
        {
            "variant": "RGB student only",
            "meaning": "no verifier, no protocol, no fusion",
            "auc": student_only["claimed_pairwise_auc"],
            "eer": student_only["claimed_eer"],
            "hard_auc": student_only["claimed_hard_auc"],
            "tar": student_only["claimed_tar_at_far_1e2"],
        },
    ]


def load_process_ablation_rows() -> List[Dict]:
    itf = load_json(PROJECT_ROOT / "data" / "itf_eval_live32_v4_stage12.json")
    topo = load_json(PROJECT_ROOT / "data" / "topology_eval_live32_v4_tuned_stage12.json")
    joint = load_json(PROJECT_ROOT / "data" / "joint_scaling_summary.json")
    return [
        {
            "variant": "ITF field verifier",
            "primary": itf["ordering_checks"]["ordered_triplet_accuracy"],
            "aux": itf["derived_scores"]["cross_over_isp_ratio"],
            "note": "field response only",
        },
        {
            "variant": "Topology verifier",
            "primary": topo["topology"]["ordering_checks"]["ordered_triplet_accuracy"],
            "aux": topo["topology"]["derived_scores"]["cross_over_isp_ratio"],
            "note": "persistent summary only",
        },
        {
            "variant": "Joint process verifier",
            "primary": joint["weighted_64"]["v4_acc"],
            "aux": joint["weighted_128"]["v4_acc"],
            "note": "joint verifier acc@64/128",
        },
    ]


def load_main_results_summary() -> Dict | None:
    if not MAIN_RESULTS_SUMMARY_PATH.exists():
        return None
    return load_json(MAIN_RESULTS_SUMMARY_PATH)


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_internal_table_tex(rows: List[Dict]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Verification pipeline used throughout the experimental chapter, reported from process-only analysis to the deployed active verifier.}",
        "\\label{tab:internal_pipeline}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Method & Verification input & Evaluation setting & Main result & Supporting result \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["stage"] == "P3":
            value_b = row["value_b"]
            setting = "Closed registry (32/64/128 sources)"
            value = f"Accuracy = {format_float(value_b['32'],3)}/{format_float(value_b['64'],3)}/{format_float(value_b['128'],3)}"
            aux = "Reported across 32/64/128 registered sources"
        elif row["stage"] in {"P4", "P5"}:
            setting = "Closed-registry upper bound"
            value = f"{row['primary_metric']}: {format_float(row['value_a'], 4)}"
            aux_parts = []
            if "value_b_name" in row:
                aux_parts.append(f"{row['value_b_name']}: {format_float(row['value_b'], 4)}")
            if "value_c_name" in row:
                aux_parts.append(f"{row['value_c_name']}: {format_float(row['value_c'], 4)}")
            aux = "; ".join(aux_parts) if aux_parts else "--"
        elif row["stage"] in {"P6a", "P6b", "P6c"}:
            setting = "Deployed verification"
            value = f"{row['primary_metric']}: {format_float(row['value_a'], 4)}"
            aux_parts = []
            if "value_b_name" in row:
                aux_parts.append(f"{row['value_b_name']}: {format_float(row['value_b'], 4)}")
            if "value_c_name" in row:
                aux_parts.append(f"{row['value_c_name']}: {format_float(row['value_c'], 4)}")
            aux = "; ".join(aux_parts) if aux_parts else "--"
        else:
            setting = "Ordered source consistency"
            value = f"{row['primary_metric']}: {format_float(row['value_a'], 4)}"
            aux = f"{row['value_b_name']}: {format_float(row['value_b'], 4)}"
        lines.append(
            f"{latex_escape(row['name'])} & {latex_escape(row['input'])} & "
            f"{latex_escape(setting)} & {latex_escape(value)} & {latex_escape(aux)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_internal_compact_table_tex(rows: List[Dict]) -> str:
    keep = {"P3", "P5", "P6a", "P6b"}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Core verification results, from process-side upper bounds to the deployed verifier.}",
        "\\label{tab:core_verification}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllll}",
        "\\toprule",
        "Method & Verification input & Evaluation setting & Main result & Supporting result \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["stage"] not in keep:
            continue
        if row["stage"] == "P3":
            metrics = row["value_b"]
            value = (
                f"Accuracy (32/64/128) = "
                f"{format_float(metrics['32'],3)}/{format_float(metrics['64'],3)}/{format_float(metrics['128'],3)}"
            )
            aux = "Process-side closed-registry verifier"
            setting = "Closed registry"
        elif row["stage"] == "P5":
            value = f"AUC = {format_float(row['value_a'],4)}"
            aux = f"EER = {format_float(row['value_b'],4)}"
            setting = "Closed-registry upper bound"
        elif row["stage"] == "P6a":
            value = f"AUC = {format_float(row['value_a'],4)}"
            aux = f"EER = {format_float(row['value_b'],4)}"
            setting = "Deployed verification"
        else:
            value = f"AUC = {format_float(row['value_a'],4)}"
            aux = (
                f"EER = {format_float(row['value_b'],4)}; "
                f"TAR@FAR$=10^{{-2}}$ = {format_float(row['value_c'],4)}"
            )
            setting = "Deployed verification"
        lines.append(
            f"{latex_escape(row['name'])} & {latex_escape(row['input'])} & "
            f"{latex_escape(setting)} & {value} & {aux} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_external_metric_table_tex(rows: List[Dict]) -> str:
    # Compact manuscript table for the proxy external comparison. Every row is
    # backed by a JSON result produced by evaluate_external_rgb_claim_baseline.py
    # under the same split/protocol.
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{External RGB-only baselines under the shared proxy protocol.}",
        "\\label{tab:external_compact_main}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Method & Input & AUC & EER & Hard AUC & TAR@1\\% & Gallery@Prop & Gallery@Shift \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = latex_escape(row["method"])
        if row["method"] == "Ours-FV":
            method = f"\\textbf{{{method}}}"
        pairwise_auc = format_percent(row["pairwise_auc"])
        tar_at_far_1e2 = format_percent(row.get("tar_at_far_1e2"))
        hard_auc = format_percent(row["hard_auc"])
        if row["method"] == "Ours-FV":
            hard_auc = f"\\textbf{{{hard_auc}}}"
        lines.append(
            f"{method} & {latex_escape(row.get('input', row['family']))} & "
            f"{pairwise_auc} & "
            f"{format_percent(row['eer'])} & "
            f"{hard_auc} & "
            f"{tar_at_far_1e2} & "
            f"{format_percent(row['anchor_full_gallery_acc'])} & "
            f"{format_percent(row['shift_full_gallery_acc'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\vspace{0.15em}",
        "\\parbox{\\textwidth}{\\footnotesize\\textit{Note.} Values are percentages. All rows use the same \\texttt{seed=42}, \\texttt{max\\_raws=1024}, and \\texttt{val\\_raws=77} split with stored RGB reference embeddings. TruFor and DIRE are excluded because the local copies do not contain a final usable checkpoint or reconstruction cache.}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_system_compare_tex(rows: List[Dict]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{System-level comparison between our active authentication setting and representative prior active/content-authentication pipelines.}",
        "\\label{tab:active_system_compare}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Method & Setting & Verification input & Camera-side change & Credential form & Benign postproc & Comparison role \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["usage"] != "narrative reference":
            continue
        lines.append(
            f"{latex_escape(row['method'])} & {latex_escape(row['problem_type'])} & {latex_escape(row['input'])} & "
            f"{latex_escape(row['camera_change'])} & {latex_escape(row['watermark_or_signature'])} & "
            f"{latex_escape(row['benign_postproc'])} & {latex_escape(row['usage'])} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_official_ablation_table_tex(rows: List[Dict]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Component ablations of the official active-authentication scorer under the primary batch-local same-image protocol. All rows use the same checkpoint and split; only the scoring components are removed or replaced.}",
        "\\label{tab:official_ablation}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l l c c c c}",
        "\\toprule",
        "Variant & Components kept & AUC & EER & Hard AUC & TAR (FAR=$10^{-2}$) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{latex_escape(row['variant'])} & {latex_escape(row['meaning'])} & "
            f"{format_float(row['auc'],4)} & {format_float(row['eer'],4)} & "
            f"{format_float(row['hard_auc'],4)} & {format_float(row['tar'],4)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_process_ablation_table_tex(rows: List[Dict]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Process-certificate construction ablation. This table isolates the contribution of topology before RGB-side recovery is introduced.}",
        "\\label{tab:process_ablation}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Variant & Evaluation setting & Main result & Supporting result \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row["variant"] == "Joint process verifier":
            setting = "Closed registry"
            main_result = f"Accuracy (64 sources) = {format_float(row['primary'],4)}"
            support_result = f"Accuracy (128 sources) = {format_float(row['aux'],4)}"
        else:
            setting = "Ordered source consistency"
            main_result = f"Ordered-triplet acc. = {format_float(row['primary'],4)}"
            support_result = f"Cross-to-shift ratio = {format_float(row['aux'],4)}"
        lines.append(
            f"{latex_escape(row['variant'])} & {latex_escape(setting)} & "
            f"{latex_escape(main_result)} & {latex_escape(support_result)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def build_main_results_table_tex(summary: Dict | None) -> str:
    if summary is None:
        return "% Main results table not generated yet.\n"

    conditions = ["Ref", "Prop", "Shift-1", "Shift-2"]
    settings = [
        ("Process", ["Joint", "ITF", "Topology"], {"Joint"}),
        ("RGB Projection", ["Joint", "Geometric", "Topological"], {"Joint"}),
        ("Active Verification", ["Final", "Main", "Protocol"], {"Final"}),
    ]

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.4pt}",
        "\\caption{Main results under lawful same-source dissemination.}",
        "\\label{tab:main_results}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll|cccc|cccc|cccc|cccc}",
        "\\toprule",
        "\\multirow{2}{*}{Setting} & \\multirow{2}{*}{Branch}",
        "& \\multicolumn{4}{c|}{Ref}",
        "& \\multicolumn{4}{c|}{Prop}",
        "& \\multicolumn{4}{c|}{Shift-1}",
        "& \\multicolumn{4}{c}{Shift-2} \\\\",
        "& ",
        "& AUC$\\uparrow$ & EER$\\downarrow$ & TAR@1\\%$\\uparrow$ & TAR@0.1\\%$\\uparrow$",
        "& AUC$\\uparrow$ & EER$\\downarrow$ & TAR@1\\%$\\uparrow$ & TAR@0.1\\%$\\uparrow$",
        "& AUC$\\uparrow$ & EER$\\downarrow$ & TAR@1\\%$\\uparrow$ & TAR@0.1\\%$\\uparrow$",
        "& AUC$\\uparrow$ & EER$\\downarrow$ & TAR@1\\%$\\uparrow$ & TAR@0.1\\%$\\uparrow$ \\\\",
        "\\midrule",
    ]
    table = summary["primary_table"] if "primary_table" in summary else summary["table"]
    for setting_idx, (setting, branches, bold_branches) in enumerate(settings):
        for branch_idx, branch in enumerate(branches):
            row_prefix = f"\\multirow{{3}}{{*}}{{{latex_escape(setting)}}}" if branch_idx == 0 else ""
            branch_label = latex_escape(branch)
            if branch in bold_branches:
                branch_label = f"\\textbf{{{branch_label}}}"
            row = [row_prefix, branch_label]
            for condition in conditions:
                metrics = table[setting][branch][condition]
                values = [
                    format_float(metrics["auc"], 4),
                    format_float(metrics["eer"], 4),
                    format_float(metrics["tar_at_far_1e2"], 4),
                    format_float(metrics["tar_at_far_1e3"], 4),
                ]
                if branch in bold_branches:
                    values = [f"\\textbf{{{value}}}" for value in values]
                row.extend(values)
            lines.append(" & ".join(row) + " \\\\")
        if setting_idx != len(settings) - 1:
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "",
        "\\vspace{0.2em}",
        "\\parbox{\\textwidth}{\\footnotesize",
        "\\textit{Note.}",
        "Ref/Prop/Shift-1/2 denote reference, benign propagation, and two lawful rendering shifts.",
        "Bold marks the primary row in each setting; TAR@1\\%/0.1\\% use FAR $10^{-2}/10^{-3}$.}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_official_scaling_table_tex(analysis: Dict | None) -> str:
    if analysis is None:
        return "% Official protocol analysis not generated yet.\n"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Harder full-gallery claimed-source verification using the deployed official scorer. Unlike the batch-local metric, this supplementary protocol compares each query against the full registered gallery.}",
        "\\label{tab:official_scaling}",
        "\\small",
        "\\begin{tabular}{rccccc}",
        "\\toprule",
        "Registered sources & Validation sources & AUC & EER & Hard AUC & TAR (FAR=$10^{-2}$) \\\\",
        "\\midrule",
    ]
    for row in analysis.get("scaling", []):
        metrics = row["official_metrics"]
        lines.append(
            f"{int(row['max_raws'])} & {int(row['val_raws'])} & {format_float(metrics['pairwise_auc'],4)} & "
            f"{format_float(metrics['eer'],4)} & {format_float(metrics['hard_auc'],4)} & "
            f"{format_float(metrics['tar_at_far_1e2'],4)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def build_official_stress_table_tex(analysis: Dict | None) -> str:
    if analysis is None:
        return "% Official stress analysis not generated yet.\n"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Supplementary threat-oriented stress tests under the same full-gallery protocol. Random-swap ASR and hard-swap ASR are measured at the clean operating point with FAR$=10^{-2}$.}",
        "\\label{tab:official_stress}",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Scenario & AUC & EER & TAR (FAR=$10^{-2}$) & Random-swap ASR & Hard-swap ASR \\\\",
        "\\midrule",
    ]
    pretty_names = {
        "clean": "Clean full-gallery",
        "jpeg_q90": "JPEG Q=90",
        "jpeg_q75": "JPEG Q=75",
        "resize75_jpeg90": "Resize75 + JPEG90",
    }
    for row in analysis.get("stress_tests", []):
        metrics = row["official_metrics"]
        lines.append(
            f"{latex_escape(pretty_names.get(row['stress_name'], row['stress_name']))} & "
            f"{format_float(metrics['pairwise_auc'],4)} & {format_float(metrics['eer'],4)} & "
            f"{format_float(metrics['tar_at_far_1e2'],4)} & "
            f"{format_float(row['random_swap_asr_at_far_1e2'],4)} & "
            f"{format_float(row['hard_swap_asr_at_far_1e2'],4)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def build_protocol_stress_compact_table_tex(analysis: Dict | None) -> str:
    if analysis is None:
        return "% Compact protocol/stress table not generated yet.\n"
    stress_rows = {row["stress_name"]: row for row in analysis.get("stress_tests", [])}
    selected = [name for name in ["clean", "jpeg_q90", "resize75_jpeg90"] if name in stress_rows]
    pretty_names = {
        "clean": "Clean full-gallery",
        "jpeg_q90": "JPEG re-encoding",
        "resize75_jpeg90": "Resize + JPEG",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Supplementary harder-protocol and stress results under the full-gallery verifier.}",
        "\\label{tab:protocol_stress_compact}",
        "\\small",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Scenario & AUC & EER & TAR (FAR=$10^{-2}$) & Hard-swap ASR \\\\",
        "\\midrule",
    ]
    for name in selected:
        row = stress_rows[name]
        metrics = row["official_metrics"]
        lines.append(
            f"{latex_escape(pretty_names[name])} & "
            f"{format_float(metrics['pairwise_auc'],4)} & "
            f"{format_float(metrics['eer'],4)} & "
            f"{format_float(metrics['tar_at_far_1e2'],4)} & "
            f"{format_float(row['hard_swap_asr_at_far_1e2'],4)} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def build_narrative_text(internal_rows: List[Dict], external_metric_rows: List[Dict], analysis: Dict | None) -> str:
    code_oracle = next(row for row in internal_rows if row["stage"] == "P5")
    official = next(row for row in internal_rows if row["stage"] == "P6b")
    official_variant = next(row for row in internal_rows if row["stage"] == "P6c")
    joint = next(row for row in internal_rows if row["stage"] == "P3")
    external_methods = ", ".join(row["method"] for row in external_metric_rows) if external_metric_rows else "no external baseline"
    ours_proxy = next((row for row in external_metric_rows if row["method"] == "Ours-FV"), None)
    ours_proxy_text = ""
    if ours_proxy is not None:
        ours_proxy_text = (
            f" Under the same easy RGB-reference proxy protocol, our own student also saturates at "
            f"AUC={format_float(ours_proxy['pairwise_auc'],4)} and EER={format_float(ours_proxy['eer'],4)}."
        )
    stress_text = ""
    if analysis is not None:
        clean = next((row for row in analysis.get("stress_tests", []) if row.get("stress_name") == "clean"), None)
        jpeg90 = next((row for row in analysis.get("stress_tests", []) if row.get("stress_name") == "jpeg_q90"), None)
        if clean is not None and jpeg90 is not None:
            stress_text = (
                f" In the harder full-gallery stress protocol, the same official scorer reaches "
                f"AUC={format_float(clean['official_metrics']['pairwise_auc'],4)} and EER={format_float(clean['official_metrics']['eer'],4)} on clean data, "
                f"and remains near that level after JPEG re-encoding (AUC={format_float(jpeg90['official_metrics']['pairwise_auc'],4)}, "
                f"EER={format_float(jpeg90['official_metrics']['eer'],4)})."
            )
    return (
        "The experimental chapter follows a fixed evidence chain: ITF validity, topology contribution, joint process verification, "
        "teacher/oracle upper bounds, the deployed active-authentication system, and finally external baselines. "
        f"The internal pipeline already shows that ITF alone preserves perfect ordered-triplet consistency, while topology trades a small amount of sensitivity "
        "for additional structural robustness. Once the two are fused, the closed-registry protocol remains strong across 32/64/128 registered sources "
        f"with joint accuracies {format_float(joint['value_b']['32'],3)}/{format_float(joint['value_b']['64'],3)}/{format_float(joint['value_b']['128'],3)}. "
        f"Under the simple assumption of a closed registry, the teacher code oracle reaches AUC={format_float(code_oracle['value_a'],4)} and "
        f"EER={format_float(code_oracle['value_b'],4)}, which confirms that the process-side authentication signal itself is strong enough to reach the target operating point. "
        f"The fully deployed official system remains harder, currently reaching claimed AUC={format_float(official['value_a'],4)} and EER={format_float(official['value_b'],4)}, "
        "which means the remaining gap primarily comes from RGB-side recovery rather than from the absence of process evidence. "
        f"A light verifier-only soup can further push the strongest variant to AUC={format_float(official_variant['value_a'],4)} and "
        f"EER={format_float(official_variant['value_b'],4)}, so we treat the single checkpoint as the conservative deployable main result and the soup row as the strongest supplementary variant. "
        + ours_proxy_text +
        stress_text +
        " "
        f"For external literature baselines, we separate protocol-adapted RGB methods ({external_methods}) from active/content-authentication systems, "
        "and interpret them only as easy proxy baselines, so that methods with different inputs and certificate modalities are not mixed into the same main quantitative claim."
    )


def main():
    GENERATED_ROOT.mkdir(parents=True, exist_ok=True)

    internal_rows = build_internal_rows()
    external_comparison_rows = build_external_comparison_rows()
    external_metric_rows = build_external_metric_rows()
    credential_stress_rows = build_credential_binding_stress_rows()
    external_table_rows = external_metric_rows
    official_protocol_analysis = load_official_protocol_analysis()
    official_ablation_rows = load_official_ablation_rows()
    process_ablation_rows = load_process_ablation_rows()
    main_results_summary = load_main_results_summary()

    write_json(GENERATED_ROOT / "internal_pipeline_summary.json", {"rows": internal_rows})
    write_json(
        GENERATED_ROOT / "final_experiment_release.json",
        {
            "official_main_result_path": str(OFFICIAL_RESULT_PATH),
            "official_strongest_variant_path": str(OFFICIAL_STRONGEST_VARIANT_PATH),
            "official_main_result": load_json(OFFICIAL_RESULT_PATH),
            "official_strongest_variant": load_json(OFFICIAL_STRONGEST_VARIANT_PATH),
            "credential_binding_stress": credential_stress_rows,
            "student_only": load_json(STUDENT_ONLY_COSINE_PATH),
        },
    )
    write_json(GENERATED_ROOT / "external_comparison_matrix.json", {"rows": external_comparison_rows})
    write_json(GENERATED_ROOT / "external_metric_summary.json", {"rows": external_metric_rows})
    write_json(GENERATED_ROOT / "credential_binding_stress_summary.json", {"rows": credential_stress_rows})
    write_json(GENERATED_ROOT / "official_ablation_summary.json", {"rows": official_ablation_rows})
    write_json(GENERATED_ROOT / "process_ablation_summary.json", {"rows": process_ablation_rows})

    (GENERATED_ROOT / "internal_pipeline_table.tex").write_text(
        build_internal_table_tex(internal_rows), encoding="utf-8"
    )
    (GENERATED_ROOT / "main_results_table.tex").write_text(
        build_main_results_table_tex(main_results_summary), encoding="utf-8"
    )
    (GENERATED_ROOT / "internal_pipeline_compact_table.tex").write_text(
        build_internal_compact_table_tex(internal_rows), encoding="utf-8"
    )
    (GENERATED_ROOT / "external_rgb_table.tex").write_text(
        build_external_metric_table_tex(external_table_rows), encoding="utf-8"
    )
    (GENERATED_ROOT / "active_system_compare_table.tex").write_text(
        build_system_compare_tex(external_comparison_rows), encoding="utf-8"
    )
    (GENERATED_ROOT / "official_ablation_table.tex").write_text(
        build_official_ablation_table_tex(official_ablation_rows), encoding="utf-8"
    )
    (GENERATED_ROOT / "process_ablation_table.tex").write_text(
        build_process_ablation_table_tex(process_ablation_rows), encoding="utf-8"
    )
    (GENERATED_ROOT / "official_scaling_table.tex").write_text(
        build_official_scaling_table_tex(official_protocol_analysis), encoding="utf-8"
    )
    (GENERATED_ROOT / "official_stress_table.tex").write_text(
        build_official_stress_table_tex(official_protocol_analysis), encoding="utf-8"
    )
    (GENERATED_ROOT / "protocol_stress_compact_table.tex").write_text(
        build_protocol_stress_compact_table_tex(official_protocol_analysis), encoding="utf-8"
    )
    (GENERATED_ROOT / "experiment_argument.txt").write_text(
        build_narrative_text(internal_rows, external_metric_rows, official_protocol_analysis), encoding="utf-8"
    )

    print(f"[saved] {GENERATED_ROOT}")


if __name__ == "__main__":
    main()
