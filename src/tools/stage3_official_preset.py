OFFICIAL_STAGE3_PRESET_NAME = "official_sameimage_fusion_v1"
OFFICIAL_STAGE3_EVAL_MAX_RAWS = 1024

OFFICIAL_STAGE3_PRESET_FIELDS = (
    "claim_reference_mode",
    "official_claim_score_mode",
    "protocol_score_mode",
    "bit_scale",
    "claim_verifier_weight",
    "claim_main_score_norm_mode",
    "protocol_score_norm_mode",
)

OFFICIAL_STAGE3_PRESETS = {
    OFFICIAL_STAGE3_PRESET_NAME: {
        "claim_reference_mode": "same_image",
        "official_claim_score_mode": "fusion_head",
        "protocol_score_mode": "code_cosine",
        "bit_scale": 1.2,
        "claim_verifier_weight": 1.0,
        "claim_main_score_norm_mode": "none",
        "protocol_score_norm_mode": "none",
    }
}


def normalize_stage3_official_preset_name(name):
    if name is None:
        return None
    normalized = str(name).strip()
    if not normalized or normalized.lower() == "none":
        return None
    if normalized not in OFFICIAL_STAGE3_PRESETS:
        raise ValueError(f"Unsupported stage3 official preset: {normalized}")
    return normalized


def resolve_stage3_official_config(config=None, preset_name=None, cli_overrides=None):
    config = dict(config or {})
    cli_overrides = dict(cli_overrides or {})
    resolved = {}
    for key in OFFICIAL_STAGE3_PRESET_FIELDS:
        value = config.get(key)
        if value is not None:
            resolved[key] = value

    resolved_preset_name = normalize_stage3_official_preset_name(
        preset_name if preset_name is not None else config.get("official_eval_preset")
    )
    if resolved_preset_name is not None:
        resolved.update(OFFICIAL_STAGE3_PRESETS[resolved_preset_name])

    for key, value in cli_overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved_preset_name, resolved


def apply_stage3_official_config_to_args(args, preset_name=None):
    resolved_preset_name, resolved = resolve_stage3_official_config(
        config=vars(args),
        preset_name=preset_name if preset_name is not None else getattr(args, "official_eval_preset", None),
        cli_overrides=None,
    )
    setattr(args, "official_eval_preset", resolved_preset_name)
    for key, value in resolved.items():
        setattr(args, key, value)
    return resolved_preset_name, resolved
