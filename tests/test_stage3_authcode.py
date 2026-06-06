import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.auth_code_head import AuthCodeHead
from src.models.auth_codebook import AuthenticationCodebook
from src.models.claim_score_fusion import ClaimScoreFusionHead
from src.models.prototype_verifier import PrototypeVerifier
from src.models.student_code_head import StudentCodeHead
from src.models.student_token_head import StudentTokenHead
from src.models.teacher_tokenizer import TeacherTokenTokenizer
from src.models.visual_encoder import VisualEncoder
from src.train.train_stage3_authcode import (
    aggregate_claim_reference,
    build_claim_score_outputs,
    build_claim_score_fusion_features,
    build_claim_verifier_repr,
    build_same_sample_mask,
    build_same_raw_mask,
    claim_score_fusion_input_dim,
    claim_verifier_dim,
    combine_score_matrices,
    compute_combined_scores,
    compute_claim_score_components,
    compute_claim_scores,
    compute_claim_verifier_scores,
    compute_deterministic_gated_scores,
    compute_protocol_score_bundle,
    compute_hard_score_distill_loss,
    compute_logit_recovery_loss,
    compute_optional_claim_hardcase_weights,
    compute_optional_claim_positive_tail_rescue_loss,
    compute_operating_point_proxy_loss,
    estimate_balanced_batch_threshold,
    compute_protocol_score_matrix,
    compute_score_distill_loss,
    compute_token_match_matrix,
    compute_token_prototype_alignment_loss,
    compute_total_authcode_loss,
    forward_student_branches,
    fuse_claim_score_matrices,
    inherit_init_checkpoint_config,
    maybe_expand_repetition_state_dict,
    resolve_student_match_targets,
    should_load_claim_score_fusion_state_dict,
    sweep_same_raw_claim_calibration,
    summarize_target_scores,
    summarize_masked_scores,
    soft_bits,
    parse_args as parse_train_stage3_authcode_args,
)
from src.tools.evaluate_stage3_authcode import parse_args as parse_evaluate_stage3_authcode_args
from src.tools.ecc_codec import build_auth_codec
from src.tools.stage3_official_preset import (
    OFFICIAL_STAGE3_PRESET_NAME,
    apply_stage3_official_config_to_args,
    resolve_stage3_official_config,
)


def test_visual_encoder_can_return_backbone_features():
    encoder = VisualEncoder(d_out=8, backbone_type="resnet18", pretrained=False)
    out = encoder(
        torch.randn(2, 3, 64, 64),
        return_logits=True,
        return_features=True,
    )
    assert out["global_logits"].shape == (2, 8)
    assert out["feature_vec"].shape == (2, encoder.feature_dim)


def test_auth_code_head_outputs_expected_shapes():
    head = AuthCodeHead(d_in=6, code_dim=4, hidden_dim=8, use_sequence=True)
    vec = torch.randn(3, 6)
    seq = torch.randn(3, 5, 6)
    out = head(vec, seq, return_sequence=True, return_logits=True)
    assert out["global_logits"].shape == (3, 4)
    assert out["global_repr"].shape == (3, 4)
    assert out["stage_logits"].shape == (3, 5, 4)
    assert out["stage_repr"].shape == (3, 5, 4)


def test_student_code_head_outputs_expected_shapes():
    head = StudentCodeHead(d_in=6, code_dim=4, hidden_dim=8, dropout=0.1)
    vec = torch.randn(3, 6)
    base_logits = torch.randn(3, 4)
    out = head(vec, base_logits=base_logits, return_logits=True)
    assert out["global_logits"].shape == (3, 4)
    assert out["global_repr"].shape == (3, 4)


def test_student_token_head_outputs_expected_shapes():
    head = StudentTokenHead(d_in=6, num_tokens=5, hidden_dim=8, dropout=0.1)
    vec = torch.randn(3, 6)
    out = head(vec)
    assert out["token_logits"].shape == (3, 5)
    assert out["token_probs"].shape == (3, 5)
    assert torch.allclose(out["token_probs"].sum(dim=1), torch.ones(3), atol=1e-6)


def test_student_token_head_can_use_recovery_logits_as_context():
    head = StudentTokenHead(d_in=6, num_tokens=5, code_dim=4, hidden_dim=8, dropout=0.0)
    vec = torch.randn(3, 6)
    base_logits = torch.randn(3, 4)
    out = head(vec, base_logits=base_logits)
    assert out["token_logits"].shape == (3, 5)
    assert out["token_probs"].shape == (3, 5)


def test_forward_student_branches_supports_token_head_without_code_head():
    student = VisualEncoder(d_out=4, backbone_type="resnet18", pretrained=False)
    token_head = StudentTokenHead(d_in=student.feature_dim, num_tokens=3, code_dim=4, hidden_dim=8, dropout=0.0)

    def encode_student_logits(logits):
        return logits

    out = forward_student_branches(
        student=student,
        rgb_image=torch.randn(2, 3, 64, 64),
        use_sequence=False,
        encode_student_logits=encode_student_logits,
        auth_codebook=None,
        apply_codebook_to_scores=False,
        student_code_head=None,
        student_token_head=token_head,
    )
    assert out["recovery_logits"].shape == (2, 4)
    assert out["token_logits"].shape == (2, 3)
    assert out["token_probs"].shape == (2, 3)


def test_inherit_init_checkpoint_config_recovers_architecture_flags():
    args = argparse.Namespace(
        teacher_key="teacher_joint_seq",
        code_dim=32,
        payload_dim=None,
        ecc_scheme="identity",
        ecc_repetition=2,
        student_code_space="payload",
        teacher_hidden_dim=256,
        teacher_dropout=0.0,
        teacher_codebook_size=0,
        teacher_codebook_temperature=12.0,
        codebook_mode="replace",
        use_claim_verifier_head=False,
        claim_verifier_hidden_dim=128,
        claim_verifier_dropout=0.0,
        claim_verifier_input_mode="bits",
        use_student_code_head=False,
        student_code_head_hidden_dim=256,
        student_code_head_dropout=0.0,
        use_claim_score_fusion_head=False,
        claim_score_fusion_hidden_dim=32,
        claim_score_fusion_dropout=0.0,
        claim_score_fusion_protocol_modes=None,
        claim_score_fusion_mode="direct",
        claim_score_fusion_residual_scale=1.0,
        backbone_type="resnet18",
        input_mode="residual_only",
        residual_scale=1.75,
        residual_kernel=9,
        local_crop_mode="none",
        local_crop_size=160,
        local_patch_offset=24,
        image_size=224,
        resize_size=320,
        augmentation_preset="center",
        include_versions=[1, 2, 3, 4],
        anchor_versions=[1, 2],
        shift_versions=[3, 4],
        sequence_score_weight=0.2,
        teacher_sequence_match_weight=0.5,
        student_sequence_match_weight=0.5,
    )
    checkpoint = {
        "config": {
            "use_student_code_head": True,
            "student_code_head_hidden_dim": 192,
            "backbone_type": "resnet50",
            "input_mode": "rgb_residual",
            "image_size": 256,
            "sequence_score_weight": 0.0,
            "claim_score_fusion_mode": "residual",
            "claim_score_fusion_residual_scale": 0.75,
        }
    }
    inherited = inherit_init_checkpoint_config(args, checkpoint)
    assert "use_student_code_head" in inherited
    assert args.use_student_code_head is True
    assert args.student_code_head_hidden_dim == 192
    assert args.backbone_type == "resnet50"
    assert args.input_mode == "rgb_residual"
    assert args.image_size == 256
    assert args.sequence_score_weight == 0.0
    assert args.claim_score_fusion_mode == "residual"
    assert args.claim_score_fusion_residual_scale == 0.75


def test_inherit_init_checkpoint_config_recovers_loss_weights():
    args = argparse.Namespace(
        teacher_consistency_weight=1.0,
        teacher_pair_weight=0.5,
        teacher_bank_pair_weight=0.5,
        teacher_bank_hard_weight=0.25,
        teacher_claim_pair_weight=0.0,
        teacher_claim_hard_weight=0.0,
        teacher_codebook_commit_weight=0.0,
        teacher_positive_margin=0.55,
        teacher_negative_margin=0.10,
        teacher_hard_margin=0.15,
        student_bit_weight=1.0,
        student_sequence_match_weight=0.5,
        student_pair_weight=1.0,
        student_target_mode="canonical",
        student_target_blend_alpha=0.5,
        student_claim_bit_weight=0.0,
        student_claim_pair_weight=0.0,
        student_claim_bank_pair_weight=0.0,
        student_protocol_claim_pair_weight=0.0,
        student_protocol_claim_bank_pair_weight=0.0,
        student_hard_weight=0.0,
        student_claim_hard_weight=0.0,
        student_claim_bank_hard_weight=0.0,
        student_protocol_claim_hard_weight=0.0,
        student_protocol_claim_bank_hard_weight=0.0,
        student_soft_align_weight=0.5,
        student_claim_align_weight=0.0,
        student_recovery_bit_weight=0.0,
        student_recovery_stage_weight=0.0,
        student_recovery_align_weight=0.0,
        student_token_class_weight=1.0,
        student_token_proto_weight=0.5,
        student_score_distill_weight=0.0,
        student_hard_score_distill_weight=0.0,
        student_claim_score_distill_weight=0.0,
        student_codebook_class_weight=0.0,
        student_codebook_proto_weight=0.0,
        student_positive_margin=0.55,
        student_negative_margin=0.10,
        student_hard_margin=0.15,
        claim_reference_mode="same_raw",
        claim_reference_versions=None,
        claim_bank_mode="mean_bits",
        claim_anchor_aux_weight=0.0,
        teacher_balance_weight=0.1,
        teacher_decorrelation_weight=0.05,
        teacher_uniformity_weight=0.05,
        codebook_usage_weight=0.0,
        codebook_separation_weight=0.0,
        uniformity_temperature=2.0,
        protocol_score_mode="none",
        protocol_score_alpha=0.0,
        official_claim_score_mode="deterministic_gate",
        claim_main_score_norm_mode="none",
        protocol_score_norm_mode="none",
        bit_scale=3.0,
        use_joint_claim_loss=False,
    )
    checkpoint = {
        "config": {
            "student_claim_bit_weight": 0.5,
            "student_claim_pair_weight": 1.0,
            "student_claim_bank_pair_weight": 0.25,
            "student_claim_hard_weight": 1.25,
            "student_claim_bank_hard_weight": 0.5,
            "student_claim_align_weight": 0.5,
            "student_recovery_bit_weight": 0.25,
            "student_recovery_align_weight": 0.25,
            "protocol_score_mode": "code_cosine",
            "bit_scale": 1.0,
        }
    }
    inherited = inherit_init_checkpoint_config(args, checkpoint)
    assert "student_claim_bit_weight" in inherited
    assert "student_claim_pair_weight" in inherited
    assert "student_claim_bank_pair_weight" in inherited
    assert "student_claim_hard_weight" in inherited
    assert "student_claim_bank_hard_weight" in inherited
    assert "student_claim_align_weight" in inherited
    assert "student_recovery_bit_weight" in inherited
    assert "student_recovery_align_weight" in inherited
    assert "protocol_score_mode" in inherited
    assert "bit_scale" in inherited
    assert args.student_claim_bit_weight == 0.5
    assert args.student_claim_pair_weight == 1.0
    assert args.student_claim_bank_pair_weight == 0.25
    assert args.student_claim_hard_weight == 1.25
    assert args.student_claim_bank_hard_weight == 0.5
    assert args.student_claim_align_weight == 0.5
    assert args.student_recovery_bit_weight == 0.25
    assert args.student_recovery_align_weight == 0.25
    assert args.protocol_score_mode == "code_cosine"
    assert args.bit_scale == 1.0


def test_stage3_official_preset_resolution_prioritizes_cli_over_preset_over_checkpoint():
    preset_name, resolved = resolve_stage3_official_config(
        config={
            "claim_reference_mode": "same_raw",
            "protocol_score_mode": "none",
            "bit_scale": 1.0,
            "claim_verifier_weight": 0.25,
        },
        preset_name=OFFICIAL_STAGE3_PRESET_NAME,
        cli_overrides={
            "bit_scale": 1.5,
            "claim_reference_mode": None,
        },
    )
    assert preset_name == OFFICIAL_STAGE3_PRESET_NAME
    assert resolved["claim_reference_mode"] == "same_image"
    assert resolved["protocol_score_mode"] == "code_cosine"
    assert resolved["claim_verifier_weight"] == 1.0
    assert resolved["bit_scale"] == 1.5


def test_apply_stage3_official_config_to_args_overrides_training_official_fields():
    args = argparse.Namespace(
        official_eval_preset=OFFICIAL_STAGE3_PRESET_NAME,
        claim_reference_mode="same_raw",
        official_claim_score_mode="deterministic_gate",
        protocol_score_mode="none",
        bit_scale=3.0,
        claim_verifier_weight=0.5,
        claim_main_score_norm_mode="row_center",
        protocol_score_norm_mode="row_zscore",
    )
    preset_name, resolved = apply_stage3_official_config_to_args(args)
    assert preset_name == OFFICIAL_STAGE3_PRESET_NAME
    assert resolved["claim_reference_mode"] == "same_image"
    assert args.claim_reference_mode == "same_image"
    assert args.official_claim_score_mode == "fusion_head"
    assert args.protocol_score_mode == "code_cosine"
    assert args.bit_scale == 1.2
    assert args.claim_verifier_weight == 1.0
    assert args.claim_main_score_norm_mode == "none"
    assert args.protocol_score_norm_mode == "none"


def test_train_parse_args_defaults_to_official_selection_metric():
    with mock.patch.object(sys, "argv", ["train_stage3_authcode.py"]):
        args = parse_train_stage3_authcode_args()
    assert args.official_eval_preset == OFFICIAL_STAGE3_PRESET_NAME
    assert args.selection_metric == "val_student_claimed_official_pairwise_auc"
    assert args.claim_score_fusion_mode == "direct"
    assert args.claim_score_fusion_residual_scale == 1.0


def test_evaluate_parse_args_leaves_claim_reference_unset_until_resolution():
    with mock.patch.object(sys, "argv", ["evaluate_stage3_authcode.py", "--checkpoint", "dummy.pt"]):
        args = parse_evaluate_stage3_authcode_args()
    assert args.official_eval_preset is None
    assert args.claim_reference_mode is None


def test_teacher_tokenizer_assigns_and_looks_up_tokens():
    tokenizer = TeacherTokenTokenizer(code_dim=3, num_tokens=2, temperature=8.0)
    with torch.no_grad():
        tokenizer.prototypes.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.7, 0.0]], dtype=torch.float32)
    indices, token_logits = tokenizer.assign(logits)
    assert indices.tolist() == [0, 1]
    assert token_logits.shape == (2, 2)
    looked_up = tokenizer.lookup(indices)
    assert looked_up.shape == (2, 3)


def test_repetition_codec_encodes_and_decodes_payload_bits():
    codec = build_auth_codec(code_dim=6, ecc_scheme="repetition", ecc_repetition=3, payload_dim=2)
    payload_logits = torch.tensor([[1.5, -2.0]], dtype=torch.float32)
    codeword_logits = codec.encode_logits(payload_logits)
    assert torch.allclose(codeword_logits, torch.tensor([[1.5, 1.5, 1.5, -2.0, -2.0, -2.0]], dtype=torch.float32))

    noisy_codeword = torch.tensor([[1.2, -0.4, 0.9, -1.0, 0.3, -2.2]], dtype=torch.float32)
    decoded_logits = codec.decode_logits(noisy_codeword)
    decoded_bits = codec.hard_payload_bits_from_codeword(noisy_codeword)
    assert torch.allclose(decoded_logits, torch.tensor([[0.5666667, -0.9666667]], dtype=torch.float32), atol=1e-5)
    assert decoded_bits.tolist() == [[1, 0]]


def test_repetition_state_dict_expands_output_layers():
    source_state = {
        "head.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "head.bias": torch.tensor([0.1, -0.2], dtype=torch.float32),
    }
    target_state = {
        "head.weight": torch.zeros(4, 2, dtype=torch.float32),
        "head.bias": torch.zeros(4, dtype=torch.float32),
    }
    expanded = maybe_expand_repetition_state_dict(source_state, target_state, repetition=2)
    assert torch.allclose(
        expanded["head.weight"],
        torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]], dtype=torch.float32),
    )
    assert torch.allclose(expanded["head.bias"], torch.tensor([0.1, 0.1, -0.2, -0.2], dtype=torch.float32))


def test_auth_codebook_quantizes_to_nearest_prototype():
    codebook = AuthenticationCodebook(code_dim=3, num_codes=2, temperature=8.0)
    with torch.no_grad():
        codebook.codes.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        )
    logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.7, 0.0]], dtype=torch.float32)
    quantized, indices, sims, prototypes = codebook.quantize(logits, straight_through=False)
    assert indices.tolist() == [0, 1]
    assert sims.shape == (2, 2)
    assert torch.allclose(quantized, prototypes)
    assert torch.allclose(prototypes, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32))


def test_compute_combined_scores_uses_sequence_branch():
    query_logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    bank_bits = soft_bits(torch.tensor([[2.0, -2.0], [-2.0, 2.0]], dtype=torch.float32), scale=2.0)
    query_stage_logits = torch.tensor([[[2.0, -2.0], [2.0, -2.0]]], dtype=torch.float32)
    bank_stage_bits = soft_bits(
        torch.tensor(
            [
                [[2.0, -2.0], [2.0, -2.0]],
                [[-2.0, 2.0], [-2.0, 2.0]],
            ],
            dtype=torch.float32,
        ),
        scale=2.0,
    )
    logits, _ = compute_combined_scores(
        query_logits,
        bank_bits,
        query_stage_logits=query_stage_logits,
        bank_stage_bits=bank_stage_bits,
        bit_scale=2.0,
        sequence_score_weight=0.5,
    )
    assert logits.shape == (1, 2)
    assert logits.argmax(dim=1).tolist() == [0]


def test_claim_score_helpers_support_same_raw_and_verifier():
    mask = build_same_raw_mask(["raw_a", "raw_a", "raw_b"], device=torch.device("cpu"))
    assert mask.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]

    sample_mask = build_same_sample_mask(["s1", "s2", "s1"], device=torch.device("cpu"))
    assert sample_mask.tolist() == [
        [True, False, True],
        [False, True, False],
        [True, False, True],
    ]

    student_bits = torch.randn(3, 4)
    teacher_bits = torch.randn(3, 4)
    verifier = PrototypeVerifier(d_model=4, hidden_dim=8)
    scores = compute_claim_scores(student_bits, teacher_bits, verifier)
    assert scores.shape == (3, 3)


def test_claim_score_helpers_support_stage_sequence_scores():
    student_bits = torch.tensor([[1.0, -1.0], [0.9, -0.8]], dtype=torch.float32)
    teacher_bits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
    student_stage_logits = torch.tensor(
        [
            [[2.0, -2.0], [2.0, -2.0]],
            [[1.8, -1.6], [1.9, -1.7]],
        ],
        dtype=torch.float32,
    )
    teacher_stage_logits = torch.tensor(
        [
            [[2.0, -2.0], [2.0, -2.0]],
            [[-2.0, 2.0], [-2.0, 2.0]],
        ],
        dtype=torch.float32,
    )
    teacher_stage_bits = soft_bits(
        teacher_stage_logits,
        scale=2.0,
    )
    bit_scores = compute_claim_scores(
        student_bits,
        teacher_bits,
        claim_verifier=None,
        verifier_weight=0.0,
        query_stage_logits=student_stage_logits,
        reference_stage_bits=teacher_stage_bits,
        bit_scale=2.0,
        sequence_score_weight=0.5,
    )
    cosine_scores = compute_claim_scores(
        student_bits,
        teacher_bits,
        claim_verifier=None,
        verifier_weight=0.0,
        query_stage_logits=student_stage_logits,
        reference_stage_logits=teacher_stage_logits,
        bit_scale=2.0,
        sequence_score_weight=0.5,
    )
    assert bit_scores.shape == (2, 2)
    assert bit_scores.argmax(dim=1).tolist() == [0, 0]
    assert cosine_scores.shape == (2, 2)
    assert cosine_scores.argmax(dim=1).tolist() == [0, 0]


def test_protocol_score_helpers_support_cosine_and_decode_modes():
    codec = build_auth_codec(code_dim=4, ecc_scheme="identity", ecc_repetition=1, payload_dim=4)
    query_logits = torch.tensor([[3.0, -3.0, 2.0, -2.0], [3.0, 3.0, -2.0, -2.0]], dtype=torch.float32)
    reference_logits = torch.tensor([[3.0, -3.0, 2.0, -2.0], [-3.0, 3.0, -2.0, 2.0]], dtype=torch.float32)
    cosine_scores = compute_protocol_score_matrix(query_logits, reference_logits, codec, mode="code_cosine")
    decode_scores = compute_protocol_score_matrix(query_logits, reference_logits, codec, mode="decode_success")
    assert cosine_scores.shape == (2, 2)
    assert cosine_scores.argmax(dim=1).tolist() == [0, 0]
    assert decode_scores.tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_claim_verifier_repr_modes_cover_bits_logits_and_concat():
    bits = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
    logits = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    assert claim_verifier_dim(2, "bits") == 2
    assert claim_verifier_dim(2, "logits") == 2
    assert claim_verifier_dim(2, "bits_logits") == 4
    assert torch.allclose(build_claim_verifier_repr(bits, logits, "bits"), bits)
    assert torch.allclose(build_claim_verifier_repr(bits, logits, "logits"), torch.tensor([[0.6, 0.8]]))
    concat = build_claim_verifier_repr(bits, logits, "bits_logits")
    assert concat.shape == (1, 4)
    assert torch.allclose(concat, torch.tensor([[0.5, -0.5, 0.6, 0.8]]), atol=1e-6)


def test_compute_claim_scores_can_use_custom_verifier_representations():
    verifier = PrototypeVerifier(d_model=4, hidden_dim=8, dropout=0.0)
    query_bits = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    reference_bits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
    query_repr = torch.tensor([[1.0, -1.0, 0.6, 0.8]], dtype=torch.float32)
    reference_repr = torch.tensor([[1.0, -1.0, 0.6, 0.8], [-1.0, 1.0, -0.6, 0.8]], dtype=torch.float32)
    scores = compute_claim_scores(
        query_bits,
        reference_bits,
        verifier,
        verifier_weight=0.5,
        query_verifier_repr=query_repr,
        reference_verifier_repr=reference_repr,
    )
    assert scores.shape == (1, 2)


def test_claim_verifier_feature_only_mode_keeps_main_scores_clean():
    verifier = PrototypeVerifier(d_model=2, hidden_dim=8, dropout=0.0)
    query_bits = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    reference_bits = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32)
    plain_scores = compute_claim_scores(query_bits, reference_bits, claim_verifier=None)
    feature_only_scores = compute_claim_scores(
        query_bits,
        reference_bits,
        verifier,
        verifier_weight=1.0,
        verifier_score_mode="feature_only",
    )
    assert torch.allclose(feature_only_scores, plain_scores)
    verifier_scores = compute_claim_verifier_scores(
        query_bits,
        reference_bits,
        verifier,
        verifier_weight=1.0,
        verifier_score_mode="feature_only",
    )
    assert verifier_scores is not None
    components = compute_claim_score_components(
        query_bits,
        reference_bits,
        verifier,
        verifier_weight=1.0,
        verifier_score_mode="feature_only",
    )
    assert torch.allclose(components["claim_scores"], plain_scores)
    assert torch.allclose(components["verifier_scores"], verifier_scores)


def test_compute_total_authcode_loss_includes_protocol_claim_terms():
    args = SimpleNamespace(
        teacher_consistency_weight=0.0,
        teacher_sequence_match_weight=0.0,
        teacher_pair_weight=0.0,
        teacher_bank_pair_weight=0.0,
        teacher_bank_hard_weight=0.0,
        teacher_claim_pair_weight=0.0,
        teacher_claim_hard_weight=0.0,
        teacher_codebook_commit_weight=0.0,
        student_bit_weight=0.0,
        student_sequence_match_weight=0.0,
        student_pair_weight=0.0,
        student_claim_bit_weight=0.0,
        student_claim_pair_weight=0.0,
        student_claim_bank_pair_weight=0.0,
        student_protocol_claim_pair_weight=2.0,
        student_protocol_claim_bank_pair_weight=3.0,
        student_verifier_claim_pair_weight=0.0,
        student_hard_weight=0.0,
        student_claim_hard_weight=0.0,
        student_claim_bank_hard_weight=0.0,
        student_protocol_claim_hard_weight=5.0,
        student_protocol_claim_bank_hard_weight=7.0,
        student_verifier_claim_hard_weight=0.0,
        student_verifier_claim_bank_hard_weight=0.0,
        student_soft_align_weight=0.0,
        student_claim_align_weight=0.0,
        student_recovery_bit_weight=0.0,
        student_recovery_stage_weight=0.0,
        student_recovery_align_weight=0.0,
        student_score_distill_weight=0.0,
        student_hard_score_distill_weight=0.0,
        student_claim_score_distill_weight=0.0,
        student_codebook_class_weight=0.0,
        student_codebook_proto_weight=0.0,
        teacher_balance_weight=0.0,
        teacher_decorrelation_weight=0.0,
        teacher_uniformity_weight=0.0,
        codebook_usage_weight=0.0,
        codebook_separation_weight=0.0,
    )
    total = compute_total_authcode_loss(
        args,
        student_protocol_claim_pair_loss=torch.tensor(1.0),
        student_protocol_claim_bank_pair_loss=torch.tensor(1.0),
        student_protocol_claim_hard_loss=torch.tensor(1.0),
        student_protocol_claim_bank_hard_loss=torch.tensor(1.0),
    )
    assert torch.allclose(total, torch.tensor(17.0))


def test_compute_total_authcode_loss_includes_verifier_claim_terms():
    args = SimpleNamespace(
        teacher_consistency_weight=0.0,
        teacher_sequence_match_weight=0.0,
        teacher_pair_weight=0.0,
        teacher_bank_pair_weight=0.0,
        teacher_bank_hard_weight=0.0,
        teacher_claim_pair_weight=0.0,
        teacher_claim_hard_weight=0.0,
        teacher_codebook_commit_weight=0.0,
        student_bit_weight=0.0,
        student_sequence_match_weight=0.0,
        student_pair_weight=0.0,
        student_claim_bit_weight=0.0,
        student_claim_pair_weight=0.0,
        student_claim_bank_pair_weight=0.0,
        student_protocol_claim_pair_weight=0.0,
        student_protocol_claim_bank_pair_weight=0.0,
        student_verifier_claim_pair_weight=2.0,
        student_hard_weight=0.0,
        student_claim_hard_weight=0.0,
        student_claim_bank_hard_weight=0.0,
        student_protocol_claim_hard_weight=0.0,
        student_protocol_claim_bank_hard_weight=0.0,
        student_verifier_claim_hard_weight=3.0,
        student_verifier_claim_bank_hard_weight=5.0,
        student_soft_align_weight=0.0,
        student_claim_align_weight=0.0,
        student_recovery_bit_weight=0.0,
        student_recovery_stage_weight=0.0,
        student_recovery_align_weight=0.0,
        student_score_distill_weight=0.0,
        student_hard_score_distill_weight=0.0,
        student_claim_score_distill_weight=0.0,
        student_codebook_class_weight=0.0,
        student_codebook_proto_weight=0.0,
        teacher_balance_weight=0.0,
        teacher_decorrelation_weight=0.0,
        teacher_uniformity_weight=0.0,
        codebook_usage_weight=0.0,
        codebook_separation_weight=0.0,
    )
    total = compute_total_authcode_loss(
        args,
        student_verifier_claim_pair_loss=torch.tensor(1.0),
        student_verifier_claim_hard_loss=torch.tensor(1.0),
        student_verifier_claim_bank_hard_loss=torch.tensor(1.0),
    )
    assert torch.allclose(total, torch.tensor(10.0))


def test_combine_score_matrices_blends_main_and_protocol_scores():
    main_scores = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    protocol_scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    joint_scores = combine_score_matrices(main_scores, protocol_scores, alpha=0.25)
    expected = torch.tensor([[0.85, 0.15], [0.225, 0.775]], dtype=torch.float32)
    assert torch.allclose(joint_scores, expected)


def test_combine_score_matrices_supports_row_zscore_normalization():
    main_scores = torch.tensor([[1.0, 3.0], [2.0, 6.0]], dtype=torch.float32)
    protocol_scores = torch.tensor([[10.0, 14.0], [4.0, 12.0]], dtype=torch.float32)
    joint_scores = combine_score_matrices(
        main_scores,
        protocol_scores,
        alpha=0.5,
        main_normalization="none",
        auxiliary_normalization="row_zscore",
    )
    expected = torch.tensor(
        [
            [0.0, 2.0],
            [0.5, 3.5],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(joint_scores, expected, atol=1e-6)


def test_claim_score_fusion_head_outputs_expected_shape():
    head = ClaimScoreFusionHead(input_dim=12, hidden_dim=16, dropout=0.1)
    features = torch.randn(5, 7, 12)
    scores = head(features)
    assert scores.shape == (5, 7)


def test_claim_score_fusion_head_residual_mode_starts_from_base_scores():
    head = ClaimScoreFusionHead(
        input_dim=12,
        hidden_dim=16,
        dropout=0.0,
        mode="residual",
        residual_scale=2.0,
    )
    features = torch.randn(5, 7, 12)
    base_scores = torch.randn(5, 7)
    scores = head(features, base_scores=base_scores)
    assert torch.allclose(scores, base_scores)


def test_build_claim_score_fusion_features_returns_contextual_features():
    main_scores = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float32)
    protocol_scores = torch.tensor([[0.8, 0.2, -0.4]], dtype=torch.float32)
    features = build_claim_score_fusion_features(main_scores, protocol_scores)
    assert features.shape == (1, 3, 12)
    assert torch.allclose(features[0, :, 0], main_scores[0])
    assert torch.allclose(features[0, :, 1], protocol_scores[0])
    assert features[0, 0, 4] > features[0, 1, 4] > features[0, 2, 4]


def test_claim_score_fusion_features_support_extra_protocol_scores():
    main_scores = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float32)
    protocol_scores = torch.tensor([[0.8, 0.2, -0.4]], dtype=torch.float32)
    payload_scores = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
    decode_scores = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    features = build_claim_score_fusion_features(
        main_scores,
        protocol_scores,
        extra_protocol_scores=[payload_scores, decode_scores],
    )
    assert claim_score_fusion_input_dim(["payload_agreement", "decode_success"]) == 20
    assert features.shape == (1, 3, 20)
    assert torch.allclose(features[0, :, 12], payload_scores[0])
    assert torch.allclose(features[0, :, 16], decode_scores[0])


def test_claim_score_fusion_input_dim_supports_verifier_auxiliary_score():
    assert claim_score_fusion_input_dim(include_verifier_score=True) == 16
    assert claim_score_fusion_input_dim(["payload_agreement"], include_verifier_score=True) == 20


def test_claim_score_fusion_features_support_token_scores():
    main_scores = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float32)
    protocol_scores = torch.tensor([[0.8, 0.2, -0.4]], dtype=torch.float32)
    token_scores = torch.tensor([[0.95, 0.3, 0.05]], dtype=torch.float32)
    features = build_claim_score_fusion_features(
        main_scores,
        protocol_scores,
        token_scores=token_scores,
    )
    assert claim_score_fusion_input_dim(include_token_score=True) == 16
    assert features.shape == (1, 3, 16)
    assert torch.allclose(features[0, :, 12], token_scores[0])


def test_claim_score_fusion_features_pad_missing_extra_protocol_scores():
    main_scores = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float32)
    protocol_scores = torch.tensor([[0.8, 0.2, -0.4]], dtype=torch.float32)
    features = build_claim_score_fusion_features(
        main_scores,
        protocol_scores,
        extra_protocol_scores=[None, torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)],
    )
    assert features.shape == (1, 3, 20)
    assert torch.allclose(features[0, :, 12], torch.zeros(3))
    assert torch.allclose(features[0, :, 16], torch.tensor([1.0, 0.5, 0.0]))


def test_fuse_claim_score_matrices_can_use_learned_head():
    main_scores = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    protocol_scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    fusion = torch.nn.Linear(12, 1, bias=False)
    with torch.no_grad():
        fusion.weight.zero_()
        fusion.weight[0, 0] = 1.0
    fused_scores = fuse_claim_score_matrices(
        main_scores,
        protocol_scores,
        claim_score_fusion_head=fusion,
    )
    assert torch.allclose(fused_scores, main_scores)


def test_fuse_claim_score_matrices_residual_head_starts_from_combined_scores():
    main_scores = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
    protocol_scores = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    head = ClaimScoreFusionHead(
        input_dim=12,
        hidden_dim=16,
        dropout=0.0,
        mode="residual",
        residual_scale=1.5,
    )
    fused_scores = fuse_claim_score_matrices(
        main_scores,
        protocol_scores,
        claim_score_fusion_head=head,
        alpha=0.25,
    )
    expected_scores = combine_score_matrices(main_scores, protocol_scores, alpha=0.25)
    assert torch.allclose(fused_scores, expected_scores)


def test_should_skip_claim_score_fusion_load_when_mode_mismatch():
    args = argparse.Namespace(claim_score_fusion_mode="residual")
    checkpoint = {
        "config": {"claim_score_fusion_mode": "direct"},
        "claim_score_fusion_state_dict": {"dummy": torch.tensor([1.0])},
    }
    should_load, reason = should_load_claim_score_fusion_state_dict(args, checkpoint)
    assert should_load is False
    assert "mode mismatch" in reason


def test_compute_protocol_score_bundle_returns_primary_and_extra_scores():
    codec = build_auth_codec(code_dim=4, ecc_scheme="identity", ecc_repetition=1, payload_dim=4)
    query_logits = torch.tensor([[3.0, -3.0, 2.0, -2.0]], dtype=torch.float32)
    reference_logits = torch.tensor([[3.0, -3.0, 2.0, -2.0], [-3.0, 3.0, -2.0, 2.0]], dtype=torch.float32)
    primary, extras = compute_protocol_score_bundle(
        query_protocol_logits=query_logits,
        reference_protocol_logits=reference_logits,
        auth_codec=codec,
        primary_mode="code_cosine",
        extra_modes=["payload_agreement", "decode_success"],
    )
    assert primary.shape == (1, 2)
    assert len(extras) == 2
    assert extras[0].shape == (1, 2)
    assert extras[1].shape == (1, 2)


def test_compute_protocol_score_bundle_keeps_missing_extra_modes_as_placeholders():
    codec = build_auth_codec(code_dim=4, ecc_scheme="identity", ecc_repetition=1, payload_dim=4)
    query_logits = torch.tensor([[3.0, -3.0, 2.0, -2.0]], dtype=torch.float32)
    reference_logits = torch.tensor([[3.0, -3.0, 2.0, -2.0], [-3.0, 3.0, -2.0, 2.0]], dtype=torch.float32)
    primary, extras = compute_protocol_score_bundle(
        query_protocol_logits=query_logits,
        reference_protocol_logits=reference_logits,
        auth_codec=codec,
        primary_mode="code_cosine",
        extra_modes=["index_match", "payload_agreement"],
    )
    assert primary.shape == (1, 2)
    assert len(extras) == 2
    assert extras[0] is None
    assert extras[1].shape == (1, 2)


def test_compute_token_match_matrix_uses_reference_token_indices():
    query_token_logits = torch.tensor([[3.0, 1.0], [0.5, 2.5]], dtype=torch.float32)
    reference_token_indices = torch.tensor([0, 1], dtype=torch.long)
    scores = compute_token_match_matrix(query_token_logits, reference_token_indices)
    assert scores.shape == (2, 2)
    assert scores[0, 0] > scores[0, 1]
    assert scores[1, 1] > scores[1, 0]


def test_compute_deterministic_gated_scores_penalizes_token_mismatch():
    main_scores = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    token_scores = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    residual_scores = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    gated = compute_deterministic_gated_scores(
        main_scores,
        token_scores,
        residual_scores=residual_scores,
        gate_penalty=1.0,
        residual_weight=0.25,
    )
    assert gated.shape == (1, 2)
    assert gated[0, 0] > gated[0, 1]


def test_build_claim_score_outputs_supports_deterministic_gate_and_fusion():
    main_scores = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    protocol_scores = torch.tensor([[0.6, -0.4]], dtype=torch.float32)
    token_scores = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    outputs = build_claim_score_outputs(
        main_scores,
        protocol_scores,
        token_scores=token_scores,
        official_mode="deterministic_gate",
        gate_penalty=1.0,
        residual_weight=0.25,
    )
    assert set(outputs.keys()) == {"official_scores", "fusion_scores", "gated_scores", "hard_gated_scores"}
    assert torch.allclose(outputs["official_scores"], outputs["gated_scores"])
    assert outputs["official_scores"][0, 0] > outputs["official_scores"][0, 1]


def test_token_prototype_alignment_loss_is_small_for_matching_tokens():
    tokenizer = TeacherTokenTokenizer(code_dim=2, num_tokens=2, temperature=8.0)
    with torch.no_grad():
        tokenizer.prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    query_token_logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]], dtype=torch.float32)
    target_indices = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.ones(2, dtype=torch.float32)
    loss = compute_token_prototype_alignment_loss(query_token_logits, tokenizer, target_indices, weights)
    assert float(loss.item()) < 0.1


def test_summarize_target_scores_extracts_pos_and_negatives():
    scores = torch.tensor(
        [
            [0.9, 0.1, 0.2],
            [0.3, 0.8, 0.4],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    pos, hard_neg, all_neg = summarize_target_scores(scores, targets)
    assert torch.allclose(pos, torch.tensor([0.9, 0.8]))
    assert torch.allclose(hard_neg, torch.tensor([0.2, 0.4]))
    assert torch.allclose(all_neg, torch.tensor([0.1, 0.2, 0.3, 0.4]))


def test_summarize_masked_scores_can_ignore_neutral_pairs():
    scores = torch.tensor(
        [
            [0.9, 0.8, 0.2],
            [0.7, 0.95, 0.1],
            [0.3, 0.2, 0.85],
        ],
        dtype=torch.float32,
    )
    positive_mask = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        dtype=torch.bool,
    )
    negative_mask = torch.tensor(
        [
            [False, False, True],
            [False, False, True],
            [True, True, False],
        ],
        dtype=torch.bool,
    )
    pos, hard_neg, all_neg = summarize_masked_scores(scores, positive_mask, negative_mask=negative_mask)
    assert torch.allclose(pos, torch.tensor([0.9, 0.95, 0.85]))
    assert torch.allclose(hard_neg, torch.tensor([0.2, 0.1, 0.3]))
    assert torch.allclose(all_neg, torch.tensor([0.2, 0.1, 0.3, 0.2]))


def test_aggregate_claim_reference_supports_sharp_bank_modes():
    bit_list = [
        torch.tensor([0.2, -0.4, 0.6], dtype=torch.float32),
        torch.tensor([0.4, -0.2, 0.8], dtype=torch.float32),
    ]
    logit_list = [
        torch.tensor([0.3, -0.7, 0.5], dtype=torch.float32),
        torch.tensor([0.1, -0.5, 0.9], dtype=torch.float32),
    ]

    mean_bits = aggregate_claim_reference(bit_list, logit_list, bit_scale=1.5, claim_bank_mode="mean_bits")
    mean_logits_tanh = aggregate_claim_reference(bit_list, logit_list, bit_scale=1.5, claim_bank_mode="mean_logits_tanh")
    sign_mean_logits = aggregate_claim_reference(bit_list, logit_list, bit_scale=1.5, claim_bank_mode="sign_mean_logits")

    assert torch.allclose(mean_bits, torch.tensor([0.3, -0.3, 0.7]))
    assert torch.allclose(mean_logits_tanh, torch.tanh(torch.tensor([0.3, -0.9, 1.05])), atol=1e-5)
    assert torch.all((sign_mean_logits == 1.0) | (sign_mean_logits == -1.0))
    assert sign_mean_logits.tolist() == [1.0, -1.0, 1.0]


def test_sweep_same_raw_claim_calibration_returns_best_candidate():
    batch_logits_records = [
        {
            "student_logits": torch.tensor([[0.9, -0.3], [0.8, -0.2], [-0.8, 0.2]], dtype=torch.float32),
            "teacher_logits": torch.tensor([[1.2, -0.4], [1.0, -0.3], [-1.1, 0.4]], dtype=torch.float32),
            "raw_anchors": ["raw_a", "raw_a", "raw_b"],
        }
    ]
    best = sweep_same_raw_claim_calibration(
        batch_logits_records=batch_logits_records,
        claim_verifier=None,
        device=torch.device("cpu"),
        bit_scales=[0.5, 1.0],
        verifier_weights=[0.0],
        sequence_weights=[0.0, 0.5],
    )
    assert best is not None
    assert best["bit_scale"] in {0.5, 1.0}
    assert best["verifier_weight"] == 0.0
    assert best["sequence_weight"] in {0.0, 0.5}
    assert "tar_at_far_1e2" in best["metrics"]


def test_estimate_balanced_batch_threshold_stays_between_tail_anchors():
    pos_scores = torch.tensor([0.6, 0.7, 0.8, 0.9])
    neg_scores = torch.tensor([-0.2, -0.1, 0.1, 0.2, 0.3])
    threshold = estimate_balanced_batch_threshold(
        pos_scores,
        neg_scores,
        positive_quantile=0.25,
        negative_quantile=0.75,
    )
    assert 0.2 <= float(threshold) <= 0.8


def test_operating_point_proxy_loss_drops_for_better_separation():
    sample_weights = torch.ones(3)
    worse_loss = compute_operating_point_proxy_loss(
        pos_scores=torch.tensor([0.15, 0.20, 0.25]),
        hard_neg_scores=torch.tensor([0.10, 0.12, 0.14]),
        all_neg_scores=torch.tensor([0.05, 0.08, 0.10, 0.12, 0.14]),
        sample_weights=sample_weights,
        positive_quantile=0.1,
        negative_quantile=0.9,
        margin=0.02,
        scale=10.0,
    )
    better_loss = compute_operating_point_proxy_loss(
        pos_scores=torch.tensor([0.45, 0.50, 0.55]),
        hard_neg_scores=torch.tensor([-0.10, -0.08, -0.06]),
        all_neg_scores=torch.tensor([-0.20, -0.16, -0.12, -0.10, -0.06]),
        sample_weights=sample_weights,
        positive_quantile=0.1,
        negative_quantile=0.9,
        margin=0.02,
        scale=10.0,
    )
    assert float(better_loss) < float(worse_loss)


def test_claim_hardcase_reweighting_upweights_small_gap_samples():
    scores = torch.tensor(
        [
            [0.9, 0.2, 0.1],
            [0.3, 0.29, 0.28],
            [0.1, 0.2, 0.8],
        ],
        dtype=torch.float32,
    )
    positive_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [False, False, True],
        ]
    )
    base_weights = torch.ones(3)
    reweighted = compute_optional_claim_hardcase_weights(
        scores,
        claim_reference_mode="same_image",
        sample_weights=base_weights,
        strength=1.0,
        margin=0.05,
        scale=10.0,
        claim_positive_mask=positive_mask,
    )
    assert float(reweighted[1]) > float(reweighted[0])
    assert float(reweighted[1]) > float(reweighted[2])


def test_positive_tail_rescue_loss_is_smaller_for_stronger_positives():
    positive_mask = torch.tensor(
        [
            [True, False, False],
            [True, False, False],
            [False, False, True],
        ]
    )
    sample_weights = torch.ones(3)
    weak_scores = torch.tensor(
        [
            [0.15, 0.10, 0.05],
            [0.18, 0.12, 0.11],
            [0.05, 0.08, 0.20],
        ],
        dtype=torch.float32,
    )
    strong_scores = torch.tensor(
        [
            [0.55, 0.10, 0.05],
            [0.58, 0.12, 0.11],
            [0.05, 0.08, 0.60],
        ],
        dtype=torch.float32,
    )
    weak_loss = compute_optional_claim_positive_tail_rescue_loss(
        weak_scores,
        claim_reference_mode="same_image",
        sample_weights=sample_weights,
        positive_quantile=0.1,
        negative_quantile=0.9,
        margin=0.01,
        scale=10.0,
        claim_positive_mask=positive_mask,
    )
    strong_loss = compute_optional_claim_positive_tail_rescue_loss(
        strong_scores,
        claim_reference_mode="same_image",
        sample_weights=sample_weights,
        positive_quantile=0.1,
        negative_quantile=0.9,
        margin=0.01,
        scale=10.0,
        claim_positive_mask=positive_mask,
    )
    assert float(strong_loss) < float(weak_loss)


def test_resolve_student_match_targets_supports_sample_and_blend_modes():
    canonical_logits = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    sample_logits = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)
    canonical_stage = canonical_logits.unsqueeze(1).repeat(1, 2, 1)
    sample_stage = sample_logits.unsqueeze(1).repeat(1, 2, 1)

    sample_target_logits, sample_target_stage, sample_target_bits = resolve_student_match_targets(
        canonical_logits=canonical_logits,
        sample_logits=sample_logits,
        canonical_stage_logits=canonical_stage,
        sample_stage_logits=sample_stage,
        bit_scale=1.5,
        target_mode="sample",
    )
    blend_target_logits, blend_target_stage, blend_target_bits = resolve_student_match_targets(
        canonical_logits=canonical_logits,
        sample_logits=sample_logits,
        canonical_stage_logits=canonical_stage,
        sample_stage_logits=sample_stage,
        bit_scale=1.5,
        target_mode="blend",
        blend_alpha=0.25,
    )

    assert torch.allclose(sample_target_logits, sample_logits)
    assert torch.allclose(sample_target_stage, sample_stage)
    assert torch.allclose(sample_target_bits, soft_bits(sample_logits, 1.5))
    assert torch.allclose(blend_target_logits, torch.tensor([[1.25, -1.25]], dtype=torch.float32))
    assert torch.allclose(blend_target_stage, torch.tensor([[[1.25, -1.25], [1.25, -1.25]]], dtype=torch.float32))
    assert torch.allclose(blend_target_bits, soft_bits(blend_target_logits, 1.5))


def test_logit_recovery_loss_is_small_for_identical_logits():
    logits = torch.tensor([[0.3, -0.7], [1.2, -1.0]], dtype=torch.float32)
    weights = torch.ones(2, dtype=torch.float32)
    loss = compute_logit_recovery_loss(logits, logits, weights)
    assert float(loss.item()) < 1e-6


def test_score_distill_loss_is_small_for_identical_scores():
    scores = torch.tensor([[2.0, 0.5, -1.0], [1.5, -0.5, -2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 0], dtype=torch.long)
    weights = torch.ones(2, dtype=torch.float32)
    loss = compute_score_distill_loss(
        student_scores=scores,
        teacher_scores=scores,
        targets=targets,
        sample_weights=weights,
        temperature=0.5,
        topk=2,
    )
    assert float(loss.item()) < 1e-5


def test_hard_score_distill_loss_is_small_for_identical_scores():
    scores = torch.tensor([[2.0, 0.5, -1.0], [1.5, -0.5, -2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 0], dtype=torch.long)
    weights = torch.ones(2, dtype=torch.float32)
    loss = compute_hard_score_distill_loss(
        student_scores=scores,
        teacher_scores=scores,
        targets=targets,
        sample_weights=weights,
        temperature=0.5,
    )
    assert float(loss.item()) < 1e-5


def test_total_authcode_loss_includes_hard_and_score_distill_terms():
    args = argparse.Namespace(
        teacher_consistency_weight=1.0,
        teacher_sequence_match_weight=0.0,
        teacher_pair_weight=0.0,
        teacher_bank_pair_weight=0.0,
        teacher_bank_hard_weight=0.0,
        student_bit_weight=0.0,
        student_sequence_match_weight=0.0,
        student_pair_weight=0.0,
        student_claim_bit_weight=0.0,
        student_claim_pair_weight=0.0,
        student_hard_weight=2.0,
        student_claim_hard_weight=0.0,
        student_soft_align_weight=0.0,
        student_claim_align_weight=0.0,
        student_score_distill_weight=3.0,
        student_hard_score_distill_weight=5.0,
        student_claim_score_distill_weight=0.0,
        teacher_balance_weight=0.0,
        teacher_decorrelation_weight=0.0,
        teacher_uniformity_weight=0.0,
    )
    total = compute_total_authcode_loss(
        args,
        teacher_consistency_loss=torch.tensor(1.0),
        student_hard_loss=torch.tensor(2.0),
        student_score_distill_loss=torch.tensor(4.0),
        student_hard_score_distill_loss=torch.tensor(6.0),
    )
    assert torch.isclose(total, torch.tensor(47.0))
