import numpy as np
import pytest
import torch

from dvsrc.baselines import DTWBaseline, dtw_distance
from dvsrc.config import ModelConfig
from dvsrc.losses import T2Loss, project_conflicting
from dvsrc.model import (
    ConditionalABCT2Head, ConditionalSENT2Head, DualEvidenceT2Head, DVSRNet, MaskedBatchNorm1d,
    MultiScaleCoordinateFusion, ProgressiveBayesianT2Head, SequenceEncoder,
    SharedPairEvidenceMatcher, SignatureEncoder, SignatureEncoding, SpatialTaskAdapter,
    StatefulBayesianT2Head,
    T1Head, T2Head, T2InternalAdapter, T2ResidualAdapter, UnifiedEvidenceT2Head,
    T2RelationEnergyAdapter, temporal_resample,
)


def encoding(batch: int, members: int | None = None, tokens: int = 8, dim: int = 32) -> SignatureEncoding:
    shape = (batch, tokens, dim) if members is None else (batch, members, tokens, dim)
    global_shape = (batch, dim) if members is None else (batch, members, dim)
    mask_shape = shape[:-1]
    spatial_shape = (*shape[:-1], 2)
    return SignatureEncoding(
        global_shared=torch.randn(global_shape), local_shared=torch.randn(shape),
        valid_mask=torch.ones(mask_shape, dtype=torch.bool), spatial=torch.rand(spatial_shape) * 2 - 1,
        global_sequence=torch.randn(global_shape), global_image=torch.randn(global_shape), diagnostics={},
    )


def config() -> ModelConfig:
    return ModelConfig(hidden_dim=32, conformer_heads=4, conformer_ffn=64, use_tsa=True,
                       sinkhorn_iters=8, sinkhorn_epsilon=0.1)


def permute_members(value: SignatureEncoding, permutation: torch.Tensor) -> SignatureEncoding:
    return SignatureEncoding(
        value.global_shared[:, permutation], value.local_shared[:, permutation],
        value.valid_mask[:, permutation], value.spatial[:, permutation],
        value.global_sequence[:, permutation], value.global_image[:, permutation], {},
    )


def test_t1_pair_is_symmetric():
    torch.manual_seed(1)
    head = T1Head(config()).eval()
    reference = encoding(2, members=1)
    query = encoding(2)
    mask = torch.ones(2, 1, dtype=torch.bool)
    first = head(reference, query, mask)["case_logit"]
    query_as_set = SignatureEncoding(
        query.global_shared[:, None], query.local_shared[:, None], query.valid_mask[:, None], query.spatial[:, None],
        query.global_sequence[:, None], query.global_image[:, None], {},
    )
    reference_as_query = SignatureEncoding(
        reference.global_shared[:, 0], reference.local_shared[:, 0], reference.valid_mask[:, 0], reference.spatial[:, 0],
        reference.global_sequence[:, 0], reference.global_image[:, 0], {},
    )
    second = head(query_as_set, reference_as_query, mask)["case_logit"]
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


def test_qrsa_is_reference_permutation_invariant():
    torch.manual_seed(2)
    head = T1Head(config()).eval()
    references, query = encoding(2, members=5), encoding(2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    first = head(references, query, mask)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    second = head(permute_members(references, permutation), query, mask)
    torch.testing.assert_close(first["case_logit"], second["case_logit"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(first["reliability"][:, permutation], second["reliability"], atol=2e-6, rtol=2e-6)


def test_v3_t1_uses_protocol_specific_final_heads():
    torch.manual_seed(22)
    model_config = config()
    model_config.variant = "v3"
    head = T1Head(model_config).train()
    one = head(encoding(2, members=1), encoding(2), torch.ones(2, 1, dtype=torch.bool))
    one["case_logit"].sum().backward()
    assert any(parameter.grad is not None for parameter in head.one_to_one.parameters())
    assert all(parameter.grad is None for parameter in head.qrsa.parameters())

    head.zero_grad(set_to_none=True)
    five = head(encoding(2, members=5), encoding(2), torch.ones(2, 5, dtype=torch.bool))
    five["case_logit"].sum().backward()
    assert all(parameter.grad is None for parameter in head.one_to_one.parameters())
    assert any(parameter.grad is not None for parameter in head.qrsa.parameters())


def test_v4_one_to_one_head_starts_as_exact_v1_residual():
    torch.manual_seed(24)
    model_config = config()
    model_config.variant = "v4"
    head = T1Head(model_config).eval()
    output = head(encoding(2, members=1), encoding(2), torch.ones(2, 1, dtype=torch.bool))

    torch.testing.assert_close(output["case_logit"], output["pair_logits"][:, 0])


def test_v5r1_set_head_starts_from_permutation_invariant_pair_consensus():
    torch.manual_seed(27)
    model_config = config()
    model_config.variant = "v5r1"
    head = T1Head(model_config).eval()
    references, query = encoding(2, members=5), encoding(2)
    mask = torch.ones(2, 5, dtype=torch.bool)

    first = head(references, query, mask)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    second = head(permute_members(references, permutation), query, mask)
    expected = 0.5 * (first["pair_logits"].mean(1) + first["pair_logits"].median(1).values)

    torch.testing.assert_close(first["case_logit"], expected)
    torch.testing.assert_close(first["evidence_anchor_logit"], expected)
    torch.testing.assert_close(first["case_logit"], second["case_logit"], atol=2e-6, rtol=2e-6)


def test_t2_candidate_permutation_equivariance_and_unknown_invariance():
    torch.manual_seed(3)
    head = T2Head(config()).eval()
    candidates, query = encoding(2, members=4), encoding(2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    second = head(permute_members(candidates, permutation), query, mask)
    torch.testing.assert_close(first["rank_logits"][:, permutation], second["rank_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["unknown_probability"], second["unknown_probability"], atol=3e-5, rtol=3e-5)
    assert torch.isfinite(first["sinkhorn_error"]).all()


def test_conditional_abc_factorization_and_candidate_permutation_contract():
    torch.manual_seed(31)
    model_config = config()
    model_config.variant = "t2_abc_v1"
    head = ConditionalABCT2Head(model_config).eval()
    candidates, query = encoding(3, members=8), encoding(3)
    mask = torch.ones(3, 8, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([5, 0, 7, 2, 1, 6, 4, 3])
    second = head(permute_members(candidates, permutation), query, mask)

    torch.testing.assert_close(first["joint_probability"].sum(1), torch.ones(3))
    expected_present = (1 - first["rf_probability"]) * first["in_set_probability"]
    torch.testing.assert_close(first["exist_probability"], expected_present)
    torch.testing.assert_close(
        first["unknown_probability"],
        first["rf_probability"] + (1 - first["rf_probability"]) * (1 - first["in_set_probability"]),
    )
    torch.testing.assert_close(first["rf_probability"], second["rf_probability"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["in_set_probability"], second["in_set_probability"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(
        first["rank_probability"][:, permutation], second["rank_probability"], atol=3e-5, rtol=3e-5,
    )


def test_conditional_abc_exposes_seven_class_official_probability():
    torch.manual_seed(33)
    model_config = config()
    model_config.variant = "unified_abc_v11"
    head = ConditionalABCT2Head(model_config).eval()
    candidates, query = encoding(2, members=5), encoding(2)
    mask = torch.ones(2, 5, dtype=torch.bool)

    output = head(candidates, query, mask)

    assert output["official_probability"].shape == (2, 7)
    torch.testing.assert_close(output["official_probability"].sum(1), torch.ones(2))
    torch.testing.assert_close(output["official_probability"][:, :5], output["candidate_probability"])
    torch.testing.assert_close(output["official_probability"][:, 5], output["absent_probability"])
    torch.testing.assert_close(output["official_probability"][:, 6], output["rf_probability"])


def test_unified_sen_v20_shares_pair_evidence_and_preserves_abc_factorization():
    torch.manual_seed(39)
    model_config = config()
    model_config.variant = "unified_sen_v20"
    matcher = SharedPairEvidenceMatcher(32, 4, 0.0).eval()
    head = ConditionalSENT2Head(model_config).eval()
    candidates, query = encoding(2, members=5), encoding(2)
    mask = torch.ones(2, 5, dtype=torch.bool)
    first_relation = matcher(candidates, query)
    first = head(first_relation, query, mask)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = permute_members(candidates, permutation)
    second = head(matcher(permuted, query), query, mask)

    assert first["source_pair_embedding"].shape == (2, 5, 32)
    torch.testing.assert_close(first["official_probability"].sum(1), torch.ones(2))
    torch.testing.assert_close(
        first["exist_probability"],
        (1 - first["rf_probability"]) * first["in_set_probability"],
    )
    torch.testing.assert_close(
        first["rank_probability"][:, permutation], second["rank_probability"],
        atol=3e-5, rtol=3e-5,
    )
    torch.testing.assert_close(
        first["in_set_probability"], second["in_set_probability"],
        atol=3e-5, rtol=3e-5,
    )


def test_shared_pair_evidence_zero_variance_has_finite_gradients():
    matcher = SharedPairEvidenceMatcher(32, 4, 0.0).eval()
    references, query = encoding(2, members=1), encoding(2)
    references.local_shared = torch.zeros_like(
        references.local_shared, requires_grad=True,
    )
    query.local_shared = torch.zeros_like(query.local_shared, requires_grad=True)

    output = matcher(references, query)
    loss = output["pair_embedding"].sum() + output["relation_stats"].sum()
    loss.backward()

    assert torch.isfinite(references.local_shared.grad).all()
    assert torch.isfinite(query.local_shared.grad).all()


def test_unified_v12_t2_adapter_starts_as_mask_preserving_identity():
    torch.manual_seed(34)
    source = encoding(2, tokens=6, dim=32)
    source.valid_mask[:, -2:] = False
    adapter = T2ResidualAdapter(dim=32, bottleneck=8).eval()

    adapted = adapter(source)

    torch.testing.assert_close(adapted.global_shared, source.global_shared)
    torch.testing.assert_close(adapted.global_sequence, source.global_sequence)
    torch.testing.assert_close(adapted.global_image, source.global_image)
    torch.testing.assert_close(adapted.local_shared, source.local_shared)
    assert adapted.valid_mask.data_ptr() == source.valid_mask.data_ptr()


def test_unified_v14_internal_adapter_starts_as_encoder_identity():
    torch.manual_seed(36)
    model_config = config()
    model_config.variant = "unified_abc_v14"
    model_config.pretrained = False
    model_config.fusion = "sequence_only"
    model_config.conformer_layers = 2
    model_config.sequence_tokens = 8
    encoder = SignatureEncoder(model_config).eval()
    adapter = T2InternalAdapter(model_config).eval()
    sequence = torch.randn(2, 12, model_config.feature_dim)
    sequence_mask = torch.ones(2, 12, dtype=torch.bool)
    image = torch.zeros(2, 3, 32, 32)
    anchors = torch.zeros(2, 8, 6)
    anchor_mask = torch.ones(2, 8, dtype=torch.bool)

    baseline = encoder(sequence, sequence_mask, image, anchors, anchor_mask)
    adapted = encoder(
        sequence, sequence_mask, image, anchors, anchor_mask, task_adapter=adapter,
    )

    torch.testing.assert_close(adapted.global_shared, baseline.global_shared)
    torch.testing.assert_close(adapted.local_shared, baseline.local_shared)
    assert len(adapter.sequence) == model_config.conformer_layers
    spatial = torch.randn(2, 32, 4, 4)
    torch.testing.assert_close(adapter.image_stage3(spatial), spatial)
    assert isinstance(adapter.image_stage3, SpatialTaskAdapter)


def test_unified_v14_t1_path_bypasses_nonzero_t2_internal_adapter():
    torch.manual_seed(37)
    model_config = config()
    model_config.variant = "unified_abc_v14"
    model_config.t1_variant = "v5r1"
    model_config.pretrained = False
    model_config.fusion = "sequence_only"
    model_config.conformer_layers = 2
    model_config.sequence_tokens = 8
    model = DVSRNet(model_config).eval()
    batch = {
        "protocol": "t1_1v1",
        "sequence": torch.randn(2, 12, model_config.feature_dim),
        "sequence_mask": torch.ones(2, 12, dtype=torch.bool),
        "image": torch.zeros(2, 3, 32, 32),
        "anchors": torch.zeros(2, 8, 6),
        "anchor_mask": torch.ones(2, 8, dtype=torch.bool),
        "query_index": torch.tensor([1]),
        "set_index": torch.tensor([[0]]),
        "set_mask": torch.ones(1, 1, dtype=torch.bool),
    }

    baseline = model(batch)["case_logit"]
    assert isinstance(model.t2_adapter, T2InternalAdapter)
    with torch.no_grad():
        for adapter in model.t2_adapter.sequence:
            adapter.up.bias.fill_(1.0)
    after_t2_change = model(batch)["case_logit"]

    torch.testing.assert_close(after_t2_change, baseline)


def test_unified_scratch_null_source_head_preserves_probability_contract():
    torch.manual_seed(38)
    model_config = config()
    model_config.variant = "unified_abc_v14"
    model_config.t1_variant = "v5r1"
    model_config.t2_null_source_enabled = True
    model_config.pretrained = False
    model_config.fusion = "sequence_only"
    model_config.conformer_layers = 2
    model_config.sequence_tokens = 8
    model = DVSRNet(model_config).eval()
    batch = {
        "protocol": "t2",
        "sequence": torch.randn(6, 12, model_config.feature_dim),
        "sequence_mask": torch.ones(6, 12, dtype=torch.bool),
        "image": torch.zeros(6, 3, 32, 32),
        "anchors": torch.zeros(6, 8, 6),
        "anchor_mask": torch.ones(6, 8, dtype=torch.bool),
        "query_index": torch.tensor([5]),
        "set_index": torch.tensor([[0, 1, 2, 3, 4]]),
        "set_mask": torch.ones(1, 5, dtype=torch.bool),
        "target_index": torch.tensor([2]),
        "episode_type_index": torch.tensor([0]),
    }

    output = model(batch)

    assert model.t2_adapter is not None
    assert model.t2_energy_adapter is None
    assert model.t2.null_source_head is not None
    torch.testing.assert_close(output["official_probability"].sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(
        output["full_hierarchical_probability"].sum(dim=-1), torch.ones(1),
    )
    torch.testing.assert_close(
        output["full_hierarchical_probability"][:, 0], output["t1_genuine_probability"],
    )
    torch.testing.assert_close(
        output["full_hierarchical_probability"][:, 1:],
        output["t1_forgery_probability"][:, None] * output["official_probability"],
    )


def test_relation_energy_adapter_is_identity_at_initialization_and_preserves_probabilities():
    torch.manual_seed(39)
    batch, members, dim = 2, 5, 32
    adapter = T2RelationEnergyAdapter(dim, dropout=0.0).eval()
    query = encoding(batch, dim=dim)
    set_mask = torch.ones(batch, members, dtype=torch.bool)
    rank_logits = torch.randn(batch, members)
    rank_probability = rank_logits.softmax(dim=-1)
    rf_probability = torch.sigmoid(torch.randn(batch))
    in_set_logit = torch.randn(batch)
    output = {
        "source_pair_embedding": torch.randn(batch, members, dim),
        "rank_logits": rank_logits,
        "rank_probability": rank_probability,
        "rf_probability": rf_probability,
        "in_set_logit": in_set_logit.clone(),
        "in_set_probability": torch.sigmoid(in_set_logit),
    }

    adapted = adapter(output, query, set_mask)

    torch.testing.assert_close(adapted["in_set_logit"], in_set_logit)
    torch.testing.assert_close(adapted["relation_energy_residual"], torch.zeros(batch))
    torch.testing.assert_close(adapted["official_probability"].sum(dim=-1), torch.ones(batch))
    assert adapted["official_probability"].shape == (batch, members + 2)


def test_unified_v14_teacher_and_student_collect_matching_fusion_layers():
    torch.manual_seed(38)
    model_config = config()
    fusion = MultiScaleCoordinateFusion(
        model_config.hidden_dim, model_config.conformer_heads, dropout=0.0,
    ).eval()
    adapter = T2InternalAdapter(model_config).eval()
    temporal = torch.randn(2, 4, 32)
    temporal_mask = torch.ones(2, 4, dtype=torch.bool)
    anchors = torch.rand(2, 4, 6)
    anchor_valid = torch.ones(2, 4, dtype=torch.bool)
    stage3 = torch.randn(2, 32, 8, 8)
    stage4 = torch.randn(2, 32, 4, 4)
    image_global = torch.randn(2, 32)
    arguments = (
        temporal, temporal_mask, anchors, anchor_valid, stage3, stage4, image_global,
    )

    _, teacher_stages = fusion.forward_adapted(
        *arguments, None, None, collect_stages=True,
    )
    _, student_stages = fusion.forward_adapted(
        *arguments, adapter.fusion_input, adapter.fusion_tokens, collect_stages=True,
    )

    assert teacher_stages.keys() == student_stages.keys()
    assert {"fusion.layer.0", "fusion.layer.1"} <= teacher_stages.keys()


def test_unified_v12_conditional_head_keeps_a_b_c_probability_contract():
    torch.manual_seed(35)
    model_config = config()
    model_config.variant = "unified_abc_v12"
    head = ConditionalABCT2Head(model_config).eval()
    candidates, query = encoding(2, members=5), encoding(2)

    output = head(candidates, query, torch.ones(2, 5, dtype=torch.bool))

    expected_present = (1 - output["rf_probability"]) * output["in_set_probability"]
    torch.testing.assert_close(output["exist_probability"], expected_present)
    torch.testing.assert_close(output["official_probability"].sum(1), torch.ones(2))
    torch.testing.assert_close(output["rank_probability"].sum(1), torch.ones(2))


def test_unified_v13_direct_factors_margin_pairing_and_teacher_distillation_affect_loss():
    batch = {
        "target_index": torch.tensor([1, -1]),
        "episode_type_index": torch.tensor([0, 1]),
        "set_mask": torch.ones(2, 3, dtype=torch.bool),
        "metadata": [
            {"counterfactual_pair_id": "pair-1", "counterfactual_role": "present"},
            {"counterfactual_pair_id": "pair-1", "counterfactual_role": "absent"},
        ],
    }
    common = {
        "rf_logit": torch.tensor([-3.0, -3.0]),
        "joint_probability": torch.full((2, 4), 0.25),
        "official_probability": torch.full((2, 5), 0.20),
        "teacher_rank_logits": torch.tensor([[0.0, 4.0, -1.0], [0.0, 0.0, 0.0]]),
        "teacher_in_set_logit": torch.tensor([3.0, -3.0]),
    }
    good = {
        **common,
        "rank_logits": torch.tensor([[0.0, 3.0, -1.0], [0.0, 0.0, 0.0]]),
        "in_set_logit": torch.tensor([2.0, -2.0]),
    }
    bad = {
        **common,
        "rank_logits": torch.tensor([[3.0, -1.0, 0.0], [0.0, 0.0, 0.0]]),
        "in_set_logit": torch.tensor([-2.0, 2.0]),
    }
    loss = T2Loss(
        mode="conditional_abc_v1", joint_weight=0.0, official_weight=0.0,
        factor_b_weight=0.75, rank_weight=1.25, rank_margin_weight=0.25,
        pair_weight=0.25, rank_distillation_weight=0.5,
        b_distillation_weight=0.25, distillation_temperature=2.0,
    )

    good_loss, parts = loss(good, batch)
    bad_loss, _ = loss(bad, batch)

    assert good_loss < bad_loss
    assert parts["t2_counterfactual_pair"] > 0
    assert parts["t2_rank_distillation"] >= 0
    assert parts["t2_teacher_hard_negative"] >= 0
    assert parts["t2_b_distillation"] >= 0
    assert parts["t2_rank_margin"] >= 0


def test_unified_v14_rank_and_b_losses_are_phase_isolated():
    batch = {
        "target_index": torch.tensor([0, -1]),
        "episode_type_index": torch.tensor([0, 1]),
        "set_mask": torch.ones(2, 3, dtype=torch.bool),
        "metadata": [{}, {}],
    }
    output = {
        "rf_logit": torch.zeros(2),
        "rank_logits": torch.tensor([[3.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        "in_set_logit": torch.tensor([2.0, -2.0]),
        "joint_probability": torch.full((2, 4), 0.25),
        "official_probability": torch.full((2, 5), 0.20),
        "teacher_rank_logits": torch.tensor([[4.0, 2.0, -1.0], [0.0, 0.0, 0.0]]),
        "teacher_in_set_logit": torch.tensor([3.0, -3.0]),
    }
    loss = T2Loss(
        mode="conditional_abc_v1", rank_weight=1.0, rank_margin_weight=0.5,
        rank_distillation_weight=1.0, hard_negative_weight=0.5,
        factor_b_weight=1.0, b_distillation_weight=0.25,
    )

    loss.set_training_phase("rank_recovery")
    rank_total, rank_parts = loss(output, batch)
    loss.set_training_phase("b_calibration")
    b_total, b_parts = loss(output, batch)

    assert torch.isfinite(rank_total)
    assert torch.isfinite(b_total)
    assert rank_parts["t2_teacher_hard_negative"] > 0
    assert b_parts["t2_b_distillation"] > 0


def test_conditional_abc_a_only_path_matches_full_query_decision():
    torch.manual_seed(32)
    model_config = config()
    model_config.variant = "t2_abc_v1"
    head = ConditionalABCT2Head(model_config).eval()
    candidates, query = encoding(2, members=8), encoding(2)
    mask = torch.ones(2, 8, dtype=torch.bool)

    a_only = head.forward_a(query)
    full = head(candidates, query, mask)

    torch.testing.assert_close(a_only["rf_logit"], full["rf_logit"])
    torch.testing.assert_close(a_only["rf_probability"], full["rf_probability"])


def test_v3_t2_absolute_evidence_is_candidate_permutation_equivariant():
    torch.manual_seed(23)
    model_config = config()
    model_config.variant = "v3"
    head = T2Head(model_config).eval()
    candidates, query = encoding(2, members=4), encoding(2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    second = head(permute_members(candidates, permutation), query, mask)

    torch.testing.assert_close(first["match_logits"][:, permutation], second["match_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["rank_logits"][:, permutation], second["rank_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["exist_probability"], second["exist_probability"], atol=3e-5, rtol=3e-5)


def test_v4_open_set_evidence_is_permutation_equivariant():
    torch.manual_seed(25)
    model_config = config()
    model_config.variant = "v4"
    head = T2Head(model_config).eval()
    candidates, query = encoding(2, members=4), encoding(2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    second = head(permute_members(candidates, permutation), query, mask)

    torch.testing.assert_close(first["match_logits"][:, permutation], second["match_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["rank_logits"][:, permutation], second["rank_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["unknown_probability"], second["unknown_probability"], atol=3e-5, rtol=3e-5)


def test_v5_subtype_evidence_is_permutation_equivariant_and_normalized():
    torch.manual_seed(26)
    model_config = config()
    model_config.variant = "v5"
    head = T2Head(model_config).eval()
    candidates, query = encoding(2, members=4), encoding(2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    second = head(permute_members(candidates, permutation), query, mask)

    torch.testing.assert_close(first["match_logits"][:, permutation], second["match_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["rank_logits"][:, permutation], second["rank_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["type_probability"], second["type_probability"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["joint_probability"].sum(1), torch.ones(2))


def test_v6_bayesian_probabilities_are_normalized_and_permutation_equivariant():
    torch.manual_seed(27)
    model_config = config()
    model_config.variant = "v6"
    head = T2Head(model_config).eval()
    candidates, query = encoding(2, members=4), encoding(2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    second = head(permute_members(candidates, permutation), query, mask)

    torch.testing.assert_close(first["rank_logits"][:, permutation], second["rank_logits"], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["joint_probability"][:, permutation], second["joint_probability"][:, :-1], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["joint_probability"][:, -1], second["joint_probability"][:, -1], atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(first["joint_probability"].sum(1), torch.ones(2))
    torch.testing.assert_close(first["type_probability"].sum(1), torch.ones(2))


def test_v6_loss_rewards_counterfactual_in_set_ordering():
    batch = {
        "target_index": torch.tensor([1, -1]),
        "episode_type_index": torch.tensor([0, 1]),
        "set_mask": torch.ones(2, 3, dtype=torch.bool),
        "metadata": [
            {"counterfactual_pair_id": "pair-1", "counterfactual_role": "present"},
            {"counterfactual_pair_id": "pair-1", "counterfactual_role": "absent"},
        ],
    }
    common = {
        "rf_logit": torch.tensor([-2.0, -2.0]),
        "rank_logits": torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 0.0]]),
        "match_logits": torch.tensor([[0.0, 2.0, 0.0], [-2.0, -2.0, -2.0]]),
        "training_joint_probability": torch.tensor([[0.03, 0.85, 0.03, 0.09], [0.03, 0.03, 0.03, 0.91]]),
    }
    good = {**common, "in_set_logit": torch.tensor([2.0, -2.0])}
    reversed_pair = {**common, "in_set_logit": torch.tensor([-2.0, 2.0])}
    loss = T2Loss(mode="bayesian_v6")

    good_loss, parts = loss(good, batch)
    bad_loss, _ = loss(reversed_pair, batch)

    assert good_loss < bad_loss
    assert parts["t2_counterfactual_pair"] > 0


def test_v7_progressive_posteriors_are_normalized_and_candidate_permutation_invariant():
    torch.manual_seed(28)
    model_config = config()
    model_config.variant = "v7"
    head = ProgressiveBayesianT2Head(model_config).eval()
    candidates, query = encoding(2, members=4), encoding(2)
    mask = torch.ones(2, 4, dtype=torch.bool)
    first = head(candidates, query, mask)
    permutation = torch.tensor([2, 0, 3, 1])
    second = head(permute_members(candidates, permutation), query, mask)

    torch.testing.assert_close(first["rank_logits"][:, permutation], second["rank_logits"], atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(
        first["bayesian_stage_probability"], second["bayesian_stage_probability"],
        atol=4e-5, rtol=4e-5,
    )
    torch.testing.assert_close(first["bayesian_stage_probability"].sum(-1), torch.ones(2, 4))
    torch.testing.assert_close(first["joint_probability"].sum(-1), torch.ones(2))


def test_v7_training_phase_never_updates_rank_and_open_parameters_together():
    model_config = config()
    model_config.variant = "v7"
    head = ProgressiveBayesianT2Head(model_config)

    head.set_training_phase("rank")
    assert all(parameter.requires_grad for parameter in head.rank_branch.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in head.named_parameters() if not name.startswith("rank_branch.")
    )

    head.set_training_phase("open")
    assert not any(parameter.requires_grad for parameter in head.rank_branch.parameters())
    assert all(
        parameter.requires_grad
        for name, parameter in head.named_parameters() if not name.startswith("rank_branch.")
    )


def test_v7_loss_rewards_progressive_correct_class_evidence():
    batch = {
        "target_index": torch.tensor([1, -1, -1]),
        "episode_type_index": torch.tensor([0, 1, 2]),
        "set_mask": torch.ones(3, 3, dtype=torch.bool),
        "metadata": [{}, {}, {}],
    }
    rank_logits = torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    good_stage_logits = torch.tensor([
        [[0.0, -1.0, -0.5], [0.5, -0.5, -1.0], [1.5, -1.0, -1.0], [3.0, -1.0, -1.0]],
        [[0.0, -1.0, -0.5], [0.5, -0.5, -1.0], [-0.5, 1.5, -1.0], [-1.0, 3.0, -1.0]],
        [[0.0, -1.0, -0.5], [-1.0, -1.0, 1.5], [-1.0, -1.0, 2.0], [-1.0, -1.0, 3.0]],
    ])
    bad_stage_logits = good_stage_logits.flip(1)
    common = {"rank_logits": rank_logits, "in_set_logit": torch.tensor([2.0, -2.0, 0.0])}
    good = {
        **common, "bayesian_stage_logits": good_stage_logits,
        "bayesian_stage_probability": good_stage_logits.softmax(-1),
    }
    bad = {
        **common, "bayesian_stage_logits": bad_stage_logits,
        "bayesian_stage_probability": bad_stage_logits.softmax(-1),
    }
    loss = T2Loss(mode="progressive_bayesian_v7")

    good_loss, parts = loss(good, batch)
    bad_loss, _ = loss(bad, batch)

    assert good_loss < bad_loss
    assert {"t2_query_rf", "t2_global_type", "t2_final_type", "t2_monotonic"} <= set(parts)


def test_v8_append_state_matches_one_shot_full_candidate_state():
    torch.manual_seed(29)
    model_config = config()
    model_config.variant = "v8"
    model_config.stateful_prefix_sizes = (1, 2, 4)
    head = StatefulBayesianT2Head(model_config).eval()
    candidates, query = encoding(2, members=4), encoding(2)

    def candidate_slice(start, stop):
        return SignatureEncoding(
            candidates.global_shared[:, start:stop], candidates.local_shared[:, start:stop],
            candidates.valid_mask[:, start:stop], candidates.spatial[:, start:stop],
            candidates.global_sequence[:, start:stop], candidates.global_image[:, start:stop], {},
        )

    one_shot = head(candidates, query, torch.ones(2, 4, dtype=torch.bool))
    state = head.begin_state(query)
    state, first = head.append_state(state, candidate_slice(0, 2), torch.ones(2, 2, dtype=torch.bool))
    state, second = head.append_state(state, candidate_slice(2, 4), torch.ones(2, 2, dtype=torch.bool))

    assert state.set_mask.shape == (2, 4)
    assert len(state.posterior_history) == 3
    assert first["rank_logits"].shape == (2, 2)
    torch.testing.assert_close(second["type_probability"], one_shot["type_probability"], atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(second["rank_logits"], one_shot["rank_logits"], atol=4e-5, rtol=4e-5)


def test_v8_dynamic_prefix_loss_changes_target_when_source_arrives():
    batch = {
        "target_index": torch.tensor([2]),
        "episode_type_index": torch.tensor([0]),
        "set_mask": torch.ones(1, 4, dtype=torch.bool),
        "metadata": [{}],
    }
    stage_logits = torch.tensor([[[0.0, 0.0, 0.0], [-1.0, 2.0, -1.0],
                                  [-1.0, 2.0, -1.0], [3.0, -1.0, -1.0]]])
    good_prefix = torch.tensor([[[-1.0, 3.0, -1.0], [-1.0, 3.0, -1.0], [3.0, -1.0, -1.0]]])
    bad_prefix = good_prefix.flip(1)
    common = {
        "rank_logits": torch.tensor([[0.0, 0.0, 3.0, 0.0]]),
        "in_set_logit": torch.tensor([2.0]),
        "bayesian_stage_logits": stage_logits,
        "bayesian_stage_probability": stage_logits.softmax(-1),
        "stateful_prefix_sizes": torch.tensor([1, 2, 4]),
    }
    good = {**common, "stateful_prefix_logits": good_prefix,
            "stateful_prefix_probability": good_prefix.softmax(-1)}
    bad = {**common, "stateful_prefix_logits": bad_prefix,
           "stateful_prefix_probability": bad_prefix.softmax(-1)}
    loss = T2Loss(mode="stateful_bayesian_v8")

    good_loss, parts = loss(good, batch)
    bad_loss, _ = loss(bad, batch)

    assert good_loss < bad_loss
    assert {"t2_stateful", "t2_stateful_monotonic", "t2_source_arrival"} <= set(parts)


def test_v9_unifies_candidates_and_null_in_one_conditional_distribution():
    torch.manual_seed(31)
    model_config = config()
    model_config.variant = "v9"
    model_config.stateful_prefix_sizes = (1, 2, 4)
    head = UnifiedEvidenceT2Head(model_config).eval()
    output = head(
        encoding(2, members=4), encoding(2), torch.ones(2, 4, dtype=torch.bool),
    )

    assert output["conditional_logits"].shape == (2, 5)
    assert output["joint_probability"].shape == (2, 5)
    torch.testing.assert_close(output["conditional_probability"].sum(-1), torch.ones(2))
    torch.testing.assert_close(output["joint_probability"].sum(-1), torch.ones(2))
    torch.testing.assert_close(output["type_probability"].sum(-1), torch.ones(2))
    torch.testing.assert_close(
        output["exist_probability"], output["joint_probability"][:, :-1].sum(-1),
    )


def test_v9_append_state_matches_one_shot_and_preserves_old_candidate_scores():
    torch.manual_seed(37)
    model_config = config()
    model_config.variant = "v9"
    model_config.stateful_prefix_sizes = (1, 2, 4)
    head = UnifiedEvidenceT2Head(model_config).eval()
    candidates, query = encoding(1, members=4), encoding(1)

    def candidate_slice(start, stop):
        return SignatureEncoding(
            candidates.global_shared[:, start:stop], candidates.local_shared[:, start:stop],
            candidates.valid_mask[:, start:stop], candidates.spatial[:, start:stop],
            candidates.global_sequence[:, start:stop], candidates.global_image[:, start:stop], {},
        )

    one_shot = head(candidates, query, torch.ones(1, 4, dtype=torch.bool))
    state = head.begin_state(query)
    state, first = head.append_state(state, candidate_slice(0, 2), torch.ones(1, 2, dtype=torch.bool))
    state, second = head.append_state(state, candidate_slice(2, 4), torch.ones(1, 2, dtype=torch.bool))

    torch.testing.assert_close(second["candidate_logits"], one_shot["candidate_logits"], atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(second["type_probability"], one_shot["type_probability"], atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(first["candidate_logits"], second["candidate_logits"][:, :2], atol=4e-5, rtol=4e-5)


def test_v9_loss_rewards_candidate_null_and_rf_factorization():
    batch = {
        "target_index": torch.tensor([1, -1, -1]),
        "episode_type_index": torch.tensor([0, 1, 2]),
        "set_mask": torch.ones(3, 3, dtype=torch.bool),
        "metadata": [{}, {}, {}],
    }
    good_conditional = torch.tensor([
        [0.0, 3.0, 0.0, -1.0], [-1.0, -1.0, -1.0, 3.0], [0.0, 0.0, 0.0, 0.0],
    ])
    bad_conditional = torch.tensor([
        [0.0, -1.0, 0.0, 3.0], [0.0, 3.0, 0.0, -1.0], [0.0, 0.0, 0.0, 0.0],
    ])

    def make_output(conditional_logits, rf_logit):
        conditional_probability = conditional_logits.softmax(-1)
        rf_probability = torch.sigmoid(rf_logit)
        joint = torch.cat([
            (1 - rf_probability)[:, None] * conditional_probability[:, :-1],
            (rf_probability + (1 - rf_probability) * conditional_probability[:, -1])[:, None],
        ], dim=-1)
        prefix = conditional_logits[:, None].expand(-1, 2, -1)
        return {
            "training_phase": "unified", "rank_logits": conditional_logits[:, :-1],
            "candidate_logits": conditional_logits[:, :-1], "conditional_logits": conditional_logits,
            "rf_logit": rf_logit, "in_set_logit": torch.logsumexp(conditional_logits[:, :-1], -1) - conditional_logits[:, -1],
            "joint_probability": joint, "prefix_conditional_logits": prefix,
            "stateful_prefix_sizes": torch.tensor([1, 3]),
        }

    loss = T2Loss(mode="unified_evidence_v9")
    good_loss, parts = loss(make_output(good_conditional, torch.tensor([-3.0, -3.0, 3.0])), batch)
    bad_loss, _ = loss(make_output(bad_conditional, torch.tensor([3.0, 3.0, -3.0])), batch)

    assert good_loss < bad_loss
    assert {"t2_conditional", "t2_rf", "t2_prefix", "t2_brier"} <= set(parts)


def test_v10_combines_independent_rank_and_open_probabilities():
    torch.manual_seed(41)
    model_config = config()
    model_config.variant = "v10"
    model_config.stateful_prefix_sizes = (1, 2, 4)
    head = DualEvidenceT2Head(model_config).eval()
    output = head(
        encoding(2, members=4), encoding(2),
        encoding(2, members=4), encoding(2),
        torch.ones(2, 4, dtype=torch.bool),
    )

    torch.testing.assert_close(output["joint_probability"].sum(-1), torch.ones(2))
    assert output["pim_statistics"].shape[:2] == (2, 4)
    assert output["bayesian_evidence_increment"].shape == (2, 4, 3)
    torch.testing.assert_close(output["type_probability"].sum(-1), torch.ones(2))
    torch.testing.assert_close(
        output["joint_probability"][:, :-1],
        output["type_probability"][:, :1] * output["rank_probability"],
    )
    torch.testing.assert_close(
        output["stateful_prefix_probability"][:, -1], output["type_probability"],
        atol=1e-6, rtol=1e-6,
    )


def test_v10_training_phases_have_disjoint_trainable_parameters():
    model_config = config()
    model_config.variant = "v10"
    head = DualEvidenceT2Head(model_config)

    head.set_training_phase("rank")
    assert any(parameter.requires_grad for parameter in head.rank_relation.parameters())
    assert not any(parameter.requires_grad for parameter in head.open_relation.parameters())
    assert not any(parameter.requires_grad for parameter in head.rf_gate.parameters())

    head.set_training_phase("open")
    assert not any(parameter.requires_grad for parameter in head.rank_relation.parameters())
    assert any(parameter.requires_grad for parameter in head.open_relation.parameters())
    assert any(parameter.requires_grad for parameter in head.rf_gate.parameters())

    head.set_training_phase("calibration")
    trainable = {name for name, parameter in head.named_parameters() if parameter.requires_grad}
    assert trainable == {"rank_log_temperature", "open_log_temperature", "query_present_logit"}


def test_masked_batch_norm_ignores_padding_values():
    torch.manual_seed(4)
    first = MaskedBatchNorm1d(3).train()
    second = MaskedBatchNorm1d(3).train()
    second.load_state_dict(first.state_dict())
    valid = torch.randn(2, 3, 5)
    short_mask = torch.ones(2, 5, dtype=torch.bool)
    padded = torch.cat([valid, torch.full((2, 3, 4), 1e4)], dim=-1)
    padded_mask = torch.cat([short_mask, torch.zeros(2, 4, dtype=torch.bool)], dim=-1)
    output_first = first(valid, short_mask)
    output_second = second(padded, padded_mask)[..., :5]
    torch.testing.assert_close(output_first, output_second)
    torch.testing.assert_close(first.running_mean, second.running_mean)
    torch.testing.assert_close(first.running_var, second.running_var)


def test_pcgrad_projection_matches_conflict_rule():
    first = torch.tensor([1.0, 0.0])
    aligned = torch.tensor([0.5, 0.5])
    opposed = torch.tensor([-1.0, 1.0])

    aligned_result, aligned_cosine = project_conflicting([first], [aligned])
    opposed_result, opposed_cosine = project_conflicting([first], [opposed])

    torch.testing.assert_close(aligned_result[0], first + aligned)
    dot = (first * opposed).sum()
    expected = first - dot / opposed.square().sum() * opposed + opposed - dot / first.square().sum() * first
    torch.testing.assert_close(opposed_result[0], expected)
    assert aligned_cosine > 0
    assert opposed_cosine < 0


def test_pcgrad_projection_stays_finite_when_float32_squares_would_overflow():
    first = torch.tensor([1.0e30, 0.0])
    opposed = torch.tensor([-1.0e30, 1.0e30])

    projected, cosine = project_conflicting([first], [opposed])

    assert torch.isfinite(projected[0]).all()
    assert torch.isfinite(cosine)
    assert cosine.item() == pytest.approx(-2 ** -0.5)


def test_pcgrad_projection_rejects_non_finite_task_gradients():
    with pytest.raises(FloatingPointError, match="non-finite task gradients"):
        project_conflicting([torch.tensor([float("inf")])], [torch.ones(1)])


def test_dtw_pair_distance_is_symmetric_and_cached():
    class Store:
        def __init__(self):
            base = np.arange(6, dtype=np.float64)
            self.rows = {
                "a": np.stack([base, base, base, base + 1, base + 2, base, np.ones(6)], axis=1),
                "b": np.stack([base, base + 1, base, base + 2, base + 1, base, np.ones(6)], axis=1),
            }

        def load_csv(self, sample_id):
            return self.rows[sample_id]

    baseline = DTWBaseline(Store())
    first = baseline.score_pair("a", "b")
    second = baseline.score_pair("b", "a")

    assert first == second
    assert len(baseline.pair_cache) == 1


def test_vectorized_dtw_matches_reference_recurrence():
    first = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    second = np.asarray([[0.0], [1.5], [3.0]])
    costs = np.sqrt(np.square(first[:, None, :] - second[None, :, :]).sum(axis=-1))
    n, m = costs.shape
    window = max(abs(n - m), int(max(n, m) * 0.5))
    reference = np.full((n + 1, m + 1), np.inf)
    reference[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            reference[i, j] = costs[i - 1, j - 1] + min(
                reference[i, j - 1], reference[i - 1, j], reference[i - 1, j - 1],
            )

    assert dtw_distance(first, second, window_fraction=0.5) == reference[n, m] / (n + m)


def test_temporal_resample_ignores_padding_and_preserves_valid_extent():
    valid = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    padded = torch.cat([valid, torch.full((1, 5, 3), 1e4)], dim=1)
    mask = torch.tensor([[True] * 4 + [False] * 5])

    output, output_mask = temporal_resample(padded, mask, tokens=3)

    assert output_mask.tolist() == [[True, True, True]]
    torch.testing.assert_close(output[:, 0], valid[:, 0])
    torch.testing.assert_close(output[:, -1], valid[:, -1])
    assert output.max() < 1e4


def test_raw_multiscale_sequence_encoder_accepts_five_primitives():
    model_config = ModelConfig(
        feature_dim=5, hidden_dim=32, conformer_layers=1, conformer_heads=4,
        conformer_ffn=64, conformer_kernel=15, sequence_tokens=8,
        sequence_stem="raw_multiscale", sequence_model_tokens=16,
    )
    encoder = SequenceEncoder(model_config).eval()
    sequence = torch.rand(2, 80, 5)
    mask = torch.ones(2, 80, dtype=torch.bool)
    mask[1, 55:] = False

    global_token, local, local_mask, model_mask = encoder(sequence, mask)

    assert global_token.shape == (2, 32)
    assert local.shape == (2, 8, 32)
    assert local_mask.all()
    assert model_mask.shape == (2, 16)
    assert torch.isfinite(global_token).all()


def test_factorized_t2_loss_does_not_use_absolute_rank_logit_margin():
    batch = {
        "target_index": torch.tensor([1, -1]),
        "exist_label": torch.tensor([1.0, 0.0]),
    }
    output = {
        "rank_logits": torch.tensor([[0.1, 0.8, -0.2], [0.5, 0.4, 0.3]]),
        "exist_logit": torch.tensor([1.0, -1.0]),
        "view_global_loss": torch.tensor(0.0),
        "view_local_loss": torch.tensor(0.0),
    }
    shifted = {**output, "rank_logits": output["rank_logits"] + 100}
    loss = T2Loss(mode="factorized_v2")

    first, _ = loss(output, batch)
    second, _ = loss(shifted, batch)

    torch.testing.assert_close(first, second)


def test_v3_t2_loss_supervises_absolute_candidate_evidence():
    batch = {
        "target_index": torch.tensor([1, -1]),
        "exist_label": torch.tensor([1.0, 0.0]),
        "set_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    common = {
        "rank_logits": torch.tensor([[0.1, 0.8, -0.2], [0.5, 0.4, 0.3]]),
        "exist_logit": torch.tensor([1.0, -1.0]),
        "view_global_loss": torch.tensor(0.0),
        "view_local_loss": torch.tensor(0.0),
    }
    good = {**common, "match_logits": torch.tensor([[-1.0, 2.0, -1.0], [-2.0, -2.0, -2.0]])}
    collapsed = {**common, "match_logits": torch.tensor([[-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]])}
    loss = T2Loss(mode="hierarchical_v3")

    good_loss, _ = loss(good, batch)
    collapsed_loss, _ = loss(collapsed, batch)

    assert good_loss < collapsed_loss


def test_v4_open_set_loss_prefers_correct_candidate_and_unknown_competition():
    batch = {
        "target_index": torch.tensor([1, -1]),
        "exist_label": torch.tensor([1.0, 0.0]),
        "set_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    common = {
        "rank_logits": torch.tensor([[0.1, 2.0, -0.2], [0.5, 0.4, 0.3]]),
        "match_logits": torch.tensor([[-1.0, 2.0, -1.0], [-2.0, -2.0, -2.0]]),
        "exist_logit": torch.tensor([2.0, -2.0]),
        "view_global_loss": torch.tensor(0.0),
        "view_local_loss": torch.tensor(0.0),
    }
    good = {
        **common,
        "unknown_logit": torch.tensor([-1.0, 2.0]),
        "open_set_logits": torch.tensor([[0.1, 2.0, -0.2, -1.0], [-2.0, -2.0, -2.0, 2.0]]),
    }
    collapsed = {
        **common,
        "unknown_logit": torch.tensor([2.0, -1.0]),
        "open_set_logits": torch.tensor([[-2.0, -2.0, -2.0, 2.0], [0.1, 2.0, -0.2, -1.0]]),
    }
    loss = T2Loss(mode="open_set_v4")

    good_loss, _ = loss(good, batch)
    collapsed_loss, _ = loss(collapsed, batch)

    assert good_loss < collapsed_loss


def test_v5_loss_prefers_hard_candidate_and_correct_unknown_subtype():
    batch = {
        "target_index": torch.tensor([1, -1, -1]),
        "exist_label": torch.tensor([1.0, 0.0, 0.0]),
        "episode_type_index": torch.tensor([0, 1, 2]),
        "set_mask": torch.ones(3, 3, dtype=torch.bool),
    }
    common = {
        "match_logits": torch.tensor([[-1.0, 2.0, 0.5], [-2.0, -2.0, -2.0], [-2.0, -2.0, -2.0]]),
        "view_global_loss": torch.tensor(0.0),
        "view_local_loss": torch.tensor(0.0),
    }
    good = {
        **common,
        "rank_logits": torch.tensor([[0.0, 3.0, 1.0], [0.5, 0.4, 0.3], [0.2, 0.1, 0.0]]),
        "type_logits": torch.tensor([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        "exist_logit": torch.tensor([3.0, -3.0, -3.0]),
        "unknown_logit": torch.tensor([0.0, 3.0, 3.0]),
        "open_set_logits": torch.tensor([
            [0.0, 3.0, 1.0, 0.0], [-2.0, -2.0, -2.0, 3.0], [-2.0, -2.0, -2.0, 3.0],
        ]),
    }
    collapsed = {
        **common,
        "rank_logits": torch.tensor([[3.0, 0.0, 1.0], [0.5, 0.4, 0.3], [0.2, 0.1, 0.0]]),
        "type_logits": torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [0.0, 3.0, 0.0]]),
        "exist_logit": torch.tensor([-3.0, -3.0, -3.0]),
        "unknown_logit": torch.tensor([3.0, 3.0, 3.0]),
        "open_set_logits": torch.tensor([
            [3.0, 0.0, 1.0, 3.0], [-2.0, -2.0, -2.0, 3.0], [-2.0, -2.0, -2.0, 3.0],
        ]),
    }
    loss = T2Loss(mode="open_set_v5")

    good_loss, good_parts = loss(good, batch)
    collapsed_loss, _ = loss(collapsed, batch)

    assert good_loss < collapsed_loss
    assert {"t2_subtype", "t2_hard_margin", "t2_all_negative"} <= set(good_parts)
    assert good_parts["t2_all_negative"] >= 0
