import itertools
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from dvsrc.config import ExperimentConfig, ModelConfig
from dvsrc.calibration import Calibrator
from dvsrc.losses import T1Loss
from dvsrc.model import DVSRNet, GlobalPairRelation, SignatureEncoding, SortedLogitSetHead, T1Head
from dvsrc.task_package import export_task_package, load_task_package
from dvsrc.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]


def simple_config(**changes):
    args = dict(
        variant="v5r1", t1_variant="simple_v1", pretrained=False,
        hidden_dim=32, conformer_layers=1, conformer_heads=4,
        conformer_ffn=64, sequence_tokens=8, local_tokens=4, dropout=0.0,
        use_qrsa=False, use_t1_residual_verification=False,
        use_t1_evidence_anchor=False,
    )
    args.update(changes)
    return ModelConfig(**args)


def encoding(members=None):
    shape = (2, 32) if members is None else (2, members, 32)
    z = torch.randn(shape, requires_grad=True)
    local_shape = (*shape[:-1], 8, 32)
    # NaNs expose accidental consumption of local pair features by the new head.
    return SignatureEncoding(
        global_shared=z, local_shared=torch.full(local_shape, float("nan")),
        valid_mask=torch.ones(local_shape[:-1], dtype=torch.bool),
        spatial=torch.zeros(*local_shape[:-1], 2),
        global_sequence=z, global_image=z, diagnostics={},
    )


def test_simple_config_and_no_legacy_modules():
    config = ExperimentConfig.from_yaml(ROOT / "models/simple_v1/config.yaml")
    assert config.model.t1_variant == "simple_v1"
    assert config.train.seed == 42
    assert not config.train.stage_a_finalize_test
    assert not config.train.stage_a_evaluate_test_each_epoch
    assert config.train.stage_a_selection_policy == "dual_t1"
    head = T1Head(config.model)
    assert isinstance(head.relation, GlobalPairRelation)
    assert head.relation.mlp[0].in_features == 512
    assert sum(p.numel() for p in head.five_to_one.parameters()) == 113
    assert not any(hasattr(head, attr) for attr in (
        "qrsa", "local_adapter", "one_to_one", "residual_one_to_one", "evidence_anchored_set",
    ))


def test_global_pair_is_symmetric():
    torch.manual_seed(42)
    relation = GlobalPairRelation(32, 0).eval()
    a, b = torch.randn(2, 32), torch.randn(2, 32)
    ab, ba = relation(a, b), relation(b, a)
    for left, right in zip(ab, ba):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_five_reference_head_is_permutation_invariant_and_learned():
    torch.manual_seed(42)
    head = SortedLogitSetHead().eval()
    logits = torch.tensor([[-2., -1., 0., 1., 2.], [-3., -1., 0., 1., 3.]])
    mask = torch.ones_like(logits, dtype=torch.bool)
    expected = head(logits, mask)
    torch.testing.assert_close(expected, head.mlp(logits.sort(dim=1).values).squeeze(-1))
    assert not torch.isclose(expected[0], expected[1])  # Equal means, different distributions.
    for permutation in itertools.permutations(range(5)):
        torch.testing.assert_close(head(logits[:, permutation], mask), expected, rtol=0, atol=0)


@pytest.mark.parametrize("count", [0, 2, 4, 6])
def test_five_reference_head_rejects_wrong_count(count):
    with pytest.raises(ValueError, match="exactly five"):
        SortedLogitSetHead()(torch.zeros(2, count), torch.ones(2, count, dtype=torch.bool))


@pytest.mark.parametrize("count", [1, 5])
def test_simple_head_outputs_and_backward(count):
    torch.manual_seed(42)
    config = simple_config()
    head = T1Head(config, config.t1_variant).train()
    reference, query = encoding(count), encoding()
    mask = torch.ones(2, count, dtype=torch.bool)
    result = head(reference, query, mask)
    assert set(result) == {"pair_embedding", "pair_logits", "case_logit", "identity_embedding"}
    assert result["pair_logits"].shape == (2, count)
    assert result["case_logit"].shape == (2,)
    if count == 1:
        torch.testing.assert_close(result["case_logit"], result["pair_logits"][:, 0])
    result["case_logit"].sum().backward()
    for tensor in (query.global_shared, reference.global_shared):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert all(p.grad is not None for p in head.relation.parameters())
    if count == 1:
        assert all(p.grad is None for p in head.five_to_one.parameters())
    else:
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in head.five_to_one.parameters())


def test_simple_head_rejects_invalid_references_and_shared_local_matcher():
    config = simple_config()
    head = T1Head(config, config.t1_variant)
    refs, query = encoding(5), encoding()
    mask = torch.ones(2, 5, dtype=torch.bool)
    bad = mask.clone()
    bad[0, 2] = False
    with pytest.raises(ValueError, match="every reference"):
        head(refs, query, bad)
    with pytest.raises(ValueError, match="boolean"):
        head(refs, query, mask.float())
    with pytest.raises(ValueError, match="global-only"):
        head(refs, query, mask, shared_relation=torch.nn.Identity())
    with pytest.raises(ValueError, match="exactly one or five"):
        head(encoding(3), query, torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="shared local matcher"):
        DVSRNet(simple_config(variant="unified_sen_v20"))


def batch(count):
    return {
        "protocol": f"t1_{count}v1" if count == 5 else "t1_1v1",
        "sequence": torch.randn(count + 1, 48, 10),
        "sequence_mask": torch.ones(count + 1, 48, dtype=torch.bool),
        "image": torch.randn(count + 1, 3, 32, 32),
        "anchors": torch.zeros(count + 1, 8, 6),
        "anchor_mask": torch.ones(count + 1, 8, dtype=torch.bool),
        "query_index": torch.tensor([count]),
        "set_index": torch.arange(count)[None],
        "set_mask": torch.ones(1, count, dtype=torch.bool),
        "label": torch.ones(1), "metadata": [{"target_writer_id": "writer"}],
    }


@pytest.mark.parametrize("count", [1, 5])
def test_simple_training_loss_through_multimodal_encoder(count):
    torch.manual_seed(42)
    model = DVSRNet(simple_config()).train()
    data = batch(count)
    result = model(data)
    loss, parts = T1Loss()(result, data)
    assert torch.isfinite(loss)
    for key in ("t1_subset", "t1_quality", "t1_set_aux", "t1_anchor_alignment"):
        assert parts[key].item() == 0
    loss.backward()
    for module in (model.encoder.sequence, model.encoder.image, model.encoder.fusion):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)


def test_simple_checkpoint_export_and_reload(tmp_path):
    torch.manual_seed(42)
    config = simple_config(fusion="sequence_only")
    model = DVSRNet(config).eval()
    checkpoint, package = tmp_path / "best_t1.pt", tmp_path / "t1.pt"
    torch.save({"model": model.state_dict(), "config": ExperimentConfig(model=config).to_dict()}, checkpoint)
    export_task_package(checkpoint, "t1", package)
    restored, _ = load_task_package(package)
    assert restored.config.t1_variant == "simple_v1"
    with torch.no_grad():
        for count in (1, 5):
            data = batch(count)
            torch.testing.assert_close(restored(data)["case_logit"], model(data)["case_logit"])


@pytest.mark.parametrize("variant", ["simple_v1", "v5r1"])
def test_prediction_and_calibration_support_optional_reliability(variant):
    config = simple_config(fusion="sequence_only", t1_variant=variant)
    trainer = Trainer.__new__(Trainer)
    trainer.model = DVSRNet(config).eval()
    trainer.device = torch.device("cpu")
    trainer._autocast = nullcontext
    calibrator = Calibrator(t1_1v1_temperature=1.5, t1_5v1_temperature=2.0)
    for count in (1, 5):
        data = batch(count)
        prediction = trainer.predict([data], calibrator=calibrator)[0]
        with torch.no_grad():
            logit = trainer.model(data)["case_logit"]
            expected = calibrator.calibrate_t1(logit, count).item()
        assert prediction["score"] == pytest.approx(expected)
        assert len(prediction["pair_scores"]) == count
        assert ("reference_reliability" in prediction) == (variant != "simple_v1")
