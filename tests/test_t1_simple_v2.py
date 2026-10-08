import itertools
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from dvsrc.calibration import Calibrator
from dvsrc.config import ExperimentConfig
from dvsrc.losses import T1Loss, supervised_contrastive
from dvsrc.metrics import binary_roc, binary_summary, roc_auc, t1_metrics
from dvsrc.model import DVSRNet, StabilizedGlobalPairRelation
from dvsrc.task_package import export_task_package, load_task_package
from dvsrc.trainer import Trainer
from test_t1_simple import simple_config, batch


def config(**changes):
    return simple_config(t1_variant="simple_v2", t1_sequence_residual=True,
                         t1_pair_stabilize=True, **changes)


def test_roc_ties_match_threshold_counts_and_are_order_invariant():
    y, s = np.array([1, 0, 1, 0]), np.array([0.5] * 4)
    assert roc_auc(y, s) == 0.5
    assert binary_summary(y, s)["eer"] == 0.5
    for perm in itertools.permutations(range(4)):
        assert roc_auc(y[list(perm)], s) == 0.5
    y, s = np.array([0, 1, 1, 0, 1]), np.array([0.8, 0.8, 0.2, 0.2, 0.1])
    fpr, tpr, thresholds = binary_roc(y, s)
    for fp, tp, threshold in zip(fpr, tpr, thresholds):
        assert fp == pytest.approx((s[y == 0] >= threshold).mean())
        assert tp == pytest.approx((s[y == 1] >= threshold).mean())


def test_saturated_probabilities_do_not_destroy_ranking():
    rows = [{"label": y, "score": 1.0, "raw_logit": s, "query_state": "NW",
             "attack_type": "genuine" if y else "SF"}
            for y, s in [(0, 20.), (0, 21.), (1, 22.), (1, 23.)]]
    metrics = t1_metrics(rows)
    assert metrics["overall"]["roc_auc"] == 1
    assert metrics["overall"]["eer"] == 0
    assert metrics["overall"]["accuracy"] == 0.5
    assert metrics["genuine_vs_SF"]["roc_auc"] == 1
    assert metrics["conditions"]["unknown"]["roc_auc"] == 1
    rows[0].pop("raw_logit")
    with pytest.raises(ValueError, match="mix"):
        t1_metrics(rows)


def test_fp32_prediction_and_raw_logit_calibration(tmp_path, monkeypatch):
    class FakeModel:
        def eval(self):
            pass
        def __call__(self, data):
            return {"case_logit": torch.tensor([8., 9.], dtype=torch.bfloat16),
                    "pair_logits": torch.tensor([[8.], [9.]], dtype=torch.bfloat16)}
    trainer = Trainer.__new__(Trainer)
    trainer.model, trainer.device, trainer._autocast = FakeModel(), torch.device("cpu"), nullcontext
    data = {"protocol": "t1_1v1", "set_index": torch.zeros(2, 1, dtype=torch.long),
            "set_mask": torch.ones(2, 1, dtype=torch.bool), "metadata": [{}, {}]}
    rows = trainer.predict([data])
    assert rows[0]["score"] < rows[1]["score"] < 1
    assert rows[0]["raw_logit"] == 8
    assert Calibrator().calibrate_t1(torch.tensor([8.], dtype=torch.bfloat16), 1).dtype == torch.float32
    trainer.output = tmp_path
    trainer.loader = lambda *args: []
    trainer.predict = lambda loader: [{"score": 1., "raw_logit": 100., "label": 0},
                                      {"score": 1., "raw_logit": 200., "label": 1}]
    seen = []
    def temperature(logits, labels):
        seen.append(logits.clone())
        return 20.
    monkeypatch.setattr("dvsrc.trainer.fit_binary_temperature", temperature)
    trainer.fit_t1_calibration()
    assert len(seen) == 2
    for logits in seen:
        torch.testing.assert_close(logits, torch.tensor([100., 200.]))


@pytest.mark.parametrize("mode", ["diff_product", "diff_only", "product_only"])
def test_masked_pair_features_equal_budget_and_symmetry(mode):
    torch.manual_seed(42)
    relation = StabilizedGlobalPairRelation(32, 0, mode, True).eval()
    base = StabilizedGlobalPairRelation(32, 0, "diff_product", True)
    assert sum(p.numel() for p in base.parameters()) == sum(p.numel() for p in relation.parameters())
    assert relation.mlp[0].in_features == 64
    assert relation.product_mix_logit.sigmoid().item() == pytest.approx(.1)
    a, b = torch.randn(2, 32, requires_grad=True), torch.randn(2, 32, requires_grad=True)
    features = []
    hook = relation.mlp[0].register_forward_pre_hook(lambda m, args: features.append(args[0]))
    _, logit = relation(a, b)
    torch.testing.assert_close(logit, relation(b, a)[1])
    hook.remove()
    if mode == "diff_only":
        assert features[0][:, 32:].count_nonzero() == 0
    elif mode == "product_only":
        assert features[0][:, :32].count_nonzero() == 0
    logit.sum().backward()
    assert a.grad.abs().sum() > 0
    if mode != "diff_only":
        assert relation.product_mix_logit.grad.abs() > 0


def test_corrected_control_reproduces_v1_forward_with_same_weights():
    torch.manual_seed(42)
    v1 = DVSRNet(simple_config()).eval()
    cfg = config()
    cfg.t1_sequence_residual = cfg.t1_pair_stabilize = False
    torch.manual_seed(42)
    v2 = DVSRNet(cfg).eval()
    for name, tensor in v1.state_dict().items():
        torch.testing.assert_close(v2.state_dict()[name], tensor)
    with torch.no_grad():
        for n in (1, 5):
            data = batch(n)
            torch.testing.assert_close(v1(data)["case_logit"], v2(data)["case_logit"])


@pytest.mark.parametrize("fusion", ["ms_caf", "sequence_only", "image_only", "late"])
def test_encoder_residual_paths_and_training(fusion):
    torch.manual_seed(42)
    model = DVSRNet(config(fusion=fusion)).train()
    assert model.encoder.fusion_mix_logit.sigmoid().item() == pytest.approx(.1)
    result = model(batch(5))
    loss, _ = T1Loss(True)(result, batch(5))
    assert torch.isfinite(loss)
    loss.backward()
    for module, active in ((model.encoder.sequence, fusion != "image_only"),
                           (model.encoder.image, fusion != "sequence_only")):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert bool(grads) == active
        assert all(torch.isfinite(g).all() for g in grads)
    if fusion in ("ms_caf", "late"):
        assert model.encoder.fusion_mix_logit.grad.abs() > 0


def test_genuine_identity_loss_excludes_forgery_gradients():
    z = torch.randn(5, 32, requires_grad=True)
    out = {"identity_embedding": z, "pair_logits": torch.zeros(5, 1),
           "case_logit": torch.zeros(5), "view_global_loss": torch.tensor(0.),
           "view_local_loss": torch.tensor(0.)}
    data = {"label": torch.tensor([1., 1., 1., 0., 0.]),
            "metadata": [{"target_writer_id": w} for w in ("A", "A", "B", "A", "B")],
            "set_mask": torch.ones(5, 1, dtype=torch.bool)}
    _, parts = T1Loss(True)(out, data)
    torch.testing.assert_close(parts["t1_metric"], supervised_contrastive(z[:3], ["A", "A", "B"]))
    parts["t1_metric"].backward()
    assert z.grad[3:].count_nonzero() == 0
    data["label"].zero_()
    assert T1Loss(True)(out, data)[1]["t1_metric"] == 0


def test_v2_package_roundtrip_and_direct_1v1(tmp_path):
    cfg = config(fusion="sequence_only")
    model = DVSRNet(cfg).eval()
    checkpoint, package = tmp_path / "best.pt", tmp_path / "package.pt"
    torch.save({"model": model.state_dict(), "config": ExperimentConfig(model=cfg).to_dict()}, checkpoint)
    export_task_package(checkpoint, "t1", package)
    restored, _ = load_task_package(package)
    with torch.no_grad():
        data = batch(1)
        result = model(data)
        torch.testing.assert_close(result["case_logit"], result["pair_logits"][:, 0])
        torch.testing.assert_close(result["case_logit"], restored(data)["case_logit"])
    assert sum(p.numel() for p in model.t1.five_to_one.parameters()) == 113


def test_training_step_updates_model_and_records_gates():
    trainer = Trainer.__new__(Trainer)
    trainer.config = ExperimentConfig(model=config())
    trainer.config.train.stage_a_grad_accumulation = 2
    trainer.model = DVSRNet(trainer.config.model)
    trainer.device, trainer._autocast = torch.device("cpu"), nullcontext
    trainer.t1_loss = T1Loss(True)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda step: 1)
    class Loader(list):
        batch_sampler = object()
    old = trainer.model.t1.relation.mlp[0].weight.detach().clone()
    stats = trainer._stage_a_epoch({"t1_1v1": Loader([batch(1)]),
                                   "t1_5v1": Loader([batch(5)])}, scheduler, 0)
    assert np.isfinite(stats["t1_total"])
    assert 0 < stats["t1_fusion_alpha"] < 1
    assert 0 < stats["t1_product_beta"] < 1
    assert not torch.equal(old, trainer.model.t1.relation.mlp[0].weight)
