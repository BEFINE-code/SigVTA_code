# TRACE-Net code release

This folder contains the code, experiment configurations, and tests needed to inspect and rerun TRACE-Net. It intentionally excludes datasets, caches, checkpoints, predictions, logs, and generated results.

## Included

- `dvsrc/`: model, data processing, training, evaluation, and command-line interface.
- `configs/`: SigVTA T1 experiment configurations.
- `ablation/`: T1, T1 SF-hack, and T2 ablation runners and configurations.
- `baselines/shared/repro_baselines_snapshot/`: external baseline implementations.
- `tests/`: code-level regression tests.

Datasets and trained weights must be prepared separately. Several launchers expect the local dataset and protocol directories used by the original experiments.
