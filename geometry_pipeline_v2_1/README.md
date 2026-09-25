# Geometry Pipeline v2

Dataset-agnostic orchestration of the validated Schatten geometry-response pipeline.

## Design

The numerical chain is unchanged: density operators -> pairwise singular spectra -> Schatten distances -> RBF/SVM nested CV -> OOF decision space -> perturbative XAI -> cross-level summary. Dataset-specific information lives in YAML configuration.

## Configurable quantities

- positive/negative class labels and metadata column
- matrix glob/format, node count, diagonal and symmetrization conventions
- Schatten p grid and reference p
- repeated nested-CV design
- spectral resampling and XAI plotting/validation options

`configs/ad.yaml` is the regression-oracle configuration and must reproduce the validated AD fingerprint from v1.1. `configs/template_binary_dataset.yaml` is the starting point for new binary matrix-valued datasets.

## Run

```bash
python3 run_pipeline.py --config configs/ad.yaml
```

Selected stages:

```bash
python3 run_pipeline.py --config configs/ad.yaml --stages spectral decision xai summary
```

Reuse is enabled by default. Use `--no-reuse` only when intentional recomputation is required.

## Validation policy

Before using a new dataset, first run AD and compare `05_summary/pipeline_summary.csv` with the validated v1.1 fingerprint. The intended invariant quantities are AUC(p), gamma*(p), Q(p), leading-mode share, C_f(p), and C_I(p).

## Important scope

V2 supports generic **binary matrix-valued datasets**. The perturbative XAI assumes that removing/suppressing a matrix node is scientifically meaningful. For a new domain this semantic assumption must be checked rather than applied automatically.

## v2.1 performance mode

The scientific formulas and cache layout are unchanged. The expensive XAI pairwise-node contribution stage can now parallelize independent subject pairs using shared-memory threads. By default (`parallel_backend: auto`, `workers: 0`) a short calibration selects the fastest option on the current machine while avoiding nested BLAS/LAPACK oversubscription.

To benchmark the optimized kernel on an existing AD run, keep the classifier outputs, remove only `04_xai/pairwise_node_contributions.dat` and its metadata file, then rerun `xai summary`. Do not delete the classifier spectra/checkpoints.

For a forced serial regression run set `parallel_backend: serial`; for a fixed thread count set `parallel_backend: threads` and e.g. `workers: 4`.

