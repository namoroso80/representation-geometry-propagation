# Changelog

## v2.1-performance
- Preserves the v2 scientific pipeline and cache format.
- Optimizes the pairwise XAI node-sensitivity kernel only.
- Adds thread-level pair parallelism with shared read-only density matrices.
- Prevents nested BLAS oversubscription with `threadpoolctl`.
- Adds hardware autotuning among serial / 2 / 4 / 6 / 8 workers (bounded by CPU count).
- Uses safe `overwrite_a=True` for the disposable pair-difference matrix.
- Adds XAI throughput reporting in pairs/s.

- Dataset labels are configuration-driven; no AD/NC requirement in the numerical engines.
- Schatten p grid and reference p are configuration-driven.
- Outer/inner CV design and random seed are configuration-driven.
- Optional C and absolute-gamma grids can be supplied in YAML; omission preserves the validated AD grids.
- Spectral, decision and XAI stages consume the same configured p grid.
- XAI class-specific outputs use configured class names.
- Exact p=2 XAI validation is performed when p=2 is present and skipped otherwise.
- Added a generic binary-dataset template and a fingerprint comparison utility.
- AD configuration remains the regression oracle; mathematical definitions are unchanged.