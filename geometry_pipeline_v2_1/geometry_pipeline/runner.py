from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from .config import PipelineConfig


class PipelineRunner:
    def __init__(self, cfg: PipelineConfig, project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root).resolve()
        self.engines = self.project_root / "engines"
        self.cfg.output_root.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: list[str], log_name: str) -> None:
        log_dir = self.cfg.output_root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_name
        print("\n[RUN]", " ".join(cmd))
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n\n$ " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                log.write(line)
            rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Stage failed with exit code {rc}. See {log_path}")

    @staticmethod
    def _flag(cmd: list[str], condition: bool, flag: str) -> None:
        if condition:
            cmd.append(flag)

    def _representation(self) -> str:
        return str(self.cfg.dataset.get("representation", "dense")).lower()

    def _prepare_lowrank_spectral_cache(self) -> None:
        """Expose exact low-rank pair spectra in the standard classifier cache schema."""
        c = self.cfg
        if self._representation() != "lowrank":
            return
        p = c.paths
        source = Path(p["geometry_dir"]).expanduser()
        out = c.stage_dir("classifier")
        out.mkdir(parents=True, exist_ok=True)
        target_sv = out / "pairwise_singular_values.npy"
        target_pairs = out / "pair_indices.npz"
        if target_sv.exists() and target_pairs.exists():
            return
        cmd = [sys.executable, str(self.engines / "prepare_lowrank_spectral_cache.py"),
               "--geometry-dir", str(source), "--output-dir", str(out)]
        self._run(cmd, "00_lowrank_spectral_cache.log")

    def classifier(self, reuse: bool = True) -> None:
        c = self.cfg
        ds, p = c.dataset, c.paths
        out = c.stage_dir("classifier")
        out.mkdir(parents=True, exist_ok=True)

        if self._representation() == "lowrank":
            cmd = [sys.executable, str(self.engines / "classifier_engine_precomputed.py"),
                   "--info-file", str(Path(p["info_file"]).expanduser()),
                   "--precomputed-distance-dir", str(Path(p["geometry_dir"]).expanduser()),
                   "--output-dir", str(out),
                   "--label-col", str(ds["label_column"]),
                   "--positive-label", str(ds["positive_label"]),
                   "--negative-label", str(ds["negative_label"]),
                   "--p-values", *[str(x) for x in c.science.get("p_values", [1,1.25,1.5,1.75,2,3,4,8,16])],
                   "--n-repeats", str(c.cv.get("n_repeats",20)),
                   "--n-splits", str(c.cv.get("n_splits",5)),
                   "--inner-splits", str(c.cv.get("inner_splits",5)),
                   "--random-state", str(c.cv.get("random_state",42))]
            if ds.get("id_column") is not None:
                cmd += ["--id-col", str(ds["id_column"])]
            if not reuse:
                cmd.append("--no-skip-existing")
            self._run(cmd, "01_classifier.log")
            self._prepare_lowrank_spectral_cache()
            return

        cmd = [sys.executable, str(self.engines / "classifier_engine.py"),
               "--data-dir", str(Path(p["data_dir"]).expanduser()),
               "--info-file", str(Path(p["info_file"]).expanduser()),
               "--output-dir", str(out),
               "--matrix-glob", str(ds.get("matrix_glob", "adjacency_*.csv")),
               "--matrix-format", str(ds.get("matrix_format", "auto")),
               "--label-col", str(ds["label_column"]),
               "--positive-label", str(ds["positive_label"]),
               "--negative-label", str(ds["negative_label"]),
               "--p-values", *[str(x) for x in c.science.get("p_values", [1,1.25,1.5,1.75,2,3,4,8,16])],
               "--n-repeats", str(c.cv.get("n_repeats",20)),
               "--n-splits", str(c.cv.get("n_splits",5)),
               "--inner-splits", str(c.cv.get("inner_splits",5)),
               "--random-state", str(c.cv.get("random_state",42))]
        model = c.raw.get("model", {})
        if model.get("C_values") is not None:
            cmd += ["--c-values", *[str(x) for x in model["C_values"]]]
        if model.get("gamma_values") is not None:
            cmd += ["--gamma-values", *[str(x) for x in model["gamma_values"]]]
        if ds.get("id_column") is not None:
            cmd += ["--id-col", str(ds["id_column"])]
        if ds.get("delimiter") is not None:
            cmd += ["--delimiter", str(ds["delimiter"])]
        if ds.get("expected_nodes") is not None:
            cmd += ["--expected-nodes", str(ds["expected_nodes"])]
        if ds.get("diagonal", "unit") == "keep":
            cmd += ["--diagonal", "keep"]
        if not bool(ds.get("symmetrize", True)):
            cmd.append("--no-symmetrize")
        cmd += ["--psd-tol", str(ds.get("psd_tolerance", 1e-8))]
        cmd.append("--cache-spectra")
        if reuse and (out / "pairwise_singular_values.npy").exists():
            cmd.append("--reuse-spectra")
        if not reuse:
            cmd.append("--no-skip-existing")
        self._run(cmd, "01_classifier.log")

    def spectral(self) -> None:
        c = self.cfg
        self._prepare_lowrank_spectral_cache()
        source = c.stage_dir("classifier")
        out = c.stage_dir("spectral")
        out.mkdir(parents=True, exist_ok=True)
        sopts = c.raw.get("spectral", {})
        cmd = [sys.executable, str(self.engines / "spectral_engine.py"),
               "--spectra-source", str(source), "--output-dir", str(out),
               "--n-resamples", str(sopts.get("n_resamples", 300)),
               "--subject-fraction", str(sopts.get("subject_fraction", 0.8)),
               "--random-state", str(sopts.get("random_state", 42)),
               "--p-values", *[str(x) for x in c.science.get("p_values", [1,1.25,1.5,1.75,2,3,4,8,16])]]
        self._run(cmd, "02_spectral.log")

    def decision(self) -> None:
        c = self.cfg
        ds = c.dataset
        source = c.stage_dir("classifier")
        out = c.stage_dir("decision")
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(self.engines / "decision_engine.py"),
               "--results-source", str(source), "--output-dir", str(out),
               "--positive-name", str(ds["positive_label"]),
               "--negative-name", str(ds["negative_label"]),
               "--p-values", *[str(x) for x in c.science.get("p_values", [1,1.25,1.5,1.75,2,3,4,8,16])],
               "--reference-p", str(c.science.get("reference_p", c.science.get("p_values", [1])[0])),
               "--n-repeats", str(c.cv.get("n_repeats",20)),
               "--n-splits", str(c.cv.get("n_splits",5)),
               "--dpi", str(c.raw.get("plotting", {}).get("dpi", 220))]
        self._run(cmd, "03_decision.log")

    def xai(self, reuse: bool = True) -> None:
        c = self.cfg
        ds, p = c.dataset, c.paths
        source = c.stage_dir("classifier")
        out = c.stage_dir("xai")
        out.mkdir(parents=True, exist_ok=True)
        xopts = c.raw.get("xai", {})
        if self._representation() == "lowrank":
            self._prepare_lowrank_spectral_cache()
            cmd = [sys.executable, str(self.engines / "xai_engine_lowrank.py"),
                   "--results-source", str(source),
                   "--precomputed-distance-dir", str(Path(p["geometry_dir"]).expanduser()),
                   "--factors-file", str(Path(p["factors_file"]).expanduser()),
                   "--info-file", str(Path(p["info_file"]).expanduser()),
                   "--output-dir", str(out),
                   "--label-col", str(ds["label_column"]),
                   "--positive-label", str(ds["positive_label"]),
                   "--negative-label", str(ds["negative_label"]),
                   "--p-values", *[str(x) for x in c.science.get("p_values", [1,1.25,1.5,1.75,2,3,4,8,16])],
                   "--reference-p", str(c.science.get("reference_p", 1)),
                   "--n-repeats", str(c.cv.get("n_repeats",20)),
                   "--n-splits", str(c.cv.get("n_splits",5)),
                   "--expected-nodes", str(ds["expected_nodes"]),
                   "--top-n-ranks", str(xopts.get("top_n_ranks",20)),
                   "--validation-pairs", str(xopts.get("validation_pairs",300)),
                   "--random-state", str(xopts.get("random_state",42)),
                   "--xai-parallel", str(xopts.get("parallel_backend","auto")),
                   "--xai-workers", str(xopts.get("workers",0)),
                   "--xai-autotune-pairs", str(xopts.get("autotune_pairs",48)),
                   "--dpi", str(c.raw.get("plotting",{}).get("dpi",220))]
            if ds.get("id_column") is not None:
                cmd += ["--id-col", str(ds["id_column"])]
            self._run(cmd, "04_xai.log")
            return

        cmd = [sys.executable, str(self.engines / "xai_engine.py"),
               "--results-source", str(source),
               "--data-dir", str(Path(p["data_dir"]).expanduser()),
               "--info-file", str(Path(p["info_file"]).expanduser()),
               "--output-dir", str(out),
               "--matrix-glob", str(ds.get("matrix_glob", "adjacency_*.csv")),
               "--matrix-format", str(ds.get("matrix_format", "auto")),
               "--label-col", str(ds["label_column"]),
               "--positive-label", str(ds["positive_label"]),
               "--negative-label", str(ds["negative_label"]),
               "--p-values", *[str(x) for x in c.science.get("p_values", [1,1.25,1.5,1.75,2,3,4,8,16])],
               "--reference-p", str(c.science.get("reference_p", c.science.get("p_values", [1])[0])),
               "--n-repeats", str(c.cv.get("n_repeats",20)),
               "--n-splits", str(c.cv.get("n_splits",5)),
               "--diagonal", str(ds.get("diagonal", "unit")),
               "--psd-tol", str(ds.get("psd_tolerance", 1e-8)),
               "--top-n-ranks", str(xopts.get("top_n_ranks", 20)),
               "--validation-pairs", str(xopts.get("validation_pairs", 300)),
               "--random-state", str(xopts.get("random_state", 42)),
               "--xai-parallel", str(xopts.get("parallel_backend", "auto")),
               "--xai-workers", str(xopts.get("workers", 0)),
               "--xai-autotune-pairs", str(xopts.get("autotune_pairs", 48)),
               "--dpi", str(c.raw.get("plotting", {}).get("dpi", 220))]
        if ds.get("id_column") is not None:
            cmd += ["--id-col", str(ds["id_column"])]
        if ds.get("delimiter") is not None:
            cmd += ["--delimiter", str(ds["delimiter"])]
        if ds.get("expected_nodes") is not None:
            cmd += ["--expected-nodes", str(ds["expected_nodes"])]
        if not bool(ds.get("symmetrize", True)):
            cmd.append("--no-symmetrize")
        if reuse and (out / "pairwise_node_contributions.dat").exists():
            cmd.append("--reuse-contributions")
        self._run(cmd, "04_xai.log")

    def write_manifest(self) -> None:
        payload = {
            "config": str(self.cfg.path),
            "dataset": self.cfg.dataset,
            "output_root": str(self.cfg.output_root),
            "engines": {x.name: str(x) for x in self.engines.glob("*.py")},
        }
        with (self.cfg.output_root / "pipeline_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def run(self, stages: Iterable[str], reuse: bool = True) -> None:
        self.write_manifest()
        for stage in stages:
            if stage == "classifier": self.classifier(reuse=reuse)
            elif stage == "spectral": self.spectral()
            elif stage == "decision": self.decision()
            elif stage == "xai": self.xai(reuse=reuse)
            elif stage == "summary":
                from .summary import build_summary
                build_summary(self.cfg)
            else:
                raise ValueError(f"Unknown stage: {stage}")
