from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml



@dataclass
class PipelineConfig:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("Configuration root must be a YAML mapping")
        cfg = cls(raw=raw, path=path)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        for section in ("dataset", "paths", "pipeline"):
            if section not in self.raw:
                raise ValueError(f"Missing config section: {section}")

        d = self.raw["dataset"]
        for key in ("name", "label_column", "positive_label", "negative_label"):
            if key not in d:
                raise ValueError(f"Missing dataset.{key}")
        if d["positive_label"] == d["negative_label"]:
            raise ValueError("positive_label and negative_label must differ")

        sci = self.raw.setdefault("science", {})
        pgrid = [float(x) for x in sci.get("p_values", [1, 1.25, 1.5, 1.75, 2, 3, 4, 8, 16])]
        if not pgrid or any(x < 1 for x in pgrid):
            raise ValueError("science.p_values must contain Schatten p values >= 1")
        if len(set(pgrid)) != len(pgrid):
            raise ValueError("science.p_values contains duplicates")
        ref = float(sci.get("reference_p", pgrid[0]))
        if not any(abs(ref-x) < 1e-12 for x in pgrid):
            raise ValueError("science.reference_p must be one of science.p_values")
        cv = self.raw.setdefault("cv", {})
        for key, default in (("n_repeats",20),("n_splits",5),("inner_splits",5),("random_state",42)):
            cv.setdefault(key, default)
        if int(cv["n_splits"]) < 2 or int(cv["inner_splits"]) < 2 or int(cv["n_repeats"]) < 1:
            raise ValueError("Invalid CV configuration")

        representation = str(d.get("representation", "dense")).strip().lower()
        if representation not in {"dense", "lowrank"}:
            raise ValueError("dataset.representation must be 'dense' or 'lowrank'")
        d["representation"] = representation

        paths = self.raw["paths"]
        common_required = ("info_file", "output_root")
        for key in common_required:
            if key not in paths:
                raise ValueError(f"Missing paths.{key}")

        if representation == "dense":
            if "data_dir" not in paths:
                raise ValueError("Missing paths.data_dir for dense representation")
        else:
            for key in ("factors_file", "geometry_dir"):
                if key not in paths:
                    raise ValueError(f"Missing paths.{key} for lowrank representation")
            if d.get("expected_nodes") is None:
                raise ValueError("dataset.expected_nodes is required for lowrank representation")

        allowed_stages = {"classifier", "spectral", "decision", "xai", "summary"}
        stages = self.raw["pipeline"].get("stages", [])
        if not stages:
            raise ValueError("pipeline.stages must contain at least one stage")
        unknown = [x for x in stages if x not in allowed_stages]
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {unknown}")

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw["paths"]

    @property
    def pipeline(self) -> dict[str, Any]:
        return self.raw["pipeline"]

    @property
    def science(self) -> dict[str, Any]:
        return self.raw.setdefault("science", {})

    @property
    def cv(self) -> dict[str, Any]:
        return self.raw.setdefault("cv", {})

    @property
    def output_root(self) -> Path:
        return Path(self.paths["output_root"]).expanduser().resolve()

    def stage_dir(self, stage: str) -> Path:
        names = {
            "classifier": "01_classifier",
            "spectral": "02_spectral",
            "decision": "03_decision",
            "xai": "04_xai",
            "summary": "05_summary",
        }
        return self.output_root / names[stage]
