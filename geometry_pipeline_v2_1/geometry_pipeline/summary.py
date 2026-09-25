from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from .config import PipelineConfig


def _first_existing(paths):
    for p in paths:
        if p.exists(): return p
    return None


def build_summary(cfg: PipelineConfig) -> Path:
    out = cfg.stage_dir("summary")
    out.mkdir(parents=True, exist_ok=True)

    cl = cfg.stage_dir("classifier")
    sp = cfg.stage_dir("spectral")
    de = cfg.stage_dir("decision")
    xa = cfg.stage_dir("xai")

    frames = []

    f = _first_existing([cl / "summary_by_p_repeat_level.csv", cl / "repeat_summary.csv"])
    if f is not None:
        a = pd.read_csv(f)
        if "p" in a.columns:
            # summary_by_p_repeat_level already one row/p. If repeat_summary, aggregate robustly.
            if a["p"].duplicated().any():
                num = a.select_dtypes(include=[np.number]).columns.tolist()
                keep = [c for c in num if c != "p"]
                a = a.groupby("p", as_index=False)[keep].median()
            frames.append(a)

    f = sp / "spectral_concentration_summary.csv"
    if f.exists(): frames.append(pd.read_csv(f))

    f = de / "09_decision_summary_by_p.csv"
    if f.exists(): frames.append(pd.read_csv(f))

    f = xa / "04_decision_vs_xai_concordance.csv"
    if f.exists(): frames.append(pd.read_csv(f))

    if not frames:
        raise FileNotFoundError("No stage summaries found; run analysis stages first")

    merged = frames[0]
    for b in frames[1:]:
        if "p" not in b.columns: continue
        overlapping = [c for c in b.columns if c in merged.columns and c != "p"]
        if overlapping:
            b = b.rename(columns={c: f"{c}_stage{len(merged.columns)}" for c in overlapping})
        merged = pd.merge(merged, b, on="p", how="outer")
    merged = merged.sort_values("p").reset_index(drop=True)

    # Canonical compact fingerprint columns. Prefer the exact columns emitted by
    # the validated AD engines; retain fallbacks for future dataset-agnostic engines.
    aliases = {
        "auc_heldout": [
            "test_oof_auc_repeat_median",
            "oof_auc_repeat_median",
            "test_auc_median",
            "pooled_oof_auc_median",
            "test_auc",
        ],
        "gamma_star": [
            "gamma_repeat_median",
            "best_gamma_median",
            "gamma_median",
            "best_gamma",
        ],
        "Q_typical": [
            "Q_kernel_repeat_median",
            "Q_median",
            "effective_Q_median",
            "Q",
        ],
        "decision_concordance": [
            "decision_spearman_vs_p1",
            "spearman_vs_p1_subject_median",
            "spearman_vs_p1",
        ],
        "xai_concordance": [
            "xai_subject_spearman_median_vs_p1",
        ],
        "leading_mode_share": [
            "leading_share_resample_median",
            "leading_share_median_pair",
        ],
    }
    fp = pd.DataFrame({"p": merged["p"]})
    for target, candidates in aliases.items():
        col = next((c for c in candidates if c in merged.columns), None)
        fp[target] = merged[col] if col is not None else np.nan
    if "decision_concordance" in fp:
        fp["decision_loss"] = 1.0 - fp["decision_concordance"]
    if "xai_concordance" in fp:
        fp["xai_loss"] = 1.0 - fp["xai_concordance"]

    merged.to_csv(out / "pipeline_summary_full.csv", index=False)
    fp.to_csv(out / "pipeline_summary.csv", index=False)
    print(f"[SUMMARY] {out / 'pipeline_summary.csv'}")
    return out / "pipeline_summary.csv"
