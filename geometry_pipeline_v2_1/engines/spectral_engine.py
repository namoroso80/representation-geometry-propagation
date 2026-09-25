#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-safe Schatten spectral-concentration engine.

Scientific definitions and output summaries are preserved from spectral_engine.py,
but the pairwise spectra are processed in chunks and the large per-pair metric
arrays are kept as disk-backed memmaps instead of being materialized in RAM.

This is intended for large-N datasets such as MIMII, where the cached singular-
value array can contain billions of floating-point values.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm.auto import tqdm


@dataclass(frozen=True)
class ScientificConfig:
    p_values: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 8.0, 16.0)
    topk_values: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    n_subject_resamples: int = 300
    subject_fraction: float = 0.80
    random_state: int = 42


SCI = ScientificConfig()
EPS = np.finfo(np.float64).tiny
METRIC_NAMES = ("max_equiv_modes", "leading_share", "entropy_neff", "ipr_neff")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory-safe geometry-only singular-mode concentration test across Schatten p.")
    p.add_argument("--spectra-source", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-resamples", type=int, default=SCI.n_subject_resamples)
    p.add_argument("--subject-fraction", type=float, default=SCI.subject_fraction)
    p.add_argument("--random-state", type=int, default=SCI.random_state)
    p.add_argument("--p-values", nargs="+", type=float, default=list(SCI.p_values))
    p.add_argument("--chunk-size", type=int, default=100_000,
                   help="Number of subject pairs processed in RAM at once (default: 100000).")
    p.add_argument("--work-dir", type=Path, default=None,
                   help="Directory for disk-backed temporary metric arrays. Default: <output-dir>/_work.")
    p.add_argument("--keep-work", action="store_true",
                   help="Keep disk-backed temporary arrays after successful completion.")
    p.add_argument("--skip-pairwise-npz", action="store_true",
                   help="Skip the very large pairwise_spectral_concentration.npz archive.")
    return p.parse_args()


def load_spectra(source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra_path = source / "pairwise_singular_values.npy"
    pairs_path = source / "pair_indices.npz"
    if not spectra_path.exists():
        raise FileNotFoundError(spectra_path)
    if not pairs_path.exists():
        raise FileNotFoundError(pairs_path)

    singvals = np.load(spectra_path, mmap_mode="r")
    pairs = np.load(pairs_path)
    pair_s = np.asarray(pairs["pair_s"], dtype=np.int32)
    pair_t = np.asarray(pairs["pair_t"], dtype=np.int32)

    if singvals.ndim != 2:
        raise ValueError(f"Expected 2D singular-value array, got {singvals.shape}")
    if len(pair_s) != singvals.shape[0] or len(pair_t) != singvals.shape[0]:
        raise ValueError("Pair-index arrays are inconsistent with singular spectra")

    n_subjects = int(max(pair_s.max(), pair_t.max()) + 1)
    print(f"[LOAD] {singvals.shape[0]:,} pairs | {n_subjects} subjects | {singvals.shape[1]} singular modes")
    print(f"[LOAD] spectra dtype={singvals.dtype} | mmap=yes")
    return singvals, pair_s, pair_t


def quantiles(x: np.ndarray) -> tuple[float, float, float]:
    q25, med, q75 = np.quantile(x, [0.25, 0.50, 0.75])
    return float(q25), float(med), float(q75)


def resample_interval(x: np.ndarray) -> tuple[float, float, float]:
    lo, med, hi = np.quantile(x, [0.025, 0.50, 0.975])
    return float(lo), float(med), float(hi)


def create_metric_memmaps(work_dir: Path, n_p: int, n_pairs: int):
    work_dir.mkdir(parents=True, exist_ok=True)
    maps = {}
    for name in METRIC_NAMES:
        maps[name] = open_memmap(
            work_dir / f"{name}.npy",
            mode="w+",
            dtype=np.float64,
            shape=(n_p, n_pairs),
        )
    return maps


def build_metrics_chunked(
    singvals: np.ndarray,
    p_values: np.ndarray,
    topk_values: tuple[int, ...],
    metric_maps: dict[str, np.memmap],
    chunk_size: int,
):
    n_pairs, n_modes = singvals.shape
    topk_acc = {(float(p), int(min(k, n_modes))): [] for p in p_values for k in topk_values}

    n_chunks = (n_pairs + chunk_size - 1) // chunk_size
    print(f"[TEST] Chunked pairwise metrics | chunk_size={chunk_size:,} | chunks={n_chunks:,}")

    for start in tqdm(range(0, n_pairs, chunk_size), total=n_chunks, desc="Spectral chunks"):
        stop = min(start + chunk_size, n_pairs)

        # Materialize only one chunk in RAM. Singular values are non-negative but
        # not necessarily stored in descending order after abs(eigvalsh()).
        s = np.array(singvals[start:stop], dtype=np.float64, copy=True)
        s.sort(axis=1)
        s = s[:, ::-1]
        lead = s[:, :1]
        if np.any(lead <= 0):
            n_zero = int(np.sum(lead[:, 0] <= 0))
            raise ValueError(f"Found {n_zero} zero pairwise spectra in chunk {start}:{stop}")
        r = s / lead
        r[:, 0] = 1.0
        del s, lead

        for pi, p in enumerate(p_values):
            rp = np.power(r, float(p))
            z = rp.sum(axis=1)
            if np.any(z <= 0):
                raise ValueError(f"Non-positive normalization encountered at p={p:g}")

            leading_share = 1.0 / z
            metric_maps["max_equiv_modes"][pi, start:stop] = z
            metric_maps["leading_share"][pi, start:stop] = leading_share

            # Entropy effective number without materializing a second full weight array:
            # H = log(z) - sum(rp * log(rp)) / z.
            # The x log x contribution is defined as zero at x=0.
            with np.errstate(divide="ignore", invalid="ignore"):
                log_rp = np.log(rp)
                xlogx = np.where(rp > 0, rp * log_rp, 0.0)
            entropy = np.log(z) - xlogx.sum(axis=1) / z
            metric_maps["entropy_neff"][pi, start:stop] = np.exp(entropy)

            # IPR Neff = 1 / sum(w^2) = z^2 / sum(rp^2)
            metric_maps["ipr_neff"][pi, start:stop] = (z * z) / np.sum(rp * rp, axis=1)

            # Store only chunk-level top-k values; concatenate one k at a time later.
            csum = np.cumsum(rp, axis=1)
            for k in topk_values:
                kk = int(min(k, n_modes))
                topk_acc[(float(p), kk)].append((csum[:, kk - 1] / z).copy())

            del rp, z, leading_share, log_rp, xlogx, entropy, csum

        del r

    for mm in metric_maps.values():
        mm.flush()
    return topk_acc


def summarize_pairwise(metric_maps, p_values: np.ndarray, n_pairs: int):
    rows = []
    print("[SUMMARY] Pairwise quartiles")
    for pi, p in enumerate(tqdm(p_values, desc="Pairwise summary")):
        row = {"p": float(p), "n_pairs": int(n_pairs)}
        for key in METRIC_NAMES:
            q25, med, q75 = quantiles(metric_maps[key][pi])
            row[f"{key}_q25_pair"] = q25
            row[f"{key}_median_pair"] = med
            row[f"{key}_q75_pair"] = q75
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_topk(topk_acc, p_values: np.ndarray, topk_values: tuple[int, ...], n_modes: int):
    rows = []
    print("[SUMMARY] Top-k cumulative contributions")
    for p in p_values:
        for k in topk_values:
            kk = int(min(k, n_modes))
            vals = np.concatenate(topk_acc[(float(p), kk)])
            q25, med, q75 = quantiles(vals)
            rows.append({
                "p": float(p),
                "k": kk,
                "topk_share_q25_pair": q25,
                "topk_share_median_pair": med,
                "topk_share_q75_pair": q75,
            })
            del vals
    return pd.DataFrame(rows)


def generate_subject_selections(n_subjects: int, n_resamples: int, subject_fraction: float, random_state: int):
    if not (0 < subject_fraction <= 1):
        raise ValueError("subject_fraction must be in (0, 1]")
    n_keep = max(2, int(round(subject_fraction * n_subjects)))
    rng = np.random.default_rng(random_state)
    selections = np.zeros((n_resamples, n_subjects), dtype=bool)
    for rep in range(n_resamples):
        chosen = rng.choice(n_subjects, size=n_keep, replace=False)
        selections[rep, chosen] = True
    return selections, n_keep


def subject_resample_summary(
    metric_maps,
    pair_s: np.ndarray,
    pair_t: np.ndarray,
    p_values: np.ndarray,
    n_resamples: int,
    subject_fraction: float,
    random_state: int,
):
    n_subjects = int(max(pair_s.max(), pair_t.max()) + 1)
    selections, n_keep = generate_subject_selections(
        n_subjects, n_resamples, subject_fraction, random_state
    )
    expected_pairs = n_keep * (n_keep - 1) // 2
    print(f"[RESAMPLE] {n_resamples} subject subsamples | fraction={subject_fraction:.2f} | kept_subjects={n_keep} | pairs/resample={expected_pairs:,}")
    print("[RESAMPLE] Exact subject-level medians are retained; this stage can be I/O intensive for very large N.")

    rows = []
    for pi, p in enumerate(p_values):
        print(f"[RESAMPLE] p={p:g}")
        for rep in tqdm(range(n_resamples), desc=f"p={p:g} resamples", leave=False):
            include = selections[rep]
            mask = include[pair_s] & include[pair_t]
            n_pairs = int(mask.sum())
            if n_pairs != expected_pairs:
                raise RuntimeError(f"Unexpected pair count in resample {rep}: {n_pairs} != {expected_pairs}")
            row = {"p": float(p), "resample": int(rep), "n_pairs": n_pairs}
            # One metric at a time keeps peak RAM bounded.
            for key in METRIC_NAMES:
                row[key] = float(np.median(metric_maps[key][pi][mask]))
            rows.append(row)
            del mask
    return pd.DataFrame(rows)


def add_resample_intervals(summary: pd.DataFrame, resamples: pd.DataFrame, p_values: np.ndarray):
    out = summary.copy()
    for key in METRIC_NAMES:
        stats = []
        for p in p_values:
            vals = resamples.loc[resamples["p"] == p, key].to_numpy(float)
            lo, med, hi = resample_interval(vals)
            stats.append((p, lo, med, hi))
        tmp = pd.DataFrame(stats, columns=["p", f"{key}_ci95_low", f"{key}_resample_median", f"{key}_ci95_high"])
        out = out.merge(tmp, on="p", how="left")
    return out


def threshold_report(summary: pd.DataFrame) -> str:
    s = summary.sort_values("p").reset_index(drop=True)
    share = s["leading_share_median_pair"].to_numpy(float)
    pvals = s["p"].to_numpy(float)
    lines = [
        "Leading-mode dominance threshold diagnostic",
        "==========================================",
        "",
        "Criterion: median leading-mode share >= 0.5, i.e. the leading singular mode",
        "accounts for at least half of d_p^p across subject pairs.",
        "",
    ]
    idx = np.where(share >= 0.5)[0]
    if len(idx) == 0:
        lines.append("The criterion is not reached on the sampled p grid.")
    else:
        j = int(idx[0])
        if j == 0:
            lines.append(f"The criterion is already reached at the smallest sampled p={pvals[j]:g}.")
        else:
            lines.append(f"The first sampled p satisfying the criterion is p={pvals[j]:g}; the transition is therefore bracketed by p={pvals[j-1]:g} and p={pvals[j]:g}.")
        lines.append(f"Median leading share at p={pvals[j]:g}: {share[j]:.4f}")
    return "\n".join(lines) + "\n"


def errorbar_from_summary(ax, summary: pd.DataFrame, key: str, ylabel: str, logy: bool = False):
    x = summary["p"].to_numpy(float)
    y = summary[f"{key}_median_pair"].to_numpy(float)
    lo = summary[f"{key}_ci95_low"].to_numpy(float)
    hi = summary[f"{key}_ci95_high"].to_numpy(float)
    yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=4)
    ax.set_xlabel("Schatten p")
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def save_fig(fig, output_dir: Path, stem: str):
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_plots(summary: pd.DataFrame, topk_summary: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    errorbar_from_summary(ax, summary, "max_equiv_modes", r"Max-equivalent contributing modes $M_p$", logy=True)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="single-mode limit")
    ax.legend(frameon=False)
    ax.set_title("Spectral concentration of the Schatten distance")
    save_fig(fig, output_dir, "01_max_equivalent_modes_vs_p")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    errorbar_from_summary(ax, summary, "leading_share", r"Leading-mode share of $d_p^p$", logy=False)
    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="50% leading-mode contribution")
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("Leading singular-mode dominance across p")
    save_fig(fig, output_dir, "02_leading_mode_share_vs_p")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for k, g in topk_summary.groupby("k"):
        ax.plot(g["p"], g["topk_share_median_pair"], marker="o", label=f"top {int(k)}")
    ax.set_xlabel("Schatten p")
    ax.set_ylabel(r"Median cumulative contribution to $d_p^p$")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax.set_title("How many singular modes carry the Schatten distance?")
    save_fig(fig, output_dir, "03_topk_contribution_vs_p")

    pvals = summary["p"].to_numpy(float)
    logM = np.log(summary["max_equiv_modes_median_pair"].to_numpy(float))
    rate = -np.diff(logM) / np.diff(pvals)
    pmid = 0.5 * (pvals[:-1] + pvals[1:])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(pmid, rate, marker="o")
    ax.set_xlabel("Schatten p (interval midpoint)")
    ax.set_ylabel(r"$-\Delta\log M_p/\Delta p$")
    ax.grid(alpha=0.25)
    ax.set_title("Rate of spectral concentration")
    save_fig(fig, output_dir, "04_concentration_rate_vs_p")


def save_pairwise_npz(output_dir: Path, p_values: np.ndarray, pair_s, pair_t, metric_maps):
    print("[SAVE] Writing pairwise_spectral_concentration.npz (large file; streamed from memmaps)")
    np.savez_compressed(
        output_dir / "pairwise_spectral_concentration.npz",
        p_values=p_values,
        pair_s=pair_s,
        pair_t=pair_t,
        max_equiv_modes=metric_maps["max_equiv_modes"],
        leading_share=metric_maps["leading_share"],
        entropy_neff=metric_maps["entropy_neff"],
        ipr_neff=metric_maps["ipr_neff"],
    )


def main():
    global SCI
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be >= 1")

    SCI = ScientificConfig(
        p_values=tuple(float(x) for x in args.p_values),
        topk_values=SCI.topk_values,
        n_subject_resamples=int(args.n_resamples),
        subject_fraction=float(args.subject_fraction),
        random_state=int(args.random_state),
    )

    source = args.spectra_source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = (args.work_dir or (output_dir / "_work")).expanduser().resolve()

    singvals, pair_s, pair_t = load_spectra(source)
    p_values = np.asarray(SCI.p_values, dtype=float)
    n_pairs, n_modes = singvals.shape

    # Four disk-backed arrays: about 4 * n_p * n_pairs * 8 bytes on disk.
    gib = 4 * len(p_values) * n_pairs * 8 / (1024 ** 3)
    print(f"[WORK] Disk-backed metric arrays require approximately {gib:.2f} GiB in {work_dir}")
    metric_maps = create_metric_memmaps(work_dir, len(p_values), n_pairs)

    topk_acc = build_metrics_chunked(
        singvals=singvals,
        p_values=p_values,
        topk_values=SCI.topk_values,
        metric_maps=metric_maps,
        chunk_size=int(args.chunk_size),
    )

    summary = summarize_pairwise(metric_maps, p_values, n_pairs)
    topk_summary = summarize_topk(topk_acc, p_values, SCI.topk_values, n_modes)
    del topk_acc

    resamples = subject_resample_summary(
        metric_maps=metric_maps,
        pair_s=pair_s,
        pair_t=pair_t,
        p_values=p_values,
        n_resamples=int(args.n_resamples),
        subject_fraction=float(args.subject_fraction),
        random_state=int(args.random_state),
    )
    summary = add_resample_intervals(summary, resamples, p_values)

    summary.to_csv(output_dir / "spectral_concentration_summary.csv", index=False)
    resamples.to_csv(output_dir / "spectral_concentration_resamples.csv", index=False)
    topk_summary.to_csv(output_dir / "topk_contribution_summary.csv", index=False)

    if not args.skip_pairwise_npz:
        save_pairwise_npz(output_dir, p_values, pair_s, pair_t, metric_maps)
    else:
        print("[SAVE] Skipping pairwise_spectral_concentration.npz by request")

    report = threshold_report(summary)
    (output_dir / "concentration_threshold.txt").write_text(report, encoding="utf-8")

    metadata = {
        "scientific_config": asdict(SCI),
        "spectra_source": str(source),
        "n_subject_resamples": int(args.n_resamples),
        "subject_fraction": float(args.subject_fraction),
        "random_state": int(args.random_state),
        "primary_definition": "M_p = sum_k (sigma_k/sigma_1)^p = d_p^p/sigma_1^p",
        "leading_share_definition": "leading_share = 1/M_p",
        "implementation": "chunked-memory-safe exact pairwise metrics with disk-backed memmaps",
        "chunk_size": int(args.chunk_size),
        "work_dir": str(work_dir),
        "pairwise_npz_saved": bool(not args.skip_pairwise_npz),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    make_plots(summary, topk_summary, output_dir)

    print("\n" + report.rstrip())
    print(f"[SAVE] Results -> {output_dir}")

    # Ensure memmaps are closed before optional cleanup.
    for mm in metric_maps.values():
        mm.flush()
    del metric_maps

    if not args.keep_work:
        try:
            shutil.rmtree(work_dir)
            print(f"[CLEAN] Removed temporary work directory -> {work_dir}")
        except Exception as exc:
            print(f"[WARN] Could not remove work directory {work_dir}: {exc}")


if __name__ == "__main__":
    main()
