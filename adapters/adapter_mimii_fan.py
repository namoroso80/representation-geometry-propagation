#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter MIMII -> correlation matrices for Geometry 2.0.

Each WAV clip is converted to a log-mel spectrogram with a fixed number of
frequency bands. The item-level matrix is the Pearson correlation across time
between mel-frequency bands:

    WAV -> log-mel X(time, band) -> C = corr_bands(X)

C is symmetric positive semidefinite up to floating-point precision.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import librosa
except ImportError as exc:
    raise RuntimeError("librosa is required. Install with: pip install librosa soundfile") from exc

ID_RE = re.compile(r"^id[_-]?(\d+)$", re.IGNORECASE)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Extracted original MIMII fan directory for one SNR condition.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--n-fft", type=int, default=1024)
    p.add_argument("--hop-length", type=int, default=256)
    p.add_argument("--fmin", type=float, default=0.0)
    p.add_argument("--fmax", type=float, default=None)
    p.add_argument("--target-sr", type=int, default=None,
                   help="Optional resampling rate. Default: keep native rate.")
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--psd-tol", type=float, default=1e-8)
    p.add_argument("--limit-per-class", type=int, default=None,
                   help="Optional deterministic cap per class for pilot runs.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def infer_metadata(path: Path, root: Path):
    rel_parts = [x.lower() for x in path.relative_to(root).parts]

    condition = None
    for part in rel_parts:
        if "abnormal" in part or "anomal" in part:
            condition = "ANOMALOUS"
            break
    if condition is None:
        for part in rel_parts:
            if "normal" in part:
                condition = "NORMAL"
                break

    machine_id = None
    for part in path.relative_to(root).parts:
        m = ID_RE.match(part)
        if m:
            machine_id = f"id_{int(m.group(1)):02d}"
            break

    return condition, machine_id


def correlation_from_wav(path: Path, args):
    y, sr = librosa.load(path, sr=args.target_sr, mono=True)
    if y.size < args.n_fft:
        raise ValueError(f"audio too short: {y.size} samples")

    power = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        fmin=args.fmin,
        fmax=args.fmax,
        power=2.0,
        center=True,
    )
    log_power = librosa.power_to_db(power, ref=np.max).T  # time x band

    if log_power.shape[0] <= args.n_mels:
        raise ValueError(
            f"too few time frames ({log_power.shape[0]}) for {args.n_mels} bands"
        )

    X = log_power.astype(np.float64, copy=False)
    X -= X.mean(axis=0, keepdims=True)

    ss = np.sum(X * X, axis=0)
    if np.any(ss <= args.eps):
        bad = np.flatnonzero(ss <= args.eps)
        raise ValueError(f"near-zero variance mel bands: {bad.tolist()}")

    # Correlation as normalized Gram matrix: PSD by construction.
    denom = np.sqrt(np.outer(ss, ss))
    C = (X.T @ X) / denom
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)

    evals = np.linalg.eigvalsh(C)
    min_eval = float(evals[0])
    rank = int(np.linalg.matrix_rank(C, tol=1e-10))
    if min_eval < -args.psd_tol:
        raise ValueError(f"PSD check failed: min eigenvalue={min_eval:.3e}")

    return C, {
        "sample_rate": int(sr),
        "n_audio_samples": int(y.size),
        "n_frames": int(log_power.shape[0]),
        "n_mels": int(args.n_mels),
        "matrix_rank": rank,
        "min_eigenvalue": min_eval,
        "trace": float(np.trace(C)),
    }


def main():
    args = parse_args()
    if not args.input_dir.exists():
        raise FileNotFoundError(args.input_dir)

    if args.output_dir.exists() and not args.overwrite:
        if list(args.output_dir.glob("adjacency_*.csv")):
            raise FileExistsError(
                f"{args.output_dir} already contains matrices. Use --overwrite."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wavs = sorted(
        p for p in args.input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".wav"
    )
    if not wavs:
        raise RuntimeError(f"No WAV files found below {args.input_dir}")

    candidates = []
    for p in wavs:
        condition, machine_id = infer_metadata(p, args.input_dir)
        if condition is not None:
            candidates.append((p, condition, machine_id))

    if args.limit_per_class is not None:
        selected = []
        for label in ("NORMAL", "ANOMALOUS"):
            rows = [x for x in candidates if x[1] == label]
            selected.extend(rows[:args.limit_per_class])
        candidates = sorted(selected, key=lambda x: str(x[0]))

    info_rows = []
    qc_rows = []
    out_idx = 1

    for wav_path, condition, machine_id in candidates:
        try:
            C, qc = correlation_from_wav(wav_path, args)
            out_name = f"adjacency_{out_idx:06d}.csv"
            np.savetxt(args.output_dir / out_name, C, delimiter=",", fmt="%.10g")
            sample_id = f"MIMII_{out_idx:06d}"
            info_rows.append({
                "sample_id": sample_id,
                "diagnosis": condition,
                "machine_id": machine_id,
                "source_wav": str(wav_path),
                "matrix_file": out_name,
            })
            qc_rows.append({
                "sample_id": sample_id,
                "diagnosis": condition,
                "machine_id": machine_id,
                "source_wav": str(wav_path),
                **qc,
                "status": "ok",
                "error": "",
            })
            out_idx += 1
        except Exception as exc:
            qc_rows.append({
                "sample_id": "",
                "diagnosis": condition,
                "machine_id": machine_id,
                "source_wav": str(wav_path),
                "status": "excluded",
                "error": str(exc),
            })

    info = pd.DataFrame(info_rows)
    qc = pd.DataFrame(qc_rows)
    info.to_csv(args.output_dir / "info.csv", index=False)
    qc.to_csv(args.output_dir / "adapter_qc.csv", index=False)

    print("[MIMII] Adapter complete")
    print(f"[MIMII] Input WAVs: {len(wavs)}")
    print(f"[MIMII] Labelled candidates: {len(candidates)}")
    print(f"[MIMII] Retained matrices: {len(info)}")
    if len(info):
        print("[MIMII] Class counts:")
        print(info["diagnosis"].value_counts().to_string())
        ok_qc = qc[qc["status"] == "ok"]
        print(f"[MIMII] Matrix shape: ({args.n_mels}, {args.n_mels})")
        print(
            "[MIMII] Rank range: "
            f"{int(ok_qc['matrix_rank'].min())}-{int(ok_qc['matrix_rank'].max())}"
        )
        print(
            "[MIMII] Minimum eigenvalue across retained matrices: "
            f"{ok_qc['min_eigenvalue'].min():.3e}"
        )
    print(f"[MIMII] Output: {args.output_dir}")


if __name__ == "__main__":
    main()
