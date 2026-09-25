#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from geometry_pipeline.config import PipelineConfig
from geometry_pipeline.runner import PipelineRunner

ALL = ["classifier", "spectral", "decision", "xai", "summary"]


def main():
    ap = argparse.ArgumentParser(description="Master runner for the Schatten geometry-response pipeline")
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="+", choices=ALL, default=ALL,
                    help="Stages to run, in requested order")
    ap.add_argument("--no-reuse", action="store_true",
                    help="Disable reuse of cached spectra/checkpoints/XAI contributions")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    root = Path(__file__).resolve().parent
    runner = PipelineRunner(cfg, root)
    runner.run(args.stages, reuse=not args.no_reuse)


if __name__ == "__main__":
    main()
