#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mock_run.py — Mock execution for CDR benchmark models.

Reads cached metrics from JSON and writes them in the standardized
output format. Used by Runner / Gradio demo to avoid 18h training runs.

Usage:
    python mock_run.py --model DeepTTA  --split_file <path>
    python mock_run.py --model TransCDR --split_file <path>

Note: --split_file is accepted for interface compatibility with real
runs but is not used in Level 1 mocking.
"""

import argparse
import json
import os
import sys
import time

CACHE_DIR  = "/home/intern1_2026_1/Common/CTS2026/cache"
OUTPUT_DIR = "/home/intern1_2026_1/Common/Output/mock"


def load_cache(model_name):
    """Load cached metrics for the given model."""
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_metrics.json")
    if not os.path.exists(cache_path):
        print(f"[mock_run] ERROR: cache file not found: {cache_path}",
              file=sys.stderr)
        sys.exit(1)
    with open(cache_path, "r") as f:
        return json.load(f)


def write_output(model_name, metrics):
    """Write metrics in the standardized 4-line text format."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_with_gexp_GDSC.csv")
    with open(out_path, "w") as f:
        f.write(f"Pearson = {metrics['Pearson']}\n")
        f.write(f"Spearman = {metrics['Spearman']}\n")
        f.write(f"RMSE = {metrics['RMSE']}\n")
        f.write(f"MSE = {metrics['MSE']}\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Mock CDR model runner")
    parser.add_argument("--model", type=str, required=True,
                        choices=["DeepTTA", "TransCDR"],
                        help="Model name to mock")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Split index file (accepted for interface "
                             "compatibility; ignored in Level 1 mocking)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Artificial delay in seconds (to feel realistic)")
    args = parser.parse_args()

    print(f"[mock_run] Model: {args.model}")
    print(f"[mock_run] Split file: {args.split_file or '(none)'}")
    print(f"[mock_run] Loading cached metrics...")

    cache = load_cache(args.model)
    metrics = cache["metrics"]

    if args.delay > 0:
        time.sleep(args.delay)

    out_path = write_output(args.model, metrics)

    print(f"[mock_run] Source: {cache.get('source', 'unknown')}")
    print(f"[mock_run] Results written to: {out_path}")
    print(f"[mock_run] Pearson={metrics['Pearson']:.4f}, "
          f"Spearman={metrics['Spearman']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, "
          f"MSE={metrics['MSE']:.4f}")


if __name__ == "__main__":
    main()
