#!/usr/bin/env python3
"""
Master Pipeline Runner for Probabilistic_MALDI-TOF_AST Experiments.

Executes the pipeline starting from binned mass spectrometry data:
  1. Preprocessing (cleaning, train/test split, normalization/scaling, label encoding)
  2. (Optional) Exploration (PCA, t-SNE, class distribution, correlation plots)
  3. Modeling & Evaluation (XGBoost, Random Forest, SVC, MLP with LPS or Independent)

Usage examples:
    # Run pipeline with default settings (RF and XGB with LPS and Independent)
    python run_pipeline.py

    # Run all algorithms (xgb, svc, rf, mlp) with LPS mode
    python run_pipeline.py --algorithms all --modes lps

    # Run preprocessing + exploration + modeling
    python run_pipeline.py --run-exploration

    # Run on a specific norm and oversampling
    python run_pipeline.py --norm standard --oversampling none --split 80-20
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Ensure working directory is set to the 'experiments' folder
EXPERIMENTS_DIR = Path(__file__).resolve().parent
os.chdir(EXPERIMENTS_DIR)
sys.path.insert(0, str(EXPERIMENTS_DIR))


def log_banner(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def check_binned_data(folder_name):
    """Check if binned data exists in data/unprocessed/<folder_name>/"""
    unprocessed_binned_dir = EXPERIMENTS_DIR / "data" / "unprocessed" / folder_name
    if not unprocessed_binned_dir.exists():
        unprocessed_binned_dir.mkdir(parents=True, exist_ok=True)
        print(f"[WARNING] Directory '{unprocessed_binned_dir}' did not exist; created it.")
        return []

    csv_files = list(unprocessed_binned_dir.glob("*.csv"))
    return csv_files


def run_command(cmd_args, step_name=""):
    cmd_str = " ".join(cmd_args)
    print(f"--> [{step_name}] Running: {cmd_str}")
    start = time.time()
    result = subprocess.run(cmd_args, cwd=str(EXPERIMENTS_DIR))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"[ERROR] Step '{step_name}' failed with exit code {result.returncode} (took {elapsed:.1f}s)")
        return False
    print(f"[SUCCESS] Step '{step_name}' completed in {elapsed:.1f}s\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for Probabilistic MALDI-TOF AST pipeline (starting from binned data)"
    )
    parser.add_argument(
        "-f", "--folder",
        default="binned",
        help="Subfolder inside data/unprocessed/ where binned CSV files are stored (default: 'binned')"
    )
    parser.add_argument(
        "-n", "--norm",
        choices=["standard", "min-max", "none"],
        default="standard",
        help="Data normalization method (default: 'standard')"
    )
    parser.add_argument(
        "-o", "--oversampling",
        choices=["none", "random", "smotetomek"],
        default="none",
        help="Oversampling technique (default: 'none')"
    )
    parser.add_argument(
        "-s", "--split",
        choices=["80-20", "5x2", "100"],
        default="80-20",
        help="Train/test split strategy (default: '80-20')"
    )
    parser.add_argument(
        "-a", "--algorithms",
        nargs="+",
        default=["rf", "xgb"],
        help="ML algorithms to train. Choices: 'xgb', 'svc', 'rf', 'mlp', or 'all' (default: rf xgb)"
    )
    parser.add_argument(
        "-m", "--modes",
        nargs="+",
        default=["lps", "independent"],
        help="Multi-label modes to evaluate. Choices: 'lps', 'independent', or 'both' (default: lps independent)"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip the preprocessing step (use already preprocessed datasets)"
    )
    parser.add_argument(
        "--run-exploration",
        action="store_true",
        help="Run data exploration graphics (heatmaps, PCA, t-SNE)"
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Filter specific species/filename substring (passed to preprocessing -c)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print steps and commands without executing them"
    )

    args = parser.parse_args()

    # Expand 'all' or 'both' keywords
    if "all" in args.algorithms:
        algorithms = ["xgb", "svc", "rf", "mlp"]
    else:
        algorithms = args.algorithms

    if "both" in args.modes:
        modes = ["lps", "independent"]
    else:
        modes = args.modes

    log_banner("PROBABILISTIC MALDI-TOF AST - PIPELINE RUNNER")
    print(f"Working Directory   : {EXPERIMENTS_DIR}")
    print(f"Binned Folder       : data/unprocessed/{args.folder}/")
    print(f"Normalization       : {args.norm}")
    print(f"Oversampling        : {args.oversampling}")
    print(f"Split Strategy      : {args.split}")
    print(f"Algorithms          : {', '.join(algorithms)}")
    print(f"Multi-label Modes   : {', '.join(modes)}")
    print(f"Run Exploration     : {args.run_exploration}")
    print(f"Skip Preprocessing  : {args.skip_preprocessing}")

    # Step 0: Check Input Binned Data
    if not args.skip_preprocessing:
        binned_files = check_binned_data(args.folder)
        if not binned_files:
            print(f"\n[ATTENTION] No CSV files found in 'data/unprocessed/{args.folder}/'.")
            print("Please place your binned .csv dataset file(s) into:")
            print(f"    {EXPERIMENTS_DIR / 'data' / 'unprocessed' / args.folder}")
            print("\nExample expected file names:")
            print("    e_coli_driams_bin5.csv")
            print("    e_coli_driams_bin20.csv")
            print("    k_pneumoniae_driams_bin5.csv")
            print("    s_aureus_driams_bin5.csv")
            if not args.dry_run:
                sys.exit(1)
        else:
            print(f"\nFound {len(binned_files)} binned file(s) in 'data/unprocessed/{args.folder}/':")
            for f in binned_files:
                print(f"  - {f.name}")

    total_start_time = time.time()

    # Step 1: Preprocessing
    if not args.skip_preprocessing:
        log_banner("STEP 1: PREPROCESSING (Cleaning, Split, Normalization, Encoding)")
        prep_cmd = [
            sys.executable,
            "processing/preprocessing.py",
            "-f", args.folder,
            "-n", args.norm,
            "-o", args.oversampling,
            "-s", args.split,
        ]
        if args.filter:
            prep_cmd.extend(["-c", args.filter])

        if args.dry_run:
            print(f"[DRY-RUN] Would execute: {' '.join(prep_cmd)}")
        else:
            ok = run_command(prep_cmd, step_name="Preprocessing")
            if not ok:
                print("[ABORT] Pipeline stopped due to preprocessing failure.")
                sys.exit(1)
    else:
        print("\n[INFO] Skipping Step 1 (Preprocessing) as requested.")

    # Step 2: (Optional) Exploration
    if args.run_exploration:
        log_banner("STEP 2: DATA EXPLORATION (Distributions, Correlations, PCA, t-SNE)")
        folder_arg = f"{args.folder}/{args.oversampling}" if args.oversampling != "none" else args.folder
        expl_cmd = [
            sys.executable,
            "exploration/exploration.py",
            "-f", folder_arg,
            "-n", args.norm,
        ]
        if args.dry_run:
            print(f"[DRY-RUN] Would execute: {' '.join(expl_cmd)}")
        else:
            run_command(expl_cmd, step_name="Data Exploration")

    # Step 3: Modeling & Evaluation
    log_banner("STEP 3: MODELING & EVALUATION (Bayesian Optimization & Benchmark)")
    total_runs = len(algorithms) * len(modes)
    current_run = 0

    for algo in algorithms:
        for mode in modes:
            current_run += 1
            step_desc = f"Modeling [{current_run}/{total_runs}]: Algorithm={algo.upper()} | Mode={mode.upper()}"
            print(f"\n--- {step_desc} ---")

            model_cmd = [
                sys.executable,
                "modeling/modeling_lps.py",
                "-a", algo,
                "-m", mode,
                "-f", args.folder,
                "-n", args.norm,
                "-o", args.oversampling,
            ]

            if args.dry_run:
                print(f"[DRY-RUN] Would execute: {' '.join(model_cmd)}")
            else:
                ok = run_command(model_cmd, step_name=f"{algo}_{mode}")
                if not ok:
                    print(f"[WARNING] Modeling for {algo} with {mode} encountered an error.")

    elapsed_total = time.time() - total_start_time
    log_banner(f"PIPELINE COMPLETED in {elapsed_total:.1f}s")
    print(f"Results saved to : {EXPERIMENTS_DIR / 'modeling' / 'results'}")
    print(f"Models saved to  : {EXPERIMENTS_DIR / 'modeling' / 'models'}")


if __name__ == "__main__":
    main()
