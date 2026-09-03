#!/usr/bin/env python3
"""Download external datasets for MCP-IDS.

Usage:
    python scripts/download_datasets.py --dataset cx_cmu
    python scripts/download_datasets.py --dataset atbench
    python scripts/download_datasets.py --dataset ras_eval
    python scripts/download_datasets.py --dataset all
"""
import argparse
import os
import shutil
import subprocess
import sys


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")


def download_cx_cmu():
    out_dir = os.path.join(DATA_DIR, "cx_cmu")
    os.makedirs(out_dir, exist_ok=True)

    trajectory_files = ["mcpbench", "tau2bench", "swebench", "terminalbench", "search"]
    all_exist = all(
        os.path.exists(os.path.join(out_dir, f"{name}.jsonl"))
        for name in trajectory_files
    )
    if all_exist:
        print("  All trajectory files already exist, skipping download")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ERROR: pip install huggingface_hub  (required for cx-cmu download)")
        sys.exit(1)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("  WARNING: HF_TOKEN not set. cx-cmu/agent_trajectories is a gated dataset.")
        print("  Set HF_TOKEN or run: huggingface-cli login")

    for name in trajectory_files:
        out_file = os.path.join(out_dir, f"{name}.jsonl")
        if os.path.exists(out_file):
            print(f"  {name}.jsonl already exists, skipping")
            continue
        print(f"  Downloading {name}.jsonl...")
        downloaded = hf_hub_download(
            repo_id="cx-cmu/agent_trajectories",
            filename=f"{name}.jsonl",
            repo_type="dataset",
            token=hf_token,
            local_dir=out_dir,
        )
        print(f"  Saved {name}.jsonl")

    print(f"cx-cmu data saved to {out_dir}/")


def download_atbench():
    out_dir = os.path.join(DATA_DIR, "atbench")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "atbench.json")
    if os.path.exists(out_file):
        print("  atbench.json already exists, skipping download")
        return
    print("  Downloading ATBench from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ERROR: pip install huggingface_hub  (required for ATBench download)")
        sys.exit(1)
    hf_hub_download(
        repo_id="AI45Research/ATBench",
        filename="ATBench/test.json",
        repo_type="dataset",
        local_dir=out_dir,
    )
    downloaded = os.path.join(out_dir, "ATBench", "test.json")
    os.rename(downloaded, out_file)
    os.rmdir(os.path.join(out_dir, "ATBench"))
    import json
    with open(out_file) as f:
        data = json.load(f)
    n = len(data) if isinstance(data, list) else len(data.get("data", []))
    print(f"  Saved {n} records to atbench.json")


def download_ras_eval():
    out_dir = os.path.join(DATA_DIR, "ras_eval")
    os.makedirs(out_dir, exist_ok=True)
    repo_url = "https://github.com/lanzer-tree/RAS-Eval.git"
    clone_dir = os.path.join(out_dir, "_repo")
    if not os.path.exists(clone_dir):
        print("  Cloning RAS-Eval repo (sparse checkout for data only)...")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--sparse", repo_url, clone_dir], check=True)
        subprocess.run(["git", "-C", clone_dir, "sparse-checkout", "set", "data"], check=True)
    for subdir in ["logs", "attacked"]:
        src = os.path.join(clone_dir, "data", subdir)
        dst = os.path.join(out_dir, subdir)
        if os.path.isdir(src) and not os.path.exists(dst):
            shutil.copytree(src, dst)
            print(f"  Copied {subdir}/")
    print(f"RAS-Eval data available at {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["cx_cmu", "atbench", "ras_eval", "all"],
                        required=True)
    args = parser.parse_args()

    if args.dataset in ("cx_cmu", "all"):
        print("Downloading cx-cmu/agent_trajectories...")
        download_cx_cmu()
    if args.dataset in ("atbench", "all"):
        print("Downloading ATBench...")
        download_atbench()
    if args.dataset in ("ras_eval", "all"):
        print("Downloading RAS-Eval...")
        download_ras_eval()

    print("\nDone! Next: python scripts/prepare_traces.py --source <dataset>")


if __name__ == "__main__":
    main()
