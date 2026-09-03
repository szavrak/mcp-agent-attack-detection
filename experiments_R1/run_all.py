#!/usr/bin/env python3
"""Run all four ablation experiments in one process (confound, edges, sequence, AUPRC).

On Colab (GPU) this produces the final, environment-consistent numbers for the
manuscript tables. Each sub-script writes its own JSON under experiments_R1/results/;
this runner just invokes them in order and prints a compact roll-up. Device is
auto-selected (cuda if available).
"""
import importlib
import torch


def banner(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    import traceback
    print(f"device = {'cuda' if torch.cuda.is_available() else 'cpu'}")
    failed = []
    for name in ["e_a_confound", "e_b_edges", "e_c_sequence", "e_d_auprc"]:
        banner(f"RUNNING {name}")
        try:
            mod = importlib.import_module(name)
            importlib.reload(mod)
            mod.main()
        except Exception:
            failed.append(name)
            print(f"!!! {name} FAILED — continuing to next experiment:\n{traceback.format_exc()}")
    banner("ALL DONE — results written to R1_RESULTS_DIR (or experiments_R1/results/)")
    if failed:
        print(f"FAILED experiments: {failed}")


if __name__ == "__main__":
    main()
