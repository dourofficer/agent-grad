"""
python -m cli.evaluate --dir 'outputs/gpt-oss-20b/agent-grad/hand-crafted' --k 5
python -m cli.evaluate --sweep
python -m cli.evaluate --sweep --by_length --n_bins 4
python -m cli.evaluate --sweep --save outputs/gpt-oss-20b/sweep_results.tsv
python -m cli.evaluate \
    --sweep \
    --by_length --n_bins 3 \
    --save outputs/gpt-oss-20b_sweep-by-length.tsv
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from utils.common import _get_sorted_json_files, _load_json_data

CONFIGS = [
    "outputs/gpt-oss-20b/all-at-once/hand-crafted",
    "outputs/gpt-oss-20b/step-by-step/hand-crafted",
    'outputs/gpt-oss-20b-v2/agent-grad/hand-crafted',
    # 'outputs/gpt-oss-20b-v2/agent-grad-2/hand-crafted',
]
CONFIGS = [
    "outputs/gpt-oss-20b/all-at-once/algorithm-generated",
    "outputs/gpt-oss-20b/step-by-step/algorithm-generated",
    'outputs/gpt-oss-20b-v2/agent-grad/algorithm-generated',
]
K_VALUES = [1, 3, 5, 10, 20]


def compute_acc(dir, k=1, save_path=None):
    """Compute accuracy@k for agent and step predictions."""
    result_dir = Path(dir)
    data = []
    for filename in _get_sorted_json_files(result_dir):
        file_data = _load_json_data(result_dir / filename)
        file_data['metadata']['filename'] = filename
        data.append(file_data)

    assert data and "predictions" in data[0], \
        "Data must contain 'predictions' field. Run infer_predictions first."

    correct_agent, correct_step = 0, 0
    correct_files, failed_files = [], []

    for entry in data:
        top_k    = entry["predictions"][:k]
        label    = entry["metadata"]
        filename = label["filename"]

        agent_match = label["mistake_agent"] in [p["role"]          for p in top_k]
        step_match  = label["mistake_step"]  in [str(p["step_idx"]) for p in top_k]

        if agent_match: correct_agent += 1
        if step_match:  correct_step  += 1

        if step_match: correct_files.append(filename)
        else:          failed_files.append(filename)

    total     = len(data)
    agent_acc = (correct_agent / total) * 100
    step_acc  = (correct_step  / total) * 100

    print(f"\n--- Accuracy@{k} ---")
    print(f"Total: {total}")
    print(f"Agent: {correct_agent}/{total} ({agent_acc:.2f}%)")
    print(f"Step:  {correct_step}/{total} ({step_acc:.2f}%)")

    results = {
        "k": k, "total": total,
        "correct_agent": correct_agent, "correct_step": correct_step,
        "agent_acc": agent_acc, "step_acc": step_acc,
        "correct_files": correct_files, "failed_files": failed_files,
    }
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved to {save_path}")

    return agent_acc, step_acc


def compute_acc_by_trajectory_length(dir, k=1, n_bins=5, save_path=None):
    """Compute accuracy@k grouped by trajectory length bins, with an overall row."""
    result_dir = Path(dir)
    data = [_load_json_data(result_dir / f) for f in _get_sorted_json_files(result_dir)]

    assert data and "predictions" in data[0], \
        "Data must contain 'predictions' field. Run infer_predictions first."

    lengths = []
    for entry in data:
        legit_steps = [x for x in entry["steps"] if x["role"] != "loss"]
        lengths.append(len(legit_steps))
    # lengths = [len(e["steps"]) for e in data if e["steps"][-1].get("from") != "loss"]
    edges   = np.unique(np.percentile(lengths, np.linspace(0, 100, n_bins + 1))).tolist()
    edges[-1] = float('inf')

    def bin_label(n):
        for lo, hi in zip(edges, edges[1:]):
            if lo <= n < hi:
                return f"{int(lo)}-{int(hi)-1}" if hi != float('inf') else f"{int(lo)}+"

    groups = defaultdict(list)
    for entry in data:
        groups[bin_label(len(entry["steps"]))].append(entry)

    def _acc_row(label, entries):
        top_ks = [e["predictions"][:k] for e in entries]
        labels = [e["metadata"]        for e in entries]
        ca = sum(l["mistake_agent"] in [p["role"]          for p in t] for l, t in zip(labels, top_ks))
        cs = sum(l["mistake_step"]  in [str(p["step_idx"]) for p in t] for l, t in zip(labels, top_ks))
        n  = len(entries)
        return {"trajectory_length": label, "k": k, "total": n,
                "agent_acc": round(ca/n*100, 2), "step_acc": round(cs/n*100, 2)}

    rows = []
    print(f"\n--- Accuracy@{k} by Trajectory Length ---")
    print(f"{'Length':<12} {'Total':>6} {'Agent':>10} {'Step':>10}")

    for length, entries in sorted(groups.items(), key=lambda x: int(x[0].split('-')[0].replace('+', ''))):
        row = _acc_row(length, entries)
        rows.append(row)
        print(f"{row['trajectory_length']:<12} {row['total']:>6} {row['agent_acc']:>9.2f}% {row['step_acc']:>9.2f}%")

    all_row = _acc_row("all", data)
    rows.append(all_row)
    print(f"{'all':<12} {all_row['total']:>6} {all_row['agent_acc']:>9.2f}% {all_row['step_acc']:>9.2f}%")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"Saved to {save_path}")

    return rows


def sweep(save_path="sweep_results.tsv", by_length=False, n_bins=5):
    """Sweep over CONFIGS and K_VALUES. If by_length, break results down by trajectory length."""
    rows = []

    for dir_path in CONFIGS:
        parts    = Path(dir_path).parts
        model    = parts[1] if len(parts) > 1 else dir_path
        strategy = parts[2] if len(parts) > 2 else ""
        subset   = parts[3] if len(parts) > 3 else ""

        for k in K_VALUES:
            if by_length:
                for row in compute_acc_by_trajectory_length(dir_path, k=k, n_bins=n_bins):
                    rows.append({"model": model, "strategy": strategy, "subset": subset, **row})
            else:
                agent_acc, step_acc = compute_acc(dir_path, k=k)
                rows.append({
                    "model": model, "strategy": strategy, "subset": subset, "k": k,
                    "agent_acc": round(agent_acc, 2), "step_acc": round(step_acc, 2),
                })

    df = pd.DataFrame(rows)

    if by_length:
        pivot = df.pivot_table(
            index=["model", "strategy", "subset", "trajectory_length"],
            columns="k",
            values="step_acc",
            aggfunc="first",
        )
        pivot.columns = [f"step_acc@{k}" for k in pivot.columns]
        print("\n--- Sweep by Trajectory Length (step_acc) ---")
    else:
        pivot = df.pivot_table(
            index=["model", "strategy", "subset"],
            columns="k",
            values="step_acc",
        )
        pivot.columns = [f"step_acc@{k}" for k in pivot.columns]
        print("\n--- Sweep Results (step_acc) ---")

    print(pivot.to_string())
    df.to_csv(save_path, sep="\t", index=False)
    print(f"\nFull results saved to {save_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',       help='Directory with prediction files')
    parser.add_argument('--k',         type=int, default=1, help='Top-k accuracy (default: 1)')
    parser.add_argument('--n_bins',    type=int, default=5, help='Trajectory length bins (default: 5)')
    parser.add_argument('--save',      help='Path to save results')
    parser.add_argument('--sweep',     action='store_true', help='Run full config sweep')
    parser.add_argument('--by_length', action='store_true', help='Break down accuracy by trajectory length')
    args = parser.parse_args()

    if args.sweep:
        sweep(
            save_path=args.save or ("sweep_by_length.tsv" if args.by_length else "sweep_results.tsv"),
            by_length=args.by_length,
            n_bins=args.n_bins,
        )
    else:
        assert args.dir, "--dir is required when not sweeping"
        if args.by_length:
            compute_acc_by_trajectory_length(args.dir, k=args.k, n_bins=args.n_bins, save_path=args.save)
        else:
            compute_acc(args.dir, k=args.k, save_path=args.save)

if __name__ == '__main__':
    main()