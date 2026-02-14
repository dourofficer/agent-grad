"""
python -m cli.evaluate --dir 'outputs/gpt-oss-20b/all-at-once/hand-crafted' --k 1
python -m cli.evaluate --dir 'outputs/gpt-oss-20b/step-by-step/hand-crafted' --k 1
python -m cli.evaluate --dir 'outputs/gpt-oss-20b/text-grad/hand-crafted' --k 1
python -m cli.evaluate --dir 'outputs/gpt-oss-20b/agent-grad/hand-crafted' --k 1
python -m cli.evaluate --save outputs/gpt-oss-20b/sweep_results.tsv --sweep
"""
import json
import os
import argparse
import pandas as pd
from pathlib import Path
from utils.common import _get_sorted_json_files, _load_json_data

def compute_acc(dir, k=1, save_path=None):
    """Compute accuracy@k for agent and step predictions."""
    result_dir = Path(dir)
    json_files = _get_sorted_json_files(result_dir)
    data = []
    for filename in json_files:
        file_data = _load_json_data(result_dir / filename)
        file_data['metadata']['filename'] = filename
        data.append(file_data)
    # data = [_load_json_data(result_dir / fp) for fp in json_files]

    assert data and "predictions" in data[0], \
        "Data must contain 'predictions' field. Run infer_predictions first."
    
    correct_agent, correct_step = 0, 0
    correct_files, failed_files = [], []
    
    for entry in data:
        top_k = entry["predictions"][:k]
        label = entry["metadata"]
        filename = label["filename"]
        
        agent_match = label["mistake_agent"] in [p["role"] for p in top_k]
        step_match = label["mistake_step"] in [str(p["step_idx"]) for p in top_k]
        
        if agent_match: correct_agent += 1
        if step_match: correct_step += 1
        
        # only care about step accuracy for now
        if step_match: correct_files.append(filename)
        else: failed_files.append(filename)
    
    total = len(data)
    agent_acc = (correct_agent / total) * 100
    step_acc = (correct_step / total) * 100

    results = {
        "k"            : k,
        "total"        : total,
        "correct_agent": correct_agent,
        "correct_step" : correct_step,
        "agent_acc"    : agent_acc,
        "step_acc"     : step_acc,
        "correct_files": correct_files,
        "failed_files" : failed_files
    }

    print(f"\n--- Accuracy@{k} ---")
    print(f"Total: {total}")
    print(f"Agent: {correct_agent}/{total} ({agent_acc:.2f}%)")
    print(f"Step:  {correct_step}/{total} ({step_acc:.2f}%)")
    print(f"Correct: {len(correct_files)}")
    print(f"Failed: {len(failed_files)}")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {save_path}")

    return agent_acc, step_acc

def sweep(save_path="sweep_results.tsv"):
    """
    Sweep over hardcoded configs and k values, saving results to CSV.
    """
    CONFIGS = [
        "outputs/gpt-oss-20b/all-at-once/long-context",
        "outputs/gpt-oss-20b/step-by-step/long-context",
        'outputs/gpt-oss-20b/agent-grad/long-context',
        'outputs/gpt-oss-20b/text-grad/long-context',
        "outputs/gpt-oss-20b/all-at-once/hand-crafted",
        "outputs/gpt-oss-20b/step-by-step/hand-crafted",
        'outputs/gpt-oss-20b/agent-grad/hand-crafted',
        'outputs/gpt-oss-20b/text-grad/hand-crafted',
    ]

    K_VALUES = [1, 3, 5, 10]

    rows = []
    for dir_path in CONFIGS:
        # Parse readable labels from the path structure
        parts = Path(dir_path).parts  # e.g. ('outputs', 'gpt-oss-20b', 'all-at-once', 'hand-crafted')
        model    = parts[1] if len(parts) > 1 else dir_path
        strategy = parts[2] if len(parts) > 2 else ""
        subset   = parts[3] if len(parts) > 3 else ""

        for k in K_VALUES:
            agent_acc, step_acc = compute_acc(dir_path, k=k)
            rows.append({
                "model":      model,
                "strategy":   strategy,
                "subset":     subset,
                "k":          k,
                "agent_acc":  round(agent_acc, 2),
                "step_acc":   round(step_acc, 2),
            })

    df = pd.DataFrame(rows, columns=["model", "strategy", "subset", "k", "agent_acc", "step_acc"])

    # Pretty-print pivot: rows = configs, cols = k values (step_acc focus)
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
    parser.add_argument('--dir', help='Directory with prediction files')
    parser.add_argument('--k', type=int, default=1, help='Top-k accuracy (default: 1)')
    parser.add_argument('--save', help='Path to save results JSON')
    parser.add_argument('--sweep', action='store_true', help='Run full sweep')
    args = parser.parse_args()
    
    if args.sweep:
        sweep(args.save or "sweep_results.tsv")
    else:
        compute_acc(args.dir, args.k, args.save)

if __name__ == '__main__':
    main()