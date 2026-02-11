import json
import os
import pandas as pd
from pathlib import Path
from utils.prompting2 import _get_sorted_json_files, _load_json_data

def compute_acc(dir, k=1):
    """Compute accuracy@k for agent and step predictions."""
    result_dir = Path(dir)
    json_files = _get_sorted_json_files(result_dir)
    data = [_load_json_data(result_dir / fp) for fp in json_files]

    assert data and "predictions" in data[0], \
        "Data must contain 'predictions' field. Run infer_predictions first."
    
    correct_agent, correct_step = 0, 0
    
    for entry in data:
        top_k = entry["predictions"][:k]
        label = entry["metadata"]
        
        if label["mistake_agent"] in [p["role"] for p in top_k]:
            correct_agent += 1
        if label["mistake_step"] in [str(p["step_idx"]) for p in top_k]:
            
            correct_step += 1
        else:
            print(label['filename'], label['mistake_step'], [str(p["step_idx"]) for p in top_k])
    
    total = len(data)
    agent_acc = (correct_agent / total) * 100
    step_acc = (correct_step / total) * 100

    print(f"\n--- Accuracy@{k} ---")
    print(f"Total files: {total}")
    print(f"Files with predictions: {sum(1 for e in data if e['predictions'])}")
    print(f"Agent: {correct_agent}/{total} ({agent_acc:.2f}%)")
    print(f"Step:  {correct_step}/{total} ({step_acc:.2f}%)")

    return agent_acc, step_acc

if __name__ == "__main__":

    agent_acc, step_acc = compute_acc('/home/hoang/agent-grad/outputs/gpt-oss-20b/all-at-once/hand-crafted')
    print(agent_acc, step_acc)