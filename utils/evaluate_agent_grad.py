import re
import json
import os
import argparse
from pathlib import Path
import pandas as pd

def compute_acc(data, k=1):
    """
    Compute accuracy@k for agent gradient attribution.
    
    Top-k is defined as the k earliest steps with ORIGINATING_ERROR attribution.
    Checks whether the ground truth mistake appears in these top-k steps.
    
    Args:
        data: List of result entries with nodes containing attribution
        k: Number of top predictions to consider (k=1 equivalent to standard accuracy)
    
    Returns:
        tuple: (agent_accuracy, step_accuracy) as percentages
    """
    assert data and 'nodes' in data[0], \
        "Data must contain 'nodes' field with attribution results."
    
    correct_agent = 0
    correct_step = 0
    total = len(data)
    files_with_predictions = 0
    
    lengths = []
    for entry in data:
        metadata = entry['metadata']
        nodes = entry['nodes']
        lengths.append(len(nodes))
        
        # Extract ground truth
        gt_agent = metadata['mistake_agent']
        gt_step = metadata['mistake_step']  # This is a string like "5"
        
        # Find all nodes with ORIGINATING_ERROR attribution
        originating_errors = []
        for node in nodes:
            # Check if any attribution in the list is ORIGINATING_ERROR
            if any(attr == 'ORIGINATING_ERROR' for attr in node.get('attribution', [])):
                originating_errors.append({
                    'step_idx': node['step_idx'],
                    'role': node['role'],
                    'suspicion_score': node.get('suspicion_score', 0.0)
                })
        
        if not originating_errors:
            # No predictions made for this entry
            continue
        
        files_with_predictions += 1
        
        # Sort by step_idx (earliest first) - this gives us chronological order
        # If you want to use suspicion_score instead, change sort key
        originating_errors.sort(key=lambda x: x['step_idx'])
        
        # Take top k earliest originating errors
        top_k = originating_errors[:k]
        
        # Check if ground truth appears in top k
        top_k_agents = [pred['role'] for pred in top_k]
        top_k_steps = [str(pred['step_idx']) for pred in top_k]
        
        if gt_agent in top_k_agents:
            correct_agent += 1
        
        if gt_step in top_k_steps:
            correct_step += 1
    
    agent_accuracy = (correct_agent / total) * 100 if total > 0 else 0.0
    step_accuracy = (correct_step / total) * 100 if total > 0 else 0.0

    print(f"\n--- Agent Gradient Accuracy@{k} ---")
    print(f"Total files: {total}")
    print(f"Files with ORIGINATING_ERROR predictions: {files_with_predictions}")
    print(f"Correct Agent in top-{k}: {correct_agent}/{total}")
    print(f"Correct Step in top-{k}: {correct_step}/{total}")
    print(f"Agent Accuracy@{k}: {agent_accuracy:.2f}%")
    print(f"Step Accuracy@{k}: {step_accuracy:.2f}%")
    print(f"average length: {sum(lengths) / len(lengths)}")

    return agent_accuracy, step_accuracy

if __name__ == "__main__":
    # Load results
    results = []
    output_dir = Path("outputs/gpt-oss-20b/graphs")
    for filepath in sorted(output_dir.glob("*.json")):
        with open(filepath) as f:
            results.append(json.load(f))

    # Compute accuracy for multiple k values
    k_values = [1, 3, 5, 10]
    reports = []
    
    for k in k_values:
        agent_acc, step_acc = compute_acc(results, k=k)
        reports.append({
            "k": k,
            "agent_acc": agent_acc,
            "step_acc": step_acc
        })
    
    # Create and display table
    df = pd.DataFrame(reports)
    df['agent@k'] = df['agent_acc'].round(2).astype(str)
    df['step@k'] = df['step_acc'].round(2).astype(str)
    
    print("\n=== Agent Gradient Results ===")
    print(df[['k', 'agent@k', 'step@k']].to_string(index=False))