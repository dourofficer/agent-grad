import json
from utils.prompting import parse_llm_json_output
import os
import pandas as pd


def _extract_prediction(response, chat_history, method, role_idx):
    """Helper function to extract a single prediction from a response."""
    parsed = parse_llm_json_output(response.get("response"))
    response["parsed"] = parsed
    step = response.get("step")
    
    # Step-by-step method
    if method == "step_by_step":
        if parsed.get("is_decisive") and 0 <= step < len(chat_history):
            return {
                "predicted_agent": chat_history[step][role_idx],
                "predicted_step": f"{step}",
                "reason": parsed.get("reason")
            }
    
    # All-at-once method
    elif method == "all_at_once":
        if all(parsed.values()):
            return {
                "predicted_agent": parsed.get("agent_name"),
                "predicted_step": f"{parsed.get('step_number')}",
                "reason": parsed.get("reason")
            }
    
    # Text-grad method
    elif method == "text_grad":
        if all(parsed.values()) and parsed.get("attribution") == "ORIGINATING_ERROR" \
           and 0 <= step < len(chat_history):
            return {
                "predicted_agent": chat_history[step][role_idx],
                "predicted_step": f"{step}",
                "reason": parsed.get("criticism")
            }
    
    return None


def infer_predictions(data, method="step_by_step", subset="handcrafted"):
    """Adds 'predictions' field (list of all valid predictions) to each entry."""
    assert subset in ("handcrafted", "alg_generated")
    assert method in ("all_at_once", "step_by_step", "text_grad")
    
    role_idx = "role" if subset == "handcrafted" else "name"

    for entry in data:
        chat_history = entry.get("chat_history", [])
        predictions = []
        
        for response in entry.get("responses", []):
            pred = _extract_prediction(response, chat_history, method, role_idx)
            if pred:
                predictions.append(pred)
        
        entry["predictions"] = predictions
    
    return data


def add_prediction(data, method="step_by_step", subset="handcrafted"):
    """Adds 'prediction' field (first valid prediction) to each entry."""
    infer_predictions(data, method, subset)
    
    # Convert predictions list to single prediction (first or None)
    for entry in data:
        predictions = entry.pop("predictions", [])
        entry["prediction"] = predictions[0] if predictions else {
            "predicted_agent": None,
            "predicted_step": None,
            "reason": None
        }
    
    return data


def compute_acc(data, k=1):
    """Compute accuracy@k for agent and step predictions."""
    assert data and "predictions" in data[0], \
        "Data must contain 'predictions' field. Run infer_predictions first."
    
    correct_agent = correct_step = 0
    
    for entry in data:
        top_k = entry["predictions"][:k]
        label = entry["labels"]
        
        if label["mistake_agent"] in [p["predicted_agent"] for p in top_k]:
            correct_agent += 1
        if label["mistake_step"] in [p["predicted_step"] for p in top_k]:
            correct_step += 1
    
    total = len(data)
    agent_acc = (correct_agent / total) * 100
    step_acc = (correct_step / total) * 100

    # print(f"\n--- Accuracy@{k} ---")
    # print(f"Total files: {total}")
    # print(f"Files with predictions: {sum(1 for e in data if e['predictions'])}")
    # print(f"Agent: {correct_agent}/{total} ({agent_acc:.2f}%)")
    # print(f"Step:  {correct_step}/{total} ({step_acc:.2f}%)")

    return agent_acc, step_acc

if __name__ == "__main__":
    FILES = [
        "./outputs/gpt-oss-20b/responses/step_by_step-alg_generated.json",
        "./outputs/gpt-oss-20b/responses/step_by_step-handcrafted.json",
        "./outputs/gpt-oss-20b/responses/all_at_once-alg_generated.json",
        "./outputs/gpt-oss-20b/responses/all_at_once-handcrafted.json",
        "./outputs/gpt-oss-20b/responses/text_grad-handcrafted.json",
        "./outputs/gpt-oss-20b/responses/text_grad-alg_generated.json",
    ]

    # Load all data
    data = []
    for file in FILES:
        parts = file.split('/')[-1].replace('.json', '').split('-')
        with open(file) as f:
            data.append({
                "method": parts[0],
                "subset": parts[1],
                "entries": json.load(f)
            })

    # Add predictions to all entries
    data_pred = [
        {
            "method": d["method"],
            "subset": d["subset"],
            "entries": infer_predictions(
                data=d["entries"], 
                method=d["method"], 
                subset=d["subset"]
            ) 
        }
        for d in data
    ]

    # Compute accuracy for multiple k values
    k_values = [1, 3, 5, 10]
    reports = []
    
    for data_cfg in data_pred:
        method = data_cfg["method"]
        subset = data_cfg["subset"]
        entries = data_cfg["entries"]
        
        for k in k_values:
            agent_acc, step_acc = compute_acc(entries, k=k)
            reports.append({
                "method": method,
                "subset": subset,
                "k": k,
                "agent_acc": agent_acc,
                "step_acc": step_acc
            })
    
    # Create DataFrame and display
    df = pd.DataFrame(reports)
    
    # Pivot table for better readability
    print("\n=== Agent Accuracy @k ===")
    agent_pivot = df.pivot_table(
        values='agent_acc', 
        index=['method', 'subset'], 
        columns='k',
        aggfunc='first'
    )
    print(agent_pivot.round(2))
    
    print("\n=== Step Accuracy @k ===")
    step_pivot = df.pivot_table(
        values='step_acc', 
        index=['method', 'subset'], 
        columns='k',
        aggfunc='first'
    )
    print(step_pivot.round(2))
    
    # Optional: Combined view
    print("\n=== Combined Results ===")
    df['agent@k'] = df['agent_acc'].round(2).astype(str)
    df['step@k'] = df['step_acc'].round(2).astype(str)
    
    combined = df.pivot_table(
        values=['agent@k', 'step@k'],
        index=['method', 'subset'],
        columns='k',
        aggfunc='first'
    )
    print(combined)
    
    # Save to CSV
    output_path = "./outputs/gpt-oss-20b/accuracy_report.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFull report saved to: {output_path}")
