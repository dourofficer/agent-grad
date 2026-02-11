import os
import json
import re
from tqdm import tqdm
from rich.console import Console
from rich.markdown import Markdown

def mdprint(text):
    console = Console()
    md = Markdown(text)
    console.print(md)

def _get_sorted_json_files(directory_path):
    """Gets and sorts JSON files numerically from a directory."""
    try:
        files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        return sorted(files, key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
    except Exception as e:
        print(f"Error reading directory: {e}")
        return []

def _load_json_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def _extract_metadata(data):
    """
    Extracts metadata for the labels field.
    Includes specific requested fields and any other metadata from the dataset,
    while excluding heavy fields like history and system_prompt.
    """
    labels = {
        "question": data.get("question"),
        "ground_truth": data.get("ground_truth"),
        "is_corrected": data.get("is_corrected"),
        "mistake_agent": data.get("mistake_agent"),
        "mistake_step": data.get("mistake_step"),
        "mistake_reason": data.get("mistake_reason"),
        "mistake_type": data.get("mistake_type"),
        "question_id": data.get("question_ID"), 
        "system_description": data.get("system_prompt"),
    }
    
    # exclude_keys = set(labels.keys()) | {"history", "system_prompt", "question_ID"}
    
    # for key, value in data.items():
    #     if key not in exclude_keys:
    #         labels[key] = value
            
    return labels

def is_handcrafted_from_path(directory_path):
    """Derive whether data is handcrafted from the directory path."""
    path_lower = directory_path.lower()
    if "hand-crafted" in path_lower or "handcrafted" in path_lower:
        return True
    elif "algorithm-generated" in path_lower or "alg_generated" in path_lower:
        return False
    else:
        raise ValueError(
            f"Cannot determine if data is handcrafted from path: {directory_path}. "
            "Path should contain 'Hand-Crafted' or 'Algorithm-Generated'."
        )

# ============================================================
# ALL-AT-ONCE STRATEGY
# ============================================================

def get_prompt_all_at_once(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    SEP = "\n\n---\n\n"
    chat_content = SEP.join([
        f"STEP {i} - {entry.get('role', 'Unknown Agent')}: {entry.get('content', '')}" 
        for i, entry in enumerate(chat_history)
    ])

    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
        f"The Problem: {problem}\n"
        f"The Ground Truth Answer: {ground_truth}\n\n"
        "Task: Identify which agent made an error, the specific step number where the error occurred, and the reason for the error.\n"
        "Here is the conversation:\n\n" + chat_content + "\n\n"
        "Based on the conversation above, provide the following predictions in a strict JSON format:\n"
        "1. 'agent_name': The name of the agent responsible for the primary mistake leading to the incorrect solution. If no specific agent is clearly at fault, select the most likely candidate.\n"
        "2. 'step_number': The step number (integer) where the mistake first occurred. (e.g., if the mistake is in the second entry of the history, the step number is 2). Your step prediction must be within the range of the steps provided in the conversation.\n"
        "3. 'reason': A concise explanation of your prediction.\n\n"
        "Your response must be a valid JSON object with keys: \"agent_name\", \"step_number\", and \"reason\"."
    )
    system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step['role'],
                'content': step['content'],
            }
            for step_idx, step in enumerate(chat_history)
        ],
        'logs': [{
            'filename': None,
            'messages': messages,
            # 'reasoning': None,
            # 'response': None
        }],
    }

# ============================================================
# STEP-BY-STEP STRATEGY
# ============================================================

def get_prompt_step_by_step(data):
    chat_history = data.get("history", [])
    problem = data.get("question", "")
    ground_truth = data.get("ground_truth", "")

    metadata = _extract_metadata(data)

    SEP = "\n\n---\n\n"
    chat_content = SEP.join([
        f"STEP {i} - {entry.get('role', 'Unknown Agent')}: {entry.get('content', '')}" 
        for i, entry in enumerate(chat_history)
    ])

    logs = []
    for idx, entry in enumerate(chat_history):
        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {ground_truth}\n\n"
            f"Here is the conversation:\n\n{chat_content}\n\n" 
            f"Task: Determine whether STEP {idx} (performed by {entry.get('role')}) is an decisive error step. "
            "A decisive error step is a mistake step where, if corrected, with all following steps are adjusted accordingly, the system would succeed. "
            "When multiple mistakes exist, the earliest one is most decisive. "
            "Focus only on errors that critically derail the process, rather than minor imperfections.\n\n"
            "Your response must be a valid JSON object with the following keys:\n"
            "1. \"is_decisive\": boolean (true or false)\n"
            "2. \"reason\": string (explanation for your judgment)"
        )
        system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        logs.append({
            'filename': None,
            'step_idx': idx,
            'messages': messages,
        })

    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step['role'],
                'content': step['content'],
            }
            for step_idx, step in enumerate(chat_history)
        ],
        'logs': logs,
    }
