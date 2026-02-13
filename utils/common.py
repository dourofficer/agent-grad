import os
import json
import re
import yaml
from tqdm import tqdm
from utils.vllm import send_request
from rich.console import Console
from rich.markdown import Markdown
from typing import Any, Dict, List, Optional, Tuple

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
            
    return labels

def _quick_vllm(prompt):
    config = {
        "model": "openai/gpt-oss-20b",
        "temperature": 0.6,
        "max_tokens": 4000,
        "reasoning_effort": "low",
        "hostname": "localhost",
        "port": 8881,
        "concurrent_requests": 16
    }
    hostname = config.pop("hostname")
    port     = config.pop("port")
    config.pop("concurrent_requests", None)
    messages = [{'role': 'user', 'content': prompt}]

    url = f"http://{hostname}:{port}/v1/chat/completions"
    _, out, _ = send_request(url, config, {"messages": messages}, request_id=0)
    return out['response']


def _call_vllm(messages: list, config_path: str) -> Dict[str, str]:
    """Send a messages list to vLLM and return {reasoning, response}."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    hostname = config.pop("hostname")
    port     = config.pop("port")
    config.pop("concurrent_requests", None)

    url = f"http://{hostname}:{port}/v1/chat/completions"
    _, out, _ = send_request(url, config, {"messages": messages}, request_id=0)
    return {"reasoning": out["reasoning"], "response": out["response"]}