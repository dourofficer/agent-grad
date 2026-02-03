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

def parse_llm_json_output(response_text):
    """
    Parses a JSON response from an LLM, handling Markdown code blocks 
    and common formatting issues.
    """
    try:
        # Strip whitespace
        text = response_text.strip()
        
        # Remove markdown code blocks if present (e.g., ```json ... ```)
        if "```" in text:
            pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = match.group(1)
        
        # Attempt to parse
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: Try to find the first '{' and last '}'
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
        except:
            pass
        print(f"Failed to parse JSON response: {response_text[:100]}...")
        return None

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
    
    exclude_keys = set(labels.keys()) | {"history", "system_prompt", "question_ID"}
    
    for key, value in data.items():
        if key not in exclude_keys:
            labels[key] = value
            
    return labels

def get_prompts_all_at_once(directory_path, is_handcrafted, model_type=None):
    """Generates prompts for the all-at-once method."""
    prompts_data = []
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Generating All-at-Once Prompts ({model_type})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history: continue

        labels = _extract_metadata(data)

        SEP = "\n\n---\n\n"
        chat_content = SEP.join([
            f"STEP {i+1} - {entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for i, entry in enumerate(chat_history)
        ])

        prompt = (
            "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {ground_truth}\n\n"
            "Task: Identify which agent made an error, the specific step number where the error occurred, and the reason for the error.\n"
            "Here is the conversation:\n\n" + chat_content + "\n\n"
            "Based on the conversation above, provide the following predictions in a strict JSON format:\n"
            "1. 'agent_name': The name of the agent responsible for the primary mistake leading to the incorrect solution. If no specific agent is clearly at fault, select the most likely candidate.\n"
            "2. 'step_number': The step number (integer) where the mistake first occurred. (e.g., if the mistake is in the second entry of the history, the step number is 2).\n"
            "3. 'reason': A concise explanation of your prediction.\n\n"
            "Your response must be a valid JSON object with keys: \"agent_name\", \"step_number\", and \"reason\"."
        )
        system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        prompts_data.append({
            "file": json_file,
            "method": "all_at_once",
            "model_type": model_type,
            # "messages": messages,
            "step_prompts": [{
                "file": json_file,
                "step": None,
                "messages": messages
            }],
            "chat_history": chat_history,
            "labels": labels
        })
    
    return prompts_data

def get_prompts_step_by_step(directory_path, is_handcrafted, model_type=None):
    """Generates prompts for the step-by-step method."""
    prompts_data = []
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Generating Step-by-step Prompts ({model_type})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history: continue

        labels = _extract_metadata(data)

        SEP = "\n\n---\n\n"
        chat_content = SEP.join([
            f"STEP {i} - {entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for i, entry in enumerate(chat_history)
        ])
        step_prompts = []
        for idx, entry in enumerate(chat_history):
            prompt = (
                "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
                f"The Problem: {problem}\n"
                f"The Ground Truth Answer: {ground_truth}\n\n"
                f"Here is the conversation:\n\n{chat_content}\n\n" 
                f"Task: Determine whether STEP {idx} (performed by {entry.get(index_agent)}) is an decisive error step. "
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
            step_prompts.append({
                "file": json_file,
                "step": idx,
                "messages": messages
            })

        prompts_data.append({
            "file": json_file,
            "method": "step_by_step",
            "model_type": model_type,
            "step_prompts": step_prompts,
            "chat_history": chat_history,
            "labels": labels
        })
    
    return prompts_data

def get_prompts_binary_search(directory_path, is_handcrafted, model_type=None):
    """
    Generates the initial (root) prompt for the binary search method.
    """
    prompts_data = []
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Generating Binary Search Prompts ({model_type} - Root Only)"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        answer = data.get("ground_truth", "")

        if not chat_history: continue
        
        labels = _extract_metadata(data)

        start = 0
        end = len(chat_history) - 1
        
        chat_content = "\n".join([
            f"{entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}"
            for entry in chat_history
        ])

        mid = start + (end - start) // 2 
        range_description = f"from step {start} to step {end}"
        upper_half_desc = f"from step {start} to step {mid}"
        lower_half_desc = f"from step {mid + 1} to step {end}"

        prompt = (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to resolve a user query.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. "
            "Determine which half of the segment contains the single step where this crucial error occurs, leading to the failure in resolving the query.\n\n"
            f"The Problem: {problem}\n"
            f"The Ground Truth Answer: {answer}\n"
            f"Review the following conversation segment {range_description}:\n\n{chat_content}\n\n"
            f"Based on your analysis, predict whether the most critical error is located in the upper half ({upper_half_desc}) or the lower half ({lower_half_desc}).\n\n"
            "Your response must be a valid JSON object with the following keys:\n"
            "1. \"predicted_half\": string (either \"upper\" or \"lower\")\n"
            "2. \"reason\": string (brief explanation)"
        )
        system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        prompts_data.append({
            "file": json_file,
            "method": "binary_search",
            "range": f"{start}-{end}",
            "model_type": model_type,
            "messages": messages,
            "chat_history": chat_history,
            "labels": labels
        })

    return prompts_data


def get_prompts_textual_grad(directory_path, is_handcrafted, model_type=None):
    """Generates prompts for the textual-grad method."""
    prompts_data = []
    json_files = _get_sorted_json_files(directory_path)
    index_agent = "role" if is_handcrafted else "name"

    for json_file in tqdm(json_files, desc=f"Generating Textual-Grad Prompts ({model_type})"):
        file_path = os.path.join(directory_path, json_file)
        data = _load_json_data(file_path)
        if not data: continue

        chat_history = data.get("history", [])
        problem = data.get("question", "")
        ground_truth = data.get("ground_truth", "")

        if not chat_history: continue

        labels = _extract_metadata(data)

        SEP = "\n\n---\n\n"
        chat_content = SEP.join([
            f"STEP {i} - {entry.get(index_agent, 'Unknown Agent')}: {entry.get('content', '')}" for i, entry in enumerate(chat_history)
        ])
        
        TASK_TEMPLATE = (
            "**Classification Guide**:\n\n"
            "- **INPUT_ERROR**: This step contains the ORIGINAL mistake. The error was created HERE, not inherited from previous steps. If this step were fixed, downstream failures would likely be prevented.\n"
            "- **PROCESSING_ERROR**: This step propagated an error from an EARLIER step. The mistake already existed before this step, and while this step failed to catch it, it did not originate the error.\n"
            "- **NEITHER**: This step is correct, or the error was introduced in later steps.\n\n"
            "**CRITICISM Guide**:\n"
            "- If you determine this step is an error, explain specifically how it should be changed to maximize the correctness of the trajectory.\n\n"
            "## Required Output Format:\n"
            "Your response must be a valid JSON object with the following keys:\n"
            "1. \"attribution\": string (\"INPUT_ERROR\", \"PROCESSING_ERROR\", or \"NEITHER\")\n"
            "2. \"criticism\": string (One paragraph explaining how STEP {input_idx} contributed to the problem, or why it didn't.)"
        )
        
        step_prompts = []
        for idx, entry in enumerate(chat_history):
            task = TASK_TEMPLATE.format(input_idx=idx)
            prompt = (
                "You are an AI assistant tasked with analyzing a multi-agent conversation history generated during the resolution of a complex problem.\n"
                f"The Problem: {problem}\n"
                f"The Ground Truth Answer: {ground_truth}\n\n"
                f"Here is the conversation:\n\n{chat_content}\n\n" 
                f"Task: Analyze how STEP {idx} (performed by {entry.get(index_agent)}) contributed to the failure of the final output.\n\n"
                f"{task}"
            )
            system_message = "You are a helpful assistant skilled in analyzing conversations. You always respond in valid JSON format."

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]

            step_prompts.append({
                "file": json_file,
                "step": idx,
                "messages": messages
            })

        prompts_data.append({
            "file": json_file,
            "method": "step_by_step",
            "model_type": model_type,
            "step_prompts": step_prompts,
            "chat_history": chat_history,
            "labels": labels
        })
    
    return prompts_data