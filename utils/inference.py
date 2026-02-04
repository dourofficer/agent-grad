import os
import argparse
import sys
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Import the new utils functions
# Assumes new_utils.py is in the same directory
from utils.prompting import (
    get_prompts_all_at_once,
    get_prompts_step_by_step,
    get_prompts_binary_search,
    get_prompts_textual_grad,
    parse_llm_json_output
)
from utils.vllm import run_inference


def flatten(prompts_data):
    flattened = []
    for entry in prompts_data:
        prefix = {
            "file": entry.get("file"),
            "method": entry.get("method"),
            "model_type": entry.get("model_type"),
            "chat_history": entry.get("chat_history"),
            "labels": entry.get("labels")
        }
        for step_prompt in entry.get("step_prompts"):
            flattened.append({
                **step_prompt,
                **prefix,
            })
    return flattened

def unflatten(responses_data):
    unflattened = {}
    for entry in responses_data:
        filename = entry.get("file")
        if filename not in unflattened:
            unflattened[filename] = {
                "file": filename,
                "method": entry.get("method"),
                "model_type": entry.get("model_type"),
                "chat_history": entry.get("chat_history"),
                "labels": entry.get("labels"),
                "responses": []
            }
        unflattened[filename]["responses"].append({
            "step": entry.get("step"),
            "messages": entry.get("messages"),
            "reasoning": entry.get("reasoning"),
            "response": entry.get("response"),
            # "parsed": parse_llm_json_output(entry.get("response"))
        })
    return list(unflattened.values())

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

def get_output_filepath(base_dir, method, is_handcrafted):
    """Generate output filepath with consistent naming."""
    os.makedirs(base_dir, exist_ok=True)
    handcrafted_suffix = "handcrafted" if is_handcrafted else "alg_generated"
    filename = f"{method}-{handcrafted_suffix}.json"
    return os.path.join(base_dir, filename)

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search", "build_ifg", "text_grad"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="gpt-oss-20b.yaml"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/who-and-when/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs",
        help="Path to the directory containing inference results."
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        help="Azure OpenAI API Key. (Ignored for prompt generation)",
        default=None
    )

    args = parser.parse_args()

    # Derive is_handcrafted from directory path
    is_handcrafted = is_handcrafted_from_path(args.input)

    # Setup output paths
    output_dir = Path(args.output)
    prompts_filepath = get_output_filepath(output_dir / "prompts", args.method, is_handcrafted)
    responses_filepath = get_output_filepath(output_dir / "responses", args.method, is_handcrafted)

    print(f"Method: {args.method}")
    print(f"Data type: {'handcrafted' if is_handcrafted else 'algorithm-generated'}")
    print(f"Generating prompts to: {prompts_filepath}")
    print(f"Saving responses to: {responses_filepath}")
    
    # Generate prompts based on method
    method_map = {
        "all_at_once": get_prompts_all_at_once,
        "step_by_step": get_prompts_step_by_step,
        "binary_search": get_prompts_binary_search,
        "text_grad": get_prompts_textual_grad,
    }
    
    if args.method in method_map:
        prompts_data = method_map[args.method](args.input, is_handcrafted)
    else:
        raise ValueError(f"Method {args.method} not implemented for prompt generation")
        
    # Wrap in a structure that preserves metadata
    final_output = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "method": args.method,
            "input_path": args.input,
            "is_handcrafted": is_handcrafted
        },
        "prompts": prompts_data
    }

    # Save prompts
    try:
        with open(prompts_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved {len(prompts_data)} entries to {prompts_filepath}")
    except Exception as e:
        print(f"Error saving prompts: {e}")
        sys.exit(1)

    # Run inference
    flattened = flatten(prompts_data)
    results = run_inference(args.config, flattened)
    unflattened = unflatten(results)

    # Save responses
    with open(responses_filepath, "w", encoding='utf-8') as f:
        json.dump(unflattened, f, indent=4, ensure_ascii=False)
    print(f"Successfully saved responses to {responses_filepath}")


if __name__ == "__main__":
    main()