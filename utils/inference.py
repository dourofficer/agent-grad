import os
import argparse
import sys
import json
import datetime
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
        if unflattened.get(filename) is None:
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
            "parsed": parse_llm_json_output(entry.get("response"))
        })
    unflattened = [v for k, v in unflattened.items()]
    return unflattened

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()

    # Replicate exact arguments from inference.py
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
        "--directory_path",
        type=str,
        default = "../data/who-and-when/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files."
    )

    parser.add_argument(
        "--is_handcrafted",
        type=str,
        default="False",
        choices=['True', 'False'], 
        help="Specify 'True' or 'False'. Default: 'False'."
    )

    # Arguments present for signature compatibility, but ignored in prompt generation
    parser.add_argument(
        "--api_key", type=str, 
        help="Azure OpenAI API Key. (Ignored for prompt generation)",
        default=None
    )

    args = parser.parse_args()

    # Convert is_handcrafted to boolean
    is_handcrafted_bool = True if args.is_handcrafted == "True" else False

    # Setup output
    output_dir = "outputs/prompts"
    os.makedirs(output_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if is_handcrafted_bool else "_alg_generated"
    output_filename = f"{args.method}-{handcrafted_suffix}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    responses_dir = "outputs/responses"
    os.makedirs(responses_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if is_handcrafted_bool else "_alg_generated"
    responses_filename = f"{args.method}-{handcrafted_suffix}.json"
    responses_filepath = os.path.join(responses_dir, responses_filename)

    print(f"Method: {args.method}")
    print(f"Generating prompts to: {output_filepath}")
    print(f"Saving responses to: {responses_filepath}")
    
    prompts_data = []

    if args.method == "all_at_once":
        prompts_data = get_prompts_all_at_once(args.directory_path, is_handcrafted_bool)
    elif args.method == "step_by_step":
        prompts_data = get_prompts_step_by_step(args.directory_path, is_handcrafted_bool)
    elif args.method == "binary_search":
        prompts_data = get_prompts_binary_search(args.directory_path, is_handcrafted_bool)
    elif args.method == "text_grad":
        prompts_data = get_prompts_textual_grad(args.directory_path, is_handcrafted_bool)
        
    # Wrap in a structure that preserves metadata
    metadata = {
        "timestamp": str(datetime.datetime.now()),
        "method": args.method,
        # "model": args.model,
        "directory_path": args.directory_path,
        "is_handcrafted": args.is_handcrafted
    },
    final_output = {
        "metadata": metadata,
        "prompts": prompts_data
    }

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved {len(prompts_data)} entries to {output_filepath}")
    except Exception as e:
        print(f"Error saving prompts: {e}")

    # Run inference
    flattened = flatten(prompts_data)[:]
    results = run_inference(
        args.config,
        flattened
    )
    unflattened = unflatten(results)

    with open(responses_filepath, "w") as f:
        json.dump(unflattened, f, indent=4, ensure_ascii=False)

    ## parse
    

if __name__ == "__main__":
    main()