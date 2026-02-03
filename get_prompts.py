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
    get_prompts_textual_grad
)

# Constants from original file to maintain signature validity checks
KNOWN_GPT_MODELS = {"gpt-4o", "gpt4", "gpt4o-mini"}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-72b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | LOCAL_MODEL_ALIASES)

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate prompts for multi-agent chat history analysis (inspection mode).")

    # Replicate exact arguments from inference.py
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search", "build_ifg", "text_grad"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default = "../Who&When/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files. Default: '../Who&When/Algorithm-Generated'."
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
        "--api_key", type=str, default= " ", 
        help="Azure OpenAI API Key. (Ignored for prompt generation)"
    )
    parser.add_argument(
        "--azure_endpoint", type=str, default=" ", 
        help="Azure OpenAI Endpoint URL. (Ignored for prompt generation)"
    )
    parser.add_argument(
        "--api_version", type=str, default="2024-08-01-preview",
        help="Azure OpenAI API Version. (Ignored for prompt generation)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for GPT API response. (Ignored for prompt generation)"
    )

    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for local model inference. (Ignored for prompt generation)"
    )

    args = parser.parse_args()

    # Determine model type (gpt vs local) to select correct prompt style
    model_type = None
    if args.model in KNOWN_GPT_MODELS:
        model_type = 'gpt'
    elif args.model in LOCAL_MODEL_ALIASES:
        model_type = 'local'
    else:
        print(f"Error: Invalid model '{args.model}' specified.")
        sys.exit(1)

    # Convert is_handcrafted to boolean
    is_handcrafted_bool = True if args.is_handcrafted == "True" else False

    # Setup output
    output_dir = "outputs/prompts"
    os.makedirs(output_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if is_handcrafted_bool else "_alg_generated"
    output_filename = f"{args.method}_{args.model.replace('/','_')}{handcrafted_suffix}.json"
    output_filepath = os.path.join(output_dir, output_filename)

    print(f"Method: {args.method}")
    print(f"Model Type: {model_type} ({args.model})")
    print(f"Generating prompts to: {output_filepath}")
    
    prompts_data = []

    if args.method == "all_at_once":
        prompts_data = get_prompts_all_at_once(args.directory_path, is_handcrafted_bool, model_type)
    elif args.method == "step_by_step":
        prompts_data = get_prompts_step_by_step(args.directory_path, is_handcrafted_bool, model_type)
    elif args.method == "binary_search":
        prompts_data = get_prompts_binary_search(args.directory_path, is_handcrafted_bool, model_type)
    elif args.method == "text_grad":
        prompts_data = get_prompts_textual_grad(args.directory_path, is_handcrafted_bool, model_type)
        
    # Wrap in a structure that preserves metadata
    final_output = {
        "metadata": {
            "timestamp": str(datetime.datetime.now()),
            "method": args.method,
            "model": args.model,
            "model_type": model_type,
            "directory_path": args.directory_path,
            "is_handcrafted": args.is_handcrafted
        },
        "prompts": prompts_data
    }

    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(prompts_data)} entries to {output_filepath}")
    except Exception as e:
        print(f"Error saving prompts: {e}")

if __name__ == "__main__":
    main()