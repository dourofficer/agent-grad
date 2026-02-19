"""
Build dependency graphs for trajectory samples.

Two modes:
  --subset hand-crafted       Use utils.graph (MagenticOneTrajectoryParser) to build
                              the structural dependency graph deterministically.
  --subset algorithm-generated  Ask the LLM to generate dependencies step-by-step,
                              mirroring the step-by-step inference flow.

Outputs are saved to:
  outputs/dependencies/{subset}/{filename}.json

Each output file has the structure:
  {
    "dependencies": {
      "0": [],
      "1": [0],
      ...
    }
  }

Example usage:
    python -m cli.build_graph \
        --subset hand-crafted \
        --input  data/ww/hand-crafted \
        --start_idx 0 --end_idx 10

    python -m cli.build_graph \
        --subset algorithm-generated \
        --input  data/ww/algorithm-generated \
        --output outputs/dependencies/algorithm-generated \
        --config configs/gpt-oss-20b.yaml \
        --start_idx 0 --end_idx 10
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml

from utils.graph import MagenticOneTrajectoryParser
from utils.common import _get_sorted_json_files, _load_json_data, _call_vllm


# ============================================================
# Hand-crafted: structural parser from utils.graph
# ============================================================

def build_graph_hand_crafted(example: dict) -> Dict[int, List[int]]:
    """
    Use MagenticOneTrajectoryParser (structural mode) to derive the
    dependency graph from the trajectory's agent communication patterns.
    """
    trajectory = example.get("history", [])
    parser = MagenticOneTrajectoryParser(dependency_mode="structural")
    events = parser.parse_trajectory(trajectory)
    dependencies = parser.build_dependency_graph(events)
    return {"dependencies": dependencies}


# ============================================================
# Algorithm-generated: LLM step-by-step dependency inference
# ============================================================

_SYSTEM_PROMPT = (
    "You are an expert at analyzing multi-agent trajectories. "
    "Your task is to determine which earlier steps a given step directly depends on. "
    "You always respond with a valid JSON object and nothing else."
)


def _build_dep_prompt(
    problem: str,
    ground_truth: str,
    system_desc: dict,
    history: List[dict],
    step_idx: int,
    prior_dependencies: Dict[int, List[int]],
) -> str:
    """
    Build the per-step dependency prompt.  Shows the full trajectory for context,
    highlights the current step, and shows the already-resolved dependency graph
    so the LLM can reason transitively if needed.
    """
    SEP = "\n---\n"
    history_text = SEP.join(
        f"STEP {i} - {entry.get('role', 'Unknown')}: {entry.get('content', '')}"
        for i, entry in enumerate(history)
    )
    system_text = SEP.join(
        f"### {k}: {v.strip('## Your role').strip()}"  for k, v in system_desc.items()
    )

    prior_deps_text = json.dumps(
        {str(k): v for k, v in prior_dependencies.items()}, indent=2
    )

    current_step = history[step_idx]
    current_role = current_step.get("role", "Unknown")
    current_content = current_step.get("content", "")

    return (
        f"## Problem\n{problem}\n\n"
        f"## Ground Truth Answer\n{ground_truth}\n\n"
        f"## Agentic System Description\n{system_text}\n\n"
        f"## Full Trajectory\n{history_text}\n\n"
        # f"## Dependencies resolved so far (step → [direct predecessors])\n"
        # f"{prior_deps_text}\n\n"
        f"## Current Step to Analyse\n"
        f"STEP {step_idx} - {current_role}: {current_content}\n\n"
        f"## Your Task\n"
        f"Identify which earlier step indices STEP {step_idx} depends on.\n"
        f"A step B depends on step A if the content of A is a necessary context or "
        f"precondition that informs the existence and/or the execution for B.\n"
        f"Return ONLY a JSON object with a single key \"dependencies\" whose value "
        f"is a list of integer step indices.\n\n"
        f"Example: {{\"dependencies\": [2, 4]}}"
    )


def _parse_dep_response(response: str, step_idx: int) -> List[int]:
    """
    Parse the LLM response for a single step's dependencies.
    Play with regex: extracts all integers from a string.
    """
    # steps[i] always depends on steps[i-1]
    default = [step_idx - 1] if step_idx - 1 >= 0 else []
    if not response or not isinstance(response, str): return default

    text = response.strip()
    nums = [int(x) for x in re.findall(r"\b(\d+)\b", text)]
    deps = list({d for d in set(default + nums) if 0 <= d < step_idx})

    return deps

def build_graph_llm(
    example: dict,
    config_path: str,
) -> Dict[int, List[int]]:
    """
    Ask the LLM to generate the dependency for each step individually,
    mirroring the step-by-step inference flow (processes steps 0..N in order,
    feeding back the already-resolved subgraph as context for each new step).
    """
    trajectory = example.get("history", [])
    problem = example.get("question", "")
    ground_truth = example.get("ground_truth", "")
    system_desc = example.get("system_prompt")

    dependencies: Dict[int, List[int]] = {}
    logs = []

    for step_idx in range(len(trajectory)):
        if step_idx == 0:
            # First step has no predecessors by definition
            dependencies[step_idx] = []
            continue

        prompt = _build_dep_prompt(
            problem=problem,
            ground_truth=ground_truth,
            system_desc=system_desc,
            history=trajectory,
            step_idx=step_idx,
            prior_dependencies=dependencies,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        result = _call_vllm(messages, config_path)
        deps = _parse_dep_response(result["response"], step_idx)
        dependencies[step_idx] = deps
        logs.append({
            "step_idx": step_idx,
            "messages": messages,
            "reasoning": result["reasoning"],
            "response": result["response"]
        })

    logs = sorted(logs, key=lambda x: x['step_idx'])
    return {"dependencies": dependencies, "logs": logs}


# ============================================================
# Serialisation helpers
# ============================================================


def _save_dependencies(deps: Dict[int, List[int]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deps, f, indent=4, ensure_ascii=False)


# ============================================================
# Main processing loop
# ============================================================

def process_all(
    input_dir: str,
    output_dir: Path,
    subset: str,
    config_path: Optional[str],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> None:
    filenames = _get_sorted_json_files(input_dir)
    end_idx = end_idx or len(filenames)
    subset_files = filenames[start_idx:end_idx]

    print(f"[build_graph] subset  : {subset}")
    print(f"[build_graph] input   : {input_dir}")
    print(f"[build_graph] output  : {output_dir}")
    print(f"[build_graph] samples : {len(subset_files)}  (idx {start_idx}–{end_idx - 1})\n")

    for i, filename in enumerate(tqdm(subset_files, desc="Building graphs")):
        output_path = output_dir / filename

        example = _load_json_data(Path(input_dir) / filename)
        if example is None:
            tqdm.write(f"[{start_idx + i}] ✗ failed to load: {filename}")
            continue

        if subset == "hand-crafted":
            deps = build_graph_hand_crafted(example)
        else:  # algorithm-generated
            deps = build_graph_llm(example, config_path)

        _save_dependencies(deps, output_path)
        tqdm.write(f"[{start_idx + i}] ✓ saved: {filename}  ({len(deps)} steps)")



def process_all_parallel(
    input_dir: str,
    output_dir: Path,
    subset: str,
    config_path: Optional[str],
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> None:
    filenames = _get_sorted_json_files(input_dir)
    end_idx = end_idx or len(filenames)
    subset_files = filenames[start_idx:end_idx]

    # Pull max_workers from config, same pattern as backprop.py
    if max_workers is None and config_path:
        with open(config_path) as f:
            max_workers = yaml.safe_load(f).get("concurrent_requests", 16)
    max_workers = max_workers or 4

    print(f"[build_graph] subset  : {subset}")
    print(f"[build_graph] input   : {input_dir}")
    print(f"[build_graph] output  : {output_dir}")
    print(f"[build_graph] workers : {max_workers}")
    print(f"[build_graph] samples : {len(subset_files)}  (idx {start_idx}–{end_idx - 1})\n")

    def _process_one(i: int, filename: str) -> str:
        example = _load_json_data(Path(input_dir) / filename)
        if example is None:
            return f"[{start_idx + i}] ✗ failed to load: {filename}"

        if subset == "hand-crafted":
            deps = build_graph_hand_crafted(example)
        else:
            deps = build_graph_llm(example, config_path)

        output_path = output_dir / filename
        _save_dependencies(deps, output_path)
        return f"[{start_idx + i}] ✓ saved: {filename}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, i, fn): fn
            for i, fn in enumerate(subset_files)
        }
        with tqdm(total=len(subset_files), desc="Building graphs") as bar:
            for future in as_completed(futures):
                tqdm.write(future.result())
                bar.update(1)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build dependency graphs for trajectory samples.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subset",
        required=True,
        choices=["hand-crafted", "algorithm-generated"],
        help="Graph construction method.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Directory of raw trajectory JSON files. "
            "Defaults to data/ww/{subset} if not specified."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Root output directory. "
            "Defaults to outputs/dependencies/{subset}."
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/gpt-oss-20b.yaml",
        help="vLLM config YAML (required for algorithm-generated).",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx",   type=int, default=None)
    args = parser.parse_args()

    # Resolve defaults
    input_dir  = args.input  or f"data/ww/{args.subset}"
    output_dir = Path(args.output or f"outputs/dependencies/{args.subset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.subset == "algorithm-generated" and args.config is None:
        parser.error("--config is required when --subset is algorithm-generated")

    # process_all(
    process_all_parallel(
        input_dir=input_dir,
        output_dir=output_dir,
        subset=args.subset,
        config_path=args.config,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )


if __name__ == "__main__":
    main()