"""
Two-phase agent-grad inference pipeline.

Phase 1  --prepare  Build the dependency graph, render all backward prompts,
                    and save a skeleton JSON per example (steps + prompt-only logs).
Phase 2  --process  Load the skeleton, run each log entry through vLLM, write
                    reasoning/response back, and accumulate grad/attribution on steps.

Example usage:
    # Phase 1: build prompt skeletons
    python -m cli.backprop --prepare \
        --config configs/gpt-oss-20b.yaml \
        --input  data/ww/hand-crafted \
        --output outputs/gpt-oss-20b/agent-grad/hand-crafted \
        --start_idx 0 --end_idx 2

    # Phase 2: run inference and fill in responses
    python -m cli.backprop --process \
        --config configs/gpt-oss-20b.yaml \
        --input  data/ww/hand-crafted \
        --output outputs/gpt-oss-20b/text-grad/hand-crafted

    # Both phases in one go (default when neither flag is given)
    python -m cli.backprop \
        --config configs/gpt-oss-20b.yaml \
        --input  data/ww/hand-crafted \
        --output outputs/gpt-oss-20b/text-grad/hand-crafted \
        --start_idx 0 --end_idx 10
"""

import re
import json
import yaml
import argparse
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_grad import Graph, agent_step, compute_loss, register_backward_template
from utils.vllm import send_request
from utils.graph import MagenticOneTrajectoryParser
from utils.prompting import _get_sorted_json_files, _load_json_data, _extract_metadata


# ============================================================
# Prompt builder & response parser  (severity-free)
# ============================================================

def build_generic_backward_prompt(
    problem: str,
    ground_truth: str,
    output: Tuple[str, int, str], # (role, idx, value)
    inputs: List[Tuple[str, int, str, bool]],  # (role, idx, value, requires_grad)
    downstream_grad: str,
) -> str:
    """
    Build a backward prompt that asks for attribution + criticism only (no severity).
    The model must return one JSON key per input step.
    """
    output_role, output_idx, output_value = output
    INPUTS = "\n---\n".join(
        f"**Input {i}** — Step {inp_idx} ({inp_role}) - Taking gradient: {inp_requires_grad}.\n{inp_value}"
        for i, (inp_role, inp_idx, inp_value, inp_requires_grad) in enumerate(inputs)
    )

    json_template = {
        f"step_{inp_idx}": {
            "attribution": "ORIGINATING_ERROR | PROPAGATING_ERROR | NEITHER",
            "criticism":   "Your detailed analysis here.",
        }
        for (_, inp_idx, _, requires_grad) in inputs
        if requires_grad
    }
    EXAMPLE_OUTPUT = json.dumps(json_template, indent=2)

    return f"""You are analyzing a failure in a multi-agent system.

## Context
PROBLEM: {problem}
GROUND TRUTH: {ground_truth}

## Output Under Analysis
Agent "{output_role}" (Step {output_idx}) produced:
```
{output_value}
```

## Inputs That Led to This Output
{INPUTS}

## Downstream Criticism
The following criticism was identified for the output of Step {output_idx}:
```
{downstream_grad}
```

## Your Task
For EACH input step listed above that takes gradient, determine how it contributed to the output's failure.

**ATTRIBUTION guide**:
- **ORIGINATING_ERROR**: The error was created in this step. If this step were corrected, \
downstream failures would likely be prevented.
- **PROPAGATING_ERROR**: This step forwarded an error that originated earlier. It did not \
create the mistake, but it failed to catch or correct it.
- **NEITHER**: This step is correct, or any errors appeared only in later steps.

**CRITICISM guide**:
- For ORIGINATING_ERROR or PROPAGATING_ERROR: explain (1) how this step caused or forwarded \
the problem, and (2) what a correct version of this step would look like.
- For NEITHER: briefly confirm why the step is correct in this context.

## Required Output Format
Respond with ONLY a valid JSON object — no preamble, no markdown fences:
{EXAMPLE_OUTPUT}""".strip()


def parse_backward_response(
    response: str,
    inputs: List[Tuple[str, int, str]],  # (role, idx, value, requires_grad)
) -> Dict[int, Dict[str, str]]:
    """
    Parse the LLM backward response into {step_idx: {attribution, criticism}}.
    Falls back to regex when the response is not valid JSON.
    """
    VALID_ATTRIBUTIONS = {"ORIGINATING_ERROR", "PROPAGATING_ERROR", "NEITHER"}
    results: Dict[int, Dict[str, str]] = {}
    active_inputs = [x for x in inputs if x[-1] is True] # only requires_grad = True

    # --- attempt JSON parse ---
    text = response.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match: text = fence_match.group(1)

    parsed_json: Optional[dict] = None
    try: parsed_json = json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try: parsed_json = json.loads(brace_match.group(0))
            except json.JSONDecodeError: pass

    if parsed_json is not None:
        for (_, inp_idx, _, _) in active_inputs:
            entry = parsed_json.get(f"step_{inp_idx}", {})
            attribution = str(entry.get("attribution", "UNKNOWN")).upper().strip()
            # if attribution not in VALID_ATTRIBUTIONS:
            #     attribution = "UNKNOWN"
            results[inp_idx] = {
                "attribution": attribution,
                "criticism":   str(entry.get("criticism", "")).strip(),
            }
        return results

    # --- fallback ---
    for (_, inp_idx, _, _) in active_inputs:
        results[inp_idx] = {
            "attribution": text, 
            "criticism": text
        }
    return results


# ============================================================
# Phase 1 — load_and_prepare_data
# ============================================================

def prepare_example(example: dict, role_id: str = "role") -> dict:
    """
    Build the dependency graph and render all backward-pass prompts for one example.

    Returns a skeleton dict:
        {
            'metadata': {...},
            'steps': [
                {
                    'step_idx': int,
                    'input_steps':  [int, ...],
                    'output_steps': [int, ...],
                    'role':    str,
                    'content': str,
                    'requires_grad': bool,
                    'grad':        [],   # filled during Phase 2
                    'attribution': [],   # filled during Phase 2
                }
            ],
            'logs': [
                {
                    'output_step_idx': int,
                    'input_step_idxs': [int, ...],
                    'messages': [...],  # fully rendered for log[0]; placeholder for rest
                    # 'reasoning', 'response', 'results' added during Phase 2
                }
            ]
        }

    `logs` are ordered in backward-pass sequence (same order process_example iterates).
    The first entry's prompt contains the seeded loss gradient.  Subsequent entries use
    a `{downstream_grad}` placeholder because those gradients are not yet available —
    process_example re-renders the prompt live with accumulated gradients.
    """
    metadata     = _extract_metadata(example)
    trajectory   = example["history"]
    problem      = metadata["question"]
    ground_truth = example["ground_truth"]

    # dependency graph
    parser       = MagenticOneTrajectoryParser(dependency_mode="structural")
    events       = parser.parse_trajectory(trajectory)
    dependencies = parser.build_dependency_graph(events)   # {idx: [pred_idxs]}

    successors: Dict[int, List[int]] = {i: [] for i in range(len(trajectory))}
    for idx, preds in dependencies.items():
        for p in preds:
            successors[p].append(idx)

    # build Tensor nodes (forward-pass only, no llm_fn)
    nodes = []
    for i, step in enumerate(trajectory):
        predecessors = [nodes[j] for j in dependencies[i]]
        node = agent_step(
            inputs=predecessors,
            output_value=step["content"],
            output_role=step[role_id],
            output_idx=i,
            problem=problem,
            ground_truth=ground_truth,
            llm_fn=None,
        )
        if i == 0:
            node.requires_grad = False
        nodes.append(node)

    # seed the loss gradient on the final node
    graph = Graph(problem=problem, ground_truth=ground_truth)
    graph.nodes = nodes
    graph.set_loss(nodes[-1])

    initial_criticism = compute_loss(
        final_output=nodes[-1],
        expected=ground_truth,
        problem=problem,
    )
    templates = graph.linearize()
    templates[0]["output_node"].grad = [{'from': 'output_loss', 'content': initial_criticism}]

    # steps
    steps = [
        {
            "step_idx":     node.step_idx,
            "input_steps":  list(dependencies[node.step_idx]),
            "output_steps": list(successors[node.step_idx]),
            "role":         node.role,
            "content":      node.value,
            "requires_grad": node.requires_grad,
            "grad":         [],
            "attribution":  [],
        }
        for node in nodes
    ]

    # logs — one per backward template, in backward order
    logs = []
    for t_idx, template in enumerate(templates):
        output_node = template["output_node"]
        input_nodes = template["input_nodes"]
        input_info  = template["input_info"]   # [(role, idx, value, requires_grad)]
        output_info = (output_node.role, output_node.step_idx, output_node.value)

        # Only the first template has a real gradient at prepare-time.
        # The rest will be re-rendered in process_example with live accumulated grads.
        if t_idx == 0:
            grad_contents = [g['content'] for g in output_node.grad]
            gradient = "\n---\n".join(grad_contents)
        else:
            gradient = "{downstream_grad}"

        prompt = build_generic_backward_prompt(
            problem=problem,
            ground_truth=ground_truth,
            output=output_info,
            inputs=input_info,
            downstream_grad=gradient,
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ]

        logs.append({
            "output_step_idx": output_node.step_idx,
            "input_step_idxs": [n.step_idx for n in input_nodes],
            "messages":        messages,
        })

    return {"metadata": metadata, "steps": steps, "logs": logs}


def load_and_prepare_data(
    input_dir: str,
    output_dir: Path,
    role_id: str = "role",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> None:
    """
    Phase 1: iterate over raw input files, build skeletons, and save to output_dir.
    Existing files are skipped to support resuming.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepaths = _get_sorted_json_files(input_dir)

    end_idx = end_idx or len(filepaths)
    subset  = filepaths[start_idx:end_idx]

    print(f"[prepare] {len(subset)} examples  ({start_idx}–{end_idx - 1})")
    print(f"[prepare] output → {output_dir}\n")

    for i, filename in enumerate(tqdm(subset, desc="Preparing")):
        output_path = output_dir / filename
        # if output_path.exists():
        #     tqdm.write(f"[{start_idx + i}] skip (exists): {filename}")
        #     continue

        example = _load_json_data(Path(input_dir) / filename)
        skeleton = prepare_example(example, role_id=role_id)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(skeleton, f, indent=4, ensure_ascii=False)
        tqdm.write(f"[{start_idx + i}] ✓ prepared: {filename}")


# ============================================================
# Phase 2 — process
# ============================================================

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


def process_example(data: dict, config_path: str) -> dict:
    """
    Phase 2 for a single skeleton file.

    Iterates over `logs` in their stored backward order.  For each log entry:
      1. Re-renders the prompt with the current accumulated gradient of the
         output node (so the actual gradient text is always live, not stale).
      2. Calls vLLM.
      3. Writes reasoning / response / results back into the log entry.
      4. Accumulates criticism / attribution into the relevant input steps.

    Already-completed log entries (those that already carry a `response` key)
    are replayed in-memory so later prompts receive correct gradient state,
    but are not re-sent to vLLM — this makes the function safe to call on a
    partially-processed file.
    """
    data     = deepcopy(data)
    steps    = data["steps"]
    logs     = data["logs"]
    metadata = data["metadata"]

    problem      = metadata["question"]
    ground_truth = metadata["ground_truth"]

    # step_map: step_idx → mutable step dict
    step_map: Dict[int, dict] = {s["step_idx"]: s for s in steps}

    # Lightweight node objects to track accumulated grad state across log entries.
    # We rebuild them in forward order so _prev links are available if needed.
    nodes: Dict[int, Any] = {}
    for step in steps:
        predecessors = [nodes[j] for j in step["input_steps"] if j in nodes]
        node = agent_step(
            inputs=predecessors,
            output_value=step["content"],
            output_role=step["role"],
            output_idx=step["step_idx"],
            problem=problem,
            ground_truth=ground_truth,
            llm_fn=None,
        )
        # restore any previously accumulated state
        node.requires_grad = step.get("requires_grad", True)
        node.grad          = list(step["grad"])
        node.attribution   = list(step["attribution"])
        nodes[step["step_idx"]] = node

    # Seed the loss gradient onto the first log's output node
    # (mirrors what prepare_example did, but now using the live node object)
    if logs:
        first_out_idx = logs[0]["output_step_idx"]
        initial_criticism = compute_loss(
            final_output=nodes[first_out_idx],
            expected=ground_truth,
            problem=problem,
        )
        if not nodes[first_out_idx].grad:
            nodes[first_out_idx].grad = [{
                'from': 'output_loss', 
                'content': initial_criticism}
            ]

    for log in logs:
        out_idx     = log["output_step_idx"]
        inp_idxs    = log["input_step_idxs"]
        out_node    = nodes[out_idx]
        input_nodes = [nodes[i] for i in inp_idxs]
        output_info = (out_node.role, out_node.step_idx, out_node.value)
        input_info  = [(n.role, n.step_idx, n.value, n.requires_grad) for n in input_nodes]

        # Re-render the prompt with live gradient (always up-to-date)
        grad_contents = [g['content'] for g in out_node.grad]
        gradient = "\n---\n".join(grad_contents) if grad_contents else "No gradient available."
        prompt = build_generic_backward_prompt(
            problem=problem,
            ground_truth=ground_truth,
            output=output_info,
            inputs=input_info,
            downstream_grad=gradient,
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ]

        # Call vLLM
        result = _call_vllm(messages, config_path)

        # Write back into log entry
        log["messages"]  = messages   # updated with live gradient
        log["reasoning"] = result["reasoning"]
        log["response"]  = result["response"]

        # Parse and distribute gradients
        parsed = parse_backward_response(result["response"], input_info)
        log["results"] = {f"step_{k}": v for k, v in parsed.items()}

        for inp_node in input_nodes:
            if inp_node.step_idx in parsed:
                entry = parsed[inp_node.step_idx]
                new_grad = {'from': out_node.step_idx, 'content': entry['criticism']}
                inp_node.grad.append(new_grad)
                new_attr = {'from': out_node.step_idx, 'content': entry['attribution']}
                inp_node.attribution.append(new_attr)

    # Flush accumulated node state back into the step dicts
    for step in steps:
        node = nodes[step["step_idx"]]
        step["grad"]        = node.grad
        step["attribution"] = node.attribution

    return data


def process_all(
    output_dir: Path,
    config_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> None:
    """
    Phase 2: iterate over skeleton files in output_dir, run inference, save in-place.
    Files where every log already has a response are skipped.
    """
    filepaths = _get_sorted_json_files(str(output_dir))

    end_idx = end_idx or len(filepaths)
    subset  = filepaths[start_idx:end_idx]

    print(f"[process] {len(subset)} examples  ({start_idx}–{end_idx - 1})")
    print(f"[process] config  → {config_path}")
    print(f"[process] output  → {output_dir}\n")

    for i, filename in enumerate(tqdm(subset, desc="Processing")):
        file_path = output_dir / filename
        if not file_path.exists():
            tqdm.write(f"[{start_idx + i}] ✗ missing skeleton: {filename}")
            continue

        data = _load_json_data(file_path)
        logs = data.get("logs", [])

        if logs and all("response" in log for log in logs):
            tqdm.write(f"[{start_idx + i}] skip (complete): {filename}")
            continue

        completed = process_example(data, config_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(completed, f, indent=4, ensure_ascii=False)
        tqdm.write(f"[{start_idx + i}] ✓ processed: {filename}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-phase agent-grad inference  (--prepare / --process / both)"
    )
    parser.add_argument(
        "--prepare", action="store_true",
        help="Phase 1: build dependency graph and save prompt skeletons",
    )
    parser.add_argument(
        "--process", action="store_true",
        help="Phase 2: call vLLM and fill reasoning/response/grad",
    )
    parser.add_argument("--config", required=True,
                        default="configs/gpt-oss-20b.yaml")
    parser.add_argument("--input",  default="data/who-and-when/Hand-Crafted",
                        help="Input directory with raw trajectory JSON files "
                             "(required for --prepare)")
    parser.add_argument("--output", default="outputs/gpt-oss-20b/text-grad/hand-crafted",
                        help="Output directory for skeleton / completed JSON files")
    parser.add_argument(
        "--role_id", choices=["name", "role"], default="role",
        help="Trajectory field for agent role "
             "('role' for hand-crafted, 'name' for alg-generated)",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx",   type=int, default=None)
    args = parser.parse_args()

    # Register new prompt template
    register_backward_template('generic', build_generic_backward_prompt)

    # Default: run both phases when neither flag is explicitly set
    run_prepare = args.prepare or (not args.prepare and not args.process)
    run_process = args.process or (not args.prepare and not args.process)

    output_dir = Path(args.output)

    if run_prepare:
        load_and_prepare_data(
            input_dir=args.input,
            output_dir=output_dir,
            role_id=args.role_id,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

    if run_process:
        process_all(
            output_dir=output_dir,
            config_path=args.config,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )


if __name__ == "__main__":
    main()