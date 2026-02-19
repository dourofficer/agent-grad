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
        --output outputs/gpt-oss-20b/agent-grad/hand-crafted

    # Both phases in one go (default when neither flag is given)
    python -m cli.backprop2 \
        --config configs/gpt-oss-20b.yaml \
        --input  data/ww/short-context \
        --output outputs/gpt-oss-20b/agent-grad/short-context \
        --start_idx 0 --end_idx 10

    python -m cli.backprop2 \
        --config configs/gpt-oss-20b.yaml \
        --input  data/ww/long-context \
        --output outputs/gpt-oss-20b/agent-grad/long-context \
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

from agent_grad import Graph, Tensor, agent_step, register_backward_template
from utils.vllm import send_request
from utils.graph import MagenticOneTrajectoryParser
from utils.common import _get_sorted_json_files, _load_json_data, _extract_metadata
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# Backpropagation logic
# ============================================================

def _toposort(graph) -> List[Tensor]:
    """Topological sort starting from loss node."""
    visited = set()
    order = []
    
    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for prev in node._prev:
            dfs(prev)
        order.append(node)
    
    loss = graph.get_loss()
    if loss:
        dfs(loss)
    else:
        for n in graph.nodes:
            dfs(n)
    
    return order

def linearize(graph) -> List[Dict[str, Any]]:
    """
    Linearize the backward pass into a list of prompt templates.
    
    Instead of recursively calling _backward(), this returns a list
    of templates that can be executed sequentially with an LLM.
    """
    
    topo = _toposort(graph)
    backward_order = list(reversed(topo))
    
    templates = []
    
    for node in backward_order:
        # Skip nodes with no predecessors (nothing to backprop to)
        if not node._prev:
            continue
        
        # Build input info
        input_nodes = list(node._prev)
        input_info = [(inp.role, inp.step_idx, inp.value, inp.requires_grad) 
                      for inp in input_nodes]
        output_info = (node.role, node.step_idx, node.value)
        # Get the template builder for this node's role
        template_builder = build_backward_prompt
        
        # Also create the template with placeholder for inspection
        prompt_template = template_builder(
            problem=graph.problem,
            ground_truth=graph.ground_truth,
            output=output_info,
            inputs=input_info,
            downstream_grad="{downstream_grad}",
        )
        
        templates.append({
            'prompt_template': prompt_template,
            'output_node': node,
            'input_nodes': input_nodes,
            'input_info': input_info,
        })
    
    return templates

# ============================================================
# Prompt builder & response parser  (severity-free)
# ============================================================
MAGENTIC_ONE = """
* Orchestrator: The lead agent, the Orchestrator is responsible for high-level \
planning, directing specialized agents, tracking progress, updating states and \
ledgers, and error recovery. As the Orchestrator operates, its planning can be \
generic or lack of specificity as long as it is not wrong.

* Assistant: This LLM-based agent is specialized in writing code, analyzing \
information collected by other agents, and creating new artifacts. It can author \
new programs and is capable of debugging its own code when provided with console \
output.

* ComputerTerminal: This agent is deterministic and does not use an LLM. Computer \
terminal performs no other action than running Python scripts (provided to it quoted \
in ```python code blocks), or sh shell scripts (provided to it quoted in ```sh code \
blocks)

* WebSurfer: This highly specialized LLM-based agent is proficient in managing a \
Chromium-based web browser. It receives natural-language requests and maps them to \
actions such as visiting URLs, performing web searches, clicking elements, typing \
into forms, and scrolling. It can also perform "reading actions" like summarizing \
content or answering questions about a document. For visual grounding, it uses \
"set-of-marks" prompting on annotated screenshots to interact with specific page \
elements. It can also be asked to sleep and wait for pages to load, in cases where \
the pages seem to be taking a while to load.

* FileSurfer: Similar in design to the WebSurfer, the FileSurfer commands a custom \
markdown-based file preview application instead of a web browser. This read-only \
application supports a wide range of file types, including PDFs, Office documents, \
images, videos, and audio. The FileSurfer can navigate folder structures and list \
directory contents to locate and process information within local files.
""".strip()

def build_backward_prompt(
    problem: str,
    ground_truth: str,
    output: Tuple[str, int, str],
    inputs: List[Tuple[str, int, str, bool]],
    downstream_grad: str,
) -> str:
    output_role, output_idx, output_value = output
    active = [(role, idx, val) for role, idx, val, rg in inputs if rg]

    inputs_block = "\n---\n".join(
        f"**Step {idx} ({role})**\n{val}" for role, idx, val, rg in inputs
    )
    json_template = {
        f"step_{idx}": {
            "jacobian": f"How step {idx} influenced step {output_idx}'s output.",
            "gradient": f"How step {idx} contributed to the failure and what should change.",
        }
        for _, idx, _ in active
    }

    return f"""You are performing a backward pass through a failed multi-agent computation graph.

## Context
Problem: {problem}
Ground truth: {ground_truth}
System: {MAGENTIC_ONE}

## Output Node — Step {output_idx} ({output_role})
```
{output_value}
```

## Input Nodes — predecessors of Step {output_idx}
{inputs_block}

## Downstream Gradient — how Step {output_idx} contributed to the failure
```
{downstream_grad}
```

## Task
For each input step, compute in order:

**1. Jacobian (dy/dx) — factual influence only, no failure judgments**
- What content from this input did Step {output_idx} use, and how (adopted directly / transformed / partially / ignored)?
- How sensitive is Step {output_idx}'s output to this input?

**2. Gradient (dL/dx) — compose jacobian with downstream gradient**
- Trace the causal chain: which content from this input flowed into the problematic aspects of Step {output_idx}'s output identified by the downstream gradient?
- If the input contributed to the failure: describe what a correct version would contain — be specific enough that the correction is unambiguous. Account for the agent's actual responsibility (e.g., an Orchestrator giving underspecified instructions is not an error; misinterpreting its inputs is).
- If the input did not contribute to the failure: state this explicitly with a justification.

## Notes
- Ground your judgement with evidence presented in the content of each step. Don't make assumption if there is no evidence given.
- Always refer to steps by their indices (e.g., "Step {output_idx}", "Step N") to maintain unambiguous references. 

## Output
Respond with ONLY a valid JSON object, no preamble or markdown:
{json.dumps(json_template, indent=2)}""".strip()


def initial_loss(final_output: Tensor, expected: str, problem: str) -> str:
    """
    Compute initial "loss" as textual criticism.
    This is the starting gradient for backpropagation.
    """
    return f"""FAILURE DETECTED

The multi-agent system attempted to solve:
{problem}

Expected answer: {expected}

The system FAILED to produce the correct result. 
Trace back through the reasoning chain to find where errors originated."""


def parse_backward_response(
    response: str,
    inputs: List[Tuple[str, int, str]],  # (role, idx, value, requires_grad)
) -> Dict[int, Dict[str, str]]:
    """
    Parse the LLM backward response into {step_idx: {attribution, criticism}}.
    Falls back to regex when the response is not valid JSON.
    """
    gradient_tmpl = (
        "Extract the gradient value for step {idx} from the following text: {text} "
        "Extract verbatim and return your answer directly without explanation, no preamble. "
        "If the information is not present, return 'UNKNOWN' only."
    )
    jacobian_tmpl = (
        "Extract the jacobian value for step {idx} from the following text: {text} "
        "Extract verbatim and return your answer directly without explanation, no preamble. "
        "If the information is not present, return 'UNKNOWN' only."
    )
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
            jacobian = str(entry.get('jacobian')).strip()
            if not jacobian:
                jacobian = _quick_vllm(jacobian_tmpl.format(idx=inp_idx, text=text))
            gradient = str(entry.get("gradient")).strip()
            if not gradient:
                gradient = _quick_vllm(gradient_tmpl.format(idx=inp_idx, text=text))

            results[inp_idx] = {
                "jacobian": jacobian,
                "gradient": gradient
            }
        return results

    # --- llm fallback ---
    for (_, inp_idx, _, _) in active_inputs:
        results[inp_idx] = {
            "jacobian": _quick_vllm(jacobian_tmpl.format(idx=inp_idx, text=text)), 
            "gradient": _quick_vllm(gradient_tmpl.format(idx=inp_idx, text=text)), 
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
        # first step is human input, no gradient.
        if i == 0:
            node.requires_grad = False
        nodes.append(node)

    # add a loss node
    loss_idx = len(trajectory)
    dependencies[loss_idx] = [loss_idx - 1]
    successors[loss_idx] = []
    loss_node = agent_step(
        inputs=[nodes[-1]],
        output_value=None,
        output_role="loss",
        output_idx=len(trajectory),
        problem=problem,
        ground_truth=ground_truth,
        llm_fn=None,
    )
    nodes.append(loss_node)

    # seed the loss gradient on the final node
    graph = Graph(problem=problem, ground_truth=ground_truth)
    graph.nodes = nodes
    graph.set_loss(nodes[-1])

    initial_criticism = initial_loss(
        final_output=nodes[-1],
        expected=ground_truth,
        problem=problem,
    )
    templates = linearize(graph)
    initial_template = templates[0]['prompt_template'].replace(
        "{downstream_grad}", initial_criticism
    )
    templates[0]['prompt_template'] = initial_template

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
        }
        for node in nodes
    ]
    steps[-1]['grad'] = [{'jacobian': None, 'gradient': initial_criticism, 'from': 'loss'}]

    # logs — one per backward template, in backward order
    logs = []
    for t_idx, template in enumerate(templates):
        output_node = template["output_node"]
        input_nodes = template["input_nodes"]
        prompt = template['prompt_template']
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
        example = _load_json_data(Path(input_dir) / filename)
        skeleton = prepare_example(example, role_id=role_id)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(skeleton, f, indent=4, ensure_ascii=False)
        tqdm.write(f"[{start_idx + i}] ✓ prepared: {filename}")


# ============================================================
# Phase 2 — process
# ============================================================

def _quick_vllm(prompt):
    config = {
        "model": "/home/hoang/resources/models/openai/gpt-oss-20b",
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


def process_example(data: dict, config_path: str) -> dict:
    data     = deepcopy(data)
    steps    = data["steps"]
    logs     = data["logs"]
    metadata = data["metadata"]

    problem      = metadata["question"]
    ground_truth = metadata["ground_truth"]

    # Lightweight node objects to track accumulated grad state across log entries.
    # We rebuild them in forward order so _prev links are available if needed.
    nodes: Dict[int, Any] = {}
    for step in steps:
        predecessors = [nodes[j] for j in step["input_steps"] if j in nodes]
        node = Tensor(
            value=step['content'],
            role=step['role'],
            step_idx=step['step_idx'],
            requires_grad=step['requires_grad'],
            grad=step['grad'],
        )
        node._prev = set(predecessors)
        nodes[step["step_idx"]] = node

    for log_idx, log in enumerate(tqdm(logs, desc=f"Backpropagating ...")):
        out_idx     = log["output_step_idx"]
        inp_idxs    = log["input_step_idxs"]
        input_nodes = [nodes[i] for i in inp_idxs]
        out_node    = nodes[out_idx]
        output_info = (out_node.role, out_node.step_idx, out_node.value)
        input_info  = [(n.role, n.step_idx, n.value, n.requires_grad) for n in input_nodes]

        template    = log["messages"][-1]["content"]


        # Re-render the prompt with live gradient (always up-to-date)
        # import pdb; pdb.set_trace()
        grad_contents = [f"Gradient from step {g['from']}: {g['gradient']}" for g in out_node.grad]
        gradient = "\n---\n".join(grad_contents) if grad_contents else "No gradient available."
        prompt = template.replace("{downstream_grad}", gradient)
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
                new_grad = {
                    'from': out_node.step_idx, 
                    'gradient': entry['gradient'],
                    'jacobian': entry['jacobian']
                }
                inp_node.grad.append(new_grad)

    # Flush accumulated node state back into the step dicts
    for step in steps:
        node = nodes[step["step_idx"]]
        step["grad"] = node.grad

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
        data = _load_json_data(file_path)
        completed = process_example(data, config_path)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(completed, f, indent=4, ensure_ascii=False)
        tqdm.write(f"[{start_idx + i}] ✓ processed: {filename}")



def process_all_parallel(
    input_dir: Path,
    output_dir: Path,
    config_path: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> None:
    """
    Phase 2: iterate over skeleton files in `output_dir`, run inference,
    and save the completed results back in-place.
    """
    filepaths = _get_sorted_json_files(str(input_dir))
    end_idx   = end_idx or len(filepaths)
    subset    = filepaths[start_idx:end_idx]
    # import pdb; pdb.set_trace()

    if max_workers is None:
        with open(config_path) as f:
            max_workers = yaml.safe_load(f).get("concurrent_requests", 8)

    print(f"[process] {len(subset)} examples  ({start_idx}–{end_idx - 1})")
    print(f"[process] config  → {config_path}")
    print(f"[process] output  → {output_dir}")
    print(f"[process] workers → {max_workers}\n")

    def _process_one(i: int, filename: str) -> str:
        file_path = output_dir / filename
        data = _load_json_data(file_path)
        completed = process_example(data, config_path)
        tmp = file_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(completed, f, indent=4, ensure_ascii=False)
        tmp.replace(file_path)
        return f"[{start_idx + i}] ✓ processed: {filename}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, i, fn): fn for i, fn in enumerate(subset)}
        with tqdm(total=len(subset), desc="Processing ") as bar:
            for future in as_completed(futures):
                tqdm.write(future.result())
                bar.update(1)

# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Two-phase agent-grad inference  (--prepare / --process / both)")
    parser.add_argument("--prepare", action="store_true", 
        help="Phase 1: build dependency graph and save prompt skeletons",)
    parser.add_argument("--process", action="store_true", 
        help="Phase 2: call vLLM and fill reasoning/response/grad",)
    parser.add_argument("--config", required=True, default="configs/gpt-oss-20b.yaml")
    parser.add_argument("--input",  default="data/ww/hand-crafted",)
    parser.add_argument("--output", default="outputs/gpt-oss-20b/text-grad/hand-crafted",)
    parser.add_argument("--role_id", choices=["name", "role"], default="role")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx",   type=int, default=None)
    args = parser.parse_args()

    # Register new prompt template
    register_backward_template('generic', build_backward_prompt)

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
        # process_all(
        process_all_parallel(
            input_dir=args.input,
            output_dir=output_dir,
            config_path=args.config,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

if __name__ == "__main__":
    main()