import re
import json

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, fields
from utils.magentic import MAGENTIC_ONE
from utils.vllm import send_request

# ============================================================
# Prompt builder & response parser  (severity-free)
# ============================================================

@dataclass
class GradientFields:
    """Single source of truth for backward-pass output schema.
    
    To add/remove/rename a field:
      1. Edit this dataclass
      2. Update FIELD_META below
    Both prompt generation and parsing adapt automatically.
    """
    jacobian: str = ""
    gradient: str = ""
    jacobian_level: str = ""


# --- per-field metadata used by prompt builder and parser ---
FIELD_META: Dict[str, dict] = {
    "jacobian": {
        "json_hint": "How step {inp_idx} influenced step {out_idx}'s output.",
        "extract_prompt":  (
            "Extract the jacobian value for step {idx} from the following text: {text} "
            "Extract verbatim and return your answer directly without explanation, no preamble. "
            "If the information is not present, return 'UNKNOWN' only."
        ),
        "empty_default":  "UNKNOWN",
    },
    "gradient": {
        "json_hint": "How step {inp_idx} contributed to the failure and what should change.",
        "extract_prompt":  (
            "Extract the gradient value for step {idx} from the following text: {text} "
            "Extract verbatim and return your answer directly without explanation, no preamble. "
            "If the information is not present, return 'UNKNOWN' only."
        ),
        "empty_default":  "UNKNOWN",
    },
    "jacobian_level": {
        "json_hint": "ZERO | NON-ZERO",
        "extract_prompt":  (
            "Extract the jacobian_level value for step {idx} from the following text: {text} "
            "Extract verbatim and return your answer directly without explanation, no preamble. "
            "If the information is not present, return 'UNKNOWN' only."
        ),
        "empty_default":  "UNKNOWN",
    },
}

# sanity check: dataclass fields and meta keys must match
assert set(f.name for f in fields(GradientFields)) == set(FIELD_META), (
    "GradientFields and FIELD_META are out of sync"
)


def _build_output_template(
    active: List[Tuple[str, int, str]], output_idx: int
) -> dict:
    """Generate the JSON template shown in the prompt."""
    return {
        f"step_{idx}": {
            name: meta["json_hint"].format(inp_idx=idx, out_idx=output_idx)
            for name, meta in FIELD_META.items()
        }
        for _, idx, _ in active
    }


def build_backward_prompt(
    context: Dict,
    output: Tuple[str, int, str],
    inputs: List[Tuple[str, int, str, bool]],
    downstream_grad: str,
) -> str:
    problem      = context["problem"]
    ground_truth = context["ground_truth"]
    system_text  = context.get("system_description", "")

    context_block = (
        f"Problem: {problem}\n"
        f"Ground Truth: {ground_truth}\n"
        f"System:\n{system_text}"
    )
    output_role, output_idx, output_value = output
    active = [(role, idx, val) for role, idx, val, rg in inputs if rg]

    inputs_block = "\n---\n".join(
        f"**Step {idx} ({role})**\n{val}" for role, idx, val, rg in inputs
    )
    json_template = _build_output_template(active, output_idx)

    return f"""You are performing a backward pass through a failed multi-agent computation graph.

## Context
{context_block}

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

**1. Jacobian — finite-difference sensitivity across the relevant perturbation**

Define x* as the ideal version of this input: what would input step have produced had it been correct, given the problem and ground truth?

Now ask: if Step {output_idx} had received x* instead of the actual input, would its output have been different in any way that matters for the task?

- If yes: Describe what part of the output would have changed and how. The failure may be attributable (fully or partially) to the input being away from x*.
- If no: The agent's behavior is independent of this input at both the actual and the ideal operating points. The failure is localized to this step's own transformation, not to its input.

Do not ask whether the agent used the input's actual content. Ask whether the agent would have produced a better output if the input had been better. Provide your analysis in a detailed paragraph.

**2. Gradient — apply the downstream failure through the Jacobian**
The downstream gradient says how Step {output_idx}'s output needed to be different.
Use your Jacobian to push that requirement back to this input:
- If the Jacobian is non-zero for the relevant aspect: describe in a detailed paragraph what this input would have needed to contain for the agent's transformation to have produced the required output. Be specific enough that the correction is unambiguous.
- Otherwise, the failure cannot be attributed to this input. State this explicitly with detailed justification.

## Notes
- Ground your judgement with evidence presented in the content of each step. Don't make assumption if there is no evidence given.
- Always refer to steps by their indices (e.g., "Step {output_idx}", "Step N") to maintain unambiguous references. 

## Output
Respond with ONLY a valid JSON object, no preamble or markdown:
{json.dumps(json_template, indent=2)}""".strip()


def parse_backward_response(
    response: str,
    inputs: List[Tuple[str, int, str, bool]],
) -> Dict[int, Dict[str, str]]:
    """
    Parse the LLM backward response into {step_idx: {field: value}}.
    Schema-aware via FIELD_META — no manual field lists.
    """
    field_names = list(FIELD_META.keys())
    results: Dict[int, Dict[str, str]] = {}
    active_inputs = [x for x in inputs if x[-1] is True]

    # --- null response ---
    if not response:
        for (_, inp_idx, _, _) in active_inputs:
            results[inp_idx] = {
                name: meta["empty_default"] for name, meta in FIELD_META.items()
            }
        return results

    # --- attempt JSON parse ---
    text = response.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)

    parsed_json: Optional[dict] = None
    try:
        parsed_json = json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            try:
                parsed_json = json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    if parsed_json is not None:
        for (_, inp_idx, _, _) in active_inputs:
            entry = parsed_json.get(f"step_{inp_idx}", {})
            step_result = {}
            for name, meta in FIELD_META.items():
                val = str(entry.get(name, "")).strip()
                if not val:
                    val = _quick_vllm(
                        meta["extract_prompt"].format(idx=inp_idx, text=text)
                    )
                step_result[name] = val
            results[inp_idx] = step_result
        return results

    # --- llm fallback ---
    for (_, inp_idx, _, _) in active_inputs:
        results[inp_idx] = {
            name: _quick_vllm(
                meta["extract_prompt"].format(idx=inp_idx, text=text)
            )
            for name, meta in FIELD_META.items()
        }
    return results


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