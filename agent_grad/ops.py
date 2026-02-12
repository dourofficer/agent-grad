"""
Operations for TextGrad

Each operation is analogous to a differentiable function in deep learning.
Operations return Tensors with attached _backward functions that propagate
textual gradients to their inputs.

Design pattern (mirrors simplegrad):
    def some_op(input1, input2, ...):
        out = Tensor(...)
        
        def _backward():
            # Accumulate gradients to inputs based on out.grad
            input1.grad.append(...)
            input1.suspicion_score += ...
        
        out._backward = _backward
        out._prev = {input1, input2, ...}
        return out
"""

import re
import json
from typing import List, Tuple, Callable, Optional, Dict, Any
from .core import Tensor


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
# BACKWARD TEMPLATE REGISTRY
# ============================================================

# Registry for custom backward prompt builders
_BACKWARD_TEMPLATES: Dict[str, Callable] = {
    'generic': build_generic_backward_prompt,
}


def register_backward_template(key: str, builder: Callable):
    """Register a custom backward prompt builder."""
    _BACKWARD_TEMPLATES[key.lower()] = builder


def get_backward_template(role: str) -> Callable:
    """Get the appropriate backward template builder for an agent role."""
    role_lower = role.lower()
    
    # Check for exact match first
    if role_lower in _BACKWARD_TEMPLATES:
        return _BACKWARD_TEMPLATES[role_lower]
    
    # Match by keywords
    if "orchestrat" in role_lower or "planner" in role_lower:
        return _BACKWARD_TEMPLATES.get("orchestrator", _BACKWARD_TEMPLATES["generic"])
    elif "web" in role_lower or "surf" in role_lower or "browse" in role_lower:
        return _BACKWARD_TEMPLATES.get("websurfer", _BACKWARD_TEMPLATES["generic"])
    elif "code" in role_lower or "terminal" in role_lower or "computer" in role_lower:
        return _BACKWARD_TEMPLATES.get("coder", _BACKWARD_TEMPLATES["generic"])
    elif "assistant" in role_lower:
        return _BACKWARD_TEMPLATES.get("assistant", _BACKWARD_TEMPLATES["generic"])
    
    return _BACKWARD_TEMPLATES["generic"]


# ============================================================
# OPERATIONS - Agent actions as differentiable functions
# ============================================================

def agent_step(
    inputs: List[Tensor],
    output_value: str,
    output_role: str,
    output_idx: int,
    problem: str = "",
    ground_truth: str = "",
    llm_fn: Callable[[str], str] = None,
) -> Tensor:
    """
    Generic agent step operation.
    
    This is analogous to a forward pass through a layer.
    The backward pass will propagate criticism to inputs.
    
    Args:
        inputs: List of input Tensors (previous steps this depends on)
        output_value: The textual output of this step
        output_role: The agent role (e.g., "WebSurfer")
        output_idx: Step index in trajectory
        problem: Problem statement for context
        ground_truth: Expected answer for context
        llm_fn: Optional LLM function for computing gradients during backward
    
    Returns:
        Output Tensor with _backward attached
    """
    out = Tensor(
        value=output_value,
        role=output_role,
        step_idx=output_idx,
        requires_grad=True,
    )
    
    def _backward():
        if not inputs:
            return
        
        # Get downstream gradient
        downstream_grad = "\n---\n".join(out.grad) if out.grad else "Output was incorrect."
        
        # Build input info for prompt
        input_info = [(inp.role, inp.step_idx, inp.value) for inp in inputs]
        
        # Get appropriate template builder
        template_builder = get_backward_template(output_role)
        
        # Build the backward prompt
        prompt = template_builder(
            problem=problem,
            ground_truth=ground_truth,
            output_role=output_role,
            output_idx=output_idx,
            output_value=output_value,
            inputs=input_info,
            downstream_grad=downstream_grad,
        )
        
        if llm_fn is not None:
            # Execute backward with LLM
            response = llm_fn(prompt)
            parsed = parse_backward_response(response, input_info)
            
            # Accumulate gradients to each input
            for inp in inputs:
                if inp.step_idx in parsed:
                    result = parsed[inp.step_idx]
                    inp.grad.append(result['criticism'])
                    inp.suspicion_score += result['severity_score']
        else:
            # Without LLM, just propagate the prompt as the gradient
            # This allows for manual/deferred execution
            for inp in inputs:
                inp.grad.append(f"[Backward prompt for step {inp.step_idx}]:\n{prompt}")
    
    out._backward = _backward
    out._prev = set(inputs)
    out._op_name = output_role.lower()
    
    return out


def llm_call(
    input_tensor: Tensor,
    output_value: str,
    role: str = "LLM",
    step_idx: int = 0,
    problem: str = "",
    ground_truth: str = "",
    llm_fn: Callable[[str], str] = None,
) -> Tensor:
    """
    Single-input LLM call operation.
    
    Simplified version of agent_step for single input.
    """
    return agent_step(
        inputs=[input_tensor],
        output_value=output_value,
        output_role=role,
        output_idx=step_idx,
        problem=problem,
        ground_truth=ground_truth,
        llm_fn=llm_fn,
    )


def web_search(
    query_tensor: Tensor,
    results_value: str,
    step_idx: int = 0,
    problem: str = "",
    ground_truth: str = "",
    llm_fn: Callable[[str], str] = None,
) -> Tensor:
    """
    Web search operation.
    """
    return agent_step(
        inputs=[query_tensor],
        output_value=results_value,
        output_role="WebSurfer",
        output_idx=step_idx,
        problem=problem,
        ground_truth=ground_truth,
        llm_fn=llm_fn,
    )


def code_execution(
    code_tensor: Tensor,
    output_value: str,
    step_idx: int = 0,
    problem: str = "",
    ground_truth: str = "",
    llm_fn: Callable[[str], str] = None,
) -> Tensor:
    """
    Code execution operation.
    """
    return agent_step(
        inputs=[code_tensor],
        output_value=output_value,
        output_role="CodeExecutor",
        output_idx=step_idx,
        problem=problem,
        ground_truth=ground_truth,
        llm_fn=llm_fn,
    )


def orchestrator_decision(
    context_tensors: List[Tensor],
    decision_value: str,
    step_idx: int = 0,
    problem: str = "",
    ground_truth: str = "",
    llm_fn: Callable[[str], str] = None,
) -> Tensor:
    """
    Orchestrator decision operation.
    
    Takes multiple context inputs and produces a decision.
    """
    return agent_step(
        inputs=context_tensors,
        output_value=decision_value,
        output_role="Orchestrator",
        output_idx=step_idx,
        problem=problem,
        ground_truth=ground_truth,
        llm_fn=llm_fn,
    )


# ============================================================
# LOSS OPERATIONS
# ============================================================

def textual_diff(generated: Tensor, expected: str, problem: str = "") -> Tensor:
    """
    Compute textual difference as loss.
    
    This is analogous to a loss function in deep learning.
    The resulting tensor's grad will be set during backward.
    
    Args:
        generated: The final output tensor
        expected: The expected/correct answer
        problem: The problem statement
    
    Returns:
        A loss Tensor that can have .backward() called on it
    """
    loss_value = f"""LOSS COMPUTATION:
Generated: {generated.value}
Expected: {expected}
Match: {'YES' if generated.value.strip() == expected.strip() else 'NO'}
"""
    
    loss = Tensor(
        value=loss_value,
        role="Loss",
        step_idx=generated.step_idx + 1,
        requires_grad=True,
    )
    
    # Initial criticism that will propagate backward
    initial_criticism = f"""FAILURE DETECTED

The multi-agent system attempted to solve:
{problem}

Expected answer: {expected}

Actual final output (Step {generated.step_idx}):
{generated.value}

The system FAILED to produce the correct result. 
Trace back through the reasoning chain to find where errors originated."""
    
    def _backward():
        # Propagate criticism to the generated output
        generated.grad.append(initial_criticism)
    
    loss._backward = _backward
    loss._prev = {generated}
    loss._op_name = "loss"
    
    # Pre-set the loss gradient so backward knows where to start
    loss.grad.append("This is the loss node - the final output was incorrect.")
    
    return loss