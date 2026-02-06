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

from typing import List, Tuple, Callable, Optional, Dict, Any
from .core import Tensor


def build_generic_backward_prompt(
    problem: str,
    ground_truth: str,
    output_role: str,
    output_idx: int,
    output_value: str,
    inputs: List[Tuple[str, int, str]],  # list of (role, idx, value)
    downstream_grad: str
) -> str:
    """
    Build a backward prompt that computes gradients for ALL inputs at once.
    Uses JSON output format for reliable parsing.
    """
    
    # Format all inputs
    INPUTS = "\n---\n".join([
        f"**Input {i+1}** - Step {input_idx} ({input_role}):\n"
        f"{input_value}\n"
        for i, (input_role, input_idx, input_value) in enumerate(inputs)
    ])

    # Build JSON template showing expected structure
    json_template = {
        f"step_{input_idx}": {
            "attribution": "ORIGINATING_ERROR | PROPAGATING_ERROR | NEITHER",
            "severity": "HIGH | MEDIUM | LOW | NONE",
            "criticism": "Your detailed analysis here"
        }
        for (input_role, input_idx, input_value) in inputs
    }
    
    import json
    EXAMPLE_OUTPUT = json.dumps(json_template, indent=2)

    TASK_GUIDE = """**ATTRIBUTION Guide**:
- **ORIGINATING_ERROR**: This step contains the ORIGINAL mistake. The error was created HERE, not inherited from previous steps. If this step were fixed, downstream failures would likely be prevented.
- **PROPAGATING_ERROR**: This step propagated an error from an EARLIER step. The mistake already existed before this step, and while this step failed to catch/correct it, it did not originate the error.
- **NEITHER**: This step is correct, or the error was introduced in later steps.

**SEVERITY Guide**:
- **HIGH**: This step is likely the decisive error (root cause)
- **MEDIUM**: This step contributed but may not be the root cause
- **LOW**: Minor contribution or uncertain
- **NONE**: Not responsible

**CRITICISM Guide**:
Provide a one-paragraph analysis based on your attribution:
- Maintain a global view from criticism of STEP {output_idx} ({output_role}).
- If **Error**: Explicitly provide explanation in two parts: (1) how this step contributed to the problem and (2) how it should be changed to maximize the correctness of the whole trajectory.
- If **Neither**: Briefly explain why the step validates as correct."""

    prompt = f"""You are analyzing a failure in a multi-agent system.

## Context
PROBLEM: {problem}
GROUND TRUTH: {ground_truth}

## The Step Being Analyzed
Agent "{output_role}" (Step {output_idx}) produced:
```
{output_value}
```

This was based on the following INPUTS:
{INPUTS}

## Downstream Issue
The following criticism was identified for the output:
```
{downstream_grad}
```

## Your Task
Analyze how EACH INPUT contributed to the OUTPUT's problem.

{TASK_GUIDE}

## CRITICAL: Output Format
You MUST respond with ONLY valid JSON in this exact structure (no additional text before or after):
```json
{EXAMPLE_OUTPUT}
```

For each step, replace the placeholder values with:
- attribution: exactly one of: ORIGINATING_ERROR, PROPAGATING_ERROR, or NEITHER
- severity: exactly one of: HIGH, MEDIUM, LOW, or NONE
- criticism: your detailed analysis as a string

OUTPUT ONLY THE JSON OBJECT. Do not include explanations, preambles, or any text outside the JSON code block.""".strip()
    
    return prompt


def parse_backward_response(response: str, inputs: List[Tuple[str, int, str]]) -> Dict[int, Dict[str, Any]]:
    """
    Parse LLM response containing gradients for multiple inputs.
    Expects JSON format but includes fallback parsing.
    
    Returns:
        dict mapping input_idx -> {attribution, severity, severity_score, criticism}
    """
    import re
    import json
    
    results = {}
    severity_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3, 'NONE': 0.0}
    
    # Try to extract JSON from response
    json_str = response.strip()
    
    # Remove markdown code blocks if present
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
    
    # Attempt JSON parsing
    try:
        parsed_json = json.loads(json_str)
        
        # Extract data for each input
        for (input_role, input_idx, input_value) in inputs:
            step_key = f"step_{input_idx}"
            
            if step_key in parsed_json:
                data = parsed_json[step_key]
                
                # Normalize attribution (handle variations)
                attribution = str(data.get('attribution', 'UNKNOWN')).upper().strip()
                if 'ORIGINAT' in attribution:
                    attribution = 'ORIGINATING_ERROR'
                elif 'PROPAGAT' in attribution:
                    attribution = 'PROPAGATING_ERROR'
                elif 'NEITHER' in attribution or 'NONE' in attribution:
                    attribution = 'NEITHER'
                
                # Normalize severity
                severity = str(data.get('severity', 'UNKNOWN')).upper().strip()
                if severity not in severity_map:
                    # Try to match partial
                    for key in severity_map:
                        if key in severity:
                            severity = key
                            break
                    else:
                        severity = 'UNKNOWN'
                
                severity_score = severity_map.get(severity, 0.0)
                criticism = str(data.get('criticism', '')).strip()
                
                results[input_idx] = {
                    'attribution': attribution,
                    'severity': severity,
                    'severity_score': severity_score,
                    'criticism': criticism,
                }
        
        # Check if we got all expected inputs
        if len(results) == len(inputs):
            return results
        
    except json.JSONDecodeError:
        pass  # Fall through to regex fallback
    
    # Fallback: Regex-based parsing for non-JSON responses
    for (input_role, input_idx, input_value) in inputs:
        if input_idx in results:
            continue  # Already parsed from JSON
        
        # Try to find any mention of this step
        patterns = [
            rf"step[\s_-]*{input_idx}[^\n]*?attribution[:\s]*([A-Z_]+)",
            rf"step[\s_-]*{input_idx}.*?attribution[:\s]*\[?([A-Z_]+)\]?",
            rf"###\s*(?:GRADIENT\s+FOR\s+)?STEP\s+{input_idx}.*?ATTRIBUTION[:\s]*\[?([A-Z_]+)\]?",
        ]
        
        attribution = 'UNKNOWN'
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                attribution = match.group(1).upper()
                break
        
        # Similar for severity
        severity = 'UNKNOWN'
        sev_patterns = [
            rf"step[\s_-]*{input_idx}[^\n]*?severity[:\s]*([A-Z]+)",
            rf"step[\s_-]*{input_idx}.*?severity[:\s]*\[?([A-Z]+)\]?",
        ]
        for pattern in sev_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                severity = match.group(1).upper()
                break
        
        severity_score = severity_map.get(severity, 0.0)
        
        # Extract criticism (any text associated with this step)
        crit_pattern = rf"step[\s_-]*{input_idx}.*?criticism[:\s]*(.+?)(?=step[\s_-]*\d+|$)"
        crit_match = re.search(crit_pattern, response, re.IGNORECASE | re.DOTALL)
        criticism = crit_match.group(1).strip() if crit_match else f"Unable to parse gradient for step {input_idx}"
        
        results[input_idx] = {
            'attribution': attribution,
            'severity': severity,
            'severity_score': severity_score,
            'criticism': criticism,
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