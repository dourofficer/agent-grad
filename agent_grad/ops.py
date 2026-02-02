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


# ============================================================
# PROMPT BUILDERS
# ============================================================

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
    
    This mirrors how matmul's _backward() computes gradients for both tensor1 and tensor2
    in a single backward call, using the relationship between inputs and output.
    """
    
    # Format all inputs
    INPUTS = "\n---\n".join([
        f"**Input {i+1}** - Step {input_idx} ({input_role}):\n"
        f"```\n{input_value}\n```"
        for i, (input_role, input_idx, input_value) in enumerate(inputs)
    ])

    # Build expected output format for each input
    OUTPUT_FORMAT_LINES = []
    for (input_role, input_idx, input_value) in inputs:
        OUTPUT_FORMAT_LINES.append(f"""### GRADIENT FOR STEP {input_idx} ({input_role})
ATTRIBUTION: [INPUT_ERROR | PROCESSING_ERROR | NEITHER]
SEVERITY: [HIGH | MEDIUM | LOW | NONE]
CRITICISM: <One paragraph explaining specifically how STEP {input_idx} ({input_role} contributed to the problem, or why it didn't.>
""")
    
    OUTPUTS = "\n".join(OUTPUT_FORMAT_LINES)

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

**Classification Guide**:

- **INPUT_ERROR** (High blame): This input contains the ORIGINAL mistake. 
  - The error was created HERE, not inherited from elsewhere
  - If we fixed THIS input, downstream failures would be prevented
  - Usually only ONE input is the true source

- **PROCESSING_ERROR** (Partial blame): This input propagated an error from an EARLIER step.
  - The mistake already existed before this step
  - This step failed to catch/correct it, but didn't originate it

- **NEITHER** (No blame): This input is correct OR the error was introduced by Step {output_idx} itself (not its inputs).

**SEVERITY Guide**:
- HIGH: This step is likely the decisive error (root cause)
- MEDIUM: This step contributed but may not be the root cause  
- LOW: Minor contribution or uncertain
- NONE: Not responsible

**CRITICISM Guide**:
- If needed, only include relevant information from the output criticism of STEP {output_idx} ({output_role}).
- If your analyzed step is concluded an error step, explain how it should be changed to maximize the correctness of the trajectory.

Note: Multiple inputs may share responsibility, or the error may be purely in the processing step itself.

## Required Output Format (provide analysis for EACH input):
{OUTPUTS}""".strip()
    
    return prompt


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
# RESPONSE PARSING
# ============================================================

def parse_backward_response(response: str, inputs: List[Tuple[str, int, str]]) -> Dict[int, Dict[str, Any]]:
    """
    Parse LLM response that contains gradients for multiple inputs.
    
    Returns:
        dict mapping input_idx -> {attribution, severity, severity_score, criticism}
    """
    import re
    
    results = {}
    severity_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3, 'NONE': 0.0}
    
    for (input_role, input_idx, input_value) in inputs:
        # Find the section for this input
        pattern = rf"###\s*GRADIENT\s+FOR\s+STEP\s+{input_idx}.*?(?=###\s*GRADIENT|$)"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if match:
            section = match.group(0)
            
            # Extract attribution
            attr_match = re.search(r'ATTRIBUTION:\s*\[?([A-Z_]+)\]?', section.upper())
            attribution = attr_match.group(1) if attr_match else 'UNKNOWN'
            
            # Extract severity
            sev_match = re.search(r'SEVERITY:\s*\[?([A-Z]+)\]?', section.upper())
            severity = sev_match.group(1) if sev_match else 'UNKNOWN'
            severity_score = severity_map.get(severity, 0.0)
            
            # Extract criticism
            crit_match = re.search(r'CRITICISM:\s*(.+?)(?=\n\n|###|$)', section, re.DOTALL)
            criticism = crit_match.group(1).strip() if crit_match else section
            
            results[input_idx] = {
                'attribution': attribution,
                'severity': severity,
                'severity_score': severity_score,
                'criticism': criticism,
            }
        else:
            # Fallback if section not found
            results[input_idx] = {
                'attribution': 'UNKNOWN',
                'severity': 'UNKNOWN', 
                'severity_score': 0.0,
                'criticism': f'Could not parse gradient for step {input_idx}',
            }
    
    return results


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