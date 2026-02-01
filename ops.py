# ============================================================
# OPERATIONS - Define backward prompts for different agent types
# ============================================================
# Each operation is a prompt template for computing "textual gradient"
# The backward prompt asks: "How did the INPUT contribute to the OUTPUT's problem?"

# from core import Tensor

def build_generic_prompt(
        problem, 
        ground_truth, 
        output_role, 
        output_idx, 
        output_value,
        inputs,  # list of tuples (input_role, input_idx, input_value)
        downstream_grad
    ):
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
ATTRIBUTION: [INPUT_ERROR | PROCESSING_ERROR | BOTH | NEITHER]
SEVERITY: [HIGH | MEDIUM | LOW | NONE]
CRITICISM: <One paragraph explaining specifically how this input contributed to the problem, or why it didn't>
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

For each input, consider:
1. Did this input contain errors that propagated forward?
2. Did this input lack critical information that was needed?
3. Did this input provide misleading or incorrect guidance?
4. Or was this input fine, and the error originated in the current step's processing?

Note: Multiple inputs may share responsibility, or the error may be purely in the processing step itself.

## Required Output Format (provide analysis for EACH input):
{OUTPUTS}""".strip()
    
    return prompt


def parse_response(response: str, inputs: list) -> dict:
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

