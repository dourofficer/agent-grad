"""
TextGrad for Multi-Agent System Failure Attribution

A minimal framework that models agent trajectories as computational graphs
and uses textual backpropagation to identify decisive error steps.

Key concepts:
- Tensor: A node holding textual content (one step in trajectory)
- Operation: Defines how an agent transforms inputs to outputs, with backward prompts
- Graph: The computational structure linking steps
- Backward pass: Propagate criticism from loss to identify error sources
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Callable


# ============================================================
# TENSOR - A node in the computational graph
# ============================================================
@dataclass
class Tensor:
    """
    A node representing one step in the trajectory.
    
    Attributes:
        value: The complete textual content of this step
        role: Agent name/type (e.g., "WebSurfer", "Orchestrator")
        step_idx: Position in original trajectory
        grad: Accumulated criticisms from backward pass
        suspicion_score: Higher = more likely to be the error source
    """
    value: str
    role: str = ""
    step_idx: int = 0
    requires_grad: bool = True
    
    # Gradient accumulation (list of criticism strings)
    grad: List[str] = field(default_factory=list)
    suspicion_score: float = 0.0
    attribution: List[str] = field(default_factory=list)
    
    # Graph structure (like simplegrad)
    _prev: Set['Tensor'] = field(default_factory=set)
    _backward: Callable = field(default=lambda: None)
    _op_name: str = ""  # Operation type that produced this
    
    def __hash__(self):
        return id(self)
    
    def __repr__(self):
        preview = self.value[:60].replace('\n', ' ') + "..." if len(self.value) > 60 else self.value
        return f"Tensor(step={self.step_idx}, role='{self.role}', score={self.suspicion_score:.2f})"
    
    def short_str(self):
        """Short representation for prompts."""
        return f"[Step {self.step_idx}] {self.role}: {self.value[:200]}..."

    def input_nodes(self):
        return [node.step_idx for node in self._prev]
    
    def backward(self, initial_grad: str = None):
        """
        Perform backward pass through the computation graph.
        Like simplegrad's backward(), but with textual gradients.
        
        Args:
            initial_grad: The starting criticism (if None, uses default loss message)
        """
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for prev in node._prev:
                    build_topo(prev)
                topo.append(node)
        
        build_topo(self)
        
        # Initialize this node's gradient if not already set
        if initial_grad is not None:
            self.grad.append(initial_grad)
        elif not self.grad:
            self.grad.append("This output was incorrect and needs analysis.")
        
        # Backward pass in reverse topological order
        for tensor in reversed(topo):
            tensor._backward()


# ============================================================
# COMPUTATION GRAPH
# ============================================================
class Graph:
    """
    Computational graph representing a trajectory.
    
    Each node is a Tensor (step), edges represent dependencies.
    """
    
    def __init__(self, problem: str = "", ground_truth: str = ""):
        self.nodes: List[Tensor] = []
        self.problem = problem
        self.ground_truth = ground_truth
        self._loss_node: Optional[Tensor] = None
    
    def add_node(
        self,
        value: str,
        role: str = "",
        step_idx: int = None,
        predecessors: List[Tensor] = None,
        op_name: str = None,
    ) -> Tensor:
        """Add a node to the graph."""
        if step_idx is None:
            step_idx = len(self.nodes)
        
        node = Tensor(
            value=value,
            role=role,
            step_idx=step_idx,
        )
        
        if predecessors:
            node._prev = set(predecessors)
        
        node._op_name = op_name or role.lower()
        self.nodes.append(node)
        return node
    
    def set_loss(self, node: Tensor):
        """Set the loss node (final output that was wrong)."""
        self._loss_node = node
    
    def get_loss(self) -> Optional[Tensor]:
        """Get the loss node."""
        return self._loss_node or (self.nodes[-1] if self.nodes else None)
    
    def _toposort(self) -> List[Tensor]:
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
        
        loss = self.get_loss()
        if loss:
            dfs(loss)
        else:
            for n in self.nodes:
                dfs(n)
        
        return order

    def linearize(self, initial_grad: str = None) -> List[Dict[str, Any]]:
        """
        Linearize the backward pass into a list of prompt templates.
        
        Instead of recursively calling _backward(), this returns a list
        of templates that can be executed sequentially with an LLM.
        
        Args:
            initial_grad: The starting criticism for the loss node
        
        Returns:
            List of dicts, each containing:
                - 'prompt_template': The prompt with {downstream_grad} placeholder
                - 'output_node': The node whose output is being analyzed
                - 'input_nodes': List of input nodes to receive gradients
                - 'format_fn': Function to format the prompt with a gradient string
        
        Usage:
            templates = graph.linearize(initial_criticism)
            gradient = initial_criticism
            for t in templates:
                prompt = t['format_fn'](gradient)
                response = llm(prompt)
                gradient = parse_and_update(response, t['input_nodes'])
        """
        from . import ops
        
        topo = self._toposort()
        backward_order = list(reversed(topo))
        
        templates = []
        
        for node in backward_order:
            # Skip nodes with no predecessors (nothing to backprop to)
            if not node._prev:
                continue
            
            # Build input info
            input_nodes = list(node._prev)
            input_info = [(inp.role, inp.step_idx, inp.value, inp.requires_grad) for inp in input_nodes]
            output_info = (node.role, node.step_idx, node.value)
            # Get the template builder for this node's role
            template_builder = ops.get_backward_template(node.role)
            
            # Create a format function that takes downstream_grad and returns the full prompt
            def make_format_fn(builder, prob, gt, output, inputs):
                def format_fn(downstream_grad: str) -> str:
                    return builder(
                        problem=prob,
                        ground_truth=gt,
                        # output_role=out_role,
                        # output_idx=out_idx,
                        # output_value=out_val,
                        output=output,
                        inputs=inputs,
                        downstream_grad=downstream_grad,
                    )
                return format_fn
            
            format_fn = make_format_fn(
                template_builder,
                self.problem,
                self.ground_truth,
                # node.role,
                # node.step_idx,
                # node.value,
                output_info,
                input_info,
            )
            
            # Also create the template with placeholder for inspection
            prompt_template = template_builder(
                problem=self.problem,
                ground_truth=self.ground_truth,
                # output_role=node.role,
                # output_idx=node.step_idx,
                # output_value=node.value,
                output=output_info,
                inputs=input_info,
                downstream_grad="{downstream_grad}",
            )
            
            templates.append({
                'prompt_template': prompt_template,
                'format_fn': format_fn,
                'output_node': node,
                'output_idx': node.step_idx,
                'input_nodes': input_nodes,
                'input_info': input_info,
            })
        
        return templates


# ============================================================
# LOSS FUNCTION - Initial criticism
# ============================================================
def compute_loss(final_output: Tensor, expected: str, problem: str) -> str:
    """
    Compute initial "loss" as textual criticism.
    This is the starting gradient for backpropagation.
    """
    return f"""FAILURE DETECTED

The multi-agent system attempted to solve:
{problem}

Expected answer: {expected}

Actual final output (Step {final_output.step_idx}):
{final_output.value}

The system FAILED to produce the correct result. 
Trace back through the reasoning chain to find where errors originated."""


# ============================================================
# RESPONSE PARSING & SCORING
# ============================================================
def parse_response(response: str, inputs: List = None) -> Dict[str, Any]:
    """
    Parse LLM response to extract attribution, severity, criticism.
    
    If inputs is provided, parse multi-input format.
    Otherwise, parse single-input format.
    """
    import re
    
    if inputs is not None:
        # Multi-input parsing
        return _parse_multi_input_response(response, inputs)
    
    # Single-input parsing (legacy)
    result = {
        'attribution': 'UNKNOWN',
        'severity': 'UNKNOWN',
        'severity_score': 0.0,
        'criticism': response,
    }
    
    # Extract attribution
    attr_match = re.search(r'ATTRIBUTION:\s*\[?([A-Z_]+)\]?', response.upper())
    if attr_match:
        result['attribution'] = attr_match.group(1)
    
    # Extract severity
    sev_match = re.search(r'SEVERITY:\s*\[?([A-Z]+)\]?', response.upper())
    if sev_match:
        result['severity'] = sev_match.group(1)
        severity_map = {'HIGH': 1.0, 'MEDIUM': 0.6, 'LOW': 0.3, 'NONE': 0.0}
        result['severity_score'] = severity_map.get(result['severity'], 0.0)
    
    # Extract criticism
    crit_match = re.search(r'CRITICISM:\s*(.+?)(?=\n\n|\Z)', response, re.DOTALL)
    if crit_match:
        result['criticism'] = crit_match.group(1).strip()
    
    return result


def _parse_multi_input_response(response: str, inputs: List) -> Dict[int, Dict[str, Any]]:
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
# SIMPLE TRAJECTORY CONVERTER
# ============================================================
def trajectory_to_graph(
    steps: List[Dict[str, str]],
    problem: str = "",
    ground_truth: str = "",
    mode: str = "linear"
) -> Graph:
    """
    Convert a trajectory to a computation graph.
    
    Args:
        steps: List of dicts with 'role' and 'content' keys
        problem: The problem statement
        ground_truth: Expected answer
        mode: "linear" (each step depends on previous)
              or "full" (each step depends on all previous)
    
    Returns:
        Graph object
    """
    graph = Graph(problem=problem, ground_truth=ground_truth)
    
    all_nodes = []
    for i, step in enumerate(steps):
        role = step.get('role', step.get('agent', 'unknown'))
        content = step.get('content', step.get('value', step.get('message', '')))
        
        if mode == "linear":
            # Linear: each step depends only on immediate predecessor
            preds = [all_nodes[-1]] if all_nodes else []
        else:
            # Full: each step depends on all previous (expensive but thorough)
            preds = all_nodes.copy()
        
        node = graph.add_node(
            value=content,
            role=role,
            step_idx=i,
            predecessors=preds,
        )
        all_nodes.append(node)
    
    # Set last node as loss
    if all_nodes:
        graph.set_loss(all_nodes[-1])
    
    return graph