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
from abc import ABC, abstractmethod
from ops import build_generic_prompt


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
    
    # Gradient accumulation
    grad: List[str] = field(default_factory=list)
    suspicion_score: float = 0.0
    
    # Graph structure
    _prev: Set['Tensor'] = field(default_factory=set)
    _op_name: str = "generic"  # Operation type that produced this
    
    def __hash__(self):
        return id(self)
    
    def __repr__(self):
        preview = self.value[:60].replace('\n', ' ') + "..." if len(self.value) > 60 else self.value
        return f"Tensor(step={self.step_idx}, role='{self.role}', score={self.suspicion_score:.2f})"
    
    def short_str(self):
        """Short representation for prompts."""
        return f"[Step {self.step_idx}] {self.role}: {self.value[:200]}..."

def get_backward_template(role):
    return build_generic_prompt

# def get_backward_template(role: str) -> str:
#     """Get the appropriate backward template for an agent role."""
#     role_lower = role.lower()
    
#     # Match by keywords
#     if "orchestrat" in role_lower or "planner" in role_lower:
#         return BACKWARD_TEMPLATES["orchestrator"]
#     elif "web" in role_lower or "surf" in role_lower or "browse" in role_lower:
#         return BACKWARD_TEMPLATES["websurfer"]
#     elif "code" in role_lower or "terminal" in role_lower or "computer" in role_lower:
#         return BACKWARD_TEMPLATES["coder"]
#     elif "assistant" in role_lower:
#         return BACKWARD_TEMPLATES["assistant"]
#     else:
#         return BACKWARD_TEMPLATES["generic"]


# def register_backward_template(key: str, template: str):
#     """Register a custom backward template."""
#     BACKWARD_TEMPLATES[key.lower()] = template


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
    
    def linearize(self, initial_criticism: str) -> List[Dict[str, Any]]:
        """
        Linearize backward pass into a list of prompt templates.
        
        This is the key function: it returns prompts you can run manually
        or feed to an LLM to perform textual backpropagation.
        
        Args:
            initial_criticism: The starting "gradient" - why the final output failed
        
        Returns:
            List of dicts with keys:
                - 'prompt': The complete prompt to send to LLM
                - 'output_node': The node whose output is being analyzed
                - 'input_node': The node whose contribution is being assessed
                - 'step_idx': Step index of the input node
        """
        topo = self._toposort()
        backward_order = list(reversed(topo))  # Process from loss backward
        
        # Initialize loss node's gradient
        loss = self.get_loss()
        if loss:
            loss.grad.append(initial_criticism)
        
        templates = []
        
        for node in backward_order:
            if not node._prev:
                continue  # Root node, nothing to backprop to
            
            # Aggregate all gradients received so far
            # downstream_grad = "\n---\n".join(node.grad) if node.grad else initial_criticism
            downstream_grad = "\n---\n".join(node.grad) if node.grad else "{downstream_grad}"
            template_func = get_backward_template(node.role)
            inputs = [(p.role, p.step_idx, p.value) for p in node._prev]
            prompt = template_func(
                problem=self.problem,
                ground_truth=self.ground_truth,
                output_value=node.value,
                output_role=node.role,
                output_idx=node.step_idx,
                inputs=inputs,
                downstream_grad=downstream_grad,
            )
            templates.append({
                'prompt': prompt,
                'output_node': node,
                'output_idx': node.step_idx,
                'input_nodes': node._prev,
                'input_idxs': (p.step_idx for p in node._prev)
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
def parse_response(response: str) -> Dict[str, Any]:
    """Parse LLM response to extract attribution, severity, criticism."""
    import re
    
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


# ============================================================
# BACKWARD EXECUTOR
# ============================================================
class BackwardExecutor:
    """
    Execute backward pass using an LLM.
    
    Usage:
        executor = BackwardExecutor(llm_fn=my_llm_function)
        results = executor.run(graph, initial_loss)
        ranked = executor.rank_nodes(graph)
    """
    
    def __init__(self, llm_fn: Callable[[str], str]):
        """
        Args:
            llm_fn: Function that takes prompt string, returns LLM response string
        """
        self.llm_fn = llm_fn
    
    def run(self, graph: Graph, initial_criticism: str) -> List[Dict]:
        """
        Run backward pass, accumulating gradients and scores.
        
        Returns:
            List of results for each backward step
        """
        templates = graph.linearize(initial_criticism)
        results = []
        
        for item in templates:
            prompt = item['prompt']
            input_node = item['input_node']
            
            # Call LLM
            response = self.llm_fn(prompt)
            
            # Parse
            parsed = parse_response(response)
            
            # Accumulate gradient to input node
            input_node.grad.append(parsed['criticism'])
            input_node.suspicion_score += parsed['severity_score']
            
            results.append({
                **item,
                'response': response,
                'parsed': parsed,
            })
        
        return results
    
    def rank_nodes(self, graph: Graph) -> List[Tensor]:
        """Rank nodes by suspicion score (highest first)."""
        return sorted(graph.nodes, key=lambda n: n.suspicion_score, reverse=True)
    
    def get_top_suspects(self, graph: Graph, k: int = 3) -> List[Dict]:
        """Get top k suspected error nodes with details."""
        ranked = self.rank_nodes(graph)[:k]
        return [
            {
                'step_idx': n.step_idx,
                'role': n.role,
                'score': n.suspicion_score,
                'gradients': n.grad,
                'value_preview': n.value[:200] + '...' if len(n.value) > 200 else n.value,
            }
            for n in ranked
        ]


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
