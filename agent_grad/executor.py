"""
Backward Executor for TextGrad

Provides utilities for executing backward passes with LLMs
and analyzing results.
"""

from typing import Callable, List, Dict, Any
from .core import Tensor, Graph


class BackwardExecutor:
    """
    Execute backward pass using an LLM.
    
    This executor configures tensors with an LLM function so that
    when .backward() is called, gradients are computed via LLM calls.
    
    Usage:
        executor = BackwardExecutor(llm_fn=my_llm_function)
        executor.configure_graph(graph)  # Set up LLM for all nodes
        loss.backward()  # Now executes with LLM
        ranked = executor.rank_nodes(graph)
    """
    
    def __init__(self, llm_fn: Callable[[str], str]):
        """
        Args:
            llm_fn: Function that takes prompt string, returns LLM response string
        """
        self.llm_fn = llm_fn
    
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
    
    def print_analysis(self, graph: Graph, top_k: int = 5):
        """Print a summary analysis of the graph after backward pass."""
        print("=" * 60)
        print("FAILURE ATTRIBUTION ANALYSIS")
        print("=" * 60)
        print(f"\nProblem: {graph.problem[:100]}...")
        print(f"Ground Truth: {graph.ground_truth}")
        print(f"\nTotal steps analyzed: {len(graph.nodes)}")
        
        print("\n" + "-" * 60)
        print("TOP SUSPECTED ERROR STEPS (by suspicion score)")
        print("-" * 60)
        
        suspects = self.get_top_suspects(graph, k=top_k)
        for i, s in enumerate(suspects, 1):
            print(f"\n{i}. Step {s['step_idx']} ({s['role']}) - Score: {s['score']:.2f}")
            print(f"   Preview: {s['value_preview'][:80]}...")
            if s['gradients']:
                print(f"   Criticisms received: {len(s['gradients'])}")
                # Show first criticism
                first_crit = s['gradients'][0][:150] if s['gradients'] else "None"
                print(f"   First criticism: {first_crit}...")
        
        print("\n" + "=" * 60)


def build_graph_with_ops(
    steps: List[Dict[str, str]],
    problem: str,
    ground_truth: str,
    llm_fn: Callable[[str], str] = None,
    mode: str = "linear"
) -> tuple:
    """
    Build a computation graph using ops, with backward functions attached.
    
    Args:
        steps: List of dicts with 'role' and 'content' keys
        problem: The problem statement
        ground_truth: Expected answer
        llm_fn: Optional LLM function for backward passes
        mode: "linear" or "full" connectivity
    
    Returns:
        Tuple of (graph, loss_tensor)
    """
    from .ops import agent_step, textual_diff
    
    graph = Graph(problem=problem, ground_truth=ground_truth)
    
    all_nodes = []
    for i, step in enumerate(steps):
        role = step.get('role', step.get('agent', 'unknown'))
        content = step.get('content', step.get('value', step.get('message', '')))
        
        if mode == "linear":
            inputs = [all_nodes[-1]] if all_nodes else []
        else:
            inputs = all_nodes.copy()
        
        node = agent_step(
            inputs=inputs,
            output_value=content,
            output_role=role,
            output_idx=i,
            problem=problem,
            ground_truth=ground_truth,
            llm_fn=llm_fn,
        )
        
        all_nodes.append(node)
        graph.nodes.append(node)
    
    if all_nodes:
        graph.set_loss(all_nodes[-1])
        loss = textual_diff(all_nodes[-1], ground_truth, problem)
        return graph, loss
    
    return graph, None


def export_backward_prompts(
    steps: List[Dict[str, str]],
    problem: str,
    ground_truth: str,
    initial_criticism: str,
    filename: str = "prompts_output.txt",
    mode: str = "linear"
):
    """
    Export all backward prompts to a file for manual testing.
    
    This builds the graph without an LLM, runs backward to generate prompts,
    then exports them for copy-paste testing.
    """
    graph, loss = build_graph_with_ops(
        steps=steps,
        problem=problem,
        ground_truth=ground_truth,
        llm_fn=None,  # No LLM - will store prompts as gradients
        mode=mode,
    )
    
    if loss is None:
        print("No steps to analyze")
        return
    
    # Run backward - this will populate gradients with prompts
    loss.backward(initial_criticism)
    
    # Export
    with open(filename, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TEXTGRAD BACKWARD PROMPTS FOR MANUAL TESTING\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Problem: {problem}\n")
        f.write(f"Ground Truth: {ground_truth}\n\n")
        
        for i, node in enumerate(graph.nodes):
            f.write("-" * 60 + "\n")
            f.write(f"NODE {node.step_idx} ({node.role})\n")
            f.write("-" * 60 + "\n")
            f.write(f"Value preview: {node.value[:200]}...\n\n")
            
            if node.grad:
                f.write("ACCUMULATED GRADIENTS/PROMPTS:\n")
                for j, g in enumerate(node.grad):
                    f.write(f"\n--- Gradient {j+1} ---\n")
                    f.write(g + "\n")
            else:
                f.write("No gradients accumulated.\n")
            
            f.write("\n")
    
    print(f"Exported {len(graph.nodes)} node prompts to {filename}")