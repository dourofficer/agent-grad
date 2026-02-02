"""
agent-grad: TextGrad for Multi-Agent System Failure Attribution

A minimal framework that models agent trajectories as computational graphs
and uses textual backpropagation to identify decisive error steps.
"""

from .core import (
    Tensor,
    Graph,
    compute_loss,
    parse_response,
    trajectory_to_graph,
)

from .ops import (
    # Prompt builders
    build_generic_backward_prompt,
    get_backward_template,
    register_backward_template,
    parse_backward_response,
    
    # Operations
    agent_step,
    llm_call,
    web_search,
    code_execution,
    orchestrator_decision,
    textual_diff,
)

from .executor import (
    BackwardExecutor,
    build_graph_with_ops,
    export_backward_prompts,
)

__all__ = [
    # Core
    'Tensor',
    'Graph',
    'compute_loss',
    'parse_response',
    'trajectory_to_graph',
    
    # Ops
    'build_generic_backward_prompt',
    'get_backward_template',
    'register_backward_template',
    'parse_backward_response',
    'agent_step',
    'llm_call',
    'web_search',
    'code_execution',
    'orchestrator_decision',
    'textual_diff',
    
    # Executor
    'BackwardExecutor',
    'build_graph_with_ops',
    'export_backward_prompts',
]