# TextGrad for Multi-Agent System Failure Attribution

A minimal framework that applies **textual backpropagation** to identify decisive error steps in multi-agent system trajectories.

## Core Concept

Inspired by [TextGrad](https://arxiv.org/abs/2406.07496), this framework models agent trajectories as **computational graphs** where:

| Deep Learning | This Framework |
|--------------|----------------|
| Tensor (numbers) | Tensor (text - step content) |
| Operations (matmul, relu) | Operations (LLM call, web search, code execution) |
| Gradient (∂L/∂x) | Gradient (textual criticism of how input caused error) |
| Loss | Initial criticism of why final output is wrong |
| Backpropagation | Chain of prompts tracing error attribution backward |

## Quick Start

```python
from core import trajectory_to_graph, compute_loss, BackwardExecutor

# 1. Define your trajectory as list of steps
trajectory = [
    {"role": "Orchestrator", "content": "Plan: First search for X..."},
    {"role": "WebSurfer", "content": "Search results show..."},
    {"role": "Orchestrator", "content": "Final answer: Y"},
]

# 2. Build the graph
graph = trajectory_to_graph(
    steps=trajectory,
    problem="What is X?",
    ground_truth="Z"  # The correct answer
)

# 3. Compute initial loss
loss = compute_loss(graph.get_loss(), expected="Z", problem="What is X?")

# 4. Get prompts for backward pass
templates = graph.linearize(loss)

# 5. Either run manually or with LLM
for item in templates:
    print(item['prompt'])  # Copy to Claude/GPT
    # OR
    # response = your_llm(item['prompt'])
```

## Key Components

### Tensor
A node in the computational graph representing one step in the trajectory.

```python
@dataclass
class Tensor:
    value: str          # Full content of this step
    role: str           # Agent name (e.g., "WebSurfer")
    step_idx: int       # Position in trajectory
    grad: List[str]     # Accumulated criticisms
    suspicion_score: float  # Higher = more suspicious
```

### Operations (Backward Templates)
Each agent type has a specialized prompt template for computing textual gradients.

Built-in operations:
- `generic` - Default for unknown agents
- `orchestrator` - For planning/delegation decisions
- `websurfer` - For web search/browse operations
- `coder` - For code execution
- `assistant` - For general LLM responses

**Adding custom operations:**
```python
from core import register_backward_template

register_backward_template("my_agent", """
You are analyzing a failure in my custom agent...
{problem}
{ground_truth}
{output_value}
{input_value}
{downstream_grad}
...
ATTRIBUTION: [...]
SEVERITY: [...]
CRITICISM: <...>
""")
```

### Graph
The computational structure linking steps together.

```python
graph = Graph(problem="...", ground_truth="...")
node1 = graph.add_node(value="...", role="Orchestrator", step_idx=0)
node2 = graph.add_node(value="...", role="WebSurfer", step_idx=1, predecessors=[node1])
graph.set_loss(node2)
```

### Linearize
The key function that converts the backward pass into executable prompts:

```python
templates = graph.linearize(initial_criticism)
# Returns list of dicts:
# [
#   {'prompt': '...', 'output_node': Tensor, 'input_node': Tensor, 'step_idx': 0},
#   ...
# ]
```

## Chain Rule in Text

The textual chain rule works like this:

```
Given: A → B → C → Loss
       
When we have criticism of C:
  "C failed because it extracted wrong info"
  
Backward to B asks:
  "Given that C failed because of wrong info,
   how did B's output contribute to C's failure?"
   
Backward to A asks:
  "Given B's contribution to the failure,
   how did A's output contribute to B's problem?"
```

Each backward step accumulates gradients (criticisms), and nodes that receive
more severe criticisms get higher suspicion scores.

## Output Format

Each backward prompt expects responses in this format:
```
ATTRIBUTION: [INPUT_ERROR | PROCESSING_ERROR | BOTH | NEITHER]
SEVERITY: [HIGH | MEDIUM | LOW | NONE]
CRITICISM: <Specific explanation>
```

## Files

- `core.py` - The main framework
- `example_trajectory_1.py` - Example with "Human Nature" lyrics problem
- `example_trajectory_2.py` - Example with Mercedes Sosa discography problem
- `prompts_trajectory_1.txt` - Exported prompts for manual testing
- `prompts_trajectory_2.txt` - Exported prompts for manual testing

## Running

```bash
# Generate prompts for trajectory 1
python example_trajectory_1.py

# Generate prompts for trajectory 2
python example_trajectory_2.py
```

## Design Principles

1. **Minimal assumptions** - Works with any trajectory format
2. **Agent-agnostic** - Easy to add new operation types
3. **Manual-first** - Export prompts for manual testing before automation
4. **Simple scoring** - Severity-based accumulation for ranking suspects

## Graph Construction Patterns

### Linear (Default)
Each step depends only on its immediate predecessor.
```
A → B → C → D
```

### Full Context
Each step depends on all previous steps (expensive but thorough).
```
A → B
A → C
B → C
A → D
B → D
C → D
```

### Custom (Memory/Planning aware)
For trajectories with memory updates and persistent plans:
```python
# Memory update: depends on old memory + new info
memory_node._prev = {old_memory, trigger_info}

# Plan: connects to subsequent queries
for query_node in subsequent_queries:
    query_node._prev.add(plan_node)
```

## Limitations

- Graph construction is currently manual/heuristic
- Requires LLM calls for each backward step (can be expensive)
- Scoring is simple severity accumulation (could be more sophisticated)
- No automatic handling of branching/parallel agent execution

## Next Steps

1. Test with real LLM (Claude/GPT) on the exported prompts
2. Compare results with ground truth from Who&When dataset
3. Iterate on backward templates based on failure patterns
4. Consider adding gradient aggregation strategies
