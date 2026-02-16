# agent-grad

**Textual backpropagation for automated failure attribution in LLM multi-agent systems.**

agent-grad adapts the [TextGrad](https://github.com/zou-group/textgrad) framework to a diagnostic task: given a failed multi-agent trajectory, propagate criticism backward through the dependency graph to identify which agent caused the failure and exactly where it happened. It is the backbone method for the *Who&When* benchmark introduced in [Zhang et al., ICML 2025](https://arxiv.org/abs/2505.00212).

---

## How it works

A failed trajectory is modeled as a directed acyclic computation graph where each node is a `Tensor` — one agent step holding its textual content, role, and accumulated gradient. A "loss" is computed at the final node (the wrong answer), then propagated backward edge by edge: at each edge, an LLM receives the downstream criticism, the output step, and its input step(s), and returns an attribution verdict (`ORIGINATING_ERROR`, `PROPAGATING_ERROR`, or `NEITHER`) plus a natural-language criticism. Steps accumulate `suspicion_score` as criticism flows through them; the top-ranked steps are the predicted decisive error locations.

```
Trajectory:  Step 0 → Step 1 → Step 2 → … → Step N  (wrong answer)
                                                  ↓
                                              compute_loss()
                                                  ↓ backward pass
               ← ← ← ← textual gradient ← ← ← ← ←
```

The backward pass is two-phased to support large-scale inference on a vLLM server:

1. **Prepare** (`--prepare`) — build the dependency graph, render all backward prompts, and write a skeleton JSON per example.
2. **Process** (`--process`) — send each prompt to vLLM and fill in reasoning/response; accumulate attributions.

---

## Repository layout

```
agent_grad/          Core library
  core.py            Tensor, Graph, compute_loss, trajectory_to_graph
  ops.py             Operation definitions and backward prompt builders
  executor.py        BackwardExecutor, build_graph_with_ops, export_backward_prompts
  __init__.py

cli/                 Entry-point scripts
  backprop.py        Two-phase inference pipeline  (--prepare / --process)
  build_graph.py     Dependency graph construction (hand-crafted or LLM-generated)
  predict.py         Populate predictions from LLM responses into output JSON
  evaluate.py        Accuracy@k evaluation and cross-config sweep

utils/
  graph.py           MagenticOneTrajectoryParser — structural dependency extraction
  vllm.py            Threaded vLLM client (send_request, run_inference)
  common.py          Shared helpers (_load_json_data, _extract_metadata, …)

configs/
  gpt-oss-20b.yaml   Example vLLM server config

data/ww/
  hand-crafted/      Who&When Magentic-One failure logs (58 tasks)
  algorithm-generated/ CaptainAgent failure logs (126 tasks)

outputs/             Inference results, organized by model / method / subset
```

---

## Installation

```bash
git clone <repo>
cd agent-grad
pip install -r requirements.txt
```

The pipeline assumes a running [vLLM](https://github.com/vllm-project/vllm) server. Configure it in `configs/`:

```yaml
# configs/gpt-oss-20b.yaml
hostname: localhost
port: 8881
model: openai/gpt-oss-20b
temperature: 0.6
max_tokens: 4000
reasoning_effort: high
concurrent_requests: 16
```

---

## Quickstart

### Library usage

```python
from agent_grad import trajectory_to_graph, compute_loss, BackwardExecutor

steps = [
    {"role": "Orchestrator", "content": "..."},
    {"role": "WebSurfer",    "content": "..."},
    {"role": "Assistant",    "content": "... (wrong answer)"},
]

graph = trajectory_to_graph(steps, problem="...", ground_truth="...", mode="linear")
loss  = compute_loss(graph.get_loss(), expected="...", problem="...")

# Export prompts for manual inspection (no LLM needed)
templates = graph.linearize(loss)

# Or run automated backward pass
def my_llm(prompt: str) -> str: ...

executor = BackwardExecutor(llm_fn=my_llm)
executor.run(graph, loss)
top_3 = executor.get_top_suspects(graph, k=3)
```

### Full pipeline (vLLM)

```bash
# 1. Build dependency graphs
python -m cli.build_graph \
    --subset hand-crafted \
    --input  data/ww/hand-crafted

# 2. Phase 1 – build prompt skeletons
python -m cli.backprop --prepare \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --start_idx 0 --end_idx 58

# 3. Phase 2 – run inference
python -m cli.backprop --process \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/agent-grad/hand-crafted

# 4. Populate predictions
python -m cli.predict \
    --dir    outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --method agent_grad

# 5. Evaluate
python -m cli.evaluate \
    --dir outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --k 1
```

Both phases can be combined by omitting `--prepare`/`--process`:

```bash
python -m cli.backprop \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/agent-grad/hand-crafted
```

---

## Core abstractions

### `Tensor`
A node in the trajectory graph. Holds the step's text, role, and — after the backward pass — a list of accumulated textual gradients (`grad`) and a `suspicion_score`.

### `Graph`
Wraps the set of `Tensor` nodes with problem/ground-truth context. Key methods:

| Method | Description |
|---|---|
| `add_node(...)` | Register a step as a node |
| `set_loss(node)` | Mark the terminal (failed) node |
| `linearize(initial_grad)` | Return ordered list of backward-prompt templates |
| `_toposort()` | Topological sort for correct backward ordering |

### `BackwardExecutor`
Wires an LLM function into the graph. Methods: `rank_nodes`, `get_top_suspects`, `print_analysis`.

### Operations (`agent_grad/ops.py`)
Each op wraps a step type (`agent_step`, `llm_call`, `web_search`, `code_execution`, `orchestrator_decision`) and registers a backward prompt template. Custom templates can be registered via `register_backward_template(role, builder_fn)`.

### Dependency graph construction (`cli/build_graph.py`)
Two modes:

- **`hand-crafted`** — `MagenticOneTrajectoryParser` deterministically extracts Magentic-One's structural communication patterns (task → plan → ledger → instruction → worker action) without any LLM call.
- **`algorithm-generated`** — an LLM infers dependencies step-by-step for arbitrary agent topologies.

---

## Output format

Each processed example is stored as a JSON file:

```json
{
  "metadata": {
    "question": "...",
    "ground_truth": "...",
    "mistake_agent": "WebSurfer",
    "mistake_step": "16"
  },
  "steps": [
    {
      "step_idx": 0,
      "role": "Orchestrator",
      "content": "...",
      "grad": [...],
      "suspicion_score": 0.0,
      "attribution": [...]
    }
  ],
  "logs": [
    {"messages": [...], "reasoning": "...", "response": "..."}
  ],
  "predictions": [
    {"step_idx": 16, "role": "WebSurfer", "score": 1.0, "reason": "..."}
  ]
}
```

---

## Evaluation

```bash
# Single config
python -m cli.evaluate --dir outputs/gpt-oss-20b/agent-grad/hand-crafted --k 1

# Sweep over all methods and k values
python -m cli.evaluate --sweep --save outputs/gpt-oss-20b/sweep_results.tsv
```

Metrics reported: **Agent-Level Accuracy** (correct failure-responsible agent) and **Step-Level Accuracy** (correct decisive error step), both at top-k. The sweep produces a pivot table over `{all-at-once, step-by-step, text-grad, agent-grad} × {hand-crafted, algorithm-generated} × k ∈ {1, 3, 5, 10}`.

---

## Who&When dataset

The `data/ww/` directory holds the [Who&When](https://arxiv.org/abs/2505.00212) benchmark: 184 annotated failure logs from 127 LLM multi-agent systems.

| Subset | Source systems | Tasks | Agents | Max log length |
|---|---|---|---|---|
| Algorithm-generated | CaptainAgent (GPT-4o) | 126 | 1–4 | 10 steps |
| Hand-crafted | Magentic-One | 58 | 1–5 | 130 steps |

Each instance includes the full conversation log, the agent system description, and human-annotated labels for the failure-responsible agent and the decisive error step.

---

## Baseline methods

For comparison, the same CLI supports three baselines from the Who&When paper:

| Method | `--method` flag | Description |
|---|---|---|
| All-at-Once | `all_at_once` | Single LLM call with the full log |
| Step-by-Step | `step_by_step` | Per-step binary is-this-decisive classification |
| Text-Grad | `text_grad` | Direct textual gradient (no dependency graph) |
| **Agent-Grad** | `agent_grad` | This method |

Use `cli/predict.py --method <flag>` to populate predictions after inference.

---

## Citation

```bibtex
@inproceedings{zhang2025whoandwhen,
  title     = {Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems},
  author    = {Zhang, Shaokun and Yin, Ming and Zhang, Jieyu and Liu, Jiale and Han, Zhiguang and Zhang, Jingyang and Li, Beibin and Wang, Chi and Wang, Huazheng and Chen, Yiran and Wu, Qingyun},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  year      = {2025}
}
```