# Common Commands

## vLLM server

Start the server (adjust `CUDA_VISIBLE_DEVICES`, `--port`, and `--tensor-parallel-size` as needed):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve openai/gpt-oss-20b \
    --port 8881 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    --max-model-len 32000 \
    --max-num-batched-tokens 16000 \
    --generation-config auto
```

Smoke-test the server:

```bash
curl -X POST http://localhost:8881/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 1000,
        "temperature": 0.9,
        "reasoning_effort": "high"
    }'
```

---

## Dependency graphs  (`cli/build_graph.py`)

```bash
# Hand-crafted (Magentic-One) — deterministic, no LLM needed
python -m cli.build_graph \
    --subset hand-crafted \
    --input  data/ww/hand-crafted

# Algorithm-generated — LLM infers step dependencies
python -m cli.build_graph \
    --subset algorithm-generated \
    --input  data/ww/algorithm-generated \
    --output outputs/dependencies/algorithm-generated \
    --config configs/gpt-oss-20b.yaml \
    --start_idx 0 --end_idx 10
```

---

## agent-grad backprop  (`cli/backprop.py`)

```bash
# Phase 1 only — build prompt skeletons
python -m cli.backprop --prepare \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --start_idx 0 --end_idx 10

# Phase 2 only — run inference against vLLM
python -m cli.backprop --process \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/agent-grad/hand-crafted

# Both phases in one go
python -m cli.backprop \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --start_idx 0 --end_idx 10

# algorithm-generated subset
python -m cli.backprop \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/algorithm-generated \
    --output outputs/gpt-oss-20b/agent-grad/algorithm-generated \
    --start_idx 0 --end_idx 10
```

---

## Baseline inference  (`cli/inference.py`)

```bash
# all-at-once
python -m cli.inference \
    --method all_at_once \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/all-at-once/hand-crafted \
    --start_idx 0 --end_idx 10

python -m cli.inference \
    --method all_at_once \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/algorithm-generated \
    --output outputs/gpt-oss-20b/all-at-once/algorithm-generated \
    --start_idx 0 --end_idx 10

# step-by-step
python -m cli.inference \
    --method step_by_step \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/step-by-step/hand-crafted \
    --start_idx 0 --end_idx 10

python -m cli.inference \
    --method step_by_step \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/algorithm-generated \
    --output outputs/gpt-oss-20b/step-by-step/algorithm-generated \
    --start_idx 0 --end_idx 10

# text-grad (direct, no dependency graph)
python -m cli.inference \
    --method text_grad \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/hand-crafted \
    --output outputs/gpt-oss-20b/text-grad/hand-crafted \
    --start_idx 0 --end_idx 10

python -m cli.inference \
    --method text_grad \
    --config configs/gpt-oss-20b.yaml \
    --input  data/ww/algorithm-generated \
    --output outputs/gpt-oss-20b/text-grad/algorithm-generated \
    --start_idx 0 --end_idx 10
```

---

## Populate predictions  (`cli/predict.py`)

Run after inference to write `predictions` into each output JSON.

```bash
python -m cli.predict \
    --dir    outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --method agent_grad

python -m cli.predict \
    --dir    outputs/gpt-oss-20b/all-at-once/hand-crafted \
    --method all_at_once

python -m cli.predict \
    --dir    outputs/gpt-oss-20b/step-by-step/hand-crafted \
    --method step_by_step

python -m cli.predict \
    --dir    outputs/gpt-oss-20b/text-grad/hand-crafted \
    --method text_grad
```

---

## Evaluation  (`cli/evaluate.py`)

```bash
# Single config, top-1
python -m cli.evaluate \
    --dir outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --k 1

# Single config with saved results
python -m cli.evaluate \
    --dir  outputs/gpt-oss-20b/agent-grad/hand-crafted \
    --k    5 \
    --save outputs/gpt-oss-20b/results_agent-grad_hand-crafted.json

python -m cli.evaluate \
    --dir  outputs/gpt-oss-20b/text-grad/hand-crafted \
    --k    5 \
    --save outputs/gpt-oss-20b/results_text-grad_hand-crafted.json

python -m cli.evaluate \
    --dir  outputs/gpt-oss-20b/all-at-once/hand-crafted \
    --k    1 \
    --save outputs/gpt-oss-20b/results_all-at-once_hand-crafted.json

python -m cli.evaluate \
    --dir  outputs/gpt-oss-20b/step-by-step/hand-crafted \
    --k    1 \
    --save outputs/gpt-oss-20b/results_step-by-step_hand-crafted.json

python -m cli.evaluate \
    --dir  outputs/gpt-oss-20b/step-by-step/algorithm-generated \
    --k    1 \
    --save outputs/gpt-oss-20b/results_step-by-step_algorithm-generated.json

# Full sweep over all methods × subsets × k values → TSV
python -m cli.evaluate --sweep \
    --save outputs/gpt-oss-20b/sweep_results.tsv
```

---

## Long-context subset

Extract files with ≥ 50 steps into a `long-context/` sub-directory for each method
(uses `copy_long_context_files` in `utils/common.py`):

```python
from utils.common import copy_long_context_files
copy_long_context_files(result_dir="outputs/gpt-oss-20b", threshold=50)
```

Then evaluate the long-context slice:

```bash
python -m cli.evaluate --dir outputs/gpt-oss-20b/agent-grad/long-context    --k 1
python -m cli.evaluate --dir outputs/gpt-oss-20b/text-grad/long-context     --k 1
python -m cli.evaluate --dir outputs/gpt-oss-20b/all-at-once/long-context   --k 1
python -m cli.evaluate --dir outputs/gpt-oss-20b/step-by-step/long-context  --k 1
```

---

## Quick prompt inspection (Python)

```python
from utils.common import mdprint
import json

# all-at-once
data = json.load(open("outputs/gpt-oss-20b/all-at-once/hand-crafted/1.json"))
mdprint(data["logs"][0]["messages"][1]["content"])

# text-grad / agent-grad — grouped by example file
data = json.load(open("outputs/gpt-oss-20b/agent-grad/hand-crafted/1.json"))
for log in data["logs"]:
    print(f"Step {log['output_step_idx']} → {log['input_step_idxs']}")
    mdprint(log["messages"][1]["content"])
    print()
```