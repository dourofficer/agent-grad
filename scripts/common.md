```bash
python get_prompts.py \
    --method all_at_once \
    --model gpt-4o \
    --is_handcrafted False \
    --directory_path "./data/who-and-when/Algorithm-Generated"

python get_prompts.py \
    --method all_at_once \ # or build_ifg
    --model gpt-4o \
    --is_handcrafted True \
    --directory_path "./data/who-and-when/Hand-Crafted"

python get_prompts.py \
    --method text_grad \
    --model gpt-4o \
    --is_handcrafted False \
    --directory_path "./data/who-and-when/Algorithm-Generated"

python get_prompts.py \
    --method text_grad
    --model gpt-4o \
    --is_handcrafted True \
    --directory_path "./data/who-and-when/Hand-Crafted"
```

```python
from Lib.new_utils import mdprint
import json
data = json.load(open("prompt_outputs/prompts_all_at_once_gpt-4o_handcrafted.json"))
prompts = [x["messages"][1]["content"] for x in data["prompts"]]
labels = [x["labels"] for x in data["prompts"]]
steps = [x["chat_history"] for x in data["prompts"]]
mdprint(prompts[0])
```

```python
from Lib.new_utils import mdprint
import json
data = json.load(open("prompt_outputs/prompts_text_grad_gpt-4o_alg_generated.json"))
prompts = [x["messages"][1]["content"] for x in data["prompts"]]
labels = [x["labels"] for x in data["prompts"]]

file_to_prompts = {}
for entry in data["prompts"]:
    if file_to_prompts.get(entry["file"]) is None:
        file_to_prompts[entry["file"]] = []
    file_to_prompts[entry["file"]].append(entry)
grouped_prompts = [v for k, v in file_to_prompts.items()]

entry_idx = 10
example = grouped_prompts[entry_idx]
error_idx = int(example[0]['labels']['mistake_step'])
step = example[error_idx]
step_prompt = step["messages"][1]["content"]

mdprint(step_prompt)
```

```python
i = 2
label = labels[i]
mistake_step = int(label['mistake_step'])
step = steps[i][mistake_step]['content']
label, mdprint(step)
```


```bash
python -m utils.inference \
    --config configs/gpt-oss-20b.yaml \
    --method text_grad \
    --input "./data/who-and-when/Algorithm-Generated/" \
    --output "./outputs/gpt-oss-20b/"

python -m utils.inference \
    --config configs/gpt-oss-20b.yaml \
    --method text_grad \
    --input "./data/who-and-when/Hand-Crafted/" \
    --output "./outputs/gpt-oss-20b/"
```

```bash
python -m utils.infer_agent_grad \
    --input data/who-and-when/Hand-Crafted \
    --output outputs/gpt-oss-20b/graphs \
    --start_idx 10 --end_idx 15
```
----------------------------------------------
```bash
python -m cli.inference \
    --method 'all_at_once' \
    --config 'configs/gpt-oss-20b.yaml' \
    --input 'data/ww/hand-crafted' \
    --output 'outputs/gpt-oss-20b/all-at-once/hand-crafted'
python -m cli.inference \
    --method 'step_by_step' \
    --config 'configs/gpt-oss-20b.yaml' \
    --input 'data/ww/hand-crafted' \
    --output 'outputs/gpt-oss-20b/step_by_step/hand-crafted' \
    --start_idx 0 --end_idx 2

python -m cli.predict \
    --dir outputs/gpt-oss-20b/all-at-once/hand-crafted \
    --method all_at_once

python -m cli.evaluate \
    --dir 'outputs/gpt-oss-20b/all-at-once/hand-crafted' \
    --k 1 \
    --save 'outputs/gpt-oss-20b/results_all-at-once_hand-crafted.json'
python -m cli.evaluate \
    --dir 'outputs/gpt-oss-20b/step-by-step/hand-crafted' \
    --k 1 \
    --save 'outputs/gpt-oss-20b/results_step-by-step_hand-crafted.json'
```
----------------------------------------------

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve openai/gpt-oss-20b \
    --port 8882 \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --disable-log-requests \
    --max-model-len 32000 \
    --max-num-batched-tokens 16000 \
    --generation-config auto


curl -X POST http://localhost:8881/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "What are the applications of LLMs in biology research?"}
        ],
        "max_tokens": 1000,
        "temperature": 0.9,
        "reasoning_effort": "high"
    }'
```

```structure of the data
outputs/gpt-oss-20b/text-grad/handcrafted/1.json
    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step.role,
                'content': step.value,
                'grad': node.grad, (grad should indicate where it receive downstream grad from)
                'suspicion_score': node.suspicion_score,
                'attribution': node.attribution
            }
            ...
        ],

        'logs': [{
            prompt: 
            reasoning: 
            response:
        }, ...],

        'predictions': [{ <- from steps
            step_idx: ...,
            role: ...,
            content: ...,
            score: ...,
            reason: ...,
        }] 
    }
outputs/gpt-oss-20b/text-grad/all-at-once/1.json
    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step.role,
                'content': step.value,
            }
            ...
        ],

        'logs': [{
            prompt: 
            reasoning: 
            response:
        }, ...],

        'predictions': [{ <- from logs
            step_idx: ...,
            role: ...,
            content: ...,
            score: ...,
            reason: ...,
        }] 
    }
outputs/gpt-oss-20b/text-grad/step-by-step/1.json
    return {
        'metadata': metadata,
        'steps': [
            {
                'step_idx': step_idx,
                'input_steps': [],
                'output_steps': [],
                'role': step.role,
                'content': step.value,
                'is_decisive': ...,
                'reason': ...
            }
            ...
        ],
        'logs': [{
            prompt: 
            reasoning: 
            response:
        }, ...],

        'predictions': [{ <- from steps
            step_idx: ...,
            role: ...,
            content: ...,
            score: ...,
            reason: ...,
        }] 
    }
```