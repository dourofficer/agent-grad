import json
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path

from agent_grad import Graph, agent_step, textual_diff, compute_loss, parse_backward_response
from utils.vllm import run_inference, send_request
from utils.graph import MagenticOneTrajectoryParser
from utils.prompting import _get_sorted_json_files, _load_json_data, _extract_metadata


def call_vllm(prompt, config_path="./configs/gpt-oss-20b.yaml"):
    """Call vLLM inference endpoint with a prompt."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    hostname = config.pop("hostname")
    port = config.pop("port")
    concurrent_requests = config.pop("concurrent_requests", 10)
    
    url = f"http://{hostname}:{port}/v1/chat/completions"
    
    data = {"messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]}
    _, out, _ = send_request(url, config, data, request_id=0)
    return {"reasoning": out["reasoning"], "response": out["response"]}


def process_example(example, config_path, role_id='role'):
    """Process a single example: build graph and compute gradients."""
    metadata = _extract_metadata(example)
    
    trajectory = example['history']
    problem = metadata['question']
    ground_truth = example['ground_truth']
    
    # Parse trajectory and build dependencies
    parser = MagenticOneTrajectoryParser(dependency_mode='structural')
    events = parser.parse_trajectory(trajectory)
    dependencies = parser.build_dependency_graph(events)
    
    # Create nodes
    nodes = []
    for i, step in enumerate(trajectory):
        predecessors = [nodes[j] for j in dependencies[i]]
        node = agent_step(
            inputs=predecessors,
            output_value=step['content'],
            output_role=step[role_id],
            output_idx=i,
            problem=problem,
            ground_truth=ground_truth,
            llm_fn=None,
        )
        nodes.append(node)
    
    # Build graph and compute initial loss
    graph = Graph(problem=problem, ground_truth=ground_truth)
    graph.nodes = nodes
    graph.set_loss(nodes[-1])
    
    initial_criticism = compute_loss(
        final_output=nodes[-1],
        expected=ground_truth,
        problem=problem
    )
    
    # Run backward pass
    templates = graph.linearize()
    templates[0]['output_node'].grad = [initial_criticism]
    backward_results = {}
    
    for template in templates:
        gradient = "\n---\n".join(template['output_node'].grad)
        prompt = template['format_fn'](gradient)
        response = call_vllm(prompt, config_path)['response']
        parsed = parse_backward_response(response, template['input_info'])
        
        step_idx = template['output_node'].step_idx
        backward_results[step_idx] = response
        
        # Accumulate gradients to input nodes
        for inp_node in template['input_nodes']:
            if inp_node.step_idx in parsed:
                inp_node.grad.append(parsed[inp_node.step_idx]['criticism'])
                inp_node.suspicion_score += parsed[inp_node.step_idx]['severity_score']
                inp_node.attribution += [parsed[inp_node.step_idx]['attribution']]
    
    # Prepare output
    return {
        'metadata': metadata,
        'problem': problem,
        'ground_truth': ground_truth,
        'nodes': [
            {
                'step_idx': node.step_idx,
                'input_nodes': node.input_nodes(),
                'role': node.role,
                'content': node.value,
                'grad': node.grad,
                'suspicion_score': node.suspicion_score,
                'attribution': node.attribution
            }
            for node in nodes
        ],
        'backward_results': backward_results
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run agent gradient computation on trajectory data"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/who-and-when/Hand-Crafted",
        help="Input data directory containing JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/agent_grad",
        help="Output directory for graph results"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/gpt-oss-20b.yaml",
        help="vLLM configuration file"
    )
    parser.add_argument(
        "--role_id",
        type=str,
        choices=['name', 'role'],
        default='role',
        help="Field name for agent role in trajectory"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Start index (for resuming interrupted runs)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="End index (None = process all remaining)"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    filepaths = _get_sorted_json_files(args.input)
    data = [_load_json_data(Path(args.input) / fp) for fp in filepaths]
    
    # Determine processing range
    end_idx = args.end_idx or len(data)
    data_subset = data[args.start_idx:end_idx]
    
    print(f"Processing {len(data_subset)} examples ({args.start_idx} to {end_idx-1})")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}\n")
    
    # Process each example
    for i, example in enumerate(tqdm(data_subset, desc="Processing")):
        idx = args.start_idx + i
        output_path = output_dir / f"{idx}.json"
        
        # Skip if already exists
        if output_path.exists():
            print(f"[{idx}] Skipping (already exists)")
            continue
        
        try:
            result = process_example(example, args.config, args.role_id)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            tqdm.write(f"[{idx}] ✓ Saved to {output_path}")
            
        except Exception as e:
            tqdm.write(f"[{idx}] ✗ Error: {e}")
            continue
    
    print(f"\n✓ Completed! Results in {output_dir}")


if __name__ == "__main__":
    main()