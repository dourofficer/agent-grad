"""
Docstring for cli.predict

python -m cli.predict --dir outputs/gpt-oss-20b/step-by-step/hand-crafted --method step_by_step
python -m cli.predict --dir outputs/gpt-oss-20b/all-at-once/hand-crafted --method all_at_once
"""

import json
import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from utils.prompting2 import _get_sorted_json_files, _load_json_data

def parse_llm_json_output(response_text):
    """ Matches patterns:
    "```json{\"key\": \"value\"}\n```"       
    "```\n{\"key\": \"value\"}\n```"            
    "{\"key\": \"value\"}"                     
    """
    if not response_text:
        return {}
    
    text = response_text.strip()
    
    # Try direct parse first (fast path)
    try: return json.loads(text)
    except json.JSONDecodeError: pass
    
    # Strip markdown code blocks
    if "```" in text:
        pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except json.JSONDecodeError: pass
    
    # Last resort: extract first {...} block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try: return json.loads(text[start:end + 1])
        except json.JSONDecodeError: pass
    
    print(f"Failed to parse JSON response: {response_text[:100]}...")
    return {}

def populate_predictions(output_dir, method='all_at_once'):
    json_files = _get_sorted_json_files(output_dir)
    
    for filename in tqdm(json_files, desc=f"Processing {method}"):
        file_path = os.path.join(output_dir, filename)
        data = _load_json_data(file_path)
        assert data is not None
        
        logs = data.get('logs', [])
        steps = data.get('steps', [])
        assert all('reasoning' in log for log in logs)
        assert all('response' in log for log in logs)
        predictions = []
        
        if method == 'all_at_once':
            # Extract the single LLM response
            assert len(logs) == 1
            response_text = logs[0].get('response', '')
            parsed = parse_llm_json_output(response_text)
            
            # Get predicted step info
            agent_name = parsed.get('agent_name', '')
            step_number = parsed.get('step_number', -1)
            reason = parsed.get('reason', '')
            
            # Build predictions list with scores for each step
            for step in steps:
                step_idx = step.get('step_idx', -1)
                if step_idx == step_number:
                    predictions.append({
                        'step_idx': step_idx,
                        'role': step.get('role', ''),
                        'content': step.get('content', ''),
                        'score': 1.0,
                        'reason': reason
                    })
            
            # Sort by step_idx
            predictions = sorted(predictions, key=lambda x: x['step_idx'])
        
        elif method == 'step_by_step':
            assert len(logs) >= 1
            for log in logs:
                step_idx = log.get('step_idx', -1)
                response_text = log.get('response', '')
                parsed = parse_llm_json_output(response_text)
                
                # Find corresponding step
                step = next((s for s in steps if s.get('step_idx') == step_idx), {})
                
                is_decisive = parsed.get('is_decisive', False)
                reason = parsed.get('reason')
                step['is_decisve'] = is_decisive
                step['reason'] = reason
                if not is_decisive: continue
                predictions.append({
                    'step_idx': step_idx,
                    'role': step.get('role'),
                    'content': step.get('content'),
                    'score': 1.0,
                    'reason': reason
                })
            
            # Sort by step_idx
            predictions = sorted(predictions, key=lambda x: x['step_idx'])
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add predictions to data structure
        data['predictions'] = predictions
        
        # Save to output directory
        output_path = os.path.join(output_dir, filename)
        os.makedirs('lovely', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(
        description='Populate predictions from LLM responses into trace data'
    )
    parser.add_argument(
        '--dir',
        type=str,
        required=True,
        help='Directory containing JSON files with LLM responses (will be modified in-place)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['all_at_once', 'step_by_step'],
        default='all_at_once',
        help='Prediction method: all_at_once or step_by_step (default: all_at_once)'
    )
    
    args = parser.parse_args()
    
    # Validate output directory exists
    if not os.path.exists(args.dir):
        raise FileNotFoundError(f"Output directory not found: {args.dir}")
    
    print(f"Processing files in: {args.dir}")
    print(f"Method: {args.method}")
    
    populate_predictions(
        output_dir=args.dir,
        method=args.method
    )

    json_files = _get_sorted_json_files(args.dir)
    print(f"Added predictions to {len(json_files)} files")

if __name__ == '__main__':
    main()