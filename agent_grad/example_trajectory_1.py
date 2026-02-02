"""
Example: Applying TextGrad to Failed Trajectory 1

This shows how to:
1. Parse a trajectory into graph format
2. Linearize backward pass into prompts
3. Export prompts for manual testing
"""

from core import (
    Graph, Tensor, 
    trajectory_to_graph, 
    compute_loss,
    BackwardExecutor,
    parse_response,
)

# ============================================================
# FAILED TRAJECTORY 1 - Parsed into steps
# Problem: King of Pop's fifth single from sixth studio album
# ============================================================

PROBLEM_1 = "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?"
GROUND_TRUTH_1 = "stare"
WRONG_ANSWER_1 = "bite"

# Parse the trajectory into steps (from Failed_Trajectory_1)
TRAJECTORY_1 = [
    {
        "role": "Orchestrator",
        "content": """Initial plan:
We are working to address the following user request:
What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?

Here is the plan to follow:
1. Identify the King of Pop's (Michael Jackson's) sixth studio album.
2. Determine the fifth single released from that album.
3. Retrieve the lyrics of the fifth single to find the last word before the second chorus.
4. Identify the last word before the second chorus from the lyrics."""
    },
    {
        "role": "Orchestrator", 
        "content": """Updated Ledger: The initial step of confirming Michael Jackson's sixth studio album has not been completed yet.
Next speaker: WebSurfer
Instruction: Please look up and confirm the sixth studio album of Michael Jackson."""
    },
    {
        "role": "WebSurfer",
        "content": """I typed 'Michael Jackson sixth studio album' into the browser search bar.
Screenshot of search results shows:
Thriller (album) - Wikipedia
Thriller is the sixth studio album by the American singer and songwriter Michael Jackson, released on November 29, 1982, by Epic Records."""
    },
    {
        "role": "Orchestrator",
        "content": """Updated Ledger: We have identified the sixth studio album as 'Thriller'.
Next speaker: WebSurfer
Instruction: Look up and confirm the fifth single released from Michael Jackson's 'Thriller' album."""
    },
    {
        "role": "WebSurfer", 
        "content": """I clicked 'Thriller (album) - Wikipedia'.
The page shows singles from Thriller:
1. "The Girl Is Mine"
2. "Billie Jean"
3. "Beat It"
4. "Wanna Be Startin' Somethin'"
5. "Human Nature"
6. "P.Y.T. (Pretty Young Thing)"
7. "Thriller"
Seven singles were released from the album."""
    },
    {
        "role": "Orchestrator",
        "content": """Updated Ledger: We have identified that 'Human Nature' is the fifth single from the 'Thriller' album.
Next speaker: WebSurfer  
Instruction: Look up the lyrics to Michael Jackson's 'Human Nature' and identify the last word before the second chorus."""
    },
    {
        "role": "WebSurfer",
        "content": """I typed 'Michael Jackson 'Human Nature' lyrics' into the browser search bar.
[Search results showing lyrics pages]"""
    },
    {
        "role": "Orchestrator",
        "content": """Updated Ledger: The lyrics are available and ready to be examined.
Next speaker: WebSurfer
Instruction: Please examine the lyrics to 'Human Nature' by Michael Jackson and identify the last word before the second chorus."""
    },
    {
        "role": "WebSurfer",
        "content": """The last word before the second chorus in Michael Jackson's song "Human Nature" is "bite." 
The lyrics preceding the second chorus are, "Then let me take a bite.\""""
    },
    {
        "role": "Orchestrator",
        "content": """Updated Ledger: The request has been successfully addressed.
The last word before the second chorus of 'Human Nature' is 'bite.'
Request satisfied: TRUE
FINAL ANSWER: bite"""
    },
]


def build_example_graph():
    """Build graph from trajectory 1."""
    return trajectory_to_graph(
        steps=TRAJECTORY_1,
        problem=PROBLEM_1,
        ground_truth=GROUND_TRUTH_1,
        mode="linear"  # Each step depends on previous
    )


def export_prompts_for_manual_testing(output_file: str = "prompts_trajectory_1.txt"):
    """
    Export all backward prompts to a file for manual testing.
    
    This is the key output - you can copy these prompts to Claude/GPT
    and run them manually to see how the backward pass works.
    """
    graph = build_example_graph()
    
    # Compute initial loss (the starting gradient)
    loss_node = graph.get_loss()
    initial_criticism = compute_loss(loss_node, GROUND_TRUTH_1, PROBLEM_1)
    
    # Get linearized prompts
    templates = graph.linearize(initial_criticism)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEXTGRAD BACKWARD PASS PROMPTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Problem: {PROBLEM_1}\n")
        f.write(f"Expected Answer: {GROUND_TRUTH_1}\n")
        f.write(f"Wrong Answer: {WRONG_ANSWER_1}\n")
        f.write(f"Number of steps: {len(TRAJECTORY_1)}\n")
        f.write(f"Number of backward prompts: {len(templates)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("INITIAL LOSS (Starting Gradient)\n")
        f.write("=" * 80 + "\n")
        f.write(initial_criticism)
        f.write("\n\n")
        
        for i, item in enumerate(templates):
            f.write("=" * 80 + "\n")
            f.write(f"BACKWARD PROMPT {i+1}\n")
            f.write(f"Analyzing: Step {item['output_step_idx']} -> Step {item['step_idx']}\n")
            f.write(f"Output Node: {item['output_node'].role} (Step {item['output_node'].step_idx})\n")
            f.write(f"Input Node: {item['input_node'].role} (Step {item['input_node'].step_idx})\n")
            f.write("=" * 80 + "\n\n")
            f.write(item['prompt'])
            f.write("\n\n")
            f.write("-" * 40 + "\n")
            f.write("YOUR RESPONSE HERE:\n")
            f.write("-" * 40 + "\n\n\n")
    
    print(f"Exported {len(templates)} prompts to {output_file}")
    return templates


def simulate_backward_pass():
    """
    Simulate the backward pass with mock LLM responses.
    
    In practice, you'd replace the mock_llm with actual LLM calls.
    """
    
    # Mock LLM responses (you'd replace this with real LLM calls)
    def mock_llm(prompt: str) -> str:
        # Simulate responses based on which step we're analyzing
        if "Step 8" in prompt and "Step 9" in prompt:
            return """ATTRIBUTION: [PROCESSING_ERROR]
SEVERITY: [HIGH]
CRITICISM: The WebSurfer agent provided incorrect information about the lyrics. The actual last word before the second chorus is "stare" not "bite". The agent may have misidentified the chorus boundaries or retrieved incorrect lyrics."""
        
        elif "Step 7" in prompt and "Step 8" in prompt:
            return """ATTRIBUTION: [INPUT_ERROR]
SEVERITY: [MEDIUM]  
CRITICISM: The orchestrator's instruction was correct, but the WebSurfer's previous search may not have found accurate lyrics sources."""
        
        elif "Step 5" in prompt or "Step 6" in prompt:
            return """ATTRIBUTION: [NEITHER]
SEVERITY: [LOW]
CRITICISM: The identification of 'Human Nature' as the fifth single appears correct based on Wikipedia data."""
        
        else:
            return """ATTRIBUTION: [NEITHER]
SEVERITY: [NONE]
CRITICISM: This step appears to be correctly executed. The error likely occurred later in the chain."""
    
    # Build graph and run
    graph = build_example_graph()
    loss_node = graph.get_loss()
    initial_criticism = compute_loss(loss_node, GROUND_TRUTH_1, PROBLEM_1)
    
    executor = BackwardExecutor(llm_fn=mock_llm)
    results = executor.run(graph, initial_criticism)
    
    print("\n" + "=" * 60)
    print("BACKWARD PASS RESULTS")
    print("=" * 60)
    
    for r in results:
        print(f"\nStep {r['input_node'].step_idx} ({r['input_node'].role}):")
        print(f"  Attribution: {r['parsed']['attribution']}")
        print(f"  Severity: {r['parsed']['severity']} ({r['parsed']['severity_score']:.1f})")
    
    print("\n" + "=" * 60)
    print("RANKED SUSPECTS (Most suspicious first)")
    print("=" * 60)
    
    suspects = executor.get_top_suspects(graph, k=3)
    for i, s in enumerate(suspects):
        print(f"\n#{i+1}: Step {s['step_idx']} ({s['role']})")
        print(f"    Suspicion Score: {s['score']:.2f}")
        print(f"    Preview: {s['value_preview'][:100]}...")


if __name__ == "__main__":
    print("Exporting prompts for manual testing...")
    export_prompts_for_manual_testing()
    
    print("\nRunning simulated backward pass...")
    simulate_backward_pass()
