"""
Example: Applying TextGrad to Failed Trajectory 2

This trajectory involves code execution failures.
Demonstrates handling of Computer Terminal operations.
"""

from core import (
    Graph, Tensor,
    trajectory_to_graph,
    compute_loss,
    register_backward_template,
)

# ============================================================
# FAILED TRAJECTORY 2 - Mercedes Sosa albums
# ============================================================

PROBLEM_2 = "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia."
GROUND_TRUTH_2 = "3"

TRAJECTORY_2 = [
    {
        "role": "Assistant",
        "content": """You are given: (1) a task and advises from your manager with a specific plan and (2) a general task.
Collect information from the general task, follow the suggestions from manager to solve the task.

# General Task
How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)?

# Task and suggestions from manager
## Task description
Determine the number of studio albums published by Mercedes Sosa between 2000 and 2009.

## Plan for solving the task
1. The music historian will provide context and background on Mercedes Sosa's discography.
2. The data analyst will extract the relevant information from the latest 2022 version of English Wikipedia.
3. The fact-checker will verify the accuracy of the extracted information."""
    },
    {
        "role": "DataAnalyst",
        "content": """Let's start with extracting the relevant information from the latest 2022 version of English Wikipedia. 
I will scrape the Wikipedia page to find the discography of Mercedes Sosa and identify the studio albums published between 2000 and 2009.

I'll use the function `get_wikipedia_text` to retrieve the text content from Mercedes Sosa's Wikipedia page."""
    },
    {
        "role": "DataAnalyst",
        "content": """I'll use the Bing Search API to fetch the specific page from Wikipedia and then parse it for the necessary discography details.

```python
from functions import perform_web_search

# Search for Mercedes Sosa's discography on Wikipedia
search_results = perform_web_search("Mercedes Sosa discography site:en.wikipedia.org")
print(search_results)
```

Computer_terminal, please run this code to retrieve the search results."""
    },
    {
        "role": "ComputerTerminal",
        "content": """exitcode: 0 (execution succeeded)
Code output: Search Result 1:
Mercedes Sosa - Wikipedia
https://en.wikipedia.org/wiki/Mercedes_Sosa
...
Search Result 4:
Cantora, un Viaje Íntimo - Wikipedia
https://en.wikipedia.org/wiki/Cantora,_un_Viaje_%C3%8Dntimo
Cantora, un Viaje Íntimo (English: Cantora, An Intimate Journey) is a double album by Argentine singer Mercedes Sosa, released on 2009...
...
Search Result 5:
La Maza - Wikipedia
...Mercedes Sosa performed the song... before releasing a studio recording of the duet on her thirty-second studio album Cantora 1 on 27 March 2009..."""
    },
    {
        "role": "DataAnalyst",
        "content": """Based on the search results, the most relevant link for obtaining Mercedes Sosa's discography is the Wikipedia page titled "Mercedes Sosa - Wikipedia".

I'll use the `scrape_wikipedia_tables` function to extract the necessary data from the Wikipedia page:

```python
from functions import scrape_wikipedia_tables

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Mercedes_Sosa"

# Keyword to find the relevant table
header_keyword = "Discography"

# Scrape the Wikipedia table
table_data = scrape_wikipedia_tables(url, header_keyword)
print(table_data)
```

Computer_terminal, please run this code to scrape the table data."""
    },
    {
        "role": "ComputerTerminal",
        "content": """exitcode: 0 (execution succeeded)
Code output: []"""
    },
    {
        "role": "DataAnalyst",
        "content": """It appears that the function `scrape_wikipedia_tables` did not return any data. This might indicate that the discography section is not in a table format on the Wikipedia page.

I'll write a Python script to scrape the text content and parse it manually:

```python
import requests
from bs4 import BeautifulSoup

# URL of the Wikipedia page
url = "https://en.wikipedia.org/wiki/Mercedes_Sosa"

# Fetch the page content
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Locate the discography section
discography_section = soup.find('span', id='Discography')
discography_content = discography_section.find_next('ul')

# Extract album names and release years
albums = []
for li in discography_content.find_all('li'):
    text = li.get_text()
    if any(year in text for year in map(str, range(2000, 2010))):
        albums.append(text)

print(albums)
```"""
    },
    {
        "role": "ComputerTerminal",
        "content": """exitcode: 1 (execution failed)
Code output: Traceback (most recent call last):
  File "/home/CaptainAgent/groupchat/tmp_code_1d854a305659e6ddcdd3fb5c230984e1.py", line 13, in <module>
    discography_content = discography_section.find_next('ul')
AttributeError: 'NoneType' object has no attribute 'find_next'"""
    },
    {
        "role": "DataAnalyst",
        "content": """It seems the specific element with id 'Discography' was not found. Let me adjust the approach:

```python
import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/Mercedes_Sosa"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Function to find the relevant section
def find_section(soup, header_text):
    for header in soup.find_all(['h2', 'h3', 'h4']):
        if header_text.lower() in header.get_text().lower():
            return header.find_next_sibling('ul')
    return None

discography_content = find_section(soup, "Discography")

albums = []
if discography_content:
    for li in discography_content.find_all('li'):
        text = li.get_text()
        if any(year in text for year in map(str, range(2000, 2010))):
            albums.append(text)

print(albums)
```"""
    },
    {
        "role": "ComputerTerminal",
        "content": """exitcode: 0 (execution succeeded)
Code output: []"""
    },
]


def build_trajectory_2_graph():
    """
    Build a graph with awareness of code execution patterns.
    
    For code execution trajectories:
    - Code request -> Terminal result is a tight dependency
    - Failed execution should have high suspicion
    """
    graph = Graph(problem=PROBLEM_2, ground_truth=GROUND_TRUTH_2)
    
    nodes = []
    for i, step in enumerate(TRAJECTORY_2):
        # Determine predecessors based on role patterns
        if i == 0:
            preds = []
        elif step['role'] == 'ComputerTerminal':
            # Terminal depends on the code that was sent (previous step)
            preds = [nodes[-1]]
        else:
            # Other agents depend on all recent context
            # But primarily the last terminal output if it exists
            preds = [nodes[-1]] if nodes else []
        
        node = graph.add_node(
            value=step['content'],
            role=step['role'],
            step_idx=i,
            predecessors=preds,
        )
        nodes.append(node)
    
    graph.set_loss(nodes[-1])
    return graph


def export_trajectory_2_prompts(output_file: str = "prompts_trajectory_2.txt"):
    """Export backward prompts for trajectory 2."""
    graph = build_trajectory_2_graph()
    loss_node = graph.get_loss()
    
    initial_criticism = f"""FAILURE DETECTED

The multi-agent system attempted to solve:
{PROBLEM_2}

Expected answer: {GROUND_TRUTH_2} (3 studio albums)

The system returned an empty result, failing to find any albums.

The actual albums published by Mercedes Sosa between 2000-2009 include:
- Cantora 1 (2009)
- Cantora 2 (2009)  
- Corazón Libre (2005)

The system's web scraping approach failed to extract this information.
Trace back through the execution chain to identify the decisive error."""
    
    templates = graph.linearize(initial_criticism)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TEXTGRAD BACKWARD PASS PROMPTS - TRAJECTORY 2\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Problem: {PROBLEM_2}\n")
        f.write(f"Expected Answer: {GROUND_TRUTH_2}\n")
        f.write(f"Number of steps: {len(TRAJECTORY_2)}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("INITIAL LOSS\n")
        f.write("=" * 80 + "\n")
        f.write(initial_criticism)
        f.write("\n\n")
        
        for i, item in enumerate(templates):
            f.write("=" * 80 + "\n")
            f.write(f"BACKWARD PROMPT {i+1}\n")
            f.write(f"Output: Step {item['output_step_idx']} ({item['output_node'].role})\n")
            f.write(f"Input: Step {item['step_idx']} ({item['input_node'].role})\n")
            f.write("=" * 80 + "\n\n")
            f.write(item['prompt'])
            f.write("\n\n")
    
    print(f"Exported {len(templates)} prompts to {output_file}")
    return templates


if __name__ == "__main__":
    export_trajectory_2_prompts()
