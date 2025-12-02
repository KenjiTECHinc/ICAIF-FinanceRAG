import json
import pandas as pd

def load_jsonl(file_path):
    """
    Load data from JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    print(f"Loading data from {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} items")
    return data