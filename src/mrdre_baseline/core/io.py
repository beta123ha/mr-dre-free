"""
Utility functions for the multi-turn deep research agent project.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file into a list of dictionaries.
    
    Args:
        filepath: Path to the JSONL file
    
    Returns:
        List of dictionaries, one per line in the JSONL file
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """
    Save list of dictionaries to JSONL file.
    
    Args:
        data: List of dictionaries to save
        filepath: Path where the JSONL file should be saved
    """
    # Atomic checkpoint: a Kaggle session may be interrupted mid-write.
    import os
    import tempfile
    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for item in data:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def append_jsonl(entry: Dict[str, Any], filepath: str):
    """Atomically append one JSONL row (small academic datasets only).

    A checkpoint archive can be written concurrently with the evaluator.
    Replacing a complete file avoids the snapshot observing half a JSON line.
    """
    target = Path(filepath)
    old = load_jsonl(str(target)) if target.exists() else []
    old.append(entry)
    save_jsonl(old, str(target))


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file into a dictionary.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary containing the JSON data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2):
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path where the JSON file should be saved
        indent: Number of spaces for indentation (default: 2)
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_dataset(dataset_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load dataset JSONL file and index by question ID.
    
    Args:
        dataset_path: Path to dataset JSONL file
    
    Returns:
        Dict mapping question ID to dataset item
    """
    dataset_items = load_jsonl(dataset_path)
    return {item["id"]: item for item in dataset_items}


def model_name(model: Any) -> str:
    """
    Extract model name from a model object or string.
    
    Args:
        model: Model object (with model_name attribute), string, or any object
    
    Returns:
        String representation of the model name
    """
    if hasattr(model, "model_name"):
        return getattr(model, "model_name")
    if isinstance(model, str):
        return model
    return getattr(model, "__class__", type(model)).__name__


def get_file_path(dir_name: str, model_name: str, round_num: int, mode: str, num_items: int = None, use_reviser: bool = False) -> str:
    """
    Generate the output file path for the current round.

    The resulting path follows the pattern:
        {dir_name}/{mode}/{model_name}/gen_turn{round_num}.jsonl

    Args:
        dir_name: Base directory for outputs
        model_name: Name of the model
        round_num: Current round number
        mode: Round type (init, reflection, checklist_feedback, formal_feedback)
        num_items: Number of checklist items (only for checklist_feedback)
        use_reviser: Whether to use reviser agent

    Returns:
        Generated file path as string
    """
    import os
    
    if round_num == 1:
        mode = 'init'

    # Handle refined modes
    original_mode = mode
    if mode.startswith('refined_'):
        original_mode = mode[len('refined_'):]
    
    if num_items is not None and original_mode == 'checklist_feedback':
        mode = f'{mode}_k={num_items}'

    folder_model_name = f"{model_name}_reviser" if use_reviser else model_name
    file_name = f"gen_turn{round_num}.jsonl"

    return os.path.join(dir_name, mode, folder_model_name, file_name)

