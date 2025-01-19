import json
from pathlib import Path

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from json file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            "Please copy config_tmp.json to config.json and fill in your settings."
        ) 