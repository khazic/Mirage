import base64
from pathlib import Path
from typing import Union

def encode_file(file_path: Union[str, Path]) -> str:
    """Encode file to base64 string
    
    Args:
        file_path: Path to image file
        
    Returns:
        base64 encoded string with data URI scheme
    """
    file_path = Path(file_path)
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')