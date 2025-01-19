# Example test cases for different tasks

negative_cases = [
    {
        "id": "neg_001",
        "prompt": "Is there a banana on the table?",
        "image_path": "images/apple_on_table.jpg",
        "ground_truth": "No, there is no banana on the table. There is only an apple.",
        "difficulty": "easy"
    },
    {
        "id": "neg_002",
        "prompt": "Are there any green bananas in the image?",
        "image_path": "images/yellow_bananas.jpg",
        "ground_truth": "No, there are only yellow bananas in the image.",
        "difficulty": "medium"
    },
    {
        "id": "neg_003",
        "prompt": "Is this a street with streetlights?",
        "image_path": "images/dark_street.jpg",
        "ground_truth": "No, this is a dark street without any streetlights.",
        "difficulty": "hard"
    }
]

multi_entity_cases = [
    {
        "id": "ent_001",
        "prompt": "List all objects you can see in this classroom.",
        "image_path": "images/classroom.jpg",
        "ground_truth": {
            "main_objects": ["students", "teacher", "blackboard", "desks", "chairs"],
            "secondary_objects": ["backpacks", "books", "pencils", "windows", "curtains"],
            "environmental": ["fluorescent lights", "wall clock", "bulletin board"]
        },
        "difficulty": "medium"
    }
]

position_cases = [
    {
        "id": "pos_001",
        "prompt": "Where is the cup relative to the laptop?",
        "image_path": "images/desk_setup.jpg",
        "ground_truth": "The cup is to the right of the laptop",
        "difficulty": "easy"
    }
]

from Mmoment.utils.config import load_config
from Mmoment.request import ModelAPI

# Load configuration
config = load_config()

# Initialize API with config
model_api = ModelAPI(
    api_key=config["api"]["openai"]["api_key"],
    model=config["api"]["openai"]["model"],
    max_tokens=config["api"]["openai"]["max_tokens"],
    temperature=config["api"]["openai"]["temperature"],
    concurrency=config["api"]["openai"]["concurrency"]
) 