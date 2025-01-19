import os
import json
from request import ModelAPI
from prompts import Sample
from utils.config import load_config

def test_single_image(config):
    """Test single image generation"""
    
    # Initialize API with config
    server = config.get("server", "openai")
    api_config = config["api"][server]
    
    model_api = ModelAPI(
        api_key=api_config["api_key"],
        base_url=api_config.get("base_url", "https://api.openai.com/v1"),
        image_config=api_config["image_config"],
        video_config=api_config["video_config"],
        concurrency=api_config["concurrency"]
    )
    
    # Test image path from config
    # image_path = config["test_image_path"]
    image_path = "/Users/ubec/Downloads/test.png"
    
    # Test prompt
    prompt = "What objects can you see in this image? Please describe them in detail."
    
    # Create a sample
    sample = Sample(
        id="test_001",
        prompt=prompt,
        parameter={"image_path": image_path},
        history=[],
        generate_model=config["api"][server]["model"]
    )
    
    # Test batch processing
    output_file = "test_output.jsonl"
    model_api([sample], output_file)
    
    # Test direct generation
    response = model_api.generate(prompt, image_path)
    print("\nDirect generation response:")
    print(response)

if __name__ == "__main__":
    config = load_config()
    test_single_image(config) 