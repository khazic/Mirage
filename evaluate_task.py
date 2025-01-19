import os
import json
import logging
from pathlib import Path
from datetime import datetime
from request import ModelAPI
from evaluator import MmomentEvaluator, NegativeDescriptionTask
from utils.config import load_config

def setup_logging(config):
    """Setup logging configuration"""
    log_dir = Path(config["logging"]["file"]).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=config["logging"]["level"],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["logging"]["file"]),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories(config):
    """Ensure required directories exist"""
    for dir_path in [
        config["data"]["input_dir"],
        config["data"]["output_dir"],
        config["data"]["results_dir"]
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def run_evaluation(config, logger):
    """Run evaluation pipeline"""
    # Initialize API with new config format
    server = config.get("server", "openai")
    api_config = config["api"][server]
    
    # Initialize API with both configs
    model_api = ModelAPI(
        api_key=api_config["api_key"],
        base_url=api_config.get("base_url", "https://api.openai.com/v1"),
        image_config=api_config["image_config"],
        video_config=api_config["video_config"],
        concurrency=api_config["concurrency"]
    )
    
    # Setup task
    test_file = os.path.join(config["data"]["input_dir"], "negative_test.json")
    logger.info(f"Loading test cases from {test_file}")
    
    # Verify test file exists
    if not os.path.exists(test_file):
        logger.warning(f"Test file not found at {test_file}, creating example test file")
        # Create example test data
        example_data = {
            "image_tests": [
                {
                    "id": "neg_img_001",
                    "prompt": "Is there a banana on the table?",
                    "image_path": "data/images/apple_on_table.jpg",
                    "ground_truth": "No, there is no banana on the table. There is only an apple.",
                    "difficulty": "easy"
                }
            ],
            "video_tests": [
                {
                    "id": "neg_vid_001",
                    "prompt": "Is there a person running in the video at 5 seconds?",
                    "video_path": "data/videos/walking_scene.mp4",
                    "timestamp": 5,
                    "ground_truth": "No, at 5 seconds there is only a person walking slowly.",
                    "difficulty": "medium"
                }
            ]
        }
        os.makedirs(os.path.dirname(test_file), exist_ok=True)
        with open(test_file, 'w') as f:
            json.dump(example_data, f, indent=2)
    
    tasks = {
        "negative": NegativeDescriptionTask(test_file)
    }
    
    # Initialize evaluator
    evaluator = MmomentEvaluator(model_api, tasks)
    
    # Run evaluation
    logger.info("Starting evaluation")
    results = evaluator.evaluate(["negative"])
    
    # Generate report
    logger.info("Generating evaluation report")
    report = evaluator.generate_report(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(config["data"]["results_dir"])
    
    # Save metrics with correct model information
    metrics_file = result_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "task": "negative",
            "image_model": api_config["image_config"]["model"],
            "video_model": api_config["video_config"]["model"],
            "metrics": results["negative"].metrics,
            "score": results["negative"].score,
            "config": {
                "image": api_config["image_config"],
                "video": api_config["video_config"]
            },
            "timestamp": timestamp
        }, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Save HTML report
    report_file = result_dir / f"report_{timestamp}.html"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")
    
    return results

def main():
    config = load_config()
    logger = setup_logging(config)
    ensure_directories(config)
    
    try:
        logger.info("Starting evaluation pipeline")
        results = run_evaluation(config, logger)
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 