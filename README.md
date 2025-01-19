# Mmoment
Multi-Modal Modeling and Evaluation on Novel Tasks

## Introduction
This project aims to evaluate and improve Large Vision Language Models (LVLMs) in multimodal tasks. We designed a series of benchmarks to identify model strengths and limitations in handling complex tasks.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/khazic/Mmoment
cd Mmoment
```

2. Install dependencies:
```bash
pip install -e .
```

3. Configure:
```bash
cp config_tmp.json config.json
# Edit config.json with your API key and settings
```

## Quick Start

1. Prepare test data:
```bash
mkdir -p data/inputs/images
mkdir -p data/inputs/videos
# Place your test images and videos in respective directories
```

2. Run single test:
```bash
python test_api.py
```

3. Run full evaluation:
```bash
python evaluate_task.py
```

## Project Structure
```
Mmoment/
├── data/
│   ├── inputs/          # Test data
│   │   ├── images/      # Test images
│   │   └── videos/      # Test videos
│   └── outputs/         # Model outputs
├── results/             # Evaluation results
├── logs/               # Log files
└── config.json         # Configuration file
```

## Benchmark Tasks

### 1. Negative Description Accuracy
- **Purpose**: Evaluate model accuracy in handling negative samples
- **Example**:
```json
{
    "id": "neg_001",
    "prompt": "Is there a banana on the table?",
    "image_path": "data/inputs/images/apple_on_table.jpg",
    "ground_truth": "No, there is no banana on the table."
}
```

### 2. Multi-Entity Recall
- **Purpose**: Test model's recall ability in multi-object scenes
- **Metrics**: Entity recall rate, attribute accuracy, relationship coverage

[Additional tasks as described in README-desc.md...]

## Configuration

Main config.json settings:
```json
{
    "api": {
        "openai": {
            "api_key": "your-api-key",
            "image_config": {
                "model": "gpt-4-vision-preview",
                "max_tokens": 1024
            },
            "video_config": {
                "model": "gpt-4-vision-preview",
                "max_tokens": 2048
            }
        }
    }
}
```

## Evaluation Results

Results are saved in results/ directory:
- metrics_{timestamp}.json: Detailed metrics
- report_{timestamp}.html: Visual report

## Custom Evaluation

1. Create new test data:
```json
# data/inputs/custom_test.json
{
    "image_tests": [],
    "video_tests": []
}
```

2. Implement new evaluation task:
```python
from evaluator import TaskBase

class CustomTask(TaskBase):
    def evaluate_response(self, sample: Sample) -> Dict[str, float]:
        # Implement evaluation logic
        pass
```

## Notes

1. API Usage
- Ensure sufficient API quota
- Video evaluation may require more tokens
- Set reasonable concurrency

2. Data Preparation
- Supported image formats: jpg, png
- Supported video formats: mp4
- Control media file size

3. Security
- Don't commit config.json to version control
- Protect API keys

## TODO
- [ ] Add more evaluation scenarios
- [ ] Support more model backends
- [ ] Improve evaluation metrics
- [ ] Add batch testing support
- [ ] Enhance report visualization

## Contributing

Issues and Pull Requests are welcome!

## License

MIT License
