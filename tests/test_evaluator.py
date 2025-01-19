import unittest
from pathlib import Path
import json
import tempfile
import shutil
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator import MmomentEvaluator, NegativeDescriptionTask, MultiEntityTask
from request import ModelAPI

class MockModelAPI:
    """Mock model API for testing"""
    def __init__(self):
        self.called_with = []
    
    def __call__(self, samples, output_file):
        self.called_with.append((samples, output_file))
        # Mock responses
        responses = {
            "neg_001": "No, there is no banana on the table. Instead, I see an apple.",
            "neg_002": "No, there are no green bananas. I see only yellow bananas.",
            "ent_001": """Main Objects:
- students (approximately 20)
- teacher at front
- blackboard with writing

Secondary Objects:
- backpacks under desks
- books on desks
- pencils and stationery

Environmental Features:
- fluorescent lighting
- windows with curtains
- clock on wall"""
        }
        
        with open(output_file, 'w') as f:
            for sample in samples:
                sample.response = responses.get(sample.id, "Mock response")
                json.dump({
                    "id": sample.id,
                    "response": sample.response
                }, f)
                f.write('\n')

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.negative_cases = [
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
            }
        ]
        
        self.multi_entity_cases = [
            {
                "id": "ent_001",
                "prompt": "List all objects you can see in this classroom.",
                "image_path": "images/classroom.jpg",
                "ground_truth": {
                    "main_objects": ["students", "teacher", "blackboard"],
                    "secondary_objects": ["backpacks", "books", "pencils"],
                    "environmental": ["lights", "windows", "clock"]
                },
                "difficulty": "medium"
            }
        ]
        
        self.negative_file = Path(self.temp_dir) / "negative_cases.json"
        self.multi_entity_file = Path(self.temp_dir) / "multi_entity_cases.json"
        
        with open(self.negative_file, 'w') as f:
            json.dump(self.negative_cases, f)
        with open(self.multi_entity_file, 'w') as f:
            json.dump(self.multi_entity_cases, f)
        
        self.model_api = MockModelAPI()
        
        self.tasks = {
            "negative": NegativeDescriptionTask(str(self.negative_file)),
            "multi_entity": MultiEntityTask(str(self.multi_entity_file))
        }
        
        self.evaluator = MmomentEvaluator(self.model_api, self.tasks)

    def test_negative_task_evaluation(self):
        """Test negative description task evaluation"""
        results = self.evaluator.evaluate(["negative"])
        
        self.assertIn("negative", results)
        result = results["negative"]
        
        self.assertIsInstance(result.score, float)
        self.assertIn("logical_consistency", result.metrics)
        self.assertIn("relevance", result.metrics)
        self.assertIn("clarity", result.metrics)
        self.assertIn("overall", result.metrics)

    def test_multi_entity_task_evaluation(self):
        """Test multi-entity recall task evaluation"""
        results = self.evaluator.evaluate(["multi_entity"])
        
        self.assertIn("multi_entity", results)
        result = results["multi_entity"]
        
        self.assertIsInstance(result.score, float)
        self.assertIn("entity_recall", result.metrics)
        self.assertIn("attribute_accuracy", result.metrics)
        self.assertIn("relationship_coverage", result.metrics)
        self.assertIn("overall", result.metrics)

    def test_report_generation(self):
        """Test evaluation report generation"""
        results = self.evaluator.evaluate()
        report = self.evaluator.generate_report(results)
        
        self.assertIsInstance(report, str)
        self.assertIn("<html>", report)
        self.assertIn("Evaluation Report", report)
        self.assertIn("negative", report)
        self.assertIn("multi_entity", report)

    def test_invalid_task(self):
        """Test handling of invalid task names"""
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(["invalid_task"])

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main() 
    
# run all tasks
# python -m unittest tests/test_evaluator.py

# run specific tasks
# python -m unittest tests.test_evaluator.TestEvaluator.test_negative_task_evaluation