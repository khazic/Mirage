from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import jsonlines
from prompts import *

@dataclass
class EvalResult:
    """Evaluation result data class"""
    score: float
    metrics: Dict[str, float] 
    details: Optional[Dict[str, Any]] = None
    html: Optional[str] = None

class TaskBase(ABC):
    """Base class for evaluation tasks"""
    
    def __init__(self, data_path: str):
        """Initialize task with test cases"""
        self.test_cases = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """Load test cases from file"""
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                return json.load(f)
        elif data_path.endswith('.jsonl'):
            return [line for line in jsonlines.open(data_path)]
        else:
            raise ValueError("Unsupported file format. Use .json or .jsonl")

    @abstractmethod
    def evaluate_response(self, sample: Sample) -> Dict[str, float]:
        """Evaluate a single response"""
        pass
    
    def __call__(self, model_api) -> EvalResult:
        """Execute evaluation and return results"""
        # Create samples
        samples = [self._create_sample(case) for case in self.test_cases]

        # Get model responses
        output_file = f"results/{self.__class__.__name__}.jsonl"
        model_api(samples, output_file)

        # Evaluate responses
        metrics_list = [self.evaluate_response(sample) for sample in samples]
        
        # Aggregate metrics
        aggregated = self._aggregate_metrics(metrics_list)
        
        return EvalResult(
            score=aggregated["overall"],
            metrics=aggregated,
            details={"samples": samples}
        )

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across samples"""
        aggregated = {}
        for metric in metrics_list[0].keys():
            aggregated[metric] = sum(m[metric] for m in metrics_list) / len(metrics_list)
        return aggregated

    @abstractmethod
    def _create_sample(self, test_case: Dict) -> Sample:
        """Create sample from test case"""
        pass

class NegativeDescriptionTask(TaskBase):
    """Task for evaluating negative description accuracy"""

    def evaluate_response(self, sample: Sample) -> Dict[str, float]:
        return {
            "logical_consistency": 0.0,
            "relevance": 0.0,
            "clarity": 0.0,
            "overall": 0.0
        }

    def _create_sample(self, test_case: Dict) -> Sample:
        return NegativePrompt.create_sample(test_case)

class MultiEntityTask(TaskBase):
    """Task for evaluating multi-entity recall"""

    def evaluate_response(self, sample: Sample) -> Dict[str, float]:
        return {
            "entity_recall": 0.0,
            "attribute_accuracy": 0.0,
            "relationship_coverage": 0.0,
            "overall": 0.0
        }

    def _create_sample(self, test_case: Dict) -> Sample:
        return MultiEntityPrompt.create_sample(test_case)

class MmomentEvaluator:
    """Evaluator for executing specified evaluation tasks"""
    
    def __init__(self, model_api, tasks: Dict[str, TaskBase]):
        """
        Args:
            model_api: Model API interface
            tasks: Mapping of task names to task instances
        """
        self.model_api = model_api
        self.tasks = tasks
        
    def evaluate(self, task_names: Optional[List[str]] = None) -> Dict[str, EvalResult]:
        """Execute specified evaluation tasks
        Args:
            task_names: List of task names to execute, None for all tasks
        Returns:
            Dict mapping task names to evaluation results
        """
        if task_names is None:
            task_names = list(self.tasks.keys())
            
        results = {}
        for name in task_names:
            if name not in self.tasks:
                raise ValueError(f"Unknown task: {name}")
            results[name] = self.tasks[name](self.model_api)
            
        return results

    def generate_report(self, results: Dict[str, EvalResult]) -> str:
        """Generate evaluation report
        Args:
            results: Evaluation results by task
        Returns:
            HTML formatted report
        """
        report = "<html><body>"
        report += "<h1>Evaluation Report</h1>"
        
        for task_name, result in results.items():
            report += f"<h2>{task_name}</h2>"
            report += "<h3>Overall Score: {:.2f}</h3>".format(result.score)
            
            report += "<h3>Metrics:</h3><ul>"
            for metric, value in result.metrics.items():
                report += f"<li>{metric}: {value:.2f}</li>"
            report += "</ul>"
            
            if result.html:
                report += result.html
                
        report += "</body></html>"
        return report
