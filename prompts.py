from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Conversation:
    """Represents a conversation turn"""
    role: str
    content: str

@dataclass
class Sample:
    """Represents a test sample for evaluation"""
    id: str
    prompt: str
    history: List[Dict[str, str]]
    parameter: Optional[Dict] = None
    generate_model: Optional[str] = None
    response: Optional[str] = None

class PromptTemplate:
    """Base class for prompt templates"""
    
    @staticmethod
    def format_conversation(history: List[Conversation], prompt: str) -> str:
        """Format conversation history and prompt into model input format"""
        inputs = ""
        for turn in history:
            role_mapper = {
                "system": "<|system|>",
                "user": "<|user|>",
                "assistant": "<|assistant|>"
            }
            inputs += f"{role_mapper[turn.role]}\n{turn.content}"
        inputs += f"<|user|>\n{prompt}<|assistant|>\n"
        return inputs

    @staticmethod
    def get_system_prompt() -> str:
        """Return system prompt for the task"""
        raise NotImplementedError

class NegativePrompt(PromptTemplate):
    """Template for negative description task"""
    
    @staticmethod
    def get_system_prompt() -> str:
        return """You are an AI assistant specialized in analyzing images and identifying what is NOT present.
Your task is to:
1. Carefully observe the image
2. Answer questions about absent objects or features
3. Be explicit and confident in stating what is NOT there
4. Provide clear reasoning based on what you DO see

Format your responses as:
"No, [absent item] is not present. Instead, I see [actual items/scene description]."

Remember:
- Be precise and avoid hedging language
- Base responses only on what you can directly observe
- Don't make assumptions about what might be outside the frame
- If you're uncertain, explain why"""
    
    @staticmethod
    def create_sample(test_case: Dict) -> Sample:
        """Create a sample from test case"""
        sample = Sample(
            id=test_case["id"],
            prompt=test_case["prompt"],
            history=test_case.get("history", [{"role": "system", "content": NegativePrompt.get_system_prompt()}]),
            parameter=test_case.get("parameter"),
            generate_model=test_case.get("generate_model")
        )
        return sample

class MultiEntityPrompt(PromptTemplate):
    """Template for multi-entity recall task"""
    
    @staticmethod
    def get_system_prompt() -> str:
        return """You are an AI assistant specialized in comprehensive scene analysis.
Your task is to:
1. List ALL visible objects in the scene, categorized as:
   - Main objects (prominent/central to the scene)
   - Secondary objects (supporting elements)
   - Environmental features (lighting, atmosphere, background)
2. Describe spatial relationships between objects
3. Note any relevant context or setting details

Format your response as:
Main Objects:
- [object 1]
- [object 2]

Secondary Objects:
- [object 1]
- [object 2]

Environmental Features:
- [feature 1]
- [feature 2]

Remember to be thorough and systematic in your analysis."""
    
    @staticmethod
    def create_sample(test_case: Dict) -> Sample:
        """Create a sample from test case"""
        sample = Sample(
            id=test_case["id"],
            prompt=test_case["prompt"],
            history=test_case.get("history", [{"role": "system", "content": MultiEntityPrompt.get_system_prompt()}]),
            parameter=test_case.get("parameter"),
            generate_model=test_case.get("generate_model")
        )
        return sample

# Add more prompt templates for other tasks...
