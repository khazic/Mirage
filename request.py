import aiohttp
import asyncio
import jsonlines
import argparse
import multiprocessing
import tqdm.asyncio
import pdb
import logging
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI
from prompts import Sample
from utils.media import encode_file

class ModelAPI:
    """Interface for model API calls"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.openai.com/v1",
                 image_config: dict = None,
                 video_config: dict = None,
                 concurrency: int = 5):
        """Initialize OpenAI API interface
        
        Args:
            api_key: OpenAI API key
            base_url: API base URL
            image_config: Configuration for image processing
            video_config: Configuration for video processing
            concurrency: Number of concurrent requests
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        # Default configs
        self.image_config = image_config or {
            "model": "gpt-4-vision-preview",
            "max_tokens": 1024,
            "temperature": 0.7
        }
        
        self.video_config = video_config or {
            "model": "gpt-4-vision-preview",
            "max_tokens": 2048,
            "temperature": 0.7
        }
        
        self.concurrency = concurrency

    def _prepare_messages(self, sample: Sample) -> List[dict]:
        """Prepare messages for OpenAI API"""
        messages = []

        for turn in sample.history:
            if turn["role"] == "system":
                messages.append({
                    "role": "system",
                    "content": turn["content"]
                })
        
        # Handle different media types
        if sample.media_type == "image":
            image_path = sample.parameter.get("image_path", "")
            if not image_path:
                raise ValueError(f"No image path provided for sample {sample.id}")
            media_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_file(image_path)}"
                }
            }
        else:  # video
            video_path = sample.parameter.get("video_path", "")
            if not video_path:
                raise ValueError(f"No video path provided for sample {sample.id}")
            media_content = {
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/mp4;base64,{encode_file(video_path)}"
                }
            }
            # Add timestamp if provided
            if "timestamp" in sample.parameter:
                media_content["timestamp"] = sample.parameter["timestamp"]
        
        messages.append({
            "role": "user",
            "content": [
                media_content,
                {
                    "type": "text",
                    "text": sample.prompt
                }
            ]
        })

        return messages

    def _get_config_for_sample(self, sample: Sample) -> dict:
        """Get appropriate configuration based on media type"""
        if sample.media_type == "video":
            if not self.video_config:
                raise ValueError("Video configuration not set")
            return self.video_config
        return self.image_config

    async def _process_sample(self, sample: Sample, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Process a single sample with rate limiting"""
        async with semaphore:
            try:
                messages = self._prepare_messages(sample)
                config = self._get_config_for_sample(sample)
                
                response = await self.async_client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    max_tokens=config["max_tokens"],
                    temperature=config["temperature"]
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error processing sample {sample.id}: {str(e)}")
                return None

    async def _batch_process(self, samples: List[Sample], output_file: str):
        """Process samples in batches with concurrency control"""
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [self._process_sample(sample, semaphore) for sample in samples]
        
        responses = await tqdm.asyncio.tqdm_asyncio.gather(*tasks)
        
        # Write results
        with jsonlines.open(output_file, 'w') as writer:
            for sample, response in zip(samples, responses):
                if response:
                    sample.response = response
                    writer.write({
                        "id": sample.id,
                        "prompt": sample.prompt,
                        "response": response,
                        "parameter": sample.parameter,
                        "generate_model": self.image_config["model"]
                    })

    def __call__(self, samples: List[Sample], output_file: str):
        """Synchronous interface for processing samples
        
        Args:
            samples: List of samples to process
            output_file: Path to output file
        """
        asyncio.run(self._batch_process(samples, output_file))

    def generate(self, prompt: str, file_path: str, type: str = "image") -> str:
        """Simple synchronous interface for single media generation"""
        config = self.video_config if type == "video" else self.image_config
        
        response = self.client.chat.completions.create(
            model=config["model"],
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": f"{type}_url",
                        f"{type}_url": {
                            "url": f"data:{type}/{'mp4' if type == 'video' else 'jpeg'};base64,{encode_file(file_path)}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"]
        )
        return response.choices[0].message.content

def parse_inputs(sample: Sample) -> str:
    """Parse sample into model input format"""
    if "<|user|>" not in sample.prompt:
        inputs = ""
        for history in sample.history:
            role_mapper = {
                "system": "<|system|>",
                "user": "<|user|>",
                "assistant": "<|assistant|>"
            }
            inputs += f"{role_mapper[history['role']]}\n{history['content']}"
        inputs += f"<|user|>\n{sample.prompt}<|assistant|>\n"
    else:
        inputs = sample.prompt
    return inputs