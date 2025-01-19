import aiohttp
import asyncio
import jsonlines
import argparse
import multiprocessing
import tqdm.asyncio
import pdb
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI
from prompts import Sample
from utils.media import encode_file

class ModelAPI:
    """Interface for model API calls"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4-vision-preview",
                 base_url: str = "https://api.openai.com",
                 max_tokens: int = 1024,
                 temperature: float = 0.7,
                 concurrency: int = 5):
        """Initialize OpenAI API interface
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4-vision-preview)
            max_tokens: Maximum tokens in response (default: 1024)
            temperature: Sampling temperature (default: 0.7)
            concurrency: Number of concurrent requests (default: 5)
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
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
        
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_file(sample.parameter.get("image_path", ""))
                    }
                },
                {
                    "type": "text",
                    "text": sample.prompt
                }
            ]
        })

        return messages

    async def _process_sample(self, sample: Sample, semaphore: asyncio.Semaphore) -> Optional[str]:
        """Process a single sample with rate limiting"""
        async with semaphore:
            try:
                messages = self._prepare_messages(sample)
                response = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error processing sample {sample.id}: {str(e)}")
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
                        "generate_model": self.model
                    })

    def __call__(self, samples: List[Sample], output_file: str):
        """Synchronous interface for processing samples
        
        Args:
            samples: List of samples to process
            output_file: Path to output file
        """
        asyncio.run(self._batch_process(samples, output_file))

    def generate(self, prompt: str, file_path: str, type: str = "image") -> str:
        """Simple synchronous interface for single image-text generation
        
        Args:
            prompt: Text prompt
            file_path: Path to path
            type: Type of input (image or video)
            
        Returns:
            Generated response
        """
        assert type in ["image", "video"], "Invalid input type, only 'image' or 'video' allowed"
        type = "image_url" if type == "image" else "video_url"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": type,
                            type: {
                                "url": encode_file(file_path)
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input file containing samples to be processed")
    parser.add_argument("--output_file", type=str, required=True, help="Output file to store the results")
    parser.add_argument("--urls", type=str, nargs="+", required=True, help="List of URLs to send the requests")
    parser.add_argument("--concurrency", type=int, default=256, help="Number of concurrent requests to be sent")
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    urls = args.urls
    concurrency = args.concurrency

    data = [line["input"] for line in jsonlines.open(input_file, "r")]
    # use multiprocessing to send requests
    PER_PROCESS_CONCURRENCY = 256
    num_process = (concurrency + PER_PROCESS_CONCURRENCY - 1) // PER_PROCESS_CONCURRENCY
    writers = [jsonlines.open(f"{output_file}_{i}", "w") for i in range(num_process)]
    datas = [data[i * len(data) // num_process: (i + 1) * len(data) // num_process] for i in range(num_process)]
    with multiprocessing.Pool(num_process) as pool:
        pool.starmap(main, [(i, datas[i], f"{output_file}_{i}", urls, PER_PROCESS_CONCURRENCY) for i in range(num_process)])
    with jsonlines.open(output_file, "w") as writer:
        for i in range(num_process):
            for line in jsonlines.open(f"{output_file}_{i}", "r"):
                writer.write(line)