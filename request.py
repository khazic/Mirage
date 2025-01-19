import aiohttp
import asyncio
import jsonlines
import argparse
import multiprocessing
import tqdm.asyncio
import pdb
from typing import List, Optional
from openai import OpenAI, AsyncOpenAI
from .prompts import Sample

async def fetch(url, headers, data, writer, semaphore):
    retry_times = 0
    while True:
        try:
            if retry_times > 5:
                print("retry 5 times, pass")
                return
            retry_times += 1
            async with semaphore:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(-1)) as session:
                    sample = data["sample"]
                    del data["sample"]
                    async with session.post(url, headers=headers, json=data, timeout=300) as response:
                        response_json = await response.json()
                        sample.response = response_json["generated_text"]
                        writer.write({"prompt": sample.prompt,
                                      "response": sample.response,
                                      "id": sample.id,
                                      "history": sample.history,
                                      "parameter": sample.parameter,
                                      "generate_model": sample.generate_model
                                      })
                        return
        except Exception as e:
            await asyncio.sleep(1)


async def async_main(pid, data, writer, urls, concurrency):
    headers = {
        "Content-Type": "application/json",
    }
    def parse_inputs(_sample):
        if "<|user|>" not in _sample.prompt:
            inputs = ""
            for _history in _sample.history:
                role_mapper = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
                inputs += f"{role_mapper[_history['role']]}\n{_history['content']}"
            inputs += f"<|user|>\n{_sample.prompt}<|assistant|>\n"
        else:
            inputs = _sample.prompt
        return inputs
    post_requests = [{
        "stream": False,
        "inputs": parse_inputs(sample),
        "parameters": {
            "best_of": 1,
            "decoder_input_details": False,
            "details": False,
            "do_sample": True,
            "max_new_tokens": 1024,
            "return_full_text": False,
            "seed": None,
            "temperature": 0.95,
            "top_p": 0.8,
            "truncate": 7168,
            "stop": ["<|endoftext|>", "<|user|>", "<|observation|>"]
        },
        "sample": sample
    } for sample in data]
    semaphore = asyncio.Semaphore(concurrency)
    # conn = aiohttp.TCPConnector(limit=concurrency*2)
    # async with aiohttp.ClientSession(connector=conn) as session:
    tasks = [fetch(urls[i % len(urls)], headers, post_request, writer, semaphore) for i, post_request in enumerate(post_requests)]
    await tqdm.asyncio.tqdm_asyncio.gather(*tasks, position=pid, desc=f"Process {pid}")


def main(pid, data, output_file, urls, concurrency):
    writer = jsonlines.open(output_file, "w")
    asyncio.run(async_main(pid, data, writer, urls, concurrency))


def request_calls(_prompts, _urls, output_file, concurrency=1024, _PER_PROCESS_CONCURRENCY=128):
    # use multiprocessing to send requests
    num_process = (concurrency + _PER_PROCESS_CONCURRENCY - 1) // _PER_PROCESS_CONCURRENCY
    writers = [jsonlines.open(output_file.replace(".jsonl", f"_{i}.jsonl"), "w") for i in range(num_process)]
    datas = [_prompts[i * len(_prompts) // num_process: (i + 1) * len(_prompts) // num_process] for i in range(num_process)]
    with multiprocessing.Pool(num_process) as pool:
        pool.starmap(main,
                     [(i, datas[i], output_file.replace(".jsonl", f"_{i}.jsonl"), _urls, _PER_PROCESS_CONCURRENCY) for i in range(num_process)])
    with jsonlines.open(output_file, "w") as writer:
        for i in range(num_process):
            for line in jsonlines.open(output_file.replace(".jsonl", f"_{i}.jsonl"), "r"):
                writer.write(line)


class ModelAPI:
    """Interface for model API calls"""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "gpt-4-vision-preview",
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
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.concurrency = concurrency
        
    def _prepare_messages(self, sample: Sample) -> List[dict]:
        """Prepare messages for OpenAI API"""
        messages = []
        
        # Add system prompt if present
        for turn in sample.history:
            if turn["role"] == "system":
                messages.append({
                    "role": "system",
                    "content": turn["content"]
                })
        
        # Add image and prompt
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image_url": sample.parameter.get("image_path", "")
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

    def generate(self, prompt: str, image_path: str) -> str:
        """Simple synchronous interface for single image-text generation
        
        Args:
            prompt: Text prompt
            image_path: Path to image
            
        Returns:
            Generated response
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image_url": image_path
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