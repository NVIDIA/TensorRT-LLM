#!/usr/bin/env python3
"""
Simple client to test TensorRT-LLM OpenAPI server using requests library.

Prerequisites:
- Start TensorRT-LLM server manually (e.g., using trtllm-serve or fastapi_server.py)
- Then run this script to test text generation

Usage:
    python disagg_serving_test.py --server-url http://localhost:8000
    python disagg_serving_test.py --server-url http://localhost:8000 --benchmark
    python disagg_serving_test.py --server-url http://localhost:8000 --benchmark --dataset-prompts
    python disagg_serving_test.py --server-url http://localhost:8000 --save-results
"""

import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock

import click
import requests


class PromptDataset:
    """Collection of meaningful prompts for benchmarking"""

    # Question Answering prompts
    QA_PROMPTS = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "How does photosynthesis work?",
        "What are the main causes of climate change?",
        "Describe the process of protein synthesis.",
        "What is the difference between AI and machine learning?",
        "How do vaccines work?",
        "What causes earthquakes?",
        "Explain the concept of compound interest.",
        "What is the water cycle?",
        "How do solar panels generate electricity?",
        "What are the benefits of regular exercise?",
        "Explain the theory of relativity.",
        "What is cryptocurrency?",
        "How does the human immune system work?",
        "What causes inflation in economics?",
        "Describe the structure of DNA.",
        "What are renewable energy sources?",
        "How do antibiotics fight infections?",
        "What is the greenhouse effect?",
    ]

    # Creative Writing prompts
    CREATIVE_PROMPTS = [
        "Write a short story about a time traveler who gets stuck in the past.",
        "Describe a world where gravity works in reverse.",
        "Create a dialogue between two AIs meeting for the first time.",
        "Write about a character who can taste emotions.",
        "Describe a city built entirely underwater.",
        "Write a story from the perspective of the last book on Earth.",
        "Create a scene where colors have sounds.",
        "Write about someone who discovers they can communicate with plants.",
        "Describe a world where memories can be traded like currency.",
        "Write a story about a library that exists between dimensions.",
        "Create a character who ages backwards.",
        "Write about a planet where it rains upwards.",
        "Describe a society where lying is physically impossible.",
        "Write a story about someone who collects lost dreams.",
        "Create a world where music has magical properties.",
        "Write about a character who can see people's lifespans.",
        "Describe a future where robots have emotions.",
        "Write a story about the last human on Mars.",
        "Create a dialogue between Earth and the Moon.",
        "Write about a world where art comes to life.",
    ]

    # Problem Solving prompts
    PROBLEM_SOLVING_PROMPTS = [
        "How would you design a sustainable city for 1 million people?",
        "Propose a solution to reduce food waste globally.",
        "Design a system to help elderly people stay connected with family.",
        "How would you make public transportation more efficient?",
        "Create a plan to reduce plastic pollution in oceans.",
        "Design an educational system for remote areas.",
        "How would you improve mental health support in workplaces?",
        "Propose a solution for affordable housing in urban areas.",
        "Design a system to reduce traffic congestion.",
        "How would you make healthcare more accessible?",
        "Create a plan to preserve endangered languages.",
        "Design a sustainable farming method for arid regions.",
        "How would you improve disaster preparedness?",
        "Propose a solution for the digital divide.",
        "Design a system to reduce energy consumption in buildings.",
        "How would you make voting more accessible and secure?",
        "Create a plan to support small businesses during economic downturns.",
        "Design a solution for space debris cleanup.",
        "How would you improve water quality in developing countries?",
        "Propose a system for fair AI algorithm development.",
    ]

    # Conversation Starters
    CONVERSATION_PROMPTS = [
        "If you could have dinner with any historical figure, who would it be and why?",
        "What's the most important lesson you've learned in life?",
        "If you could live in any time period, when would it be?",
        "What technology do you think will change the world the most?",
        "If you could solve one global problem, what would it be?",
        "What's your opinion on the future of work?",
        "If you could learn any skill instantly, what would it be?",
        "What do you think makes a good leader?",
        "If you could visit any place in the universe, where would you go?",
        "What's the most interesting book you've ever read?",
        "If you could change one thing about how schools work, what would it be?",
        "What do you think is the key to happiness?",
        "If you could have any superpower, what would it be?",
        "What's your favorite way to be creative?",
        "If you could ask the universe one question, what would it be?",
        "What do you think is humanity's greatest achievement?",
        "If you could time travel but only observe, what event would you witness?",
        "What's the most valuable advice you could give to someone?",
        "If you could redesign society from scratch, what would you change?",
        "What do you think the world will look like in 100 years?",
    ]

    # Analytical prompts
    ANALYTICAL_PROMPTS = [
        "Compare the advantages and disadvantages of remote work vs office work.",
        "Analyze the impact of social media on modern communication.",
        "Evaluate the pros and cons of electric vehicles.",
        "Compare different approaches to renewable energy.",
        "Analyze the effects of automation on employment.",
        "Evaluate the benefits and risks of genetic engineering.",
        "Compare traditional education vs online learning.",
        "Analyze the impact of streaming services on entertainment industry.",
        "Evaluate different strategies for combating climate change.",
        "Compare the effectiveness of various diet approaches.",
        "Analyze the role of artificial intelligence in healthcare.",
        "Evaluate the impact of globalization on local cultures.",
        "Compare different methods of urban planning.",
        "Analyze the effects of video games on cognitive development.",
        "Evaluate the pros and cons of universal basic income.",
        "Compare various approaches to mental health treatment.",
        "Analyze the impact of cryptocurrency on traditional banking.",
        "Evaluate different strategies for space exploration.",
        "Compare the benefits and drawbacks of nuclear energy.",
        "Analyze the role of regulation in technology development.",
    ]

    @classmethod
    def get_all_prompts(cls):
        """Return all prompts combined"""
        return (cls.QA_PROMPTS + cls.CREATIVE_PROMPTS +
                cls.PROBLEM_SOLVING_PROMPTS + cls.CONVERSATION_PROMPTS +
                cls.ANALYTICAL_PROMPTS)

    @classmethod
    def sample_prompts(cls, num_prompts: int, seed: int = None):
        """Sample prompts to match the requested number"""
        if seed is not None:
            random.seed(seed)

        all_prompts = cls.get_all_prompts()

        if num_prompts <= len(all_prompts):
            return random.sample(all_prompts, num_prompts)
        else:
            # If we need more prompts than available, sample with replacement
            return random.choices(all_prompts, k=num_prompts)

    @classmethod
    def get_prompt_stats(cls):
        """Get statistics about the prompt dataset"""
        return {
            "total_prompts": len(cls.get_all_prompts()),
            "qa_prompts": len(cls.QA_PROMPTS),
            "creative_prompts": len(cls.CREATIVE_PROMPTS),
            "problem_solving_prompts": len(cls.PROBLEM_SOLVING_PROMPTS),
            "conversation_prompts": len(cls.CONVERSATION_PROMPTS),
            "analytical_prompts": len(cls.ANALYTICAL_PROMPTS),
        }


class LLMClient:
    """Simple client for TensorRT-LLM OpenAPI server using requests"""

    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.stats_lock = Lock()
        self.results = []  # Store all request/response data

    def health_check(self):
        """Check if server is healthy"""
        print("   Trying health check...")
        # Simple health check with minimal payload
        payload = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": [{
                "role": "user",
                "content": "Hi"
            }],
            "max_tokens": 1,
            "temperature": 0.1
        }
        response = self.session.post(f"{self.server_url}/v1/chat/completions",
                                     json=payload,
                                     timeout=10)
        response.raise_for_status()
        return {"status": "healthy"}

    def generate_text(self,
                      prompt: str,
                      max_tokens: int = 100,
                      temperature: float = 0.8,
                      streaming: bool = False,
                      save_result: bool = False,
                      request_id: int = None):
        """Generate text using OpenAI chat completions format"""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()

        payload = {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "messages": [{
                "role": "user",
                "content": prompt
            }],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": streaming
        }

        result_data = {
            "request_id": request_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "streaming": streaming,
            "success": False,
            "response_time": 0,
            "text": "",
            "error": None
        }

        try:
            if streaming:
                response_result = self._stream_generate(payload)
            else:
                response = self.session.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=30)
                response.raise_for_status()
                result = response.json()
                # Extract text from OpenAI format
                text = result.get("choices",
                                  [{}])[0].get("message",
                                               {}).get("content", "")
                response_result = {"text": text, "original_response": result}

            end_time = time.time()
            result_data.update({
                "success": True,
                "response_time": end_time - start_time,
                "text": response_result.get("text", "")
            })

        except Exception as e:
            end_time = time.time()
            result_data.update({
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            })
            raise

        finally:
            if save_result:
                with self.stats_lock:
                    self.results.append(result_data)

        return response_result

    def _stream_generate(self, payload):
        """Handle streaming generation with OpenAI format"""
        response = self.session.post(f"{self.server_url}/v1/chat/completions",
                                     json=payload,
                                     stream=True,
                                     timeout=30)
        response.raise_for_status()

        full_text = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    content = data.get("choices",
                                       [{}])[0].get("delta",
                                                    {}).get("content", "")
                    if content:
                        full_text += content
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    continue
        print()
        return {"text": full_text}

    def benchmark(self,
                  num_requests: int = 1000,
                  max_workers: int = 10,
                  prompt: str = None,
                  use_dataset_prompts: bool = True,
                  save_results: bool = False):
        """Run benchmark test with multiple concurrent requests"""

        # Prepare prompts
        if use_dataset_prompts:
            prompts = PromptDataset.sample_prompts(num_requests, seed=23)
            prompt_source = "dataset"
            print(f"📚 Using {len(prompts)} diverse prompts from dataset")
            stats = PromptDataset.get_prompt_stats()
            print(
                f"   Dataset contains: {stats['total_prompts']} total prompts across {len(stats)-1} categories"
            )
        else:
            prompt = prompt or "Hello, how are you?"
            prompts = [prompt] * num_requests
            prompt_source = "single"
            print(f"   Using single prompt: '{prompt}'")

        print(
            f"🔥 Starting benchmark with {num_requests} requests using {max_workers} concurrent workers..."
        )
        print(f"   Target: {self.server_url}")
        if save_results:
            print(f"   Results will be saved to JSON file")
        print("=" * 60)

        response_times = []
        successful_requests = 0
        failed_requests = 0

        def send_request(request_id):
            """Send a single request and measure time"""
            start_time = time.time()
            try:
                # Use the prompt for this specific request
                request_prompt = prompts[request_id] if request_id < len(
                    prompts) else prompts[0]
                result = self.generate_text(request_prompt,
                                            max_tokens=20,
                                            temperature=0.8,
                                            save_result=save_results,
                                            request_id=request_id)
                end_time = time.time()
                response_time = end_time - start_time

                with self.stats_lock:
                    response_times.append(response_time)
                    return {
                        "success": True,
                        "time": response_time,
                        "id": request_id
                    }
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                return {
                    "success": False,
                    "time": response_time,
                    "id": request_id,
                    "error": str(e)
                }

        start_total = time.time()

        # Execute requests with progress tracking
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(send_request, i) for i in range(num_requests)
            ]

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["success"]:
                    successful_requests += 1
                else:
                    failed_requests += 1

                # Print progress every 100 requests
                if i % 100 == 0 or i == num_requests:
                    print(f"   Progress: {i}/{num_requests} requests completed")

        end_total = time.time()
        total_time = end_total - start_total

        # Calculate statistics
        if response_times:
            avg_time = statistics.mean(response_times)
            median_time = statistics.median(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            p95_time = statistics.quantiles(
                response_times,
                n=20)[18] if len(response_times) >= 20 else max_time
            p99_time = statistics.quantiles(
                response_times,
                n=100)[98] if len(response_times) >= 100 else max_time
        else:
            avg_time = median_time = min_time = max_time = p95_time = p99_time = 0

        # Print results
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Prompt source:       {prompt_source}")
        print(f"Total requests:      {num_requests}")
        print(f"Successful:          {successful_requests}")
        print(f"Failed:              {failed_requests}")
        print(
            f"Success rate:        {(successful_requests/num_requests)*100:.1f}%"
        )
        print(f"Total time:          {total_time:.2f}s")
        print(f"Requests per second: {num_requests/total_time:.2f}")
        print()
        print("⏱️  RESPONSE TIME STATISTICS")
        print("-" * 40)
        print(f"Average:             {avg_time*1000:.2f}ms")
        print(f"Median:              {median_time*1000:.2f}ms")
        print(f"Min:                 {min_time*1000:.2f}ms")
        print(f"Max:                 {max_time*1000:.2f}ms")
        print(f"95th percentile:     {p95_time*1000:.2f}ms")
        print(f"99th percentile:     {p99_time*1000:.2f}ms")
        print("=" * 60)

        return {
            "prompt_source": prompt_source,
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / num_requests) * 100,
            "total_time": total_time,
            "requests_per_second": num_requests / total_time,
            "avg_response_time": avg_time,
            "median_response_time": median_time,
            "min_response_time": min_time,
            "max_response_time": max_time,
            "p95_response_time": p95_time,
            "p99_response_time": p99_time
        }

    def save_results_to_file(self,
                             filename: str = "generation.json",
                             benchmark_stats: dict = None):
        """Save all collected results to a JSON file"""
        if not self.results and not benchmark_stats:
            print("   No results to save.")
            return

        # Sort results by request_id before saving
        sorted_results = sorted(self.results,
                                key=lambda x: x.get('request_id', 0))

        output_data = {
            "metadata": {
                "server_url": self.server_url,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_requests": len(sorted_results),
                "results_count": len(sorted_results)
            },
            "requests": sorted_results
        }

        if benchmark_stats:
            output_data["benchmark_statistics"] = benchmark_stats

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"   💾 Results saved to: {filename}")
            print(f"   📊 Saved {len(self.results)} request/response pairs")
        except Exception as e:
            print(f"   ❌ Error saving results: {e}")


@click.command()
@click.option("--server-url",
              default="http://localhost:8000",
              help="Server URL")
@click.option("--benchmark",
              is_flag=True,
              help="Run benchmark test with 1000 requests")
@click.option("--num-requests",
              default=1000,
              help="Number of requests for benchmark")
@click.option("--max-workers",
              default=10,
              help="Number of concurrent workers for benchmark")
@click.option("--benchmark-prompt",
              default="Hello, how are you?",
              help="Prompt to use for benchmark")
@click.option("--dataset-prompts",
              is_flag=True,
              help="Use diverse prompts from dataset instead of single prompt")
@click.option("--save-results",
              is_flag=True,
              help="Save all requests and results to generation.json")
@click.option("--results-file",
              default="generation.json",
              help="Filename for saving results")
def main(server_url: str, benchmark: bool, num_requests: int, max_workers: int,
         benchmark_prompt: str, dataset_prompts: bool, save_results: bool,
         results_file: str):
    """Test TensorRT-LLM OpenAPI server with TinyLlama model"""

    client = LLMClient(server_url)

    print(f"🤖 Testing TensorRT-LLM server at: {server_url}")
    if save_results:
        print(f"💾 Results will be saved to: {results_file}")
    print("=" * 50)

    try:
        # Health check
        print("1. Health check...")
        health = client.health_check()
        print(f"   ✅ Server healthy: {health}")

        if benchmark:
            # Run benchmark
            print(f"\n2. Running benchmark...")
            benchmark_stats = client.benchmark(
                num_requests=num_requests,
                max_workers=max_workers,
                prompt=benchmark_prompt,
                use_dataset_prompts=dataset_prompts,
                save_results=save_results)

            if save_results:
                print(f"\n3. Saving results...")
                client.save_results_to_file(results_file, benchmark_stats)
        else:
            # Test prompts
            prompts = [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ]

            print("\n2. Testing text generation...")
            for i, prompt in enumerate(prompts, 1):
                print(f"\n🎯 Test {i}: '{prompt}'")
                print(f"   Sending request...")
                result = client.generate_text(prompt,
                                              max_tokens=50,
                                              temperature=0.8,
                                              save_result=save_results,
                                              request_id=i - 1)
                print(f"   Generated: '{result.get('text', '')}'")

            # Streaming test
            print(f"\n3. Testing streaming generation...")
            streaming_prompt = "Write a short story about a robot:"
            print(f"🎯 Streaming: '{streaming_prompt}'")
            print("   Sending streaming request...")
            print("📡 Response: ", end="")
            result = client.generate_text(streaming_prompt,
                                          max_tokens=80,
                                          temperature=0.9,
                                          streaming=True,
                                          save_result=save_results,
                                          request_id=len(prompts))

            if save_results:
                print(f"\n4. Saving results...")
                client.save_results_to_file(results_file)

            print(f"\n✅ All tests completed!")

    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {server_url}")
        print("   Make sure the server is running!")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
