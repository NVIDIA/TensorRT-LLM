import argparse
import asyncio
import json
import os
import time
from typing import Dict

import aiohttp


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model name or path")
    parser.add_argument("concurrency",
                        type=int,
                        help="Number of concurrent requests")
    parser.add_argument("--host",
                        type=str,
                        default="localhost",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--output",
                        type=str,
                        default='out-strs',
                        help="Output directory")
    parser.add_argument("--num-iterations",
                        type=int,
                        default=5,
                        help="Number of test iterations")
    parser.add_argument("--wait",
                        type=float,
                        default=0.0,
                        help="Wait time between requests")
    parser.add_argument("--temperature",
                        type=float,
                        default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens",
                        type=int,
                        default=1024,
                        help="Maximum tokens to generate")
    return parser.parse_args()


# def create_prompt():
#     """Create the test prompt for nondeterminism testing"""
#     return {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": """Classify the sentiment expressed in the following text and provide the response in a single word Positive/Negative/Neutral. Explain your answer in 2 lines.
# TEXT:: Today I will exaggerate, will be melodramatic (mostly the case when I am excited) and be naive (as always). Just came out from the screening of the Avengers Endgame ("Endgame")! The journey had started in the year 2008, when Tony Stark, during his capture in a cave in Afghanistan, had created a combat suit and came out of his captivity.
# Then the combat suit made of iron was perfected and Tony Stark officially became the Iron Man!! The Marvel Cinematic Universe ("MCU") thus was initiated. The journey continued since then and in 2012 all the MCU heroes came together and formed the original "Avengers" (so much fun and good it was).
# 21 Movies in the MCU and culminating into the Infinity War (2018) and finally into the Endgame! The big adventure for me started from Jurassic Park and then came Titanic, Lagaan, Dark Knight; and then came the Avengers in 2012. Saw my absolute favorite Sholay in the hall in 2014. In the above-mentioned genre, there are good movies, great movies and then there is the Endgame.
# Today after a long long time, I came out of the hall with 100% happiness, satisfaction and over the top excitement/emotions. The movie is Epic, Marvel (in the real sense) and perfect culmination of the greatest cinematic saga of all time. It is amazing, humorous, emotional and has mind-blowing action! It is one of the finest Superhero Movie of all time.
# Just pure Awesome! It's intelligent!"""
#             }
#         ],
#         "max_tokens": 1024,
#         "temperature": 0.0,
#         "stream": False,
#         # TODO: Update for other models
#         "stop": ["</s>"],
#         "stop_token_ids": [2]
#     }

# def create_prompt():
#     """Create the test prompt for nondeterminism testing"""
#     return {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Describe the history of NVIDIA."
#             }
#         ],
#         "max_tokens": 200,
#         "temperature": 0.0,
#         #"stream": False,
#         # TODO: Update for other models
#         "stop": ["</s>"],
#         # For Llama-3.1-8B:
#         # "stop_token_ids": [2, 128001]
#         # For Mixtral-8x7B-v0.1:
#         "stop_token_ids": [2]
#     }


def create_prompt():
    """Create the test prompt for nondeterminism testing"""

    # Choose which prompt to use (comment/uncomment as needed)
    # prompt_file = "prompt.txt"  # CNN summarization task
    prompt_file = "deloitte_drilldown_1_prompt.txt"  # Code generation

    # Set max_tokens based on prompt type
    max_tokens_config = {
        "prompt.txt": 150,  # Summarization needs ~150 tokens
        "deloitte_drilldown_1_prompt.txt":
        300,  # Reasoning + JSON needs ~300 tokens
    }

    # Read prompt from file
    with open(prompt_file, 'r') as f:
        prompt_content = f.read().strip()

    return {
        "messages": [{
            "role": "user",
            "content": prompt_content
        }],
        "max_tokens": max_tokens_config.get(prompt_file, 150),
        "temperature": 0.0,
    }


async def send_request(session: aiohttp.ClientSession, prompt: dict,
                       request_id: int, host: str, port: int) -> Dict:
    """Send a single request to the server"""
    # Convert messages to prompt string
    prompt_text = prompt["messages"][0]["content"]

    payload = {
        # "model": "Qwen/Qwen2.5-14B-Instruct",
        "model": "gpt_oss/gpt-oss-20b",
        "prompt": prompt_text,
        "max_tokens": prompt["max_tokens"],
        "temperature": prompt["temperature"],
        # "stop": prompt["stop"],
        # "stop_token_ids": prompt["stop_token_ids"]
    }

    try:
        async with session.post(
                f'http://{host}:{port}/v1/completions',
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)) as response:
            if response.status == 200:
                result = await response.json()
                content = result.get("choices", [{}])[0].get("text", "")
                return {
                    "request_id": request_id,
                    "response": content,
                    "full_response": result,
                    "success": True
                }
            else:
                return {
                    "request_id": request_id,
                    "response": f"HTTP {response.status}",
                    "success": False
                }
    except Exception as e:
        return {
            "request_id": request_id,
            "response": f"Error: {str(e)}",
            "success": False
        }


async def test_nondeterminism(prompt: dict,
                              concurrency: int,
                              host: str,
                              port: int,
                              wait: float = 0.0):
    """Test for nondeterminism with given concurrency"""
    print(f"Testing with concurrency={concurrency}")

    async with aiohttp.ClientSession() as session:
        # tasks for concurrent requests
        tasks = []
        for i in range(concurrency):
            if wait > 0 and i > 0:
                await asyncio.sleep(wait)
            task = send_request(session, prompt, i, host, port)
            tasks.append(task)

        # send all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        successful_results = [r for r in results if r["success"]]
        responses = [r["response"] for r in successful_results]
        unique_responses = set(responses)

        print(f"Total requests: {len(results)}")
        print(f"Successful requests: {len(successful_results)}")
        print(f"Unique responses: {len(unique_responses)}")
        print(f"Time taken: {end_time - start_time:.2f}s")

        if len(unique_responses) == 1:
            print("DETERMINISTIC: All responses are identical")
        else:
            print("NON-DETERMINISTIC: Found different responses")
            for i, response in enumerate(responses):
                print(f"Response {i}: {response[:100]}...")

        return results


async def main():
    args = get_args()
    prompt = create_prompt()

    # if args.temperature is not None:
    #     prompt["temperature"] = args.temperature
    # if args.max_tokens is not None:
    #     prompt["max_tokens"] = args.max_tokens

    print("Starting Nondeterminism Test")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Temperature: {prompt['temperature']}")
    print(f"Max tokens: {prompt['max_tokens']}")

    os.makedirs(args.output, exist_ok=True)

    # save the prompt
    with open(os.path.join(args.output, "prompt.txt"), 'w') as f:
        f.write(json.dumps(prompt, indent=2))

    num_outputs = []

    try:
        for iteration in range(args.num_iterations):
            print(f"\n{'='*50}")
            print(f"Iteration: {iteration + 1}/{args.num_iterations}")
            print(f"{'='*50}")

            results = await test_nondeterminism(prompt, args.concurrency,
                                                args.host, args.port, args.wait)

            # save individual results
            successful_results = [r for r in results if r["success"]]

            # print(f"Debug: results = {results}")
            # print(f"Debug: successful_results = {successful_results}")

            responses = [r["response"] for r in successful_results]
            unique_responses = set(responses)

            if len(unique_responses) > 1:
                print(f"\nFOUND {len(unique_responses)} UNIQUE RESPONSES:")
                for i, unique_resp in enumerate(unique_responses):
                    print(f"\n--- UNIQUE RESPONSE {i+1} ---")
                    print(f"'{unique_resp}'")
                    print("-" * 80)
                    print(f"Length: {len(unique_resp)} chars")

                # Show which responses are which
                print(f"\nRESPONSE DISTRIBUTION:")
                for i, resp in enumerate(responses):
                    resp_num = list(unique_responses).index(resp) + 1
                    print(f"Request {i}: Response {resp_num}")
            else:
                print(f"All {len(responses)} responses are identical")

            # save results for this iteration
            iteration_results = {
                "iteration": iteration,
                "concurrency": args.concurrency,
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "unique_responses": len(unique_responses),
                "responses": responses
            }

            with open(os.path.join(args.output, f"iteration_{iteration}.json"),
                      'w') as f:
                json.dump(iteration_results, f, indent=2)

            # save individual response files
            for i, result in enumerate(successful_results):
                with open(
                        os.path.join(args.output,
                                     f"response_{iteration}_{i}.txt"),
                        'w') as f:
                    f.write(result["response"])

            num_outputs.append(len(unique_responses))

            result_str = f"Num Unique responses in {len(successful_results)}: {len(unique_responses)}"
            print(result_str)

            with open(os.path.join(args.output, f"num_outputs_{iteration}"),
                      'w') as f:
                f.write(result_str + '\n')

        print(f"\n{'='*50}")
        print("FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Num unique responses for each iteration: {num_outputs}")

        if all(n == 1 for n in num_outputs):
            print(
                "OVERALL: DETERMINISTIC - All iterations produced identical responses"
            )
        else:
            print(
                "OVERALL: NON-DETERMINISTIC - Some iterations produced different responses"
            )

    except Exception as e:
        print(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
