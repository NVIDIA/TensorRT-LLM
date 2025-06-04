import argparse
import asyncio
import json
import logging
import time

import aiohttp
import yaml

logging.basicConfig(level=logging.INFO)


async def wait_for_server(session, server_host, server_port, timeout):
    url = f"http://{server_host}:{server_port}/health"
    start_time = time.time()
    logging.info("Waiting for server to start")
    while time.time() - start_time < timeout:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    logging.info("Server is ready.")
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    raise Exception("Server did not become ready in time.")


async def send_request(session, server_host, server_port, model, prompt,
                       max_tokens, temperature, streaming, ignore_eos):
    url = f"http://{server_host}:{server_port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": ignore_eos
    }
    if streaming:
        data["stream"] = True

    async with session.post(url, headers=headers, json=data) as response:
        if response.status != 200:
            raise Exception(f"Error: {await response.text()}")

        if streaming:
            text = ""
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                        response_json = json.loads(line)
                        text += response_json["choices"][0]["text"]
            logging.info(text)
            return text
        else:
            response_json = await response.json()
            text = response_json["choices"][0]["text"]
            logging.info(text)
            return text


async def send_chat_request(session, server_host, server_port, model, prompt,
                            max_tokens, temperature, streaming):
    url = f"http://{server_host}:{server_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model":
        model,
        "messages": [{
            "role": "system",
            "content": "You are a helpfule assistant."
        }, {
            "role": "user",
            "content": prompt
        }],
        "max_tokens":
        max_tokens,
        "temperature":
        temperature
    }
    if streaming:
        data["stream"] = True

    async with session.post(url, headers=headers, json=data) as response:
        if response.status != 200:
            raise Exception(f"Error: {await response.text()}")

        if streaming:
            text = ""
            async for line in response.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line == "data: [DONE]":
                        break
                    if line.startswith("data: "):
                        line = line[len("data: "):]
                        response_json = json.loads(line)
                        if "content" in response_json["choices"][0]["delta"]:
                            text += response_json["choices"][0]["delta"][
                                "content"]
            logging.info(text)
            return text
        else:
            response_json = await response.json()
            text = response_json["choices"][0]["message"]["content"]
            logging.info(text)
            return text


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--disagg_config-file",
                        help="Path to YAML config file",
                        required=True)
    parser.add_argument("-p",
                        "--prompts-file",
                        help="Path to JSON file containing prompts",
                        required=True)
    parser.add_argument("--max-tokens",
                        type=int,
                        help="Max tokens",
                        default=100)
    parser.add_argument("--temperature",
                        type=float,
                        help="Temperature",
                        default=0.)
    parser.add_argument("--server-start-timeout",
                        type=int,
                        help="Time to wait for server to start",
                        default=None)
    parser.add_argument("-e",
                        "--endpoint",
                        type=str,
                        help="Endpoint to use",
                        default="completions")
    parser.add_argument("-o",
                        "--output-file",
                        type=str,
                        help="Output filename",
                        default="output.json")
    parser.add_argument("--streaming",
                        action="store_true",
                        help="Enable streaming responses")
    parser.add_argument("--ignore-eos", action="store_true", help="Ignore eos")
    args = parser.parse_args()

    with open(args.disagg_config_file, "r") as file:
        config = yaml.safe_load(file)

    server_host = config.get('hostname', 'localhost')
    server_port = config.get('port', 8000)
    model = config.get('model', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')

    with open(args.prompts_file, "r") as file:
        prompts = json.load(file)

    async with aiohttp.ClientSession() as session:

        if args.server_start_timeout is not None:
            await wait_for_server(session, server_host, server_port,
                                  args.server_start_timeout)

        if args.endpoint == "completions":
            tasks = [
                send_request(session, server_host, server_port, model, prompt,
                             args.max_tokens, args.temperature, args.streaming,
                             args.ignore_eos) for prompt in prompts
            ]
        elif args.endpoint == "chat":
            tasks = [
                send_chat_request(session, server_host, server_port, model,
                                  prompt, args.max_tokens, args.temperature,
                                  args.streaming) for prompt in prompts
            ]

        responses = await asyncio.gather(*tasks)

    with open(args.output_file, "w") as file:
        json.dump(responses, file, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
