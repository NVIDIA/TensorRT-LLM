# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import time

import aiohttp
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


def _prompt_summary(prompt):
    if isinstance(prompt, str):
        return f"type=str chars={len(prompt)}"
    if isinstance(prompt, list):
        return f"type=list len={len(prompt)}"
    return f"type={type(prompt).__name__}"


def _elapsed_ms(start_time):
    return (time.monotonic() - start_time) * 1000.0


async def wait_for_server(session, server_host, server_port, timeout):
    url = f"http://{server_host}:{server_port}/health"
    start_time = time.time()
    LOGGER.info("Waiting for server to start: url=%s timeout=%ss", url, timeout)
    while time.time() - start_time < timeout:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    LOGGER.info("Server is ready.")
                    return
        except aiohttp.ClientError:
            pass
        await asyncio.sleep(1)
    raise Exception("Server did not become ready in time.")


async def send_request(session, server_host, server_port, model, prompt,
                       max_tokens, temperature, streaming, ignore_eos,
                       request_index):
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

    start_time = time.monotonic()
    LOGGER.info(
        "completion request start: index=%s url=%s model=%s stream=%s "
        "max_tokens=%s temperature=%s ignore_eos=%s prompt=%s", request_index,
        url, model, streaming, max_tokens, temperature, ignore_eos,
        _prompt_summary(prompt))
    try:
        async with session.post(url, headers=headers, json=data) as response:
            LOGGER.info(
                "completion response headers: index=%s status=%s "
                "content_type=%s elapsed_ms=%.2f", request_index,
                response.status, response.headers.get("Content-Type"),
                _elapsed_ms(start_time))
            if response.status != 200:
                raise RuntimeError(f"Error: {await response.text()}")

            if streaming:
                text = ""
                chunk_count = 0
                async for line in response.content:
                    if line:
                        chunk_count += 1
                        line = line.decode('utf-8').strip()
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            line = line[len("data: "):]
                            response_json = json.loads(line)
                            choices = response_json.get("choices", [])
                            if not choices:
                                continue
                            text += choices[0].get("text", "")
                LOGGER.info(
                    "completion streaming done: index=%s chunks=%s "
                    "text_chars=%s elapsed_ms=%.2f text=%s", request_index,
                    chunk_count, len(text), _elapsed_ms(start_time), text)
                return text
            else:
                response_json = await response.json()
                choices = response_json.get("choices", [])
                if not choices:
                    raise ValueError("Missing choices in completion response")
                text = choices[0].get("text", "")
                LOGGER.info(
                    "completion done: index=%s text_chars=%s "
                    "elapsed_ms=%.2f text=%s", request_index, len(text),
                    _elapsed_ms(start_time), text)
                return text
    except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError, ValueError,
            json.JSONDecodeError):
        LOGGER.exception("completion request failed: index=%s elapsed_ms=%.2f",
                         request_index, _elapsed_ms(start_time))
        raise


async def send_chat_request(session, server_host, server_port, model, prompt,
                            max_tokens, temperature, streaming, request_index):
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

    start_time = time.monotonic()
    LOGGER.info(
        "chat request start: index=%s url=%s model=%s stream=%s "
        "max_tokens=%s temperature=%s prompt=%s", request_index, url, model,
        streaming, max_tokens, temperature, _prompt_summary(prompt))
    try:
        async with session.post(url, headers=headers, json=data) as response:
            LOGGER.info(
                "chat response headers: index=%s status=%s "
                "content_type=%s elapsed_ms=%.2f", request_index,
                response.status, response.headers.get("Content-Type"),
                _elapsed_ms(start_time))
            if response.status != 200:
                raise RuntimeError(f"Error: {await response.text()}")

            if streaming:
                text = ""
                chunk_count = 0
                async for line in response.content:
                    if line:
                        chunk_count += 1
                        line = line.decode('utf-8').strip()
                        if line == "data: [DONE]":
                            break
                        if line.startswith("data: "):
                            line = line[len("data: "):]
                            response_json = json.loads(line)
                            choices = response_json.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content is not None:
                                text += content
                LOGGER.info(
                    "chat streaming done: index=%s chunks=%s "
                    "text_chars=%s elapsed_ms=%.2f text=%s", request_index,
                    chunk_count, len(text), _elapsed_ms(start_time), text)
                return text
            else:
                response_json = await response.json()
                choices = response_json.get("choices", [])
                if not choices:
                    raise ValueError(
                        "Missing choices in chat completion response")
                text = choices[0].get("message", {}).get("content", "")
                LOGGER.info(
                    "chat done: index=%s text_chars=%s elapsed_ms=%.2f "
                    "text=%s", request_index, len(text),
                    _elapsed_ms(start_time), text)
                return text
    except (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError, ValueError,
            json.JSONDecodeError):
        LOGGER.exception("chat request failed: index=%s elapsed_ms=%.2f",
                         request_index, _elapsed_ms(start_time))
        raise


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
    LOGGER.info(
        "disagg client config: config_file=%s prompts_file=%s endpoint=%s "
        "streaming=%s output_file=%s server=%s:%s model=%s max_tokens=%s "
        "temperature=%s ignore_eos=%s server_start_timeout=%s",
        args.disagg_config_file, args.prompts_file, args.endpoint,
        args.streaming, args.output_file, server_host, server_port, model,
        args.max_tokens, args.temperature, args.ignore_eos,
        args.server_start_timeout)

    with open(args.prompts_file, "r") as file:
        prompts = json.load(file)
    LOGGER.info("loaded prompts: count=%s summaries=%s", len(prompts),
                [_prompt_summary(prompt) for prompt in prompts])

    async with aiohttp.ClientSession() as session:

        if args.server_start_timeout is not None:
            await wait_for_server(session, server_host, server_port,
                                  args.server_start_timeout)

        if args.endpoint == "completions":
            tasks = [
                send_request(session, server_host, server_port, model, prompt,
                             args.max_tokens, args.temperature, args.streaming,
                             args.ignore_eos, i)
                for i, prompt in enumerate(prompts)
            ]
        elif args.endpoint == "chat":
            tasks = [
                send_chat_request(session, server_host, server_port, model,
                                  prompt, args.max_tokens, args.temperature,
                                  args.streaming, i)
                for i, prompt in enumerate(prompts)
            ]
        else:
            raise ValueError(f"Unknown endpoint: {args.endpoint}")

        LOGGER.info("awaiting client tasks: count=%s endpoint=%s", len(tasks),
                    args.endpoint)
        responses = await asyncio.gather(*tasks)
        LOGGER.info("client tasks completed: count=%s endpoint=%s",
                    len(responses), args.endpoint)

    with open(args.output_file, "w") as file:
        json.dump(responses, file, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
