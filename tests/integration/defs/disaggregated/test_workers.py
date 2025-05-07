import asyncio
import copy
import json
import os
import subprocess
from typing import List, Optional, Tuple

import aiohttp
import pytest
import yaml
from transformers import AutoTokenizer

from tensorrt_llm import logger
from tensorrt_llm.bindings.internal.batch_manager import (BlockKey,
                                                          BlockKeyHasher)

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def get_ctx_gen_server_urls_from_cfg(config_file: str):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    ctx_servers = []
    gen_servers = []
    for server in config["context_servers"]["urls"]:
        ctx_servers.append("http://" + server)
    for server in config["generation_servers"]["urls"]:
        gen_servers.append("http://" + server)
    return ctx_servers, gen_servers


def run_disaggregated_workers(
    config_file: str,
    stdout=None,
    env: Optional[dict] = None,
    cwd: Optional[str] = None,
    num_ranks: Optional[int] = None
) -> Tuple[subprocess.Popen, List[str], List[str]]:

    ctx_servers, gen_servers = get_ctx_gen_server_urls_from_cfg(config_file)

    # TODO: auto detect num_ranks
    assert num_ranks is not None

    # Start workers
    workers_cmd = [
        'mpirun', '--allow-run-as-root', '--oversubscribe', '-n',
        str(num_ranks), 'trtllm-serve', 'disaggregated_mpi_worker', '-c',
        config_file
    ]
    logger.info(f"Running workers with command: {' '.join(workers_cmd)}")
    workers_proc = subprocess.Popen(workers_cmd,
                                    stdout=stdout,
                                    stderr=subprocess.STDOUT,
                                    env=env,
                                    cwd=cwd)
    return workers_proc, ctx_servers, gen_servers


class BasicWorkerTester:

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180):
        self.ctx_servers = ctx_servers
        self.gen_servers = gen_servers
        self.req_timeout_secs = req_timeout_secs
        self.server_start_timeout_secs = server_start_timeout_secs

    async def new_session(self):
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(
            total=self.req_timeout_secs))
        await self.wait_for_servers_ready(session)
        return session

    async def send_request(self, session: aiohttp.ClientSession, url: str,
                           request: dict) -> dict:
        # TODO: streaming support
        async with session.post(url + "/v1/completions",
                                json=request) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                raise ValueError(
                    "Received an event-stream although request stream was False"
                )

            response_dict = await response.json()
            if not response.ok:
                logger.error(f"Received failed response {response_dict}")
                response.raise_for_status()
            return response_dict

    async def send_disagg_request(self, session: aiohttp.ClientSession,
                                  ctx_url: str, gen_url: str,
                                  request: dict) -> dict:
        ctx_request = copy.deepcopy(request)
        gen_request = copy.deepcopy(request)

        ctx_request["max_tokens"] = 1
        ctx_request["disaggregated_params"] = {"request_type": "context_only"}
        ctx_response = await self.send_request(session, ctx_url, ctx_request)
        assert len(ctx_response["choices"]) == 1

        gen_request["disaggregated_params"] = ctx_response["choices"][0][
            "disaggregated_params"]
        gen_request["disaggregated_params"]["request_type"] = "generation_only"
        gen_response = await self.send_request(session, gen_url, gen_request)
        return gen_response

    async def query_kv_cache_events(self, session: aiohttp.ClientSession,
                                    url: str):
        async with session.post(url + "/kv_cache_events") as response:
            events_raw = await response.json()

        events = []
        for event_raw in events_raw:
            event = {"id": event_raw["event_id"]} | event_raw["data"]
            if event["type"] == "stored":
                for block in event["blocks"]:
                    block["token_id"] = [
                        token["token_id"] for token in block["tokens"]
                    ]
                    block["token_extra_id"] = [
                        token["token_extra_id"] for token in block["tokens"]
                    ]
                    # TODO: check by BlockKey::usesExtraIds
                    if not any(block["token_extra_id"]):
                        del block["token_extra_id"]
                    del block["tokens"]
            events.append(event)
        return events

    async def check_server_ready(self, session: aiohttp.ClientSession,
                                 server_url: str) -> bool:
        try:
            async with session.get(server_url + "/health") as response:
                return response.status == 200
        except Exception:
            return False

    async def wait_for_servers_ready(self, session: aiohttp.ClientSession):

        async def are_servers_ready(session: aiohttp.ClientSession):
            context_ready = all([
                await self.check_server_ready(session, url)
                for url in self.ctx_servers
            ])
            generation_ready = all([
                await self.check_server_ready(session, url)
                for url in self.gen_servers
            ])
            return context_ready and generation_ready

        async def check_all_servers_ready(session: aiohttp.ClientSession):
            iter = 0
            while not await are_servers_ready(session):
                wait_time = 3
                logger.info(
                    f"Context and generation servers are not ready. Waiting ({iter})..."
                )
                await asyncio.sleep(wait_time)
                iter += 1

        try:
            await asyncio.wait_for(check_all_servers_ready(session),
                                   timeout=self.server_start_timeout_secs)
        except asyncio.CancelledError:
            raise TimeoutError(
                "Timeout waiting for context and generation servers to be ready"
            )


class ConditionalWorkerTester(BasicWorkerTester):

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)

    async def multi_round_request(self, session: aiohttp.ClientSession,
                                  init_prompt: str, max_rounds: int,
                                  threshold: float):
        request = {
            "model": MODEL_NAME,
            "prompt": init_prompt,
            "max_tokens": 10,
            "temperature": 0.0,
        }
        prev_prompt_len = 0
        curr_prompt_len = 1
        for i in range(max_rounds):
            # conditional disaggregation by kv cache (estimated by prompt length)
            if prev_prompt_len > curr_prompt_len * threshold:
                logger.info(f"Sending normal request at iter {i}")
                response = await self.send_request(session, self.gen_servers[0],
                                                   request)
            else:
                logger.info(f"Sending disaggregated request at iter {i}")
                response = await self.send_disagg_request(
                    session, self.ctx_servers[0], self.gen_servers[0], request)
            logger.info(
                f"Received response {i}: {repr(response['choices'][0]['text'])}"
            )
            prev_prompt_len = response["usage"]["prompt_tokens"]
            curr_prompt_len = response["usage"]["total_tokens"]
            request["prompt"] += response["choices"][0]["text"]

    async def test_multi_round_request(self,
                                       init_prompts: List[str],
                                       max_rounds: int = 8,
                                       threshold: float = 0.75):
        async with await self.new_session() as session:
            chat_threads = [
                self.multi_round_request(session, prompt, max_rounds, threshold)
                for prompt in init_prompts
            ]
            await asyncio.gather(*chat_threads)


class CacheBlockMeta:

    def __init__(self, hash: int, parent_hash: Optional[int] = None):
        self.hash = hash
        self.parent_hash = parent_hash
        # TODO: maintain next_hashes for partial matching

    def __str__(self):
        if self.parent_hash is None:
            return f"CacheBlockMeta({self.hash:016x})"
        else:
            return f"CacheBlockMeta({self.hash:016x}, {self.parent_hash:016x})"

    def __repr__(self):
        return self.__str__()


def block_key_hasher(token_ids: List[int],
                     parent_hash: Optional[int] = None) -> int:
    block_key = BlockKey(token_ids)
    return BlockKeyHasher.hash(block_key,
                               0 if parent_hash is None else parent_hash)


class KvCacheBlockMap:

    def __init__(self):
        self.kv_blocks: dict[int, CacheBlockMeta] = {}

    def update_with_events(self, events: List[dict]):
        for event in events:
            if event["type"] == "stored":
                parent_hash = event["parent_hash"]
                for block in event["blocks"]:
                    block_hash = block["block_hash"]
                    self.kv_blocks[block_hash] = CacheBlockMeta(
                        block_hash, parent_hash)
            elif event["type"] == "removed":
                block_hashes = event["block_hashes"]
                for block_hash in block_hashes:
                    self.kv_blocks.pop(block_hash, None)

    def get_block_match_count(self, block_hashes: List[int]) -> int:
        count = 0
        for block_hash in block_hashes:
            if block_hash in self.kv_blocks:
                count += 1
            else:
                break
        return count

    def __str__(self):
        return f"ServerState(active_requests={self.active_requests}, kv_blocks={', '.join(str(block) for block in self.kv_blocks.values())})"

    def __repr__(self):
        return self.__str__()


class KvCacheEventWorkerTester(BasicWorkerTester):

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.kv_cache_block_maps = {}
        for ctx_server in ctx_servers:
            self.kv_cache_block_maps[ctx_server] = KvCacheBlockMap()
        for gen_server in gen_servers:
            if gen_server not in self.kv_cache_block_maps:
                self.kv_cache_block_maps[gen_server] = KvCacheBlockMap()

    async def send_request(self, session: aiohttp.ClientSession, url: str,
                           request: dict) -> dict:
        response = await super().send_request(session, url, request)

        events = await self.query_kv_cache_events(session, url)
        self.kv_cache_block_maps[url].update_with_events(events)
        return response

    async def multi_round_request(self,
                                  session: aiohttp.ClientSession,
                                  init_prompt: str,
                                  max_rounds: int,
                                  check_match_count: bool = True):
        request = {
            "model": MODEL_NAME,
            "prompt": init_prompt,
            "max_tokens": 64,
            "temperature": 0.0,
        }
        tokens_per_block = 32  # TODO: read from config
        prev_ctx_match_count = 0
        prev_gen_match_count = 0
        for i in range(max_rounds):
            # split tokens into blocks and check block match count by hash
            tokens = self.tokenizer(request["prompt"])["input_ids"]
            block_hashes = []
            for t in range(0, len(tokens), tokens_per_block):
                block_hashes.append(
                    block_key_hasher(tokens[t:t + tokens_per_block],
                                     None if t == 0 else block_hashes[-1]))
            ctx_match_count = self.kv_cache_block_maps[
                self.ctx_servers[0]].get_block_match_count(block_hashes)
            gen_match_count = self.kv_cache_block_maps[
                self.gen_servers[0]].get_block_match_count(block_hashes)
            assert ctx_match_count >= prev_ctx_match_count
            assert gen_match_count >= prev_gen_match_count

            response = await self.send_disagg_request(session,
                                                      self.ctx_servers[0],
                                                      self.gen_servers[0],
                                                      request)
            logger.info(
                f"Received response {i}: {repr(response['choices'][0]['text'])}"
            )
            prev_ctx_match_count = ctx_match_count
            prev_gen_match_count = gen_match_count
            request["prompt"] += response["choices"][0]["text"]

        if check_match_count:
            assert ctx_match_count > 0
            assert gen_match_count >= ctx_match_count
        return request["prompt"]

    async def test_multi_round_request(self,
                                       init_prompts: List[str],
                                       max_rounds: int = 8):
        async with await self.new_session() as session:
            chat_threads = [
                self.multi_round_request(session, prompt, max_rounds, False)
                for prompt in init_prompts
            ]
            prompts = await asyncio.gather(*chat_threads)
            await asyncio.gather(*[
                self.multi_round_request(session, prompt, 1, True)
                for prompt in prompts
            ])


def prepare_model(llama_model_root: str, llm_venv):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_workers_conditional_disaggregation(disaggregated_test_root,
                                            disaggregated_example_root,
                                            llm_venv, llama_model_root):
    config_file = os.path.join(disaggregated_test_root,
                               'test_configs/disagg_config_cache_reuse.yaml')
    prepare_model(llama_model_root, llm_venv)
    cwd = llm_venv.get_working_directory()

    with open(os.path.join(cwd, 'output_workers.log'), 'w') as log_file:
        workers_proc, ctx_servers, gen_servers = run_disaggregated_workers(
            config_file=config_file,
            stdout=log_file,
            env=llm_venv._new_env,
            cwd=cwd,
            num_ranks=2)
        try:
            tester = ConditionalWorkerTester(ctx_servers, gen_servers)
            prompts_file = os.path.join(disaggregated_example_root,
                                        'clients/prompts.json')
            with open(prompts_file, 'r') as f:
                prompts = json.load(f)
            asyncio.run(tester.test_multi_round_request(prompts))
        except Exception as e:
            raise e
        finally:
            workers_proc.terminate()
            workers_proc.wait()


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_workers_kv_cache_events(disaggregated_test_root,
                                 disaggregated_example_root, llm_venv,
                                 llama_model_root):
    config_file = os.path.join(disaggregated_test_root,
                               'test_configs/disagg_config_cache_reuse.yaml')
    prepare_model(llama_model_root, llm_venv)
    cwd = llm_venv.get_working_directory()

    with open(os.path.join(cwd, 'output_workers.log'), 'w') as log_file:
        workers_proc, ctx_servers, gen_servers = run_disaggregated_workers(
            config_file=config_file,
            stdout=log_file,
            env=llm_venv._new_env,
            cwd=cwd,
            num_ranks=2)
        try:
            tester = KvCacheEventWorkerTester(ctx_servers, gen_servers)
            prompts_file = os.path.join(disaggregated_example_root,
                                        'clients/prompts.json')
            with open(prompts_file, 'r') as f:
                prompts = json.load(f)
            asyncio.run(tester.test_multi_round_request(prompts, 6))
        except Exception as e:
            raise e
        finally:
            workers_proc.terminate()
            workers_proc.wait()
