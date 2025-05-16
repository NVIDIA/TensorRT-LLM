import asyncio
import contextlib
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
from tensorrt_llm.serve.openai_protocol import (CompletionRequest,
                                                DisaggregatedParams)
from tensorrt_llm.serve.router import (KvCacheAwareRouter,
                                       KvCacheAwareServerState,
                                       block_key_hasher)

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
            if not context_ready:
                return False
            generation_ready = all([
                await self.check_server_ready(session, url)
                for url in self.gen_servers
            ])
            return generation_ready

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


class KvCacheEventWorkerTester(BasicWorkerTester):

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.kv_cache_block_maps: dict[str, KvCacheAwareServerState] = {}
        self.kv_cache_event_maps: dict[str, list[dict]] = {}
        for ctx_server in ctx_servers:
            self.kv_cache_block_maps[ctx_server] = KvCacheAwareServerState(
                ctx_server)
            self.kv_cache_event_maps[ctx_server] = []
        for gen_server in gen_servers:
            if gen_server not in self.kv_cache_block_maps:
                self.kv_cache_block_maps[gen_server] = KvCacheAwareServerState(
                    gen_server)
                self.kv_cache_event_maps[gen_server] = []

    async def send_request(self, session: aiohttp.ClientSession, url: str,
                           request: dict) -> dict:
        response = await super().send_request(session, url, request)
        events = await self.query_kv_cache_events(session, url)
        async with self.kv_cache_block_maps[url]._lock:
            self.kv_cache_block_maps[url].update_with_events(events)
            self.kv_cache_event_maps[url].extend(events)
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
        assert len(self.ctx_servers) == 1 and len(self.gen_servers) == 1, \
            "This test assumes 1P1D"
        ctx_server = self.ctx_servers[0]
        gen_server = self.gen_servers[0]
        ctx_blocks = self.kv_cache_block_maps[ctx_server]
        gen_blocks = self.kv_cache_block_maps[gen_server]
        ctx_events = self.kv_cache_event_maps[ctx_server]
        gen_events = self.kv_cache_event_maps[gen_server]
        for i in range(max_rounds):
            # split tokens into blocks and check block match count by hash
            tokens = self.tokenizer(request["prompt"])["input_ids"]
            block_hashes = []
            for t in range(0, len(tokens) - 1, tokens_per_block):
                t_end = min(t + tokens_per_block, len(tokens) - 1)
                if t_end - t < tokens_per_block:
                    # partial block
                    break
                block_hashes.append(
                    block_key_hasher(tokens[t:t_end],
                                     None if t == 0 else block_hashes[-1]))
            ctx_match_count = await ctx_blocks.match_blocks([block_hashes])
            gen_match_count = await gen_blocks.match_blocks([block_hashes])
            ctx_evicted = False
            gen_evicted = False
            for event in ctx_events:
                if event["type"] == "removed":
                    ctx_evicted = True
                    break
            for event in gen_events:
                if event["type"] == "removed":
                    gen_evicted = True
                    break
            assert ctx_evicted or ctx_match_count >= prev_ctx_match_count
            assert gen_evicted or gen_match_count >= prev_gen_match_count
            ctx_events.clear()
            gen_events.clear()

            response = await self.send_disagg_request(session, ctx_server,
                                                      gen_server, request)
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


class KvCacheAwareRouterTester(BasicWorkerTester):

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = 180,
                 server_start_timeout_secs: int = 180):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)
        self.ctx_router = KvCacheAwareRouter(ctx_servers)
        self.gen_router = KvCacheAwareRouter(gen_servers)

    async def multi_round_request(self,
                                  session: aiohttp.ClientSession,
                                  init_prompt: str,
                                  max_rounds: int = 8,
                                  check_server_match: bool = True):
        request = {
            "model": MODEL_NAME,
            "prompt": init_prompt,
            "max_tokens": 64,
            "temperature": 0.0,
        }
        ctx_server_prev = None
        gen_server_prev = None
        for i in range(max_rounds):
            openai_request = CompletionRequest(
                model=MODEL_NAME,
                prompt=request["prompt"],
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only"))
            ctx_server, ctx_info = await self.ctx_router.get_next_server(
                openai_request)
            prompt_str = request["prompt"]
            request["prompt"] = ctx_info["token_lists"][0]
            openai_request.disaggregated_params.request_type = "generation_only"
            gen_server, _ = await self.gen_router.get_next_server(openai_request
                                                                  )
            if check_server_match and ctx_server_prev is not None:
                assert ctx_server == ctx_server_prev
                assert gen_server == gen_server_prev
            ctx_server_prev = ctx_server
            gen_server_prev = gen_server
            response = await self.send_disagg_request(session, ctx_server,
                                                      gen_server, request)
            await asyncio.gather(
                self.ctx_router.finish_request(openai_request, session),
                self.gen_router.finish_request(openai_request, session))
            logger.info(
                f"Received response {i}: {repr(response['choices'][0]['text'])}"
            )
            request["prompt"] = prompt_str + response["choices"][0]["text"]

        return request["prompt"]

    async def test_multi_round_request(self,
                                       init_prompts: List[str],
                                       max_rounds: int = 8,
                                       warm_up_rounds: int = 4):
        async with await self.new_session() as session:
            chat_threads = [
                self.multi_round_request(session, prompt, warm_up_rounds, False)
                for prompt in init_prompts
            ]
            prompts = await asyncio.gather(*chat_threads)
            logger.info("Warm up done")
            chat_threads = [
                self.multi_round_request(session, prompt, max_rounds, True)
                for prompt in prompts
            ]
            await asyncio.gather(*chat_threads)

    async def test_eviction(self):
        async with await self.new_session() as session:
            # send a dummy request for initialization
            dummy_request = {
                "model": MODEL_NAME,
                "prompt": [3] * 100,
                "max_tokens": 1,
                "temperature": 0.0,
            }
            assert len(self.gen_servers) == 1
            server = self.gen_servers[0]  # only test on this server
            server_state = self.gen_router._server_state[server]
            await self.send_request(session, server, dummy_request)
            # get block pool size from created event
            events = await self.query_kv_cache_events(session, server)
            server_state.update_with_events(events)
            block_pool_size = None
            for event in events:
                if event["type"] == "created":
                    block_pool_size = event["num_blocks_per_cache_level"][0]
                    break
            assert block_pool_size is not None
            logger.info(f"Block pool size: {block_pool_size}")

            # the dummy request can be reused
            openai_request = CompletionRequest(model=MODEL_NAME,
                                               prompt=dummy_request["prompt"])
            server, info = await self.gen_router.get_next_server(openai_request)
            first_match = info["matches"][0]
            assert first_match > 0
            await self.gen_router.finish_request(openai_request)

            # flood requests until eviction
            batch_size = 8
            blocks_per_request = 32
            requests = [copy.copy(dummy_request) for _ in range(batch_size)]
            has_evicted = False
            for i in range(0, block_pool_size // blocks_per_request + 10,
                           batch_size):
                logger.info(f"Flooding request {i} ~ {i + batch_size - 1}")
                prompt_len = self.gen_router._tokens_per_block * blocks_per_request - 10
                for j in range(batch_size):
                    prompt = [10 + i + j] * prompt_len
                    requests[j]["prompt"] = prompt
                await asyncio.gather(*[
                    self.send_request(session, server, request)
                    for request in requests
                ])
                events = await self.query_kv_cache_events(session, server)
                server_state.update_with_events(events)
                for event in events:
                    if event["type"] == "removed":
                        has_evicted = True
            assert has_evicted

            # the dummy request's reusable length decreases after eviction
            server, info = await self.gen_router.get_next_server(openai_request)
            assert info["matches"][0] < first_match


def prepare_llama_model(llama_model_root: str, llm_venv):
    src_dst_dict = {
        llama_model_root:
        f"{llm_venv.get_working_directory()}/TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)


def load_default_prompts(disaggregated_example_root: str):
    prompts_file = os.path.join(disaggregated_example_root,
                                'clients/prompts.json')
    with open(prompts_file, 'r') as f:
        return json.load(f)


@contextlib.contextmanager
def background_workers(llm_venv, config_file: str, num_ranks: int = None):
    cwd = llm_venv.get_working_directory()
    log_file = open(os.path.join(cwd, 'output_workers.log'), 'w')
    workers_proc, ctx_servers, gen_servers = run_disaggregated_workers(
        config_file=config_file,
        stdout=log_file,
        env=llm_venv._new_env,
        cwd=cwd,
        num_ranks=num_ranks)
    try:
        yield ctx_servers, gen_servers
    finally:
        workers_proc.terminate()
        workers_proc.wait()
        log_file.close()


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_workers_conditional_disaggregation(disaggregated_test_root,
                                            disaggregated_example_root,
                                            llm_venv, llama_model_root):
    config_file = os.path.join(disaggregated_test_root,
                               'test_configs/disagg_config_cache_reuse.yaml')
    prepare_llama_model(llama_model_root, llm_venv)

    with background_workers(llm_venv, config_file,
                            2) as (ctx_servers, gen_servers):
        tester = ConditionalWorkerTester(ctx_servers, gen_servers)
        prompts = load_default_prompts(disaggregated_example_root)
        asyncio.run(tester.test_multi_round_request(prompts))


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_workers_kv_cache_events(disaggregated_test_root,
                                 disaggregated_example_root, llm_venv,
                                 llama_model_root):
    config_file = os.path.join(disaggregated_test_root,
                               'test_configs/disagg_config_cache_reuse.yaml')
    prepare_llama_model(llama_model_root, llm_venv)

    with background_workers(llm_venv, config_file,
                            2) as (ctx_servers, gen_servers):
        tester = KvCacheEventWorkerTester(ctx_servers, gen_servers)
        prompts = load_default_prompts(disaggregated_example_root)
        asyncio.run(tester.test_multi_round_request(prompts, 6))


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_workers_kv_cache_aware_router(disaggregated_test_root,
                                       disaggregated_example_root, llm_venv,
                                       llama_model_root):
    config_file = os.path.join(
        disaggregated_test_root,
        'test_configs/disagg_config_cache_aware_balance.yaml')
    prepare_llama_model(llama_model_root, llm_venv)

    with background_workers(llm_venv, config_file,
                            4) as (ctx_servers, gen_servers):
        tester = KvCacheAwareRouterTester(ctx_servers, gen_servers)
        prompts = load_default_prompts(disaggregated_example_root)
        asyncio.run(tester.test_multi_round_request(prompts, 6, 4))


@pytest.mark.parametrize("llama_model_root", ['TinyLlama-1.1B-Chat-v1.0'],
                         indirect=True)
def test_workers_kv_cache_aware_router_eviction(disaggregated_test_root,
                                                disaggregated_example_root,
                                                llm_venv, llama_model_root):
    config_file = os.path.join(disaggregated_test_root,
                               'test_configs/disagg_config_cache_reuse.yaml')
    prepare_llama_model(llama_model_root, llm_venv)

    with background_workers(llm_venv, config_file,
                            2) as (ctx_servers, gen_servers):
        tester = KvCacheAwareRouterTester(ctx_servers, gen_servers)
        asyncio.run(tester.test_eviction())
