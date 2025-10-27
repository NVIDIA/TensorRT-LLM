import asyncio
import contextlib
import copy
import json
import os
import subprocess
from typing import Generator, List, Optional, Tuple

import aiohttp
import pytest
import yaml
from defs.conftest import skip_no_hopper
from defs.disaggregated.test_disaggregated_single_gpu import \
    model_path as get_model_path
from defs.trt_test_alternative import popen
from transformers import AutoTokenizer

from tensorrt_llm import logger
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (CompletionRequest,
                                                DisaggregatedParams)
from tensorrt_llm.serve.router import (KvCacheAwareRouter,
                                       KvCacheAwareServerState, ServerRole,
                                       block_key_hasher)


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
) -> Tuple[Generator[subprocess.Popen, None, None], List[str], List[str]]:

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
    workers_proc = popen(workers_cmd,
                         stdout=stdout,
                         stderr=subprocess.STDOUT,
                         env=env,
                         cwd=cwd)
    return workers_proc, ctx_servers, gen_servers


DEFAULT_TIMEOUT_SERVER_START = 900
DEFAULT_TIMEOUT_REQUEST = 180


class BasicWorkerTester:

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = DEFAULT_TIMEOUT_REQUEST,
                 server_start_timeout_secs: int = DEFAULT_TIMEOUT_SERVER_START):
        self.ctx_servers = ctx_servers
        self.gen_servers = gen_servers
        self.req_timeout_secs = req_timeout_secs
        self.server_start_timeout_secs = server_start_timeout_secs

    async def new_session(self):
        session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(force_close=True),
            timeout=aiohttp.ClientTimeout(total=self.req_timeout_secs))
        await OpenAIDisaggServer.wait_for_all_servers_ready(
            session, self.ctx_servers, self.gen_servers,
            self.server_start_timeout_secs)
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


class ConditionalWorkerTester(BasicWorkerTester):

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = DEFAULT_TIMEOUT_REQUEST,
                 server_start_timeout_secs: int = DEFAULT_TIMEOUT_SERVER_START,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)
        self.model_name = model_name

    async def multi_round_request(self, session: aiohttp.ClientSession,
                                  init_prompt: str, max_rounds: int,
                                  threshold: float):
        request = {
            "model": self.model_name,
            "prompt": init_prompt,
            "max_tokens": 10,
            "ignore_eos": True,
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
                 req_timeout_secs: int = DEFAULT_TIMEOUT_REQUEST,
                 server_start_timeout_secs: int = DEFAULT_TIMEOUT_SERVER_START,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 model_path: Optional[str] = None):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)
        if model_path is None:
            model_path = get_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_name = model_name
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
            "model": self.model_name,
            "prompt": init_prompt,
            "max_tokens": 64,
            "ignore_eos": True,
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
            ctx_match_count = await ctx_blocks.matched_tokens([block_hashes])
            gen_match_count = await gen_blocks.matched_tokens([block_hashes])
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
            assert gen_match_count > 0
            assert gen_match_count >= ctx_match_count or gen_evicted
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
            # send a request to flush events
            await self.multi_round_request(session, init_prompts[0], 1, False)
            await asyncio.gather(*[
                self.multi_round_request(session, prompt, 1, True)
                for prompt in prompts
            ])


class KvCacheAwareRouterTester(BasicWorkerTester):

    def __init__(self,
                 ctx_servers: List[str],
                 gen_servers: List[str],
                 req_timeout_secs: int = DEFAULT_TIMEOUT_REQUEST,
                 server_start_timeout_secs: int = DEFAULT_TIMEOUT_SERVER_START,
                 model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                 tokens_per_block: int = 32):
        super().__init__(ctx_servers, gen_servers, req_timeout_secs,
                         server_start_timeout_secs)
        self.ctx_router = KvCacheAwareRouter(server_role=ServerRole.CONTEXT,
                                             servers=ctx_servers,
                                             tokens_per_block=tokens_per_block)
        self.gen_router = KvCacheAwareRouter(server_role=ServerRole.GENERATION,
                                             servers=gen_servers,
                                             tokens_per_block=tokens_per_block)
        self.model_name = model_name

    async def multi_round_request(self,
                                  session: aiohttp.ClientSession,
                                  init_prompt: str,
                                  max_rounds: int = 8,
                                  check_server_match: bool = True):
        request = {
            "model": self.model_name,
            "prompt": init_prompt,
            "max_tokens": 64,
            "ignore_eos": True,
            "temperature": 0.0,
        }
        ctx_server_prev = None
        gen_server_prev = None
        ctx_match = 0
        gen_match = 0
        for i in range(max_rounds):
            openai_request = CompletionRequest(
                model=self.model_name,
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
                ctx_match += int(ctx_server == ctx_server_prev)
                gen_match += int(gen_server == gen_server_prev)
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

        if check_server_match:
            assert ctx_match > max_rounds // 2
            assert gen_match > max_rounds // 2
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
                "model": self.model_name,
                "prompt": [3] * 2000,
                "max_tokens": 1,
                "ignore_eos": True,
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
            openai_request = CompletionRequest(model=self.model_name,
                                               prompt=dummy_request["prompt"])
            server, info = await self.gen_router.get_next_server(openai_request)
            first_match = info["matches"][0]
            logger.info(f"Matched blocks: {first_match}")
            assert first_match > 0
            await self.gen_router.finish_request(openai_request)

            # flood requests until eviction
            batch_size = 64
            blocks_per_request = 32
            requests = [copy.copy(dummy_request) for _ in range(batch_size)]
            has_evicted = False
            for i in range(0, block_pool_size // blocks_per_request * 2,
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
            logger.info(
                f"Matched blocks: {first_match} -> {info['matches'][0]}")
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
    with open(os.path.join(cwd, 'output_workers.log'), 'w+') as log_file:
        workers_proc, ctx_servers, gen_servers = run_disaggregated_workers(
            config_file=config_file,
            stdout=log_file,
            env=llm_venv._new_env,
            cwd=cwd,
            num_ranks=num_ranks)
        try:
            with workers_proc as proc:
                yield ctx_servers, gen_servers
        except Exception:
            log_file.seek(0)
            logger.error("-------- Worker output --------")
            logger.error(log_file.read())
            raise
        finally:
            proc.terminate()
            proc.wait()


@pytest.mark.skip(reason="https://nvbugs/5372970")
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


@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_workers_conditional_disaggregation_deepseek_v3_lite_bf16(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    config_file = os.path.join(
        disaggregated_test_root,
        'test_configs/disagg_config_cache_reuse_deepseek_v3.yaml')
    model_root = f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16"
    src_dst_dict = {
        deepseek_v3_model_root: model_root,
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

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
        asyncio.run(tester.test_multi_round_request(prompts, 16, 4))


@skip_no_hopper
@pytest.mark.parametrize("deepseek_v3_model_root", ['DeepSeek-V3-Lite-bf16'],
                         indirect=True)
def test_workers_kv_cache_aware_router_deepseek_v3_lite_bf16(
        disaggregated_test_root, disaggregated_example_root, llm_venv,
        deepseek_v3_model_root):
    config_file = os.path.join(
        disaggregated_test_root,
        'test_configs/disagg_config_cache_aware_balance_deepseek_v3.yaml')
    model_root = f"{llm_venv.get_working_directory()}/DeepSeek-V3-Lite/bf16"
    src_dst_dict = {
        deepseek_v3_model_root: model_root,
    }
    for src, dst in src_dst_dict.items():
        if not os.path.islink(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(src, dst, target_is_directory=True)

    with background_workers(llm_venv, config_file,
                            4) as (ctx_servers, gen_servers):
        os.chdir(llm_venv.get_working_directory())
        tester = KvCacheAwareRouterTester(ctx_servers,
                                          gen_servers,
                                          model_name="DeepSeek-V3-Lite/bf16",
                                          tokens_per_block=64)
        prompts = load_default_prompts(disaggregated_example_root)
        asyncio.run(tester.test_multi_round_request(prompts, 8, 4))


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
