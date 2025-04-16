import asyncio
import copy
import json
import logging
import os
import subprocess
from typing import List, Optional, Tuple

import aiohttp
import pytest
import yaml

logging.basicConfig(level=logging.INFO)
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
                logging.error(f"Received failed response {response_dict}")
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
                logging.info(
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
                logging.info(f"Sending normal request at iter {i}")
                response = await self.send_request(session, self.gen_servers[0],
                                                   request)
            else:
                logging.info(f"Sending disaggregated request at iter {i}")
                response = await self.send_disagg_request(
                    session, self.ctx_servers[0], self.gen_servers[0], request)
            logging.info(
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

    with open(
            os.path.join(llm_venv.get_working_directory(),
                         'output_workers.log'), 'w') as log_file:
        workers_proc, ctx_servers, gen_servers = run_disaggregated_workers(
            config_file=config_file,
            stdout=log_file,
            env=llm_venv._new_env,
            cwd=llm_venv.get_working_directory(),
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
