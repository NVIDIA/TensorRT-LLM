import os
import tempfile

import pytest
import torch
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer


# Common fixtures for model and client
@pytest.fixture(scope="module")
def model_path(model_name: str):
    '''Fixture for model path, depends on model_name defined in test files'''
    return get_model_path(model_name)


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


# Common parametrized fixtures
@pytest.fixture(scope="module",
                params=[None, 'pytorch'],
                ids=["trt", "pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module",
                params=[True, False],
                ids=["extra_options", "no_extra_options"])
def extra_llm_api_options(request):
    return request.param


# Fixture for temporary options file, to be used with a dictionary defined in test files
@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(extra_llm_api_options_dict: dict):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)
        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(request, model_path, backend=None, num_postprocess_workers=None, 
           extra_llm_api_options=None, max_beam_width=None, tp_size=None, 
           max_batch_size=None, max_seq_len=None, build_engine=None, 
           engine_from_fp8_quantization=None, reasoning_parser=None, port=None):
    args = []

    if 'backend' in request.fixturenames:
        backend = request.getfixturevalue('backend')
        if backend == 'pytorch':
            args.extend(['--backend', 'pytorch'])
        else:  # trt or None
            if 'max_beam_width' in request.fixturenames:
                max_beam_width = request.getfixturevalue('max_beam_width')
                args.extend(['--max_beam_width', str(max_beam_width)])
            else:
                # default from _test_openai_chat.py
                args.extend(['--max_beam_width', '4'])

    if 'tp_size' in request.fixturenames:
        tp_size = request.getfixturevalue('tp_size')
        args.extend(['--tp_size', str(tp_size)])

    if 'num_postprocess_workers' in request.fixturenames:
        num_postprocess_workers = request.getfixturevalue(
            'num_postprocess_workers')
        args.extend(['--num_postprocess_workers', str(num_postprocess_workers)])

    if 'extra_llm_api_options' in request.fixturenames:
        extra_llm_api_options = request.getfixturevalue('extra_llm_api_options')
        if extra_llm_api_options:
            temp_file = request.getfixturevalue(
                'temp_extra_llm_api_options_file')
            args.extend(['--extra_llm_api_options', temp_file])

    if 'max_batch_size' in request.fixturenames:
        max_batch_size = request.getfixturevalue('max_batch_size')
        args.extend(['--max_batch_size', max_batch_size])

    if 'max_seq_len' in request.fixturenames:
        max_seq_len = request.getfixturevalue('max_seq_len')
        args.extend(['--max_seq_len', max_seq_len])

    if 'build_engine' in request.fixturenames:
        build_engine = request.getfixturevalue('build_engine')
        args.extend(['--build_engine', build_engine])

    if 'engine_from_fp8_quantization' in request.fixturenames:
        engine_from_fp8_quantization = request.getfixturevalue(
            'engine_from_fp8_quantization')
        args.extend(['--engine_dir', engine_from_fp8_quantization])

    if 'reasoning_parser' in request.fixturenames:
        reasoning_parser = request.getfixturevalue('reasoning_parser')
        args.extend(['--reasoning_parser', reasoning_parser])

    port = None
    if 'port' in request.fixturenames:
        port = request.getfixturevalue('port')

    with RemoteOpenAIServer(model_path, args, port=port) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_from_engine(request, model_name):
    args = []

    engine_dir = None
    if 'engine_from_fp8_quantization' in request.fixturenames:
        engine_dir = request.getfixturevalue('engine_from_fp8_quantization')
        args.extend(['--tp_size', '2'])
    elif 'build_engine' in request.fixturenames:
        engine_dir = request.getfixturevalue('build_engine')
        args.extend(['--tp_size', '4'])

    if engine_dir:
        model_path = get_model_path(model_name)
        args.extend(['--tokenizer', model_path])

        if 'force_deterministic' in request.fixturenames and request.getfixturevalue(
                'force_deterministic'):
            os.environ["FORCE_DETERMINISTIC"] = "1"

        with RemoteOpenAIServer(engine_dir, args) as remote_server:
            yield remote_server

        if 'force_deterministic' in request.fixturenames and request.getfixturevalue(
                'force_deterministic'):
            os.environ.pop("FORCE_DETERMINISTIC", None)
    else:
        # should not happen if called from right test
        pytest.skip("This server fixture requires an engine.")


@pytest.fixture(scope="module")
def multi_node_server(model_name: str, backend: str, tp_pp_size: tuple):
    os.environ["FORCE_DETERMINISTIC"] = "1"
    try:
        model_path = get_model_path(model_name)
        tp_size, pp_size = tp_pp_size
        device_count = torch.cuda.device_count()
        args = [
            "--tp_size", f"{tp_size}", "--pp_size", f"{pp_size}",
            "--gpus_per_node", f"{device_count}",
            "--kv_cache_free_gpu_memory_fraction", "0.95"
        ]
        if backend is not None:
            args.append("--backend")
            args.append(backend)
        with RemoteOpenAIServer(model_path, args, llmapi_launch=True,
                                port=8001) as remote_server:
            yield remote_server
    finally:
        os.environ.pop("FORCE_DETERMINISTIC")
