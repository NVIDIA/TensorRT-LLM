# -*- coding: utf-8 -*-

import datetime
import os
import shutil
import tempfile

import pytest

# pytest_plugins = ["pytester", "trt_test.pytest_plugin"]
USE_TURTLE = True
try:
    import trt_test  # noqa
except ImportError:
    from .test_list_parser import (CorrectionMode, get_test_name_corrections_v2,
                                   handle_corrections)
    from .trt_test_alternative import (SessionDataWriter, check_call,
                                       check_output, print_info)

    @pytest.fixture(scope="session")
    def trt_config():
        return None  # tekit shall never call this

    @pytest.fixture(scope="session")
    def gitlab_token():
        return None  # tekit shall never call this

    @pytest.fixture(scope="session")
    def versions_from_infer_device():
        pass

    USE_TURTLE = False
else:
    from trt_test.misc import check_call, check_output, print_info
    from trt_test.session_data_writer import SessionDataWriter
    USE_TURTLE = True


def llm_models_root() -> str:
    '''return LLM_MODELS_ROOT path if it is set in env, assert when it's set but not a valid path
    '''
    LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", None)
    if LLM_MODELS_ROOT is not None:
        assert os.path.isabs(
            LLM_MODELS_ROOT), "LLM_MODELS_ROOT must be absolute path"
        assert os.path.exists(
            LLM_MODELS_ROOT), "LLM_MODELS_ROOT must exists when its specified"
    return LLM_MODELS_ROOT


def venv_check_call(venv, cmd):

    def _war_check_call(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        return check_call(*args, **kwargs)

    venv.run_cmd(cmd, caller=_war_check_call, print_script=False)


def venv_check_output(venv, cmd):

    def _war_check_output(*args, **kwargs):
        kwargs["cwd"] = venv.get_working_directory()
        output = check_output(*args, **kwargs)
        return output

    return venv.run_cmd(cmd, caller=_war_check_output, print_script=False)


@pytest.fixture(scope="session")
def trt_performance_cache_name():
    return "performance.cache"


@pytest.fixture(scope="session")
def trt_performance_cache_fpath(trt_config, trt_performance_cache_name):
    fpath = os.path.join(trt_config["workspace"], trt_performance_cache_name)
    return fpath


# Get the executing turtle case name
@pytest.fixture(autouse=True)
def turtle_case_name(request):
    return request.node.nodeid


@pytest.fixture(scope="session")
def output_dir(request):
    return request.config._trt_config["output_dir"]


@pytest.fixture(scope="session")
def llm_backend_root():
    return os.path.join(os.environ["LLM_ROOT"], "triton_backend")


@pytest.fixture(scope="session")
def llm_session_data_writer(trt_config, trt_gpu_clock_lock,
                            versions_from_infer_device, output_dir):
    """
    Fixture for the SessionDataWriter, used to write session data to output directory.
    """

    # Attempt to see if we can run infer_device to get the necessary tags for perf_runner
    perf_tag_data = trt_config["perf_trt_tag"]

    if versions_from_infer_device:
        for k, v in versions_from_infer_device.items():
            if k not in perf_tag_data or perf_tag_data[k] is None:
                perf_tag_data[k] = v

    session_data_writer = SessionDataWriter(
        perf_trt_tag=perf_tag_data,
        log_output_directory=output_dir,
        output_formats=trt_config["perf_log_formats"],
        gpu_clock_lock=trt_gpu_clock_lock,
    )

    yield session_data_writer

    session_data_writer.teardown()


if USE_TURTLE:

    @pytest.fixture(scope="session")
    def trt_py3_venv_factory(trt_py_base_venv_factory):
        """
        Session-scoped fixture which provides a factory function to produce a VirtualenvRunner capable of
        running Python3 code.  Used by other session-scoped fixtures which need to modify the default VirtualenvRunner prolog.
        """

        # TODO: remove update env after TURTLE support multi devices
        # Temporarily update CUDA_VISIBLE_DEVICES visible device
        device_count = get_device_count()
        visible_devices = ",".join([str(i) for i in range(device_count)])

        print_info(f"Setting CUDA_VISIBLE_DEVICES to {visible_devices}.")

        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        def factory():
            return trt_py_base_venv_factory("python3")

        return factory

    @pytest.fixture(scope="session")
    def llm_backend_venv(trt_py3_venv_factory):
        """
        The fixture venv used for LLM tests.
        """
        venv = trt_py3_venv_factory()
        return venv
else:

    @pytest.fixture(scope="session")
    def custom_user_workspace(request):
        return request.config.getoption("--workspace")

    @pytest.fixture(scope="session")
    def llm_backend_venv(custom_user_workspace):
        workspace_dir = custom_user_workspace
        subdir = datetime.datetime.now().strftime("ws-%Y-%m-%d-%H-%M-%S")
        if workspace_dir is None:
            workspace_dir = "triton-backend-test-workspace"
        workspace_dir = os.path.join(workspace_dir, subdir)
        from defs.local_venv import PythonVenvRunnerImpl
        return PythonVenvRunnerImpl("", "", "python3",
                                    os.path.join(os.getcwd(), workspace_dir))


@pytest.fixture(scope="session")
def llm_backend_gpt_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "tools", "gpt")


@pytest.fixture(scope="session")
def llm_backend_all_models_root(llm_backend_root):
    return os.path.join(llm_backend_root, "all_models")


@pytest.fixture(scope="session")
def llm_backend_whisper_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "tools", "whisper")


@pytest.fixture(scope="session")
def llm_backend_multimodal_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "tools", "multimodal")


@pytest.fixture(scope="session")
def llm_backend_llmapi_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "tools", "llmapi")


@pytest.fixture(scope="session")
def llm_backend_inflight_batcher_llm_root(llm_backend_root):
    return os.path.join(llm_backend_root, "tools", "inflight_batcher_llm")


@pytest.fixture(scope="session")
def llm_backend_dataset_root(llm_backend_root):
    return os.path.join(llm_backend_root, "tools", "dataset")


@pytest.fixture(scope="session")
def tensorrt_llm_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples")


@pytest.fixture(scope="session")
def tensorrt_llm_gpt_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/gpt")


@pytest.fixture(scope="session")
def tensorrt_llm_gptj_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/contrib/gptj")


@pytest.fixture(scope="session")
def tensorrt_llm_multimodal_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/multimodal")


@pytest.fixture(scope="session")
def tensorrt_llm_opt_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/contrib/opt")


@pytest.fixture(scope="session")
def tensorrt_llm_medusa_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/medusa")


@pytest.fixture(scope="session")
def tensorrt_llm_eagle_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/eagle")


@pytest.fixture(scope="session")
def tensorrt_llm_enc_dec_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/enc_dec")


@pytest.fixture(scope="session")
def tensorrt_llm_whisper_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/whisper")


@pytest.fixture(scope="session")
def tensorrt_llm_llama_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/llama")


@pytest.fixture(scope="session")
def tensorrt_llm_qwen_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/qwen")


@pytest.fixture(scope="session")
def tensorrt_llm_mllama_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/mllama")


@pytest.fixture(scope="session")
def tensorrt_llm_mixtral_example_root(llm_backend_root):
    return os.path.join(llm_backend_root, "../examples/models/core/mixtral")


@pytest.fixture(scope="session")
def inflight_batcher_llm_client_root(llm_backend_root):
    inflight_batcher_llm_client_root = os.path.join(llm_backend_root,
                                                    "inflight_batcher_llm",
                                                    "client")

    assert os.path.exists(
        inflight_batcher_llm_client_root
    ), f"{inflight_batcher_llm_client_root} does not exists."
    return inflight_batcher_llm_client_root


@pytest.fixture(autouse=True)
def skip_by_device_count(request):
    if request.node.get_closest_marker('skip_less_device'):
        device_count = get_device_count()
        expected_count = request.node.get_closest_marker(
            'skip_less_device').args[0]
        if expected_count > int(device_count):
            pytest.skip(
                f'Device count {device_count} is less than {expected_count}')


def get_device_count():
    output = check_output("nvidia-smi -L", shell=True, cwd="/tmp")
    device_count = len(output.strip().split('\n'))

    return device_count


@pytest.fixture(autouse=True)
def skip_by_device_memory(request):
    "fixture for skip less device memory"
    if request.node.get_closest_marker('skip_less_device_memory'):
        device_memory = get_device_memory()
        expected_memory = request.node.get_closest_marker(
            'skip_less_device_memory').args[0]
        if expected_memory > int(device_memory):
            pytest.skip(
                f'Device memory {device_memory} is less than {expected_memory}')


def get_device_memory():
    "get gpu memory"
    memory = 0
    with tempfile.TemporaryDirectory() as temp_dirname:
        cmd = " ".join(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"])
        output = check_output(cmd, shell=True, cwd=temp_dirname)
        memory = int(output.strip().split()[0])

    return memory


@pytest.fixture(scope="session")
def models_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"

    return models_root


@pytest.fixture(scope="session")
def llama_v2_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_v2_tokenizer_model_root = os.path.join(models_root, "llama-models-v2")

    assert os.path.exists(
        llama_v2_tokenizer_model_root
    ), f"{llama_v2_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_v2_tokenizer_model_root


@pytest.fixture(scope="session")
def mistral_v1_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    mistral_v1_tokenizer_model_root = os.path.join(models_root,
                                                   "mistral-7b-v0.1")

    assert os.path.exists(
        mistral_v1_tokenizer_model_root
    ), f"{mistral_v1_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return mistral_v1_tokenizer_model_root


@pytest.fixture(scope="session")
def gpt_tokenizer_model_root(llm_backend_venv):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_tokenizer_model_root = os.path.join(models_root, "gpt2")

    assert os.path.exists(
        gpt_tokenizer_model_root
    ), f"{gpt_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_tokenizer_model_root


@pytest.fixture(scope="session")
def gptj_tokenizer_model_root(llm_backend_venv):
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gptj_tokenizer_model_root = os.path.join(models_root, "gpt-j-6b")

    assert os.path.exists(
        gptj_tokenizer_model_root
    ), f"{gptj_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gptj_tokenizer_model_root


@pytest.fixture(scope="session")
def gpt2_medium_tokenizer_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_tokenizer_model_root = os.path.join(models_root, "gpt2-medium")

    assert os.path.exists(
        gpt_tokenizer_model_root
    ), f"{gpt_tokenizer_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_tokenizer_model_root


@pytest.fixture(scope="session")
def gpt_next_ptuning_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_next_ptuning_model_root = os.path.join(models_root, "email_composition")

    assert os.path.exists(
        gpt_next_ptuning_model_root
    ), f"{gpt_next_ptuning_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_next_ptuning_model_root


@pytest.fixture(scope="session")
def gpt_2b_lora_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    gpt_2b_lora_model_root = os.path.join(models_root, "lora", "gpt-next-2b")

    assert os.path.exists(
        gpt_2b_lora_model_root
    ), f"{gpt_2b_lora_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return gpt_2b_lora_model_root


@pytest.fixture(scope="session")
def blip2_opt_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    blip2_opt_model_root = os.path.join(models_root, "blip2-opt-2.7b")

    assert os.path.exists(
        blip2_opt_model_root
    ), f"{blip2_opt_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return blip2_opt_model_root


@pytest.fixture(scope="session")
def llava_onevision_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llava_onevision_model_root = os.path.join(models_root,
                                              "llava-onevision-qwen2-7b-ov-hf")

    assert os.path.exists(
        llava_onevision_model_root
    ), f"{llava_onevision_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llava_onevision_model_root


@pytest.fixture(scope="session")
def test_video_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    test_video = os.path.join(models_root, "video-neva", "test_video")

    assert os.path.exists(
        test_video
    ), f"{test_video} does not exist under NFS LLM_MODELS_ROOT dir"
    return test_video


@pytest.fixture(scope="session")
def llava_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llava_model_root = os.path.join(models_root, "llava-1.5-7b-hf")

    assert os.path.exists(
        llava_model_root
    ), f"{llava_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llava_model_root


@pytest.fixture(scope="session")
def mllama_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    mllama_model_root = os.path.join(models_root, "llama-3.2-models",
                                     "Llama-3.2-11B-Vision")

    assert os.path.exists(
        mllama_model_root
    ), f"{mllama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return mllama_model_root


@pytest.fixture(scope="session")
def llama_v3_8b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_model_root = os.path.join(models_root, "llama-models-v3",
                                    "llama-v3-8b-instruct-hf")

    assert os.path.exists(
        llama_model_root
    ), f"{llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_model_root


@pytest.fixture(scope="session")
def llama3_v1_8b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_model_root = os.path.join(models_root, "llama-3.1-model",
                                    "Meta-Llama-3.1-8B")

    assert os.path.exists(
        llama_model_root
    ), f"{llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_model_root


@pytest.fixture(scope="session")
def mixtral_8x7b_v0_1_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    mixtral_8x7b_v0_1_model_root = os.path.join(models_root,
                                                "Mixtral-8x7B-v0.1")

    assert os.path.exists(
        mixtral_8x7b_v0_1_model_root
    ), f"{mixtral_8x7b_v0_1_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return mixtral_8x7b_v0_1_model_root


@pytest.fixture(scope="session")
def llama_v3_70b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    llama_model_root = os.path.join(models_root, "llama-models-v3",
                                    "Llama-3-70B-Instruct-Gradient-1048k")

    assert os.path.exists(
        llama_model_root
    ), f"{llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return llama_model_root


@pytest.fixture(scope="session")
def vicuna_7b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    vicuna_7b_model_root = os.path.join(models_root, "vicuna-7b-v1.3")

    assert os.path.exists(
        vicuna_7b_model_root
    ), f"{vicuna_7b_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return vicuna_7b_model_root


@pytest.fixture(scope="session")
def medusa_vicuna_7b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    medusa_vicuna_7b_model_root = os.path.join(models_root,
                                               "medusa-vicuna-7b-v1.3")

    assert os.path.exists(
        medusa_vicuna_7b_model_root
    ), f"{medusa_vicuna_7b_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return medusa_vicuna_7b_model_root


@pytest.fixture(scope="session")
def eagle_vicuna_7b_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    eagle_vicuna_7b_model_root = os.path.join(models_root,
                                              "EAGLE-Vicuna-7B-v1.3")

    assert os.path.exists(
        eagle_vicuna_7b_model_root
    ), f"{eagle_vicuna_7b_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return eagle_vicuna_7b_model_root


@pytest.fixture(scope="session")
def t5_small_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    t5_small_model_root = os.path.join(models_root, "t5-small")

    assert os.path.exists(
        t5_small_model_root
    ), f"{t5_small_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return t5_small_model_root


@pytest.fixture(scope="session")
def whisper_large_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    whisper_large_model_root = os.path.join(models_root, "whisper-models",
                                            "large-v3")

    assert os.path.exists(
        whisper_large_model_root
    ), f"{whisper_large_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return whisper_large_model_root


@pytest.fixture(scope="session")
def tiny_llama_model_root():
    models_root = llm_models_root()
    assert models_root, "Did you set LLM_MODELS_ROOT?"
    tiny_llama_model_root = os.path.join(models_root, "llama-models-v2",
                                         "TinyLlama-1.1B-Chat-v1.0")

    assert os.path.exists(
        tiny_llama_model_root
    ), f"{tiny_llama_model_root} does not exist under NFS LLM_MODELS_ROOT dir"
    return tiny_llama_model_root


# Returns an array of total memory for each available device
@pytest.fixture(scope="session")
def total_gpu_memory_mib():
    output = check_output("nvidia-smi --query-gpu memory.total --format=csv",
                          shell=True,
                          cwd="/tmp")
    lines = [l.strip() for l in output.strip().split("\n")]
    lines = lines[1:]  # skip header
    lines = [l[:-4] for l in lines]  # remove MiB suffix
    lines = [int(l) for l in lines]
    return lines


# Pytset cache mechanism can be used to store and retrieve data across test runs.
@pytest.fixture(scope="session", autouse=True)
def setup_cache_data(request, tensorrt_llm_example_root):
    # This variable will be used in hook function: pytest_runtest_teardown since
    # fixtures cannot be directly used in hooks.
    request.config.cache.set('example_root', tensorrt_llm_example_root)


def cleanup_engine_outputs(output_dir_root):
    for dirpath, dirnames, _ in os.walk(output_dir_root, topdown=False):
        for dirname in dirnames:
            if "engine_dir" in dirname or "model_dir" in dirname or "ckpt_dir" in dirname:
                folder_path = os.path.join(dirpath, dirname)
                try:
                    shutil.rmtree(folder_path)
                    print_info(f"Deleted folder: {folder_path}")
                except Exception as e:
                    print_info(f"Error deleting {folder_path}: {e}")


# Teardown hook to clean up engine outputs after each group of test cases are finished
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    current_test_basename = item.name.split(
        "[")[0] if '[' in item.name else item.name

    if nextitem:
        next_test_basename = nextitem.name.split(
            "[")[0] if '[' in nextitem.name else nextitem.name
    else:
        next_test_basename = None

    # User can set SKIP_CLEANUP_ENGINES=True to skip clean up engines.
    skip_cleanup_engines = os.getenv("SKIP_CLEANUP_ENGINES", "false")
    if skip_cleanup_engines.lower() != "true":
        if next_test_basename != current_test_basename:
            print_info(
                "SKIP_CLEANUP_ENGINES is not set to True. Cleaning up engine outputs:"
            )
            engine_outputs_root = item.config.cache.get('example_root', None)
            cleanup_engine_outputs(engine_outputs_root)
    else:
        print_info(
            "SKIP_CLEANUP_ENGINES is set to True, will not clean up engines.")

    yield


@pytest.fixture(autouse=True)
def install_root_requirements(llm_backend_root):
    """
    Fixture that automatically runs at the beginning of each test to ensure root requirements.txt is installed.
    """
    requirements_file = os.path.join(llm_backend_root, "requirements.txt")
    if os.path.exists(requirements_file):
        install_requirement_cmd = "pip3 install -r requirements.txt"
        check_call(install_requirement_cmd, shell=True, cwd=llm_backend_root)
    else:
        print_info(
            f"Warning: requirements.txt not found at {requirements_file}")


@pytest.fixture(scope="session")
def output_dir(request):
    if USE_TURTLE:
        return request.config._trt_config["output_dir"]
    else:
        return request.config.getoption("--output-dir")


if USE_TURTLE:  # perf tests can not run outside turtle for now
    # Cache all the pytest items so that we can do test list validation.
    ALL_PYTEST_ITEMS = None  # All pytest items available, before deselection.

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(session, config, items):
        # Flush the current stdout line.
        print()

        import copy

        global ALL_PYTEST_ITEMS
        ALL_PYTEST_ITEMS = copy.copy(items)
        _ = yield

else:
    #
    # When test parameters have an empty id, older versions of pytest ignored that parameter when generating the
    # test node's ID completely. This however was actually a bug, and not expected behavior that got fixed in newer
    # versions of pytest:https://github.com/pytest-dev/pytest/pull/6607. TRT test defs however rely on this behavior
    # for quite a few test names. This is a hacky WAR that restores the old behavior back so that the
    # test names do not change. Note: This might break in a future pytest version.
    #
    # TODO: Remove this hack once the test names are fixed.
    #

    from _pytest.python import CallSpec2
    CallSpec2.id = property(
        lambda self: "-".join(map(str, filter(None, self._idlist))))

    # @pytest.hookimpl(tryfirst=True, hookwrapper=True)
    # def pytest_collection_modifyitems(config, items):
    #     testlist_path = config.getoption("--test-list")
    #     waives_file = config.getoption("--waives-file")
    #     test_prefix = config.getoption("--test-prefix")
    #     if test_prefix:
    #         # Override the internal nodeid of each item to contain the correct test prefix.
    #         # This is needed for reporting to correctly process the test name in order to bucket
    #         # it into the appropriate test suite.
    #         for item in items:
    #             item._nodeid = "{}/{}".format(test_prefix, item._nodeid)

    #     regexp = config.getoption("--regexp")

    #     if testlist_path:
    #         modify_by_test_list(testlist_path, items, config)

    #     if regexp is not None:
    #         deselect_by_regex(regexp, items, test_prefix, config)

    #     if waives_file:
    #         apply_waives(waives_file, items, config)

    #     # We have to remove prefix temporarily before splitting the test list
    #     # After that change back the test id.
    #     for item in items:
    #         if test_prefix and item._nodeid.startswith(f"{test_prefix}/"):
    #             item._nodeid = item._nodeid[len(f"{test_prefix}/"):]
    #     yield
    #     for item in items:
    #         if test_prefix:
    #             item._nodeid = f"{test_prefix}/{item._nodeid}"


    def deselect_by_regex(regexp, items, test_prefix, config):
        """Filter out tests based on the patterns specified in the given list of regular expressions.
           If a test matches *any* of the expressions in the list it is considered selected."""
        compiled_regexes = []
        regex_list = []
        r = re.compile(regexp)
        compiled_regexes.append(r)
        regex_list.append(regexp)

        selected = []
        deselected = []

        corrections = get_test_name_corrections_v2(
            set(regex_list), set(it.nodeid for it in items),
            CorrectionMode.REGEX)
        handle_corrections(corrections, test_prefix)

        for item in items:
            found = False
            for regex in compiled_regexes:
                if regex.search(item.nodeid):
                    found = True
                    break
            if found:
                selected.append(item)
            else:
                deselected.append(item)

        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = selected
