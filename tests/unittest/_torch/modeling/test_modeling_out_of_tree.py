import re
from contextlib import nullcontext
from pathlib import Path
from typing import cast

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

# isort: off
from utils.util import similar
from utils.llm_data import llm_models_root
# isort: on

from llmapi.apps.openai_server import RemoteOpenAIServer


class TestOutOfTree:

    @pytest.fixture
    @staticmethod
    def oot_path() -> Path:
        return Path(
            __file__
        ).parent / ".." / ".." / ".." / ".." / "examples" / "llm-api" / "out_of_tree_example"

    @pytest.fixture
    @staticmethod
    def model_dir() -> Path:
        models_root = llm_models_root()
        assert models_root is not None
        return models_root / "opt-125m"

    @pytest.fixture
    @staticmethod
    def prompts() -> list[str]:
        return [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

    @pytest.fixture
    @staticmethod
    def references() -> list[str]:
        return [
            " J.C. and I am a student at",
            " not a racist. He is a racist.\n",
            " the capital of the French Republic.\n\nThe",
            " in the hands of the people.\n\nThe",
        ]

    @pytest.fixture
    @staticmethod
    def sampling_params() -> SamplingParams:
        return SamplingParams(max_tokens=10)

    @pytest.fixture
    @staticmethod
    def max_num_tokens() -> int:
        # estimate_max_kv_cache_tokens will create a request of max_num_tokens for forward.
        # Default 8192 will exceed the max length of absolute positional embedding in OPT, leading to out of range indexing.
        return 2048

    @pytest.mark.parametrize("import_oot_code", [False, True])
    def test_llm_api(
        self,
        import_oot_code: bool,
        oot_path: Path,
        model_dir: Path,
        prompts: list[str],
        references: list[str],
        sampling_params: SamplingParams,
        max_num_tokens: int,
        monkeypatch: pytest.MonkeyPatch,
    ):
        if import_oot_code:
            # Import out-of-tree modeling code for OPTForCausalLM
            monkeypatch.syspath_prepend(oot_path)
            import modeling_opt  # noqa

        with (nullcontext() if import_oot_code else
              pytest.raises(RuntimeError,
                            match=".*Executor worker returned error.*")) as ctx:
            with LLM(
                    model=str(model_dir),
                    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.2),
                    max_num_tokens=max_num_tokens,
            ) as llm:
                outputs = llm.generate(prompts, sampling_params=sampling_params)

            for output, ref in zip(outputs, references):
                assert similar(output.outputs[0].text, ref)

        if not import_oot_code:
            exc_val = cast(pytest.ExceptionInfo, ctx).value
            assert re.match(
                ".*Unknown architecture for AutoModelForCausalLM: OPTForCausalLM.*",
                str(exc_val.__cause__),
            ) is not None

    @pytest.mark.parametrize("import_oot_code", [False, True])
    def test_serve(
        self,
        import_oot_code: bool,
        oot_path: Path,
        model_dir: Path,
        prompts: list[str],
        references: list[str],
        sampling_params: SamplingParams,
        max_num_tokens: int,
    ):
        with (nullcontext()
              if import_oot_code else pytest.raises(RuntimeError)):
            args = []
            args.extend(["--kv_cache_free_gpu_memory_fraction",
                         "0.2"])  # for co-existence with other servers
            args.extend(["--max_num_tokens", str(max_num_tokens)])
            if import_oot_code:
                args.extend(["--custom_module_dirs", str(oot_path)])
            with RemoteOpenAIServer(str(model_dir), args) as remote_server:
                client = remote_server.get_client()
                result = client.completions.create(
                    model="model_name",
                    prompt=prompts,
                    max_tokens=sampling_params.max_tokens,
                    temperature=0.0,
                )

            for choice, ref in zip(result.choices, references):
                assert similar(choice.text, ref)
