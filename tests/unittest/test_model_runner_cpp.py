import typing as tp
from pathlib import Path

import torch
from bindings.binding_test_utils import *
from transformers import AutoTokenizer
from utils.cpp_paths import *
from utils.llm_data import llm_models_root

from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp


@pytest.fixture
def model_files(llm_root: Path, resource_path: Path, results_data_path: Path):
    # Model engines and expected outputs need to be generated.
    print(results_data_path)
    if not results_data_path.exists():
        model_cache = llm_models_root()
        model_cache_arg = ["--model_cache", str(model_cache)
                           ] if model_cache is not None else []
        prepare_model_tests(llm_root, resource_path, "gpt", model_cache_arg)


def test_logits_post_processor(model_files, model_path):

    # Define the logits post-processor callback
    def logits_post_processor(req_id: int, logits: torch.Tensor,
                              ids: tp.List[tp.List[int]], stream_ptr: int,
                              client_id: tp.Optional[int]):
        with torch.cuda.stream(torch.cuda.ExternalStream(stream_ptr)):
            logits[:] = float("-inf")
            logits[..., 42] = 0

    # Create ModelRunnerCpp
    logits_processor_map = {"my_logits_pp": logits_post_processor}
    runner = ModelRunnerCpp.from_dir(model_path,
                                     logits_processor_map=logits_processor_map)

    model_root = llm_models_root(check=True)
    hf_model_dir = Path(model_root, "gpt2")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                              padding_side="left",
                                              truncation_side="left",
                                              trust_remote_code=True,
                                              use_fast=True)

    input_text = "Born in north-east France, Soyer trained as a"
    batch_input_ids = [
        torch.tensor(tokenizer.encode(input_text,
                                      add_special_tokens=True,
                                      truncation=True),
                     dtype=torch.int32)
    ]

    pad_token_id = tokenizer.pad_token_id
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    # Create the request
    max_new_tokens = 5
    with torch.no_grad():
        outputs = runner.generate(batch_input_ids=batch_input_ids,
                                  max_new_tokens=max_new_tokens,
                                  end_id=tokenizer.eos_token_id,
                                  pad_id=pad_token_id,
                                  output_sequence_lengths=True,
                                  return_dict=True,
                                  logits_processor_names={"my_logits_pp"})

    torch.cuda.synchronize()

    # Get the new tokens
    tokens = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']

    output_begin = len(batch_input_ids[0])
    output_end = sequence_lengths[0][0]

    # check that all output tokens are 42
    assert tokens[0][0][output_begin:output_end].tolist() == [42
                                                              ] * max_new_tokens
