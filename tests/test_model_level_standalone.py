import tempfile
from pathlib import Path
from typing import Iterable

import torch
from llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm import profiler
from tensorrt_llm.models import LLaMAForCausalLM


# keep pulling data from the input_text and yield output
def GenerationEngine(input_text: Iterable[str], max_new_tokens, engine_dir,
                     tokenizer_dir):
    # prepare inputs
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                              legacy=False,
                                              padding_side='left',
                                              truncation_side='left',
                                              trust_remote_code=True,
                                              use_fast=True)

    from tensorrt_llm.runtime import ModelRunner, SamplingConfig, model_runner
    batch_input_ids = []
    # we'd like the avoid the padding and needs to know each seq's length, so tokenize them one by one

    _, config = model_runner.read_config(Path(engine_dir) / "config.json")

    for inp in input_text:
        input_ids = tokenizer.encode(inp,
                                     truncation=True,
                                     max_length=config['max_input_len'])
        batch_input_ids.append(input_ids)
    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]  # List[torch.Tensor(seq)]

    # prepare trt llm runtime
    sampling_config = SamplingConfig(
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.eos_token_id
        if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens)
    sampling_config.output_sequence_lengths = True
    sampling_config.return_dict = True

    runner = ModelRunner.from_dir(engine_dir=engine_dir)

    ###TODO: better batching
    # generate
    outputs = runner.generate(batch_input_ids, sampling_config)

    # parse and print output
    output_ids = outputs['output_ids']
    sequence_lengths = outputs['sequence_lengths']

    batch_size, num_beams, max_len = output_ids.size()
    input_lengths = [x.size(0) for x in batch_input_ids]

    for batch_idx in range(batch_size):
        for beam in range(num_beams):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text_echo = tokenizer.decode(
                inputs)  # output_ids shall contain the input ids

            output_begin = input_lengths[batch_idx]
            output_end = sequence_lengths[batch_idx][beam]
            outputs = output_ids[batch_idx][beam][
                output_begin:output_end].tolist()

            output_text = tokenizer.decode(outputs)
            assert input_text_echo == "<s> " + input_text[
                batch_idx], f"Got {input_text_echo}, expect: {input_text[batch_idx]}"
            yield input_text[batch_idx], output_text


input_text = [
    'Born in north-east France, Soyer trained as a',
    "What is large language model?"
]
expected_output = [
    "chef in Paris and London before moving to New York",
    "\nLarge language model is a model that is"
]
tensorrt_llm.logger.set_level('verbose')


def build(hf_model_dir, engine_dir):
    max_batch_size, max_isl, max_osl = 8, 256, 256
    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir, 'float16')
    llama.to_trt(max_batch_size, max_isl, max_osl)
    llama.save(engine_dir=engine_dir)


def test_stand_alone_run():
    hf_model_dir = llm_models_root() / "llama-models/llama-7b-hf"
    tokenizer_dir = hf_model_dir
    engine_dir = tempfile.TemporaryDirectory("llama")

    profiler.start("build-and-save")  # 140s on ipp1-1197
    # build and save: only once before deployment
    build(hf_model_dir, engine_dir.name)
    profiler.stop("build-and-save")

    profiler.start(
        "just-load"
    )  # about  103s on ipp1-1197, loading engine and start session itself 101s
    # load and run: can restart many time w/o rebuilding
    for idx, (inp, output) in enumerate(
            GenerationEngine(input_text, 10, engine_dir.name, tokenizer_dir)):
        print(f"Input: {inp}")
        print(f'Output: {output}')
        assert output == expected_output[
            idx], f"Expecting {expected_output[idx]}, got {output}"
    profiler.stop("just-load")
    profiler.summary()


if __name__ == "__main__":
    test_stand_alone_run()
