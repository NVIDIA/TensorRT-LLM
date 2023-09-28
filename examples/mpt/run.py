import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, T5Tokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default='gpt_outputs')
    parser.add_argument('--input_text',
                        type=str,
                        default='Born in north-east France, Soyer trained as a')
    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--tokenizer',
                        dest='tokenizer_path',
                        help="HF tokenizer config path",
                        default='EleutherAI/gpt-neox-20b')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument(
        '--prompt_table',
        type=Path,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--tasks',
        help="Comma-separated list of tasks for prompt tuning: ex 0,3,1,0")
    return parser.parse_args()


def generate(
    max_output_len: int,
    log_level: str = 'error',
    engine_dir: str = 'gpt_outputs',
    input_text: str = 'Born in north-east France, Soyer trained as a',
    input_file: str = None,
    output_csv: str = None,
    output_npy: str = None,
    tokenizer_path: str = 'gpt2',
    vocab_file=None,
    num_beams: int = 1,
    prompt_table: Path = None,
    tasks: str = None,
):
    tensorrt_llm.logger.set_level(log_level)

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    world_size = config['builder_config']['tensor_parallel']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // world_size
    hidden_size = config['builder_config']['hidden_size'] // world_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    multi_query_mode = config['builder_config']['multi_query_mode']
    paged_kv_cache = config['builder_config']['paged_kv_cache']
    tokens_per_block = config['builder_config']['tokens_per_block']
    use_prompt_tuning = config['builder_config']['use_prompt_tuning']

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    if vocab_file is not None:
        tokenizer = T5Tokenizer(vocab_file=vocab_file)
        END_ID = 50256
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        END_ID = tokenizer.eos_token_id

    model_config = ModelConfig(
        num_heads=num_heads,
        num_kv_heads=(1 if multi_query_mode else num_heads),
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        paged_kv_cache=paged_kv_cache,
        tokens_per_block=tokens_per_block,
        use_prompt_tuning=use_prompt_tuning)
    sampling_config = SamplingConfig(end_id=END_ID,
                                     pad_id=END_ID,
                                     num_beams=num_beams,
                                     temperature=1.0,
                                     top_k=1,
                                     top_p=1.0)

    engine_name = get_engine_name('mpt', dtype, world_size, runtime_rank)
    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    input_tokens = []
    if input_file is None:
        input_tokens.append(
            tokenizer.encode(input_text, add_special_tokens=False))
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_tokens.append(np.array(line, dtype='int32'))
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                row = row[row != END_ID]
                input_tokens.append(row)
        else:
            print('Input file format not supported.')
            raise SystemExit

    input_ids = None
    input_lengths = None
    if input_file is None:
        input_ids = torch.cuda.IntTensor(input_tokens)
        input_lengths = torch.cuda.IntTensor([input_ids.size(1)])
    else:
        input_lengths = torch.cuda.IntTensor([len(x) for x in input_tokens])
        if remove_input_padding:
            input_ids = np.concatenate(input_tokens)
            input_ids = torch.cuda.IntTensor(input_ids).unsqueeze(0)
        else:
            input_ids = torch.nested.to_padded_tensor(
                torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
                END_ID).cuda()

    max_input_length = torch.max(input_lengths).item()
    decoder.setup(input_lengths.size(0),
                  max_input_length,
                  max_output_len,
                  beam_width=num_beams)

    ptuning_args = []
    if use_prompt_tuning:
        if prompt_table is not None:
            prompt_table = torch.from_numpy(np.load(prompt_table))
            task_vocab_size = torch.tensor([prompt_table.shape[1]],
                                           dtype=torch.int32,
                                           device="cuda")
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1],
                 prompt_table.shape[2]))
            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(dtype))
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if tasks is not None:
            tasks = torch.tensor([int(t) for t in tasks.split(',')],
                                 dtype=torch.int32,
                                 device="cuda")
            assert tasks.shape[0] == input_ids.shape[
                0], "Number of supplied tasks must match input batch size"
        else:
            tasks = torch.zeros([input_ids.size(0)]).cuda()

        ptuning_args = [prompt_table, tasks, task_vocab_size]

    output_ids = decoder.decode(input_ids, input_lengths, sampling_config,
                                *ptuning_args)
    torch.cuda.synchronize()

    if runtime_rank == 0:
        if output_csv is None and output_npy is None:
            for b in range(input_lengths.size(0)):
                inputs = input_tokens[b]
                input_text = tokenizer.decode(inputs)
                print(f'Input: {input_text}')
                if num_beams <= 1:
                    output_begin = max_input_length
                    outputs = output_ids[b][0][output_begin:].tolist()
                    output_text = tokenizer.decode(output_ids[b][0])  #outputs)
                    print(f'Output: {output_text}')
                else:
                    for beam in range(num_beams):
                        output_begin = input_lengths[b]
                        output_end = input_lengths[b] + max_output_len
                        outputs = output_ids[b][beam][
                            output_begin:output_end].tolist()
                        output_text = tokenizer.decode(outputs)
                        print(f'Output: {output_text}')

        output_ids = output_ids.reshape((-1, output_ids.size(2)))

        if output_csv is not None:
            output_file = Path(output_csv)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            outputs = output_ids.tolist()
            with open(output_file, 'w') as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerows(outputs)

        if output_npy is not None:
            output_file = Path(output_npy)
            output_file.parent.mkdir(exist_ok=True, parents=True)
            outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
            np.save(output_file, outputs)


if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
