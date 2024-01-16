import argparse
import json
import os
from pathlib import Path

# isort: off
import torch
import tensorrt as trt
# isort: on
from transformers import AutoTokenizer

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import Session, TensorInfo


def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


def TRTOPT(args, config):
    dtype = config['pretrained_config']['dtype']
    world_size = config['pretrained_config']['mapping']['world_size']
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'

    use_gpt_attention_plugin = bool(
        config['build_config']['plugin_config']['gpt_attention_plugin'])

    num_heads = config['pretrained_config']['num_attention_heads'] // world_size
    hidden_size = config['pretrained_config']['hidden_size'] // world_size
    vocab_size = config['pretrained_config']['vocab_size']
    num_layers = config['pretrained_config']['num_hidden_layers']
    remove_input_padding = config['build_config']['plugin_config'][
        'remove_input_padding']
    max_prompt_embedding_table_size = config['build_config'].get(
        'max_prompt_embedding_table_size', 0)

    model_config = tensorrt_llm.runtime.ModelConfig(
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        hidden_size=hidden_size,
        gpt_attention_plugin=use_gpt_attention_plugin,
        remove_input_padding=remove_input_padding,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        dtype=dtype)

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    engine_name = get_engine_name(runtime_rank)
    serialize_path = os.path.join(args.opt_engine_dir, engine_name)

    tensorrt_llm.logger.set_level(args.log_level)

    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping)

    max_input_len = config['build_config']['max_input_len']
    return decoder, model_config, world_size, dtype, max_input_len


def ptuning_setup(prompt_table, dtype, hidden_size, tasks, input_ids,
                  input_lengths, remove_input_padding):
    if prompt_table is not None:
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

    num_sequences = input_lengths.size(
        0) if remove_input_padding else input_ids.size(0)

    if tasks is not None:
        tasks = torch.tensor([int(t) for t in tasks.split(',')],
                             dtype=torch.int32,
                             device="cuda")
        assert tasks.shape[
            0] == num_sequences, "Number of supplied tasks must match input batch size"
    else:
        tasks = torch.zeros([num_sequences], dtype=torch.int32).cuda()

    return [prompt_table, tasks, task_vocab_size]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, default=30)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir', type=str, default='./plan')
    parser.add_argument('--input_dir', type=str, default='image.pt')
    parser.add_argument('--query_tokens', type=str, default='query_tokens.pt')
    parser.add_argument('--opt_engine_dir',
                        type=str,
                        default='trt_engine/blip-2-opt-2.7b/fp16/1-gpu/')
    parser.add_argument('--hf_model_location',
                        type=str,
                        default="facebook/opt-2.7b")
    parser.add_argument('--input_text',
                        type=str,
                        default='Question: which city is this? Answer:')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--max_txt_len',
                        type=int,
                        help="Max text prompt length",
                        default=32)
    parser.add_argument('--top_k', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    tensorrt_llm.logger.set_level(args.log_level)

    stream = torch.cuda.current_stream().cuda_stream

    ### 0. ViT & Qformer session load ###

    vit_path = os.path.join(args.engine_dir,
                            'visual_encoder/visual_encoder_fp16.plan')
    logger.info(f'Loading engine from {vit_path}')
    with open(vit_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine {vit_path}')
    session_vit = Session.from_serialized_engine(engine_buffer)

    qformer_path = os.path.join(args.engine_dir, 'Qformer/Qformer_fp16.plan')
    logger.info(f'Loading engine from {qformer_path}')
    with open(qformer_path, 'rb') as f:
        engine_buffer_qformer = f.read()
    logger.info(f'Creating session from engine {qformer_path}')
    session_qformer = Session.from_serialized_engine(engine_buffer_qformer)

    ### 1. ViT inference session ###

    image = torch.load(args.input_dir)
    batch_size = 1
    image = image.expand(batch_size, -1, -1, -1).contiguous()
    # assert image.iscontigous()
    visual_inputs = {'input': image.half()}
    visual_output_info = session_vit.infer_shapes(
        [TensorInfo('input', trt.DataType.HALF, image.shape)])
    visual_outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
        for t in visual_output_info
    }

    ok = session_vit.run(visual_inputs, visual_outputs, stream)

    assert ok, "Runtime execution failed for vit session"

    image_embeds = visual_outputs['output']

    image_atts = torch.ones(image_embeds.size()[:-1],
                            dtype=torch.long).to(image.device)

    ### 2. Qformer inference session ###

    query_tokens = torch.load(args.query_tokens)
    query_tokens = query_tokens.expand(image_embeds.shape[0], -1,
                                       -1).contiguous()
    # assert query_tokens.is_contiguous()
    qformer_inputs = {
        'query_tokens': query_tokens.half(),
        'image_embeds': image_embeds.half(),
        'image_atts': image_atts
    }
    qformer_output_info = session_qformer.infer_shapes([
        TensorInfo('query_tokens', trt.DataType.HALF, query_tokens.shape),
        TensorInfo('image_embeds', trt.DataType.HALF, image_embeds.shape),
        TensorInfo('image_atts', trt.DataType.INT64, image_atts.shape)
    ])
    qformer_outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
        for t in qformer_output_info
    }
    ok = session_qformer.run(qformer_inputs, qformer_outputs, stream)
    assert ok, "Runtime execution failed for Qformer session"

    inputs_opt = qformer_outputs["query_output"]
    atts_opt = torch.ones(inputs_opt.size()[:-1],
                          dtype=torch.long).to(image.device)

    ### 3. OPT inference session

    prompt = args.input_text

    prompt = [prompt] * image.size(0)

    opt_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_location,
                                                  use_fast=False)
    opt_tokenizer.padding_side = "right"

    end_id = opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
    # end_id = opt_tokenizer.encode(opt_tokenizer.eos_token, add_special_tokens=False)[0]

    engine_dir = Path(args.opt_engine_dir)
    config_path = engine_dir / 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    tensorrt_llm_opt, model_config, world_size, dtype, max_input_len = TRTOPT(
        args, config)
    vocab_size = model_config.vocab_size

    def opt_blip2(prompt, inputs_opt, atts_opt):
        profiler.start("OPT")
        opt_tokens = opt_tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_txt_len,
        ).to(image.device)

        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
        input_lengths = torch.sum(attention_mask, dim=1).to(torch.int32).cuda()

        sampling_config = tensorrt_llm.runtime.SamplingConfig(
            end_id=end_id,
            pad_id=end_id,
            top_k=args.top_k,
            num_beams=args.num_beams)

        # Assemble fake prompts which points to image embedding actually

        fake_prompt_id = torch.arange(vocab_size,
                                      vocab_size +
                                      inputs_opt.shape[0] * inputs_opt.shape[1],
                                      device='cuda')
        fake_prompt_id = fake_prompt_id.reshape(inputs_opt.shape[0],
                                                inputs_opt.shape[1])
        input_ids = torch.cat([fake_prompt_id, opt_tokens.input_ids],
                              dim=1).contiguous()
        input_ids = input_ids.to(torch.int32).cuda()

        ptuning_args = ptuning_setup(inputs_opt, dtype,
                                     model_config.hidden_size, None, input_ids,
                                     input_lengths,
                                     model_config.remove_input_padding)

        with torch.no_grad():
            max_input_length = torch.max(input_lengths).item()
            tensorrt_llm_opt.setup(batch_size,
                                   max_context_length=max_input_length,
                                   max_new_tokens=args.max_output_len)

            if tensorrt_llm_opt.remove_input_padding:
                output_ids = tensorrt_llm_opt.decode_batch(
                    input_ids, sampling_config, *ptuning_args)
            else:
                output_ids = tensorrt_llm_opt.decode(input_ids, input_lengths,
                                                     sampling_config,
                                                     *ptuning_args)

            torch.cuda.synchronize()

        profiler.stop("OPT")

        # Extract a list of tensors of shape beam_width x output_ids.
        output_beams_list = [
            opt_tokenizer.batch_decode(output_ids[batch_idx, :,
                                                  input_lengths[batch_idx]:],
                                       skip_special_tokens=True)
            for batch_idx in range(batch_size)
        ]
        stripped_text = [[
            output_beams_list[batch_idx][beam_idx].strip()
            for beam_idx in range(args.num_beams)
        ] for batch_idx in range(batch_size)]
        return stripped_text

    for _ in range(100):
        stripped_text = opt_blip2(prompt, inputs_opt, atts_opt)
    logger.info("---------------------------------------------------------")
    logger.info("TensorRT-LLM BLIP-2 : ")
    logger.info(f"\n[Q] {args.input_text}")
    logger.info(f"\n[A] {stripped_text}")
    logger.info(
        f'TensorRT-LLM OPT latency: {profiler.elapsed_time_in_sec("OPT") / 100} sec'
    )
    logger.info("---------------------------------------------------------")

    for i in range(100):
        profiler.start("visual encoder")
        ok = session_vit.run(visual_inputs, visual_outputs, stream)
        profiler.stop("visual encoder")

        profiler.start("Qformer")
        ok = session_qformer.run(qformer_inputs, qformer_outputs, stream)
        profiler.stop("Qformer")

    logger.info(
        f'TensorRT-LLM ViT latency: {profiler.elapsed_time_in_sec("visual encoder") / 100} sec'
    )
    logger.info(
        f'TensorRT-LLM Qformer latency: {profiler.elapsed_time_in_sec("Qformer") / 100} sec'
    )
