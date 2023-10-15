import argparse
import time
from pathlib import Path

import torch

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.network import net_guard

from weight import load_t5_from_pytorch, parse_config  # isort:skip

MODEL_NAME = "enc_dec"


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def serialize_engine(engine, path):
    logger.info(f'Serializing engine to {path}...')
    tik = time.time()
    with open(path, 'wb') as f:
        f.write(bytearray(engine))
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Engine serialized. Total time: {t}')


def parse_arguments(args, component):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float16', 'float32', 'bfloat16'])
    parser.add_argument('--logits_dtype',
                        type=str,
                        default='float32',
                        choices=['float16', 'float32'])
    parser.add_argument(
        '--timing_cache',
        type=str,
        default='model.cache',
        help=
        'The path of to read timing cache from, will be ignored if the file does not exist'
    )
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--vocab_size', type=int, default=32128)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_positions', type=int, default=1024)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--hidden_act', type=str, default='gelu')
    parser.add_argument('--inter_size', type=int, default=None)
    parser.add_argument('--no_bias', action="store_false")
    parser.add_argument('--max_batch_size', type=int, default=8)
    parser.add_argument('--max_encoder_input_len', type=int, default=1024)
    parser.add_argument('--max_input_len', type=int, default=200)
    parser.add_argument('--max_output_len', type=int, default=200)
    parser.add_argument('--max_beam_width', type=int, default=1)
    parser.add_argument(
        '--use_bert_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates BERT attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gpt_attention_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates attention plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_gemm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates GEMM plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument(
        '--use_layernorm_plugin',
        nargs='?',
        const=None,
        type=str,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help=
        "Activates layernorm plugin. You can specify the plugin dtype or leave blank to use the model dtype."
    )
    parser.add_argument('--enable_qk_half_accum',
                        default=False,
                        action='store_true')
    parser.add_argument('--gpus_per_node', type=int, default=8)
    parser.add_argument('--builder_opt', type=int, default=None)
    parser.add_argument(
        '--output_dir',
        type=Path,
        default='trt_engines',
        help=
        'The path to save the serialized engine files, timing cache file and model configs'
    )
    parser.add_argument('--remove_input_padding',
                        default=False,
                        action='store_true')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help=
        'Seed to use when initializing the random number generator for torch.')
    parser.add_argument(
        '--use_lookup_plugin',
        nargs='?',
        const=None,
        default=False,
        choices=['float16', 'float32', 'bfloat16'],
        help="Activates the lookup plugin which enables embedding sharing.")

    args = parser.parse_args(args)
    logger.set_level(args.log_level)

    args.bias = not args.no_bias
    if args.inter_size is None:
        args.inter_size = 4 * args.n_embd

    if args.model_dir is not None:
        logger.info(f"Setting model configuration from {args.model_dir}.")
        args = parse_config(
            Path(args.model_dir) / "config.ini", component, args)
    plugins_args = [
        'use_bert_attention_plugin', 'use_gpt_attention_plugin',
        'use_gemm_plugin', 'use_layernorm_plugin', 'use_lookup_plugin'
    ]
    for plugin_arg in plugins_args:
        if getattr(args, plugin_arg) is None:
            logger.info(
                f"{plugin_arg} set, without specifying a value. Using {args.dtype} automatically."
            )
            setattr(args, plugin_arg, args.dtype)

    return args


def build_rank_engine(builder: Builder,
                      builder_config: tensorrt_llm.builder.BuilderConfig,
                      engine_name, rank, args):
    '''
       @brief: Build the engine on the given rank.
       @param rank: The rank to build the engine.
       @param args: The cmd line arguments.
       @return: The built engine.
    '''
    kv_dtype = str_dtype_to_trt(args.dtype)

    # Initialize Module
    if args.component == 'encoder':
        tllm_model = tensorrt_llm.models.EncoderModel(
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.n_positions,
            has_position_embedding=args.has_position_embedding,
            relative_attention=args.relative_attention,
            max_distance=args.max_distance,
            num_buckets=args.num_buckets,
            has_embedding_layernorm=args.has_embedding_layernorm,
            has_embedding_scale=args.has_embedding_scale,
            q_scaling=args.q_scaling,
            has_attention_qkvo_bias=args.has_attention_qkvo_bias,
            has_mlp_bias=args.has_mlp_bias,
            has_model_final_layernorm=args.has_model_final_layernorm,
            layernorm_eps=args.layernorm_eps,
            layernorm_position=args.layernorm_position,
            layernorm_type=args.layernorm_type,
            hidden_act=args.hidden_act,
            dtype=kv_dtype)
    elif args.component == 'decoder':
        tllm_model = tensorrt_llm.models.DecoderModel(
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.hidden_size,
            ffn_hidden_size=args.ffn_hidden_size,
            encoder_hidden_size=args.encoder_hidden_size,
            encoder_num_heads=args.encoder_num_heads,
            vocab_size=args.vocab_size,
            max_position_embeddings=args.n_positions,
            has_position_embedding=args.has_position_embedding,
            relative_attention=args.relative_attention,
            max_distance=args.max_distance,
            num_buckets=args.num_buckets,
            has_embedding_layernorm=args.has_embedding_layernorm,
            has_embedding_scale=args.has_embedding_scale,
            q_scaling=args.q_scaling,
            has_attention_qkvo_bias=args.has_attention_qkvo_bias,
            has_mlp_bias=args.has_mlp_bias,
            has_model_final_layernorm=args.has_model_final_layernorm,
            layernorm_eps=args.layernorm_eps,
            layernorm_position=args.layernorm_position,
            layernorm_type=args.layernorm_type,
            hidden_act=args.hidden_act,
            dtype=kv_dtype,
            logits_dtype=args.logits_dtype)

    # No support for relative attention bias in plain TRT mode
    # (If to add such support, need to add into
    #   Attention and BertAttention at tensorrt_llm/layers/attention.py)
    if args.relative_attention:
        assert args.use_bert_attention_plugin, "Relative attention bias is only supported when using BertAttention Plugin"
        assert args.use_gpt_attention_plugin, "Relative attention bias is only supported when using GPTAttention Plugin"

    if args.model_dir is not None:
        load_t5_from_pytorch(tllm_model,
                             args.model_dir,
                             args.component,
                             dtype=args.dtype)

    # Module -> Network
    network = builder.create_network()
    network.trt_network.name = engine_name
    if args.use_bert_attention_plugin:
        network.plugin_config.set_bert_attention_plugin(
            dtype=args.use_bert_attention_plugin)
    if args.use_gpt_attention_plugin:
        network.plugin_config.set_gpt_attention_plugin(
            dtype=args.use_gpt_attention_plugin)
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.use_layernorm_plugin:
        network.plugin_config.set_layernorm_plugin(
            dtype=args.use_layernorm_plugin)
    if args.enable_qk_half_accum:
        network.plugin_config.enable_qk_half_accum()
    if args.remove_input_padding:
        network.plugin_config.enable_remove_input_padding()

    if args.use_lookup_plugin:
        # Use the plugin for the embedding parallelism and sharing
        network.plugin_config.set_lookup_plugin(dtype=args.dtype)

    with net_guard(network):
        # Prepare
        network.set_named_parameters(tllm_model.named_parameters())

        # Forward
        if args.component == 'encoder':
            inputs = tllm_model.prepare_inputs(
                args.max_batch_size,
                args.max_input_len,
            )
        elif args.component == 'decoder':
            inputs = tllm_model.prepare_inputs(
                args.n_layer,
                args.max_batch_size,
                args.max_beam_width,
                args.max_input_len,
                args.max_output_len,
                args.max_encoder_input_len,
            )

        tllm_model(*inputs)

        # Adding debug outputs into the network --------------------------
        for k, v in tllm_model.named_network_outputs():
            network._mark_output(v, k,
                                 tensorrt_llm.str_dtype_to_trt(args.dtype))
        # ----------------------------------------------------------------

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    if rank == 0:
        config_path = args.output_dir / args.component / 'config.json'
        builder.save_config(builder_config, config_path)
    return engine


def build(rank, args):
    torch.cuda.set_device(rank % args.gpus_per_node)
    tensorrt_llm.logger.set_level(args.log_level)
    component_dir = args.output_dir / args.component
    component_dir.mkdir(parents=True, exist_ok=True)
    timing_cache_file = args.timing_cache if args.timing_cache else component_dir / "model.cache"
    timing_cache = timing_cache_file

    builder = Builder()
    apply_query_key_layer_scaling = False

    # Currently only support single GPU
    world_size = 1
    for cur_rank in range(world_size):
        builder_config = builder.create_builder_config(
            name=MODEL_NAME,
            precision=args.dtype,
            timing_cache=timing_cache,
            tensor_parallel=world_size,  # TP only
            num_layers=args.n_layer,
            num_heads=args.n_head,
            hidden_size=args.hidden_size,
            vocab_size=args.vocab_size,
            hidden_act=args.hidden_act,
            max_position_embeddings=args.n_positions,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            opt_level=args.builder_opt,
            cross_attention=(args.component == 'decoder'),
            has_position_embedding=args.has_position_embedding,
            has_token_type_embedding=args.has_token_type_embedding,
        )

        engine_name = get_engine_name(MODEL_NAME, args.dtype, world_size,
                                      cur_rank)
        engine = build_rank_engine(builder, builder_config, engine_name,
                                   cur_rank, args)
        assert engine is not None, f'Failed to build engine for rank {cur_rank}'

        if cur_rank == 0:
            # Use in-memory timing cache for multiple builder passes.
            timing_cache = builder_config.trt_builder_config.get_timing_cache()

        serialize_engine(engine, component_dir / engine_name)

    if rank == 0:
        ok = builder.save_timing_cache(builder_config, timing_cache_file)
        assert ok, "Failed to save timing cache."


def run_build(component, args=None):
    assert component == 'encoder' or component == 'decoder', 'Unsupported component!'
    args = parse_arguments(args, component)
    args.component = component

    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)

    logger.set_level(args.log_level)
    tik = time.time()

    # Currently only support single GPU serial build
    logger.info('Serially build TensorRT engines.')
    build(0, args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Total time of building all engines: {t}')


if __name__ == '__main__':
    run_build(component='encoder')
    run_build(component='decoder')
