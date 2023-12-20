import argparse
import configparser
import logging
import multiprocessing
import os
import re
from datetime import datetime
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

import numpy as np
import torch  # pytype: disable=import-error
from fairseq.models.transformer import TransformerModel

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

LOGGER = logging.getLogger(__name__)

extra_configs = {
    "structure": {
        "t5_with_bias": "true",
        "use_gated_activation": "false",
        "position_embedding_type": "sinusoid",
        'model_type': 'nmt'
    }
}  # TODO: remove model type as it's included in HF config's `architectures` attribute

# TODO: change name `t5_with_bias` for non-t5 model


def fuse_qkv(model, factor, saved_dir):

    def get_attn_module(component, layer, attn_type):
        m = model.models[0]
        m = getattr(m, component)
        m = m.layers[int(layer)]
        m = getattr(m, attn_type)
        return m

    for name, param in model.named_parameters():
        if 'attn.q_proj.weight' in name:
            # fuse weights of q, k, v (both self-attn and cross-attn)
            q_w = param
            _, _, component, _, layer_idx, attn_type, *_ = name.split('.')
            attn_mdl = get_attn_module(component, layer_idx, attn_type)

            # fuse qkv weight
            shape = q_w.shape  # (do, din)
            qkv_w = torch.cat(
                [q_w, attn_mdl.k_proj.weight, attn_mdl.v_proj.weight],
                dim=0).reshape([3, shape[0], shape[1]])  # (3, do, din)
            qkv_w = torch_to_numpy(qkv_w)
            split_vals = np.split(qkv_w, factor, axis=1)
            for j in range(factor):
                saved_path = saved_dir / f"{component}.layers.{layer_idx}.{attn_type}.qkv_proj.weight.{j}.bin"
                split_vals[j].tofile(saved_path.as_posix())

            # fuse qkv biases if present
            if hasattr(attn_mdl.q_proj, 'bias'):
                q_b = attn_mdl.q_proj.bias
                shape = q_b.shape[0]  # (do,)
                qkv_b = torch.cat(
                    [q_b, attn_mdl.k_proj.bias, attn_mdl.v_proj.bias],
                    dim=0).reshape([3, shape])  # (3, do)
                qkv_b = torch_to_numpy(qkv_b)
                split_vals = np.split(qkv_b, factor, axis=1)  # (3, do / n_gpus)
                for j in range(factor):
                    saved_path = saved_dir / f"{component}.layers.{layer_idx}.{attn_type}.qkv_proj.bias.{j}.bin"
                    split_vals[j].tofile(saved_path.as_posix())


# TODO: use re.compile to accelerate
def split_and_convert_process(key, val, factor, saved_dir):
    saved_key = key.replace("models.0.", "")
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    def save_splits(split_vals):
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    if re.search('embed_tokens|embed_positions|norm|(out_proj|fc2)\.bias',
                 key) is not None:
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())
    elif re.search('(output_projection|fc1)\.(weight|bias)', key) is not None:
        split_vals = np.split(val, factor, axis=0)
        save_splits(split_vals)
    elif re.search('[kqv]_proj\.(weight|bias)',
                   key) is not None:  # No need to store, fuse later!
        pass
    elif re.search(
            '(out_proj|fc2)\.weight',
            key) is not None:  # match attention o and ffn wo, split in dim 0
        split_vals = np.split(
            val, factor, axis=-1
        )  # no need to split bias, each GPU will add it individually after all reduce
        save_splits(split_vals)  # TODO: support gated activation?
    else:
        LOGGER.warning(
            f"cannot find key '{key}' with shape {val.shape}, skip weight")


def convert_checkpoint(args):
    saved_dir = Path(args.output_dir) / f"tp{args.inference_tensor_para_size}"
    saved_dir.mkdir(parents=True, exist_ok=True)

    model = TransformerModel.from_pretrained(args.input_dir)
    model = model.to(str_dtype_to_torch(args.weight_data_type))

    config = configparser.ConfigParser()

    fairseq_config = vars(model.cfg.model)  # Namespace --> dict

    config['encoder'] = dict()
    for key, val in fairseq_config.items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["encoder"]["q_scaling"] = '1'
    # NMT doesn't have final layernorm
    config['encoder']['has_model_final_layernorm'] = 'false'
    config['encoder']['vocab_size'] = str(len(model.src_dict))  # fairseq naming

    config['decoder'] = dict()
    for key, val in fairseq_config.items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"]["q_scaling"] = '1'
    config["decoder"]["rescale_before_lm_head"] = 'false'
    config['decoder']['has_model_final_layernorm'] = 'false'
    config['decoder']['vocab_size'] = str(len(model.src_dict))  # fairseq naming

    # add additional config
    for key, val in extra_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val

    with open((saved_dir / f"config.ini").as_posix(), 'w') as configfile:
        config.write(configfile)

    i_gpu_num = args.inference_tensor_para_size

    pool = multiprocessing.Pool(args.processes)
    pool.starmap_async(split_and_convert_process,
                       [(name, torch_to_numpy(param), i_gpu_num, saved_dir)
                        for name, param in model.state_dict().items()])

    # NMT uses sinusoidal positional embedding which is deterministic rather than learned embedding (see sinusoidal_positional_embedding.py), so Fairseq generates it on-the-fly instead of checkpointing it as weights.
    # for TRT-LLM, need to explicit form it and thus it has to obey a max_positions limit. Modify this later in long seqlen cases.
    # Note: when we instantiate, the +2 offset (padding_idx + 1) should be preserved and then truncated, because prepending the offset does affect the arange in sinusoidal embedding
    num_embeddings = fairseq_config['max_source_positions']
    embedding_dim = fairseq_config['encoder_embed_dim']
    padding_idx = model.models[0].encoder.embed_tokens.padding_idx  # 1
    sin_pos_embedding = model.models[0].encoder.embed_positions.get_embedding(
        padding_idx + 1 + num_embeddings,
        embedding_dim,
        padding_idx=padding_idx)  # [2 + num_embeddings, embed_dim]
    sin_pos_embedding = sin_pos_embedding[2:, :]  # remove offset embeddings
    pool.starmap_async(
        split_and_convert_process,
        [('models.0.encoder.embed_positions.weight',
          torch_to_numpy(sin_pos_embedding), i_gpu_num, saved_dir),
         ('models.0.decoder.embed_positions.weight',
          torch_to_numpy(sin_pos_embedding), i_gpu_num, saved_dir)])

    pool.close()
    pool.join()

    # fuse qkv weight and bias
    fuse_qkv(model, i_gpu_num, saved_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input_dir",
                        "-i",
                        type=str,
                        help="Path to the framework checkpoint file",
                        required=True)
    parser.add_argument("--output_dir",
                        "-o",
                        type=str,
                        help="Path to the converted TRT-LLM model weight file",
                        required=True)
    parser.add_argument("--inference_tensor_para_size",
                        "-i_g",
                        type=int,
                        help="How many gpus for inference",
                        required=True)
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4)
    parser.add_argument("--weight_data_type",
                        type=str,
                        default="float32",
                        choices=["float32", "float16",
                                 "bfloat16"])  # TODO: test support for bf16?
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format=log_format)
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = (stop_time - start_time)
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
