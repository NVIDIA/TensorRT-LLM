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
from transformers import (AutoModelForSeq2SeqLM, MBartForConditionalGeneration,
                          VisionEncoderDecoderModel)

from tensorrt_llm._utils import str_dtype_to_torch, torch_to_numpy

LOGGER = logging.getLogger(__name__)

extra_configs = {
    "structure": {
        "t5_with_bias": "true",
        "use_gated_activation": "false",
        "position_embedding_type": "learned",
        'model_type': 'bart'
    }
}  # TODO: remove model type as it's included in HF config's `architectures` attribute

# TODO: change name `t5_with_bias` for non-t5 model


def fuse_qkv(model, factor, saved_dir):

    def get_attn_module(component, layer, attn_type):
        m = model.model
        m = getattr(m, component)
        m = m.layers[int(layer)]
        m = getattr(m, attn_type)
        return m

    for name, param in model.named_parameters():
        if 'attn.q_proj.weight' in name:
            # fuse weights of q, k, v
            q_w = param
            _, component, _, layer_idx, attn_type, *_ = name.split('.')
            attn_mdl = get_attn_module(component, layer_idx, attn_type)

            # fuse qkv weight
            shape = q_w.shape  # (do, din)
            qkv_w = torch.cat(
                [q_w, attn_mdl.k_proj.weight, attn_mdl.v_proj.weight],
                dim=0).reshape([3, shape[0], shape[1]])  # (3, do, din)
            qkv_w = torch_to_numpy(qkv_w)
            split_vals = np.split(qkv_w, factor,
                                  axis=1)  # TODO: need to test using multi-gpu
            for j in range(factor):
                saved_path = saved_dir / f"model.{component}.layers.{layer_idx}.{attn_type}.qkv_proj.weight.{j}.bin"
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
                    saved_path = saved_dir / f"model.{component}.layers.{layer_idx}.{attn_type}.qkv_proj.bias.{j}.bin"
                    split_vals[j].tofile(saved_path.as_posix())


# TODO: use re.compile to accelerate
def split_and_convert_process(key, val, factor, saved_dir):
    saved_key = key
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    def save_splits(split_vals):
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    if re.search('norm|embed_positions|(out_proj|fc2)\.bias', key) is not None:
        saved_path = saved_dir / f"{saved_key}.bin"
        if 'position' in key:
            val = val[2:]  # BART does not use first two position embeddings!
        val.tofile(saved_path.as_posix())
    elif re.search('(lm_head|fc1)\.(weight|bias)', key) is not None:
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
    elif re.search('(en|de)coder.embed_tokens.weight', key) is not None:
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())
    elif 'final_logits_bias' in key:  # buffer used to manually control emission prob?
        pass
    else:
        LOGGER.warning(
            f"cannot find key '{key}' with shape {val.shape}, no skip weight")


def convert_checkpoint(args):
    saved_dir = Path(args.output_dir) / f"tp{args.inference_tensor_para_size}"
    saved_dir.mkdir(parents=True, exist_ok=True)

    if args.nougat:
        model = VisionEncoderDecoderModel.from_pretrained(args.input_dir)
        model = model.get_decoder()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.input_dir)
    model = model.to(str_dtype_to_torch(args.weight_data_type))

    config = configparser.ConfigParser()

    config['decoder'] = dict()
    for key, val in model.model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"]["q_scaling"] = '1'
    config["decoder"]["rescale_before_lm_head"] = str(False)
    config['decoder']['has_model_final_layernorm'] = str(
        args.nougat or isinstance(model, MBartForConditionalGeneration))

    if args.nougat:
        # These flags are true for mbart decoders, but missing in HF config
        config['decoder']['normalize_before'] = str(True)
        config['decoder']['normalize_embeddings'] = str(True)

        config['encoder'] = dict()
        # Init few encoder configs, needed by build, from decoder config
        encoder_config_keys = [
            "encoder_ffn_dim", "encoder_layers", "encoder_attention_heads",
            "encoder_layerdrop", "d_model"
        ]
        for key in encoder_config_keys:
            config['encoder'][key] = config['decoder'][key]
    else:
        config['encoder'] = dict()
        for key, val in model.model.encoder.config.to_dict().items():
            config["encoder"][key] = f"{val}"
        config["encoder"]["weight_data_type"] = args.weight_data_type
        config["encoder"]["q_scaling"] = '1'

        # mBART has final layernorm, BART does not
        config['encoder']['has_model_final_layernorm'] = str(
            isinstance(model, MBartForConditionalGeneration))

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
    parser.add_argument("--nougat",
                        action="store_true",
                        help="Model which uses vision encoder + mbart decoder")
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
