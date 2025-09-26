# autoflake: skip_file
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

import tensorrt_llm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        help='HF checkpoint path')
    parser.add_argument('--output_dir',
                        type=str,
                        required=True,
                        help='Save path')
    parser.add_argument(
        '--act_scales',
        type=str,
        required=True,
        help=
        'ModelOpt calibrated checkpoint dir or extracted safetensors for activation scales'
    )
    parser.add_argument('--parts',
                        type=int,
                        default=1,
                        help='devide all safetensors into parts')
    parser.add_argument('--rank',
                        type=int,
                        default=0,
                        help='which part to be quantize')
    args = parser.parse_args()
    return args


def load_and_preprocess_state_dict(modelopt_state_root, world_size=8):
    state_dict_list = []
    # load amax from state dict
    for rank in range(world_size):
        amax_file = f"{modelopt_state_root}/amax_dict_rank{rank}-mp{world_size}.pt"
        if os.path.exists(amax_file):
            state_dict_list.append(torch.load(amax_file, map_location="cuda:0"))
        else:
            print(f"WARNING: amax file not found: {amax_file}")

    if not state_dict_list:
        print("ERROR: No amax files loaded!")
        return {}

    # calculate the max across all TP ranks
    merged_state_dict = state_dict_list[0]
    for rank in range(world_size):
        for key, amax in state_dict_list[rank].items():
            if key in merged_state_dict.items():
                amax = torch.max(amax.to(0), merged_state_dict[key].to(0))
            merged_state_dict[key] = amax.to(0)

    mapping = {
        "ffn.shared_experts.w1": "mlp.shared_experts.gate_proj",
        "ffn.shared_experts.w2": "mlp.shared_experts.down_proj",
        "ffn.shared_experts.w3": "mlp.shared_experts.up_proj",
        "ffn.shared_experts": "mlp.shared_experts",
        "ffn.shared_experts": "mlp.shared_experts",
        "ffn.shared_experts": "mlp.shared_experts",
        "ffn.w1": "mlp.gate_proj",
        "ffn.w2": "mlp.down_proj",
        "ffn.w3": "mlp.up_proj",
        "head": "lm_head",
        "attn": "self_attn",
    }
    new_dict = {}
    for k, v in merged_state_dict.items():
        new_key = k.replace("layers", "model.layers")
        for original_pattern, replace_pattern in mapping.items():
            new_key = new_key.replace(original_pattern, replace_pattern)
        # ffn.experts.xx.w1/w2/w3- > mlp.experts.xx.gate_proj/down_proj/up_proj
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w1",
                         r"mlp.experts.\1.gate_proj", new_key)
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w2",
                         r"mlp.experts.\1.down_proj", new_key)
        new_key = re.sub(r"ffn\.experts\.(\d+)\.w3", r"mlp.experts.\1.up_proj",
                         new_key)
        new_dict[new_key] = v

    merged_state_dict.clear()
    merged_state_dict.update(new_dict)

    # set amax for modules to be fused and make sure they share the same input
    for key, amax in merged_state_dict.items():
        if "up_proj" in key:
            gate_proj_key = key.replace("up_proj", "gate_proj")
            if "weight_quantizer" in key:
                fused_amax = torch.max(amax, merged_state_dict[gate_proj_key])
                merged_state_dict[key] = fused_amax
                merged_state_dict[gate_proj_key] = fused_amax
            elif "input_quantizer" in key:
                assert amax == merged_state_dict[gate_proj_key]
            else:
                raise NotImplementedError

    return merged_state_dict


def get_scales_from_amax(start_layer, end_layer, renamed_state_dict):
    weight_name_dict = {"gate_proj": 1, "down_proj": 2, "up_proj": 3}
    scales = {}
    for layer_idx in range(start_layer, end_layer):
        amax_keys_per_layer = [
            x for x in renamed_state_dict.keys()
            if (x.startswith(f'model.layers.{layer_idx}.mlp.experts.')
                and x.endswith(".input_quantizer._amax"))
        ]
        for k in amax_keys_per_layer:
            expert_idx = int(k.split('.')[5])
            weight_idx = weight_name_dict[k.split('.')[6]]
            val = renamed_state_dict[k]
            scales[
                f'model.layers.{layer_idx}.mlp.experts.{expert_idx}.w{weight_idx}.input_scale'] = val.unsqueeze(
                    0) / 448

    return scales


def quantize_fp8_block_scale_to_int4(fp8_tensor, fp8_scale):
    group_size = 128
    blocked_tensor = fp8_tensor.view(fp8_tensor.shape[0] // 128, 128,
                                     fp8_tensor.shape[1] // 128,
                                     128).to(torch.float32)
    dequant_tensor = (blocked_tensor *
                      fp8_scale.unsqueeze(1).unsqueeze(3)).view(
                          fp8_tensor.shape[0],
                          fp8_tensor.shape[1] // group_size,
                          group_size).to(torch.bfloat16).to(torch.float32)
    scale_tensor = torch.abs(dequant_tensor).max(dim=2).values / 7
    quant_tensor = torch.clamp(torch.round(
        (dequant_tensor / scale_tensor.unsqueeze(-1))),
                               min=-8,
                               max=7)
    quant_tensor = quant_tensor.to(torch.int8)
    return quant_tensor.view(fp8_tensor.shape), scale_tensor


def main(args):
    model_dir = args.model_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(args.rank % num_gpus)

    model_index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
        weight_map = model_index["weight_map"]

    processed_files = {}
    for tensor_name in list(weight_map.keys()):
        if tensor_name not in weight_map:
            continue
        file_name = weight_map[tensor_name]
        if file_name in processed_files:
            continue
        processed_files[file_name] = safe_open(os.path.join(
            model_dir, file_name),
                                               "pt",
                                               device="cuda")

    with open(os.path.join(model_dir, "config.json"), 'r') as file:
        config = json.load(file)

    num_layer = config['num_hidden_layers']
    part_layer = (num_layer + args.parts - 1) // args.parts
    start_layer = args.rank * part_layer
    end_layer = min(num_layer, args.rank * part_layer + part_layer)

    def get_tensor(name):
        if name not in weight_map:
            return None
        ff = weight_map[name]
        safetensors_loader = processed_files[ff]
        return safetensors_loader.get_tensor(name).cuda()

    def get_file_name(layer):
        rank = layer // part_layer
        return "model-%05d-of-%05d.safetensors" % (rank, args.parts)

    new_safetensors = {}
    new_json = {}
    new_json['weight_map'] = {}
    new_json['metadata'] = {}
    for key in tqdm(list(weight_map.keys())):
        if "mlp.experts" in key and (key.endswith("weight")
                                     or key.endswith("weight_scale_inv")):
            if key.endswith("weight_scale_inv"):
                continue
            if args.rank == 0:
                layer = int(key.split(".")[2])
                new_json['weight_map'][key] = get_file_name(layer)
                new_json['weight_map'][key.replace(
                    "weight", "weight_scale_inv")] = get_file_name(layer)
            if int(key.split(".")[2]) < start_layer or int(
                    key.split(".")[2]) >= end_layer:
                continue
            fp8_tensor = get_tensor(key)
            fp8_scale = get_tensor(key.replace("weight", "weight_scale_inv"))
            quant_tensor, scale_tensor = quantize_fp8_block_scale_to_int4(
                fp8_tensor, fp8_scale)

            packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
            packed_tensor = packer(quant_tensor.cpu().contiguous())
            new_safetensors.update({key: packed_tensor})
            new_safetensors.update({
                key.replace("weight", "weight_scale_inv"):
                scale_tensor.contiguous()
            })
        else:
            name = key.split(".")
            if args.rank == 0:
                if len(name) < 3 or not name[2].isdigit():
                    new_safetensors.update({key: get_tensor(key)})
                    new_json['weight_map'][key] = get_file_name(0)
                    continue

                file_name = get_file_name(int(name[2]))
                new_json['weight_map'][key] = file_name

            if len(name) < 3 or not name[2].isdigit() or (int(
                    name[2]) < start_layer or int(name[2]) >= end_layer):
                continue
            new_safetensors.update({key: get_tensor(key)})

    # Process activation scales for all ranks
    if os.path.isdir(args.act_scales):
        # Extract activation scales
        renamed_state_dict = load_and_preprocess_state_dict(
            modelopt_state_root=args.act_scales, world_size=8)
        scales = get_scales_from_amax(start_layer=start_layer,
                                      end_layer=end_layer,
                                      renamed_state_dict=renamed_state_dict)
        new_safetensors.update(scales)

    if args.rank == 0:
        if not os.path.isdir(args.act_scales):
            input_scales = safe_open(args.act_scales, "pt")
            for k in input_scales.keys():
                new_safetensors.update({k: input_scales.get_tensor(k)})
                new_json['weight_map'][k] = args.act_scales.split("/")[-1]

        file_name = get_file_name(start_layer)
        print(f'saving to {file_name}...')
        save_file(new_safetensors, os.path.join(output_dir, file_name))
        with open(os.path.join(output_dir, "model.safetensors.index.json"),
                  "w") as f:
            json.dump(new_json, f)

        names = [
            "configuration_deepseek.py", "generation_config.json",
            "modeling_deepseek.py", "tokenizer.json", "tokenizer_config.json"
        ]
        for name in names:
            shutil.copy(os.path.join(model_dir, name), output_dir)
        if os.path.isdir(args.act_scales):
            shutil.copytree(args.act_scales, output_dir, dirs_exist_ok=True)
        else:
            shutil.copy(args.act_scales, output_dir)

        # config.json
        del config['quantization_config']
        with open(os.path.join(output_dir, "config.json"), 'w') as file:
            json.dump(config, file, indent=4)

        # quant_cfg.json
        attn_names = ["fused_a", "q_b_proj", "kv_b_proj", "o_proj"]
        mlp_names = ["gate_up_proj", "down_proj"]
        fp8_block_scale = {"quant_algo": "FP8_BLOCK_SCALES"}
        w4a8_awq = {"quant_algo": "W4A8_AWQ"}
        quant_cfg = {}
        quant_cfg["quant_algo"] = "MIXED_PRECISION"
        quant_cfg["kv_cache_quant_algo"] = None
        quant_cfg["quantized_layers"] = {}
        for l in range(61):
            prefix = f"model.layers.{l}"
            for n1 in attn_names:
                quant_cfg["quantized_layers"][
                    f"{prefix}.self_attn.{n1}"] = fp8_block_scale
            for n2 in mlp_names:
                quant_cfg["quantized_layers"][
                    f"{prefix}.mlp.shared_experts.{n2}"] = fp8_block_scale
            if l < 3:
                for n3 in mlp_names:
                    quant_cfg["quantized_layers"][
                        f"{prefix}.mlp.{n3}"] = fp8_block_scale
            else:
                quant_cfg["quantized_layers"][
                    f"{prefix}.mlp.experts"] = w4a8_awq
        with open(os.path.join(output_dir, "quant_cfg.json"), 'w') as file:
            json.dump(quant_cfg, file, indent=4)

        # hf_quant_config.json
        hf_quant_config = {}
        hf_quant_config['quantization'] = {}
        hf_quant_config['quantization']["quant_algo"] = "MIXED_PRECISION"
        hf_quant_config['quantization']["kv_cache_quant_algo"] = None
        with open(os.path.join(output_dir, "hf_quant_config.json"),
                  'w') as file:
            json.dump(hf_quant_config, file, indent=4)
    else:
        file_name = get_file_name(start_layer)
        print(f'saving to {file_name}...')
        save_file(new_safetensors, os.path.join(output_dir, file_name))


if __name__ == "__main__":
    args = parse_args()
    main(args)
