# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
from collections import OrderedDict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default="distil-whisper/distil-large-v3",
                        help="Model name")
    parser.add_argument("--cache_dir",
                        type=str,
                        default=None,
                        help="Cache directory")
    parser.add_argument("--output_dir",
                        type=str,
                        default="./assets/",
                        help='Store the "translated" model here')
    parser.add_argument("--output_name",
                        type=str,
                        default="distil-large-v3",
                        help="Output model name")

    args = parser.parse_args()

    model_name = args.model_name
    cache_dir = args.cache_dir
    output_dir = args.output_dir
    output_name = args.output_name

    if cache_dir is not None:
        print("Trying to load the model from the cache")
        model = AutoModel.from_pretrained(model_name,
                                          cache_dir=cache_dir,
                                          use_safetensors=True)
    else:
        print("Downloading the model:")
        model = AutoModel.from_pretrained(model_name, use_safetensors=True)
    model = model.half()  # compatible with openai's checkpoint
    config = model.config
    model_dims = {
        'n_mels': config.num_mel_bins,
        'n_vocab': config.vocab_size,
        'n_audio_ctx': config.max_source_positions,
        'n_audio_state': config.d_model,
        'n_audio_head': config.encoder_attention_heads,
        'n_audio_layer': config.encoder_layers,
        'n_text_ctx': config.max_target_positions,
        'n_text_state': config.d_model,
        'n_text_head': config.decoder_attention_heads,
        'n_text_layer': config.decoder_layers
    }

    original_model_state_dict = model.state_dict()
    new_state_dict = {}

    for key, value in tqdm(original_model_state_dict.items()):
        new_state_dict[translate(key)] = value
    print("Param keys have been changed. Saving the model...")

    pytorch_model = {"dims": model_dims, "model_state_dict": new_state_dict}

    # Create the directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created successfully!")
    else:
        print(f"Directory '{output_dir}' already exists!")

    output_path = Path(output_dir) / f"{output_name}.pt"
    torch.save(pytorch_model, output_path)
    print("Model saved to ", output_path)
    print("Kindly use that to build the tensorrt_llm engine.")


def translate(current_param):
    for pattern, repl in reverse_translation.items():
        if re.match(pattern, current_param):
            return re.sub(pattern, repl, current_param)

reverse_translation = OrderedDict({
    r"^encoder\.layers\.(\d+)\.self_attn.k_proj\.(\w+)$": r"encoder.blocks.\1.attn.key.\2",
    r"^encoder\.layers\.(\d+)\.self_attn.out_proj\.(\w+)$": r"encoder.blocks.\1.attn.out.\2",
    r"^encoder\.layers\.(\d+)\.self_attn.q_proj\.(\w+)$": r"encoder.blocks.\1.attn.query.\2",
    r"^encoder\.layers\.(\d+)\.self_attn.v_proj\.(\w+)$": r"encoder.blocks.\1.attn.value.\2",
    r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(\w+)$": r"encoder.blocks.\1.attn_ln.\2",
    r"^encoder\.layers\.(\d+)\.fc1\.(\w+)$": r"encoder.blocks.\1.mlp.0.\2",
    r"^encoder\.layers\.(\d+)\.fc2\.(\w+)$": r"encoder.blocks.\1.mlp.2.\2",
    r"^encoder\.layers\.(\d+)\.final_layer_norm\.(\w+)$": r"encoder.blocks.\1.mlp_ln.\2",
    r"^encoder\.embed_positions\.weight$": r"encoder.positional_embedding",
    r"^encoder\.layer_norm\.(\w+)$": r"encoder.ln_post.\1",
    r"^encoder\.(\w+)\.(\w+)": r"encoder.\1.\2",
\
    r"^decoder\.embed_positions\.weight$": r"decoder.positional_embedding",
    r"^decoder\.embed_tokens\.weight$": r"decoder.token_embedding.weight",
    r"^decoder\.layer_norm\.(\w+)$": r"decoder.ln.\1",
\
    r"^decoder\.layers\.(\d+)\.encoder_attn\.k_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.key.\2",
    r"^decoder\.layers\.(\d+)\.encoder_attn\.out_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.out.\2",
    r"^decoder\.layers\.(\d+)\.encoder_attn\.q_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.query.\2",
    r"^decoder\.layers\.(\d+)\.encoder_attn\.v_proj.(\w+)$": r"decoder.blocks.\1.cross_attn.value.\2",
    r"^decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.(\w+)$": r"decoder.blocks.\1.cross_attn_ln.\2",
\
    r"^decoder\.layers\.(\d+)\.self_attn\.k_proj\.(\w+)$": r"decoder.blocks.\1.attn.key.\2",
    r"^decoder\.layers\.(\d+)\.self_attn\.out_proj\.(\w+)$": r"decoder.blocks.\1.attn.out.\2",
    r"^decoder\.layers\.(\d+)\.self_attn\.q_proj\.(\w+)$": r"decoder.blocks.\1.attn.query.\2",
    r"^decoder\.layers\.(\d+)\.self_attn\.v_proj\.(\w+)$": r"decoder.blocks.\1.attn.value.\2",
    r"^decoder\.layers\.(\d+)\.self_attn_layer_norm\.(\w+)$": r"decoder.blocks.\1.attn_ln.\2",
    r"^decoder\.layers\.(\d+)\.fc1\.(\w+)$": r"decoder.blocks.\1.mlp.0.\2",
    r"^decoder\.layers\.(\d+)\.fc2\.(\w+)$": r"decoder.blocks.\1.mlp.2.\2",
    r"^decoder\.layers\.(\d+)\.final_layer_norm\.(\w+)$": r"decoder.blocks.\1.mlp_ln.\2",
})

if __name__ == "__main__":
    main()
