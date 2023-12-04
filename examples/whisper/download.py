import torch
import argparse
import configparser
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from pathlib import Path
import math


extra_configs = {
    "structure": {
        "bart_with_bias": "false",
        "use_gated_activation": "false",
        "position_embedding_type": "learned_absolute",
        'model_type': 'bart'
    }
}

def download(args):
    model = WhisperForConditionalGeneration.from_pretrained(args.input_dir)

    torch.save(model.state_dict(), f'{args.output_dir}/{args.input_dir.split("/")[-1]}.ckpt')

    dict_layers = {}
    for k, v in model.state_dict().items():
        dict_layers[k] = v.shape

    import json
    with open('decoder.json', 'w') as fp:
        json.dump(dict_layers, fp, indent=4)

    # write config
    config = configparser.ConfigParser()
    config["encoder"] = {}

    for key, val in model.model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"

    config["encoder"]["weight_data_type"] = args.weight_data_type

    def get_offset_q_scaling(config) -> str:
        d_model = config.d_model
        num_heads = config.encoder_attention_heads
        head_size = d_model // num_heads
        scaling = 1/(math.sqrt(num_heads)*(((head_size) ** -0.25)))
        return str(scaling)

    config["encoder"]["q_scaling"] = get_offset_q_scaling(model.model.encoder.config)

    config["decoder"] = {}
    for key, val in model.model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type

    config["decoder"]["q_scaling"] = get_offset_q_scaling(model.model.decoder.config)

    for key, val in extra_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val
    with open((f"{args.output_dir}/config.ini"), 'w') as configfile:
        config.write(configfile)

if __name__=="__main__":
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
                        help="Path to the save weight file",
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
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Provide verbose messages")
    args = parser.parse_args()
    download(args)