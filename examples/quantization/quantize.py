import argparse

import torch.multiprocessing as mp

from tensorrt_llm.quantization import (quantize_and_export,
                                       quantize_nemo_and_export)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        help="Specify where the HuggingFace model is",
                        default=None)
    parser.add_argument('--nemo_ckpt_path',
                        help="Specify where the NeMo checkpoint is",
                        default=None)
    parser.add_argument(
        '--decoder_type',
        type=str,
        default='gptnext',
        choices=['gptnext', 'llama'],
        help="Decoder type; effective for NeMo checkpoint only.")
    parser.add_argument(
        '--device',
        help=
        "The device to run calibration; effective for HuggingFace model only.",
        default='cuda',
        choices=['cuda', 'cpu'])
    parser.add_argument(
        "--device_map",
        help="How to map the model on the devices",
        default="auto",
        choices=["auto", "sequential", "cpu", "gpu"],
    )
    parser.add_argument(
        '--calib_dataset',
        type=str,
        default='cnn_dailymail',
        help=
        "The huggingface dataset name or the local directory of the dataset for calibration."
    )
    parser.add_argument(
        '--calib_tp_size',
        type=int,
        default=1,
        help=
        "Tensor parallel size for calibration; effective for NeMo checkpoint only."
    )
    parser.add_argument(
        '--calib_pp_size',
        type=int,
        default=1,
        help=
        "Pipeline parallel size for calibration; effective for NeMo checkpoint only."
    )

    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'float16', 'bfloat16', 'float32'],
        help=
        "The data type for the model weights and activations of the non-quantized part, e.g., embedding and lm_head. "
        "If 'auto', the data type is automatically inferred from the source model; "
        "however, if the source dtype is float32, it is converted to float16.")
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="full_prec",
        choices=[
            "nvfp4",
            "fp8",
            "fp8_pc_pt",
            "int8_sq",
            "int4_awq",
            "w4a8_awq",
            "int8_wo",
            "int4_wo",
            "full_prec",
        ],
    )
    parser.add_argument(
        "--seed",
        help="Seed the generate random numbers, the value will be used to call"
        "random.seed(value) and numpy.random.seed(value)",
        type=int,
        default=1234)
    parser.add_argument("--tokenizer_max_seq_length",
                        help="Max sequence length to init the tokenizers",
                        type=int,
                        default=2048)

    parser.add_argument("--batch_size",
                        help="Batch size for calibration.",
                        type=int,
                        default=1)
    parser.add_argument("--calib_size",
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--calib_max_seq_length",
                        help="Max sequence length for calibration",
                        type=int,
                        default=512)
    parser.add_argument("--output_dir", default="exported_model")
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--awq_block_size", type=int, default=128)
    parser.add_argument("--kv_cache_dtype",
                        help="KV Cache dtype.",
                        default=None,
                        choices=["int8", "fp8", None])
    parser.add_argument("--quantize_lm_head",
                        action='store_true',
                        default=False)
    # Medusa
    parser.add_argument('--num_medusa_heads', type=int, default=4)
    parser.add_argument('--num_medusa_layers', type=int, default=1)
    parser.add_argument('--max_draft_len', type=int, default=63)
    parser.add_argument('--medusa_hidden_act', type=str, default="silu")
    parser.add_argument('--medusa_model_dir', type=str, default=None)
    parser.add_argument('--quant_medusa_head',
                        default=False,
                        action='store_true',
                        help="whether to quantize the weights of medusa heads")

    # auto quantization
    parser.add_argument(
        '--autoq_format',
        default=None,
        type=str,
        help=
        "Specific quantization algorithms will be searched in auto quantization."
        "The algorithm must in ['fp8', 'int4_awq', 'w4a8_awq', 'int8_sq']."
        "You can use ',' to separate more than one quantization algorithms(e.g. --autoq_format fp8,int4_awq,w4a8_awq)."
        "Notice: fp8 and int8_sq can't be used at the same time.")
    parser.add_argument(
        '--auto_quantize_bits',
        type=float,
        default=None,
        help="Effective bits constraint for auto quantization. If not set, "
        "regular quantization without auto quantization search will be applied."
        "You can't set it lower than the num_bits of most aggressive quantization format."
        "For example, if 'int4_awq' is in autoq_format, it can't be lower than 4.0."
    )

    args = parser.parse_args()

    # auto_quantize_bits check
    if args.autoq_format:
        lower_bound, upper_bound = 4 if '4' in args.autoq_format else 8, 16
        if args.auto_quantize_bits is None or args.auto_quantize_bits < lower_bound or args.auto_quantize_bits > upper_bound:
            print(
                f"invalid auto_quantize_bits value, will be set to {lower_bound}"
            )
            args.auto_quantize_bits = lower_bound

    if args.model_dir is not None:
        quantize_and_export(
            model_dir=args.model_dir,
            device=args.device,
            calib_dataset=args.calib_dataset,
            dtype=args.dtype,
            qformat=args.qformat
            if args.auto_quantize_bits is None else args.autoq_format,
            kv_cache_dtype=args.kv_cache_dtype,
            calib_size=args.calib_size,
            batch_size=args.batch_size,
            calib_max_seq_length=args.calib_max_seq_length,
            awq_block_size=args.awq_block_size,
            output_dir=args.output_dir,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            cp_size=args.cp_size,
            seed=args.seed,
            tokenizer_max_seq_length=args.tokenizer_max_seq_length,
            num_medusa_heads=args.num_medusa_heads,
            num_medusa_layers=args.num_medusa_layers,
            max_draft_len=args.max_draft_len,
            medusa_hidden_act=args.medusa_hidden_act,
            medusa_model_dir=args.medusa_model_dir,
            quant_medusa_head=args.quant_medusa_head,
            auto_quantize_bits=args.auto_quantize_bits,
            device_map=args.device_map,
            quantize_lm_head=args.quantize_lm_head)
    elif args.nemo_ckpt_path is not None:
        quantize_nemo_and_export(nemo_ckpt_path=args.nemo_ckpt_path,
                                 decoder_type=args.decoder_type,
                                 calib_dataset=args.calib_dataset,
                                 calib_tp_size=args.calib_tp_size,
                                 calib_pp_size=args.calib_pp_size,
                                 dtype=args.dtype,
                                 qformat=args.qformat,
                                 kv_cache_dtype=args.kv_cache_dtype,
                                 calib_size=args.calib_size,
                                 batch_size=args.batch_size,
                                 calib_max_seq_length=args.calib_max_seq_length,
                                 awq_block_size=args.awq_block_size,
                                 output_dir=args.output_dir,
                                 tp_size=args.tp_size,
                                 pp_size=args.pp_size,
                                 cp_size=args.cp_size,
                                 seed=args.seed)
    else:
        raise ValueError(
            "One of source checkpoint (model_dir, nemo_ckpt_path) must be specified"
        )
