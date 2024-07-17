import argparse

import torch.multiprocessing as mp

from tensorrt_llm.quantization import (quantize_and_export,
                                       quantize_nemo_and_export)

mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
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

    parser.add_argument("--dtype", help="Model data type.", default="float16")
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="full_prec",
        choices=[
            "fp8", "int8_sq", "int4_awq", "w4a8_awq", "int8_wo", "int4_wo",
            "full_prec"
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
    parser.add_argument("--awq_block_size", type=int, default=128)
    parser.add_argument("--kv_cache_dtype",
                        help="KV Cache dtype.",
                        default=None,
                        choices=["int8", "fp8", None])
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

    args = parser.parse_args()

    if args.model_dir is not None:
        quantize_and_export(
            model_dir=args.model_dir,
            device=args.device,
            calib_dataset=args.calib_dataset,
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
            seed=args.seed,
            tokenizer_max_seq_length=args.tokenizer_max_seq_length,
            num_medusa_heads=args.num_medusa_heads,
            num_medusa_layers=args.num_medusa_layers,
            max_draft_len=args.max_draft_len,
            medusa_hidden_act=args.medusa_hidden_act,
            medusa_model_dir=args.medusa_model_dir,
            quant_medusa_head=args.quant_medusa_head)
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
                                 seed=args.seed)
    else:
        raise ValueError(
            "One of source checkpoint (model_dir, nemo_ckpt_path) must be specified"
        )
