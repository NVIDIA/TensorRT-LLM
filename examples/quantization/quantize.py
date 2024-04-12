import argparse

from tensorrt_llm.quantization import quantize_and_export

if __name__ == "__main__":
    DEFAULT_RAND_SEED = 1234
    DEFAULT_MAX_SEQ_LEN = 2048
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        help="Specify where the HuggingFace model is",
                        required=True)
    parser.add_argument("--device", default="cuda")
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
        default=DEFAULT_RAND_SEED)
    parser.add_argument("--max_seq_length",
                        help="Max sequence length to init the tokenizers",
                        type=int,
                        default=DEFAULT_MAX_SEQ_LEN)

    parser.add_argument("--batch_size",
                        help="Batch size for calibration.",
                        type=int,
                        default=1)
    parser.add_argument("--calib_size",
                        help="Number of samples for calibration.",
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
    args = parser.parse_args()

    quantize_and_export(model_dir=args.model_dir,
                        dtype=args.dtype,
                        output_dir=args.output_dir,
                        device=args.device,
                        tp_size=args.tp_size,
                        pp_size=args.pp_size,
                        qformat=args.qformat,
                        kv_cache_dtype=args.kv_cache_dtype,
                        calib_size=args.calib_size,
                        batch_size=args.batch_size,
                        awq_block_size=args.awq_block_size,
                        seed=args.seed,
                        max_seq_length=args.max_seq_length)
