def add_common_args(parser):
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--visual_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing visual TRT engines')
    parser.add_argument('--visual_engine_name',
                        type=str,
                        default='model.engine',
                        help='Name of visual TRT engine')
    parser.add_argument('--llm_engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing TRT-LLM engines')
    parser.add_argument('--hf_model_dir',
                        type=str,
                        default=None,
                        help="Directory containing tokenizer")
    parser.add_argument('--input_text',
                        type=str,
                        nargs='+',
                        default=None,
                        help='Text prompt to LLM')
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--run_profiling',
                        action='store_true',
                        help='Profile runtime over several iterations')
    parser.add_argument('--profiling_iterations',
                        type=int,
                        help="Number of iterations to run profiling",
                        default=20)
    parser.add_argument('--check_accuracy',
                        action='store_true',
                        help='Check correctness of text output')
    parser.add_argument(
        '--video_path',
        type=str,
        default=None,
        help=
        'Path to your local video file, using \'llava-onevision-accuracy\' to check the Llava-OneVision model accuracy'
    )
    parser.add_argument(
        '--video_num_frames',
        type=int,
        help=
        "The number of frames sampled from the video in the Llava-OneVision model.",
        default=None)
    parser.add_argument("--image_path",
                        type=str,
                        nargs='+',
                        default=None,
                        help='List of input image paths, separated by symbol')
    parser.add_argument("--path_sep",
                        type=str,
                        default=",",
                        help='Path separator symbol')
    parser.add_argument('--enable_context_fmha_fp32_acc',
                        action='store_true',
                        default=None,
                        help="Enable FMHA runner FP32 accumulation.")
    parser.add_argument(
        '--enable_chunked_context',
        action='store_true',
        help='Enables chunked context (only available with cpp session).',
    )
    parser.add_argument(
        '--use_py_session',
        default=False,
        action='store_true',
        help=
        "Whether or not to use Python runtime session. By default C++ runtime session is used for the LLM."
    )
    parser.add_argument(
        '--kv_cache_free_gpu_memory_fraction',
        default=0.9,
        type=float,
        help='Specify the free gpu memory fraction.',
    )
    parser.add_argument(
        '--cross_kv_cache_fraction',
        default=0.5,
        type=float,
        help=
        'Specify the kv cache fraction reserved for cross attention. Only applicable for encoder-decoder models. By default 0.5 for self and 0.5 for cross.',
    )
    parser.add_argument(
        '--multi_block_mode',
        type=lambda s: s.lower() in
        ("yes", "true", "t", "1"
         ),  # custom boolean function to convert input string to boolean
        default=True,
        help=
        "Distribute the work across multiple CUDA thread-blocks on the GPU for masked MHA kernel."
    )
    return parser
