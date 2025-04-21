def add_common_args(parser):
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--engine_dir',
                        type=str,
                        default=None,
                        help='Directory containing visual and LLM TRT engines')
    parser.add_argument('--visual_engine_name',
                        type=str,
                        default='model.engine',
                        help='Name of visual TRT engine')
    parser.add_argument('--audio_engine_name',
                        type=str,
                        default='model.engine',
                        help='Name of audio TRT engine')
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
    parser.add_argument("--audio_path",
                        type=str,
                        default=None,
                        help='input audio path')
    parser.add_argument("--path_sep",
                        type=str,
                        default=",",
                        help='Path separator symbol')
    parser.add_argument("--prompt_sep",
                        type=str,
                        default=",",
                        help="Prompt separator symbol")
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
        '--mm_embedding_offloading',
        type=lambda s: s.lower() == "true",
        default=None,
        help=
        'Enable position table offloading. When not specified, defaults to True if using a multimodal model with chunked context.'
    )
    parser.add_argument(
        '--session',
        default='cpp_llm_only',
        type=str,
        choices=['python', 'cpp_llm_only', 'cpp'],
        help=
        'Rumtime used to run the models. \n`cpp_llm_only`: vision engine run in python runtime, but LLM in pybind cpp runtime\n`python`: everything runs in python runtime\n`cpp`: everything runs in C++ runtime'
    )
    parser.add_argument(
        '--kv_cache_free_gpu_memory_fraction',
        default=0.7,
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
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    return parser


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_str_match_rate(s1, s2):
    distance = levenshtein_distance(s1, s2)
    max_length = max(len(s1), len(s2))
    match_rate = (1 - distance / max_length) * 100
    return match_rate
