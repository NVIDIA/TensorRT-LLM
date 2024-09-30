import argparse
import os

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_new_tokens', type=int, default=30)
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
    parser.add_argument('--video_path',
                        type=str,
                        default=None,
                        help='Path to your local video file')
    parser.add_argument("--image_path",
                        type=str,
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
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")

    return parser.parse_args()


def print_result(model, input_text, output_text, args):
    logger.info("---------------------------------------------------------")
    if model.model_type != 'nougat':
        logger.info(f"\n[Q] {input_text}")
    for i in range(len(output_text)):
        logger.info(f"\n[A]: {output_text[i]}")

    if args.num_beams == 1:
        output_ids = model.tokenizer(output_text[0][0],
                                     add_special_tokens=False)['input_ids']
        logger.info(f"Generated {len(output_ids)} tokens")

    if args.check_accuracy:
        if model.model_type != 'nougat':
            if model.model_type == "vila":
                if len(args.image_path.split(args.path_sep)) == 1:
                    assert output_text[0][0].lower(
                    ) == "the image captures a bustling city intersection teeming with life. from the perspective of a car's dashboard camera, we see"
            elif model.model_type == 'fuyu':
                assert output_text[0][0].lower() == '4'
            elif model.model_type == "pix2struct":
                assert "characteristic | cat food, day | cat food, wet | cat treats" in output_text[
                    0][0].lower()
            elif model.model_type in [
                    'blip2', 'neva', 'phi-3-vision', 'llava_next'
            ]:
                assert 'singapore' in output_text[0][0].lower()
            elif model.model_type == 'video-neva':
                assert 'robot' in output_text[0][0].lower()
            elif model.model_type == 'kosmos-2':
                assert 'snowman' in output_text[0][0].lower()
            else:
                assert output_text[0][0].lower() == 'singapore'

    if args.run_profiling:
        msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(
            name) / args.profiling_iterations
        logger.info('Latencies per batch (msec)')
        logger.info('TRT vision encoder: %.1f' % (msec_per_batch('Vision')))
        logger.info('TRTLLM LLM generate: %.1f' % (msec_per_batch('LLM')))
        logger.info('Multimodal generate: %.1f' % (msec_per_batch('Generate')))

    logger.info("---------------------------------------------------------")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_arguments()
    logger.set_level(args.log_level)

    model = MultimodalModelRunner(args)
    raw_image = model.load_test_image()

    num_iters = args.profiling_iterations if args.run_profiling else 1
    for _ in range(num_iters):
        input_text, output_text = model.run(args.input_text, raw_image,
                                            args.max_new_tokens)

    runtime_rank = tensorrt_llm.mpi_rank()
    if runtime_rank == 0:
        print_result(model, input_text, output_text, args)
