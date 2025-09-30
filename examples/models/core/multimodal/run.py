import argparse
import os

from utils import add_common_args, compute_str_match_rate

import tensorrt_llm
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime import MultimodalModelRunner


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
                for i in range(len(args.image_path.split(args.path_sep))):
                    if i % 2 == 0:
                        assert output_text[i][0].lower(
                        ) == "the image captures a bustling city intersection teeming with life. from the perspective of a car's dashboard camera, we see"
                    else:
                        assert output_text[i][0].lower(
                        ) == "the image captures the iconic merlion statue in singapore, a renowned worldwide landmark. the merlion, a mythical"
            elif model.model_type == "llava":
                for i in range(len(args.image_path.split(args.path_sep))):
                    assert output_text[i][0].lower() == 'singapore'
            elif model.model_type == 'fuyu':
                assert output_text[0][0].lower() == '4'
            elif model.model_type == "pix2struct":
                assert "characteristic | cat food, day | cat food, wet | cat treats" in output_text[
                    0][0].lower()
            elif model.model_type in [
                    'blip2', 'neva', 'phi-3-vision', 'llava_next',
                    'phi-4-multimodal', 'pixtral'
            ]:
                assert 'singapore' in output_text[0][0].lower()
            elif model.model_type == 'video-neva':
                assert 'robot' in output_text[0][0].lower()
            elif model.model_type == 'kosmos-2':
                assert 'snowman' in output_text[0][0].lower()
            elif model.model_type == "mllama":
                if "If I had to write a haiku for this one" in input_text:
                    ref_1 = ", it would be:.\\nPeter Rabbit is a rabbit.\\nHe lives in a cozy little house.\\nHe's a very good rabbit.\\"
                    ref_2 = "Here is a haiku for the image:\n\n"

                elif "Answer:" in input_text:
                    ref_1 = "2,173. <OCR/> A 1 2 3 4 5 6 Date Income 2005-12-17"
                    ref_2 = "Answer: 2,173. <OCR/> 1 2 3 4 5 6 Date Income 2005-12-17"

                elif "The key to life is" in input_text:
                    ref_1 = "to find your passion and pursue it with all your heart. For me, that passion is photography. I love capturing the beauty of the world around me"
                    ref_2 = "not to be found in the external world,"
                output = output_text[0][0]
                match_rate = max(compute_str_match_rate(ref_1, output),
                                 compute_str_match_rate(ref_2, output))
                logger.info(f"match rate: {match_rate}")
                assert match_rate >= 50, \
                    f"expected results: '{ref_1}' or '{ref_2}', generated results: '{output}'"

            elif model.model_type == 'llava_onevision':
                if args.video_path is None:
                    assert 'singapore' in output_text[0][0].lower()
                else:
                    assert 'the video is funny because the child\'s actions are' in output_text[
                        0][0].lower()
            elif model.model_type == "qwen2_vl":
                assert 'dog' in output_text[0][0].lower()
            else:
                assert output_text[0][0].lower() == 'singapore'

    if args.run_profiling:
        msec_per_batch = lambda name: 1000 * profiler.elapsed_time_in_sec(
            name) / args.profiling_iterations
        logger.info('Latencies per batch (msec)')
        logger.info('e2e generation: %.1f' % (msec_per_batch('Generate')))
        logger.info(' ' * 2 + 'Preprocessing: %.1f' %
                    (msec_per_batch('Preprocess')))
        logger.info(' ' * 4 + 'Vision encoder: %.1f' %
                    (msec_per_batch('Vision encoder')))
        if profiler.elapsed_time_in_sec('Feature transform') is not None:
            logger.info(' ' * 4 + 'Feature transform: %.1f' %
                        (msec_per_batch('Feature transform')))
        logger.info(' ' * 2 + 'LLM generate: %.1f' % (msec_per_batch('LLM')))
        logger.info(' ' * 2 + 'Tokenizer decode: %.1f' %
                    (msec_per_batch('Tokenizer decode')))

    logger.info("---------------------------------------------------------")


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    args = parser.parse_args()
    logger.set_level(args.log_level)

    model = MultimodalModelRunner(args)
    visual_data = model.load_test_data(args.image_path, args.video_path)
    audio_data = model.load_test_audio(args.audio_path)

    if args.run_profiling:
        num_warmup_iters = 3  # Multiple iterations to load both vision and LLM engines into memory
        for _ in range(num_warmup_iters):
            input_text, output_text = model.run(args.input_text, visual_data,
                                                audio_data, args.max_new_tokens)
        profiler.reset()

    num_iters = args.profiling_iterations if args.run_profiling else 1

    for _ in range(num_iters):
        input_text, output_text = model.run(args.input_text, visual_data,
                                            audio_data, args.max_new_tokens)

    runtime_rank = tensorrt_llm.mpi_rank()
    if runtime_rank == 0:
        print_result(model, input_text, output_text, args)

# TODO: raise error if VILA mode 1 with C++ runtime
