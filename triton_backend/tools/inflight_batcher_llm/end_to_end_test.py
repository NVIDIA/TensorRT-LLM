#!/usr/bin/python

import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import ast
import json
from datetime import datetime
from functools import partial

import numpy as np
from utils import utils


def callback(user_data, start_time, result, error):
    user_data._completed_requests.put((result, error))
    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    user_data._latencies.append(latency)


def verify_logits(expected_logits, input_logits, rtol=1e-02, atol=1e-02):
    torch.cuda.synchronize()
    result = np.allclose(expected_logits, input_logits, rtol, atol)
    if not result:
        ndiff = 0
        a = expected_logits.reshape(-1)
        b = input_logits.reshape(-1)
        assert a.size == b.size
        for i in range(a.size):
            if a[i] != b[i]:
                ndiff += 1
                print(f"Expect value: {a[i]}, output value: {b[i]}")
                if ndiff > 20:
                    break
    return result


# Helper function to add parameter tensors
def prepare_tensor(name, value, dtype, protocol, batch_size=1):
    if value is not None:
        shape = np.array(
            [[value]], dtype=dtype) if batch_size > 1 else np.array([value],
                                                                    dtype=dtype)
        return utils.prepare_tensor(name, shape, protocol)
    return None


def async_stream_infer(client, model_name, inputs, outputs, protocol, user_data,
                       request_id, use_llmapi):
    assert use_llmapi, "Streaming is only supported for LLMAPI model"
    assert protocol == "grpc", "Streaming is only supported for gRPC protocol"
    client.start_stream(callback=partial(callback, user_data, datetime.now()))
    client.async_stream_infer(model_name,
                              inputs,
                              outputs=outputs,
                              request_id=str(request_id))
    client.stop_stream()


def test_functionality_llmapi(
        client,
        model_name,
        prompts,
        batch_size=1,  # TODO: [JIRA-4496] support batching in llmapi backend and add tests here.
        streaming=False,
        sampling_params=None,
        output_config=None):
    """Test basic model functionality with different prompts."""
    print(f"[INFO] Start testing on {len(prompts)} prompts.")

    results = []
    user_data = utils.UserData() if streaming else None

    for i, prompt in enumerate(prompts):
        inputs = []
        # Prepare text_input
        input_data = np.array(
            [prompt], dtype=object
        )  ## TODO: [JIRA-4496] support batching in llmapi backend and add tests here.
        inputs.append(
            utils.prepare_tensor("text_input", input_data, FLAGS.protocol))

        if streaming:
            inputs.append(
                utils.prepare_tensor("streaming", np.array([True], dtype=bool),
                                     FLAGS.protocol))

        # Convert sampling_params to tensors
        if sampling_params is not None:
            for param_name, param_value in sampling_params.items():
                inputs.append(
                    prepare_tensor("sampling_param_" + param_name, param_value,
                                   type(param_value), FLAGS.protocol))

        if output_config is not None:
            for name, value in output_config.items():
                inputs.append(prepare_tensor(name, value, bool, FLAGS.protocol))

        return_finish_reason = output_config[
            "return_finish_reason"] if output_config and "return_finish_reason" in output_config else False
        return_stop_reason = output_config[
            "return_stop_reason"] if output_config and "return_stop_reason" in output_config else False

        # Only include needed outputs
        outputs = utils.prepare_outputs(
            FLAGS.protocol,
            return_finish_reason=return_finish_reason,
            return_stop_reason=return_stop_reason)

        try:
            if streaming:
                assert user_data is not None
                # async_stream_infer(client, model_name, inputs, outputs, protocol, user_data, request_id, use_llmapi)
                async_stream_infer(client, model_name, inputs, outputs,
                                   FLAGS.protocol, user_data, i, True)
            else:
                result = client.infer(model_name, inputs, request_id=str(i))
                results.append(result)
        except Exception as e:
            print(f"[Functionality test] Failed to infer with error: {e}")
            exit(1)

    if streaming:
        results = utils.get_grpc_results(user_data, len(prompts))

    for result in results:
        text_output = result.as_numpy("text_output")[0].decode("utf-8")
        assert text_output, "Text output should not be empty."
        if FLAGS.verbose:
            print(f"Text output: {text_output}", flush=True)

        if return_finish_reason:
            finish_reason = result.as_numpy("finish_reason")
            assert finish_reason, "Finish reason should not be empty."
            if FLAGS.verbose:
                print(f"Finish reason: {finish_reason}")

        if return_stop_reason:
            stop_reason = result.as_numpy("stop_reason")
            assert stop_reason, "Stop reason should not be empty."
            if FLAGS.verbose:
                print(f"Stop reason: {stop_reason}")

    print("[INFO] Functionality test succeeded.")


def test_functionality_ifb(client,
                           prompts,
                           output_lens,
                           vocabSizePadded=50257,
                           return_log_probs=False,
                           return_context_logits=False,
                           return_generation_logits=False,
                           test_bls=False):
    print(f"[INFO] Start testing on {len(prompts)} prompts.")
    for i, prompt in enumerate(prompts):

        # 1. Ensemble models manually: preprocessing -> tensorrt_llm -> postprocessing
        model_name = 'preprocessing'
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("QUERY", input0_data, FLAGS.protocol),
            utils.prepare_tensor("BAD_WORDS_DICT", bad_words_list,
                                 FLAGS.protocol),
            utils.prepare_tensor("STOP_WORDS_DICT", stop_words_list,
                                 FLAGS.protocol),
            utils.prepare_tensor("REQUEST_OUTPUT_LEN", output0_len,
                                 FLAGS.protocol),
        ]
        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("INPUT_ID")
        output1 = result.as_numpy("REQUEST_INPUT_LEN")
        output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
        decoder_input_id = result.as_numpy("DECODER_INPUT_ID")
        output_end_id = result.as_numpy("OUT_END_ID")
        output_pad_id = result.as_numpy("OUT_PAD_ID")
        inputIds = output0  # Use to check context logits shape

        model_name = "tensorrt_llm"
        inputs = [
            utils.prepare_tensor("input_ids", output0, FLAGS.protocol),
            utils.prepare_tensor("decoder_input_ids", decoder_input_id,
                                 FLAGS.protocol),
            utils.prepare_tensor("input_lengths", output1, FLAGS.protocol),
            utils.prepare_tensor("request_output_len", output2, FLAGS.protocol),
            utils.prepare_tensor("end_id", output_end_id, FLAGS.protocol),
            utils.prepare_tensor("pad_id", output_pad_id, FLAGS.protocol),
        ]
        if return_log_probs:
            return_log_probs_flag = np.array([[True]], dtype=bool)
            inputs += [
                utils.prepare_tensor("return_log_probs", return_log_probs_flag,
                                     FLAGS.protocol),
            ]

        if return_context_logits:
            return_context_logits_flag = np.array([[True]], dtype=bool)
            inputs += [
                utils.prepare_tensor("return_context_logits",
                                     return_context_logits_flag,
                                     FLAGS.protocol),
            ]
        if return_generation_logits:
            return_generation_logits_flag = np.array([[True]], dtype=bool)
            inputs += [
                utils.prepare_tensor("return_generation_logits",
                                     return_generation_logits_flag,
                                     FLAGS.protocol),
            ]

        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("output_ids").astype(np.int32)
        seq_lengths = result.as_numpy("sequence_length")
        if return_log_probs:
            cum_log_probs = result.as_numpy("cum_log_probs").astype(np.float32)
            output_log_probs = result.as_numpy("output_log_probs").astype(
                np.float32)
        if return_context_logits:
            context_logits = result.as_numpy("context_logits").astype(
                np.float32)
            print(f"context_logits.shape: {context_logits.shape}")
        if return_generation_logits:
            generation_logits = result.as_numpy("generation_logits").astype(
                np.float32)
            print(f"generation_logits.shape: {generation_logits.shape}")

        model_name = "postprocessing"
        inputs = [
            utils.prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol),
            utils.prepare_tensor("SEQUENCE_LENGTH", seq_lengths,
                                 FLAGS.protocol),
        ]
        inputs[0].set_data_from_numpy(output0)
        inputs[1].set_data_from_numpy(seq_lengths)

        result = client.infer(model_name, inputs, request_id=str(i))
        output0 = result.as_numpy("OUTPUT")

        # 2. Use ensemble model
        model_name = "ensemble"
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
            utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
            utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("stop_words", stop_words_list, FLAGS.protocol),
        ]

        if return_log_probs:
            return_log_probs_flag = np.array([[True]], dtype=bool)
            inputs += [
                utils.prepare_tensor("return_log_probs", return_log_probs_flag,
                                     FLAGS.protocol),
            ]
        if return_context_logits:
            return_context_logits_flag = np.array([[True]], dtype=bool)
            inputs += [
                utils.prepare_tensor("return_context_logits",
                                     return_context_logits_flag,
                                     FLAGS.protocol),
            ]
        if return_generation_logits:
            return_generation_logits_flag = np.array([[True]], dtype=bool)
            inputs += [
                utils.prepare_tensor("return_generation_logits",
                                     return_generation_logits_flag,
                                     FLAGS.protocol),
            ]

        outputs = utils.prepare_outputs(FLAGS.protocol, return_log_probs,
                                        return_context_logits,
                                        return_generation_logits)

        print(outputs)

        result = client.infer(model_name,
                              inputs,
                              outputs=outputs,
                              request_id=str(i))

        # 3. Check the results between manually ensembled models and the ensemble model
        ensemble_output = result.as_numpy('text_output')

        print(f"ensemble output: {ensemble_output}")
        assert output0 == ensemble_output
        if return_log_probs:
            ensemble_cum_log_probs = result.as_numpy('cum_log_probs')
            ensemble_output_log_probs = result.as_numpy('output_log_probs')
            assert cum_log_probs == ensemble_cum_log_probs
            assert (output_log_probs == ensemble_output_log_probs).all()

        if return_context_logits:
            ensemble_context_logits = result.as_numpy('context_logits')
            assert verify_logits(context_logits, ensemble_context_logits)
            ensemble_context_logits_shape = ensemble_context_logits.shape
            assert (len(ensemble_context_logits_shape) == 3)

            # Expect shape [1, prompt_length, vocabSizePadded]
            assert (ensemble_context_logits_shape[0] == 1)  # One request
            assert (ensemble_context_logits_shape[1] == inputIds.size
                    )  # Prompt length
            assert (ensemble_context_logits_shape[2] == vocabSizePadded
                    )  # VocabSizePadded

        if return_generation_logits:
            ensemble_generation_logits = result.as_numpy('generation_logits')
            assert verify_logits(generation_logits, ensemble_generation_logits)
            ensemble_generation_logits_shape = ensemble_generation_logits.shape
            assert (len(ensemble_generation_logits_shape) == 4)

            # Expect shape [1, beam_width, output_length, vocabSizePadded]
            assert (ensemble_generation_logits_shape[0] == 1)  # One request
            assert (ensemble_generation_logits_shape[1] == 1
                    )  # Beam width (default)
            assert (ensemble_generation_logits_shape[2] == output_lens[i]
                    )  # Output length
            assert (ensemble_generation_logits_shape[3] == vocabSizePadded
                    )  # VocabSizePadded

        if test_bls:
            # 4. Use bls
            model_name = "tensorrt_llm_bls"
            input0 = [[prompt]]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.int32) * output_lens[i]
            bad_words_list = np.array([[""]], dtype=object)
            stop_words_list = np.array([[""]], dtype=object)

            inputs = [
                utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
                utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
                utils.prepare_tensor("bad_words", bad_words_list,
                                     FLAGS.protocol),
                utils.prepare_tensor("stop_words", stop_words_list,
                                     FLAGS.protocol),
            ]
            if return_context_logits:
                return_context_logits_flag = np.array([[True]], dtype=bool)
                inputs += [
                    utils.prepare_tensor("return_context_logits",
                                         return_context_logits_flag,
                                         FLAGS.protocol),
                ]
            if return_generation_logits:
                return_generation_logits_flag = np.array([[True]], dtype=bool)
                inputs += [
                    utils.prepare_tensor("return_generation_logits",
                                         return_generation_logits_flag,
                                         FLAGS.protocol),
                ]

            result = client.infer(model_name,
                                  inputs,
                                  outputs=outputs,
                                  request_id=str(i))

            # 5. Check the results between manually ensembled models and the bls model
            bls_output = result.as_numpy('text_output')
            assert output0 == bls_output

            if return_log_probs:
                result.as_numpy('cum_log_probs')
                result.as_numpy('output_log_probs')
                # Disabled due to flaky results
                #assert cum_log_probs == bls_cum_log_probs
                #assert (output_log_probs == bls_output_log_probs).all()

            if return_context_logits:
                bls_context_logits = result.as_numpy('context_logits')
                bls_context_logits_shape = bls_context_logits.shape
                # Disabled due to flaky results
                #assert verify_logits(context_logits, bls_context_logits)

                assert (len(bls_context_logits_shape) == 3)
                # Expect shape [1, prompt_length, vocabSizePadded]
                assert (bls_context_logits_shape[0] == 1)  # One request
                assert (bls_context_logits_shape[1] == inputIds.size
                        )  # Prompt length
                assert (bls_context_logits_shape[2] == vocabSizePadded
                        )  # VocabSizePadded

            if return_generation_logits:
                bls_generation_logits = result.as_numpy('generation_logits')
                # Disabled due to flaky results
                #assert verify_logits(generation_logits, bls_generation_logits)

                bls_generation_logits_shape = bls_generation_logits.shape
                assert (len(bls_generation_logits_shape) == 4)

                # Expect shape [1, beam_width, output_length, vocabSizePadded]
                assert (bls_generation_logits_shape[0] == 1)  # One request
                assert (bls_generation_logits_shape[1] == 1
                        )  # Beam width (default)
                assert (bls_generation_logits_shape[2] == output_lens[i]
                        )  # Output length
                assert (bls_generation_logits_shape[3] == vocabSizePadded
                        )  # VocabSizePadded

        if FLAGS.verbose:
            print('Response: {}'.format(result.get_response()))
            print('Output: {}'.format(ensemble_output))
    print(f"[INFO] Functionality test succeed.")


def create_inputs(prompt, output_len, FLAGS, use_llmapi=False):
    inputs = []
    if not use_llmapi:
        input0 = [[prompt]]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.int32) * output_len
        bad_words_list = np.array([[""]], dtype=object)
        stop_words_list = np.array([[""]], dtype=object)

        inputs = [
            utils.prepare_tensor("text_input", input0_data, FLAGS.protocol),
            utils.prepare_tensor("max_tokens", output0_len, FLAGS.protocol),
            utils.prepare_tensor("bad_words", bad_words_list, FLAGS.protocol),
            utils.prepare_tensor("stop_words", stop_words_list, FLAGS.protocol),
        ]

    else:
        input_data = np.array(
            [prompt], dtype=object
        )  ## TODO: [JIRA-4496] support batching in llmapi backend and add tests here.
        inputs.append(
            utils.prepare_tensor("text_input", input_data, FLAGS.protocol))
        inputs.append(
            utils.prepare_tensor("sampling_param_max_tokens",
                                 np.array([output_len], dtype=np.int32),
                                 FLAGS.protocol))

    return inputs


def test_performance(client, prompts, output_lens, FLAGS, use_llmapi=False):
    print(f"[INFO] Warm up for benchmarking.")
    if FLAGS.model_name is None:
        FLAGS.model_name = "ensemble"
    print(f"FLAGS.model_name: {FLAGS.model_name}")

    for i in range(min(10, len(prompts))):
        inputs = create_inputs(prompts[0], output_lens[0], FLAGS, use_llmapi)
        outputs = utils.prepare_outputs(FLAGS.protocol)
        warmup_user_data = utils.UserData()

        if FLAGS.streaming:
            async_stream_infer(client, FLAGS.model_name, inputs, outputs,
                               FLAGS.protocol, warmup_user_data, i, use_llmapi)
        else:
            client.infer(FLAGS.model_name,
                         inputs,
                         outputs=outputs,
                         request_id=str(i))

    print(f"[INFO] Start benchmarking on {len(prompts)} prompts.")
    latency = 0
    async_requests = []
    start_time = datetime.now()
    user_data = utils.UserData()
    for i, prompt in enumerate(prompts):
        inputs = create_inputs(prompt, output_lens[i], FLAGS, use_llmapi)
        outputs = utils.prepare_outputs(FLAGS.protocol)

        if FLAGS.streaming:
            async_stream_infer(client, FLAGS.model_name, inputs, outputs,
                               FLAGS.protocol, user_data, i, use_llmapi)
        else:
            if FLAGS.protocol == "http":
                async_requests.append(
                    client.async_infer(FLAGS.model_name,
                                       inputs,
                                       outputs=outputs,
                                       request_id=str(i)))
            elif FLAGS.protocol == "grpc":
                async_requests.append(
                    client.async_infer(FLAGS.model_name,
                                       inputs,
                                       outputs=outputs,
                                       callback=partial(callback, user_data,
                                                        datetime.now()),
                                       request_id=str(i)))

    if FLAGS.protocol == "http":
        utils.get_http_results(async_requests)
    elif FLAGS.protocol == "grpc":
        utils.get_grpc_results(user_data, len(prompts))
    else:
        raise RuntimeError("Invalid protocol")

    stop_time = datetime.now()
    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    print(f"[INFO] Total Latency: {latency} ms")
    if FLAGS.protocol == "grpc":
        request_latencies = 0.0
        for latency in user_data._latencies:
            request_latencies += latency
        print(f"[INFO] Total request latencies: {request_latencies} ms")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        choices=['http', 'grpc'],
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('-c',
                        '--concurrency',
                        type=int,
                        default=128,
                        required=False,
                        help='Specify concurrency')
    parser.add_argument('--max-input-len',
                        type=int,
                        required=False,
                        help='Specify max input length')

    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')

    parser.add_argument('--return-log-probs',
                        action="store_true",
                        default=False,
                        help='Return log probs.')

    parser.add_argument('--return-context-logits',
                        action="store_true",
                        default=False,
                        help='Return context logits.')

    parser.add_argument('--return-generation-logits',
                        action="store_true",
                        default=False,
                        help='Return generation logits.')

    parser.add_argument('--test-bls',
                        action="store_true",
                        default=False,
                        help="test BLS model")
    parser.add_argument('--test-llmapi',
                        action="store_true",
                        default=False,
                        help="test LLMAPI model")
    parser.add_argument('--model-name',
                        type=str,
                        required=False,
                        help="model name")
    parser.add_argument('--streaming',
                        action="store_true",
                        default=False,
                        help="streaming")
    parser.add_argument('--output-config',
                        type=ast.literal_eval,
                        help='Output config dictionary')
    parser.add_argument('--sampling-params',
                        type=ast.literal_eval,
                        help='Sampling parameter dictionary')

    FLAGS = parser.parse_args()
    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    try:
        client = utils.create_inference_server_client(
            FLAGS.protocol,
            FLAGS.url,
            concurrency=FLAGS.concurrency,
            verbose=FLAGS.verbose)
    except Exception as e:
        print("Encountered error: " + str(e))
        sys.exit(1)

    prompts = []
    output_lens = []
    with open(FLAGS.dataset, 'r') as f:
        data_dict = json.load(f)
        for req in data_dict:
            prompt = req['input'] + ' ' + req['instruction']
            output = req['output']
            # 1.3 is a magic number that converts number of words to number of tokens
            if int(len(prompt.split(' ')) / 1.3) > FLAGS.max_input_len:
                continue
            prompts.append(prompt)
            # 1.3 is a magic number that converts number of words to number of tokens
            output_lens.append(int(len(output.split(' ')) * 1.3))

    vocabSizePadded = 50257  # gpt
    # Parse llmapi specific arguments
    if FLAGS.test_llmapi:
        assert FLAGS.model_name is not None, "model_name is required for llmapi tests"
        test_functionality_llmapi(client,
                                  FLAGS.model_name,
                                  prompts,
                                  streaming=FLAGS.streaming,
                                  sampling_params=FLAGS.sampling_params,
                                  output_config=FLAGS.output_config)
        test_performance(client, prompts, output_lens, FLAGS, use_llmapi=True)
    else:
        test_functionality_ifb(client, prompts, output_lens, vocabSizePadded,
                               FLAGS.return_log_probs,
                               FLAGS.return_context_logits,
                               FLAGS.return_generation_logits, FLAGS.test_bls)
        test_performance(client, prompts, output_lens, FLAGS, use_llmapi=False)
