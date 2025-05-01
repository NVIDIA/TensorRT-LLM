#!/usr/bin/python

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import queue
import sys

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, np_to_triton_dtype


def prepare_tensor(name, input):
    t = grpcclient.InferInput(name, input.shape,
                              np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)
        output = result.as_numpy('text_output')
        print(output, flush=True)


def get_preprocessor_inputs(prompt, output_len, bad_words, stop_words, end_id,
                            pad_id):
    input0 = [[prompt]]
    input0_data = np.array(input0).astype(object)
    output0_len = np.ones_like(input0).astype(np.int32) * output_len

    preprocessor_inputs = [
        prepare_tensor("QUERY", input0_data),
        prepare_tensor("REQUEST_OUTPUT_LEN", output0_len),
    ]

    if bad_words:
        bad_words_list = np.array([bad_words], dtype=object)
        preprocessor_inputs += [
            prepare_tensor("BAD_WORDS_DICT", bad_words_list)
        ]

    if stop_words:
        stop_words_list = np.array([stop_words], dtype=object)
        preprocessor_inputs += [
            prepare_tensor("STOP_WORDS_DICT", stop_words_list)
        ]

    if end_id:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        preprocessor_inputs += [prepare_tensor("END_ID", end_id_data)]

    if pad_id:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        preprocessor_inputs += [prepare_tensor("PAD_ID", pad_id_data)]

    return preprocessor_inputs


def extract_preprocessor_outputs(result):

    input_ids = np.squeeze(result.as_numpy("INPUT_ID").astype(np.int32), axis=0)
    bad_words_ids = result.as_numpy("BAD_WORDS_IDS").astype(np.int32)
    stop_words_ids = result.as_numpy("STOP_WORDS_IDS").astype(np.int32)
    end_id = result.as_numpy("OUT_END_ID").astype(np.int32)[0][0]
    pad_id = result.as_numpy("OUT_PAD_ID").astype(np.int32)[0][0]

    return input_ids, bad_words_ids, stop_words_ids, end_id, pad_id


def get_trtllm_inputs(input_ids,
                      input_length,
                      request_output_len,
                      draft_tokens,
                      beam_width,
                      temperature,
                      repetition_penalty,
                      presence_penalty,
                      frequency_penalty,
                      bad_words_ids,
                      stop_words_ids,
                      end_id,
                      pad_id,
                      return_draft_model_draft_logits=False,
                      return_target_model_accepted_token_logits=False):

    # These two flags correspond to the settings of draft model and target model respectively,
    # and only one of them can be true at a time.
    assert not (return_draft_model_draft_logits
                and return_target_model_accepted_token_logits)

    # input_ids is expected to have shape [input_length]
    # Add batch dimension of 1
    input_ids_data = np.expand_dims(input_ids, axis=0)
    inputs = [
        prepare_tensor("input_ids", input_ids_data),
        prepare_tensor("input_lengths",
                       np.array([[input_length]], dtype=np.int32)),
        prepare_tensor("request_output_len",
                       np.array([[request_output_len]], dtype=np.int32)),
        prepare_tensor("bad_words_list", bad_words_ids),
        prepare_tensor("stop_words_list", stop_words_ids),
        prepare_tensor("beam_width", np.array([[beam_width]], dtype=np.int32)),
        prepare_tensor("temperature", np.array([[temperature]],
                                               dtype=np.float32)),
    ]

    if draft_tokens is not None:
        draft_tokens_data = np.array([draft_tokens], dtype=np.int32)
        inputs.append(prepare_tensor("draft_input_ids", draft_tokens_data))

    if repetition_penalty is not None:
        repetition_penalty_data = np.array([[repetition_penalty]],
                                           dtype=np.float32)
        inputs.append(
            prepare_tensor("repetition_penalty", repetition_penalty_data))

    if presence_penalty is not None:
        presence_penalty_data = np.array([[presence_penalty]], dtype=np.float32)
        inputs.append(prepare_tensor("presence_penalty", presence_penalty_data))

    if frequency_penalty is not None:
        frequency_penalty_data = np.array([[frequency_penalty]],
                                          dtype=np.float32)
        inputs.append(
            prepare_tensor("frequency_penalty", frequency_penalty_data))

    if end_id is not None:
        end_id_data = np.array([[end_id]], dtype=np.int32)
        inputs.append(prepare_tensor("end_id", end_id_data))

    if pad_id is not None:
        pad_id_data = np.array([[pad_id]], dtype=np.int32)
        inputs.append(prepare_tensor("pad_id", pad_id_data))

    if return_draft_model_draft_logits:
        return_draft_model_draft_logits_data = np.array(
            [[return_draft_model_draft_logits]], dtype=bool)
        inputs.append(
            prepare_tensor("return_generation_logits",
                           return_draft_model_draft_logits_data))

    if return_target_model_accepted_token_logits:
        return_target_model_accepted_token_logits_data = np.array(
            [[return_target_model_accepted_token_logits]], dtype=bool)
        inputs.append(
            prepare_tensor("return_generation_logits",
                           return_target_model_accepted_token_logits_data))

    return inputs


def check_result(result, model_name):
    if type(result) == InferenceServerException:
        print(
            f"Received an error from server while calling {model_name}: {result}"
        )


def extract_trtllm_outputs(result):
    # Get batch 0, beam 0 output_ids
    output_ids = np.squeeze(result.as_numpy("output_ids").astype(np.int32),
                            axis=(0, 1))
    sequence_length_data = result.as_numpy("sequence_length").astype(np.int32)
    assert sequence_length_data.shape[0] == 1
    assert sequence_length_data.shape[1] == 1
    sequence_length = sequence_length_data[0, 0]
    cum_log_probs = result.as_numpy("cum_log_probs")
    if cum_log_probs is not None:
        cum_log_probs = cum_log_probs.astype(np.float32)
    output_log_probs = result.as_numpy("output_log_probs")
    if output_log_probs is not None:
        output_log_probs = output_log_probs.astype(np.float32)
    context_logits = result.as_numpy("context_logits")
    if context_logits is not None:
        context_logits.astype(np.float32)
    generation_logits = result.as_numpy("generation_logits")
    if generation_logits is not None:
        generation_logits = generation_logits.astype(np.float32)
    return output_ids, sequence_length, cum_log_probs, output_log_probs, context_logits, generation_logits


def get_postprocessor_inputs(output_ids, cum_log_probs, output_log_probs,
                             context_logits, generation_logits):
    output_ids_data = np.expand_dims(output_ids, axis=(0, 1))
    inputs = [
        prepare_tensor("TOKENS_BATCH", output_ids_data),
        prepare_tensor("SEQUENCE_LENGTH",
                       np.array([[len(output_ids)]], dtype=np.int32)),
    ]

    return inputs


def encountered_stop_words(input_ids, stop_words_ids):
    for stop_word_ids in stop_words_ids:
        if np.array_equal(input_ids[-len(stop_word_ids):], stop_word_ids):
            return True
    return False


def run_speculative_inference(
        client_draft, client_target, prompt, output_len, in_num_draft_tokens,
        request_id, repetition_penalty, presence_penalty, frequency_penalty,
        temperature, stop_words, bad_words, end_id, pad_id, beam_width,
        preprocessor_model_name, draft_tensorrt_llm_model_name,
        target_tensorrt_llm_model_name, postprocessor_model_name,
        return_draft_model_draft_logits,
        return_target_model_accepted_token_logits, verbose):

    # Call the preprocessor
    preprocessor_inputs = get_preprocessor_inputs(prompt, output_len, bad_words,
                                                  stop_words, end_id, pad_id)
    preprocessor_result = client_draft.infer(preprocessor_model_name,
                                             preprocessor_inputs,
                                             request_id=request_id)
    check_result(preprocessor_result, preprocessor_model_name)
    prompt_input_ids, bad_words_ids, stop_words_ids, end_id, pad_id = extract_preprocessor_outputs(
        preprocessor_result)

    input_ids = prompt_input_ids
    last_input_ids = None
    draft_output_ids = None

    while True:

        num_draft_tokens = min(
            in_num_draft_tokens,
            len(prompt_input_ids) + output_len - len(input_ids) - 1)

        if num_draft_tokens > 0:

            if verbose:
                print("Draft model input ids:")
                print(input_ids.tolist())

            #Generate up to num_draft_tokens with draft model
            draft_inputs = get_trtllm_inputs(
                input_ids,
                len(input_ids),
                num_draft_tokens,
                None,
                beam_width,
                temperature,
                repetition_penalty,
                presence_penalty,
                frequency_penalty,
                bad_words_ids,
                stop_words_ids,
                end_id,
                pad_id,
                return_draft_model_draft_logits=return_draft_model_draft_logits)

            draft_result = client_draft.infer(draft_tensorrt_llm_model_name,
                                              draft_inputs,
                                              request_id=request_id)
            check_result(draft_result, draft_tensorrt_llm_model_name)
            draft_output_ids, draft_seq_len, cum_log_probs, output_log_probs, context_logits, generation_logits = extract_trtllm_outputs(
                draft_result)

            if verbose:
                print("Draft model output ids:")
                print(draft_output_ids.tolist())
                print("draft_sequence_length")
                print(draft_seq_len)

            # Set the draft token and call the target model to generate up to num_draft_tokens + 1
            draft_tokens = draft_output_ids[len(input_ids):draft_seq_len]

            if verbose:
                print("draft_tokens")
                print(draft_tokens.tolist())
                if return_draft_model_draft_logits:
                    draft_model_draft_token_logits = generation_logits.squeeze(
                        0)  # [beam_width, num_draft_tokens, vocab_size]
                    print(
                        f"draft model draft tokens' logits: shape: {draft_model_draft_token_logits.shape}, value: {draft_model_draft_token_logits}"
                    )

        if verbose:
            print("Target model input ids")
            print(input_ids.tolist())

        # Generate up to len(draft_tokens) + 1 with target model
        target_inputs = get_trtllm_inputs(
            input_ids,
            len(input_ids),
            len(draft_tokens) + 1 if num_draft_tokens > 0 else 1,
            draft_tokens if num_draft_tokens > 0 else None,
            beam_width,
            temperature,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            bad_words_ids,
            stop_words_ids,
            end_id,
            pad_id,
            return_target_model_accepted_token_logits=
            return_target_model_accepted_token_logits)

        target_result = client_target.infer(target_tensorrt_llm_model_name,
                                            target_inputs,
                                            request_id=request_id)
        check_result(target_result, target_tensorrt_llm_model_name)
        target_output_ids, seq_length, cum_log_probs, output_log_probs, context_logits, generation_logits = extract_trtllm_outputs(
            target_result)

        if verbose:
            print("Target model output_ids")
            print(target_output_ids.tolist())
            print("target seq_length")
            print(seq_length)
            if return_target_model_accepted_token_logits:
                target_model_accept_token_logits = generation_logits.squeeze(
                    0).squeeze(0)  # [num_accepted_tokens, vocab_size]
                print(
                    f"target model accepted tokens' logits: shape: {target_model_accept_token_logits.shape}, value: {target_model_accept_token_logits}"
                )

        # Store the last iteration input_ids to check if EOS was encountered
        last_input_ids = input_ids
        # Update the input ids with new output_ids
        input_ids = target_output_ids

        # Evaluate criteria to stop generation loop.
        # If we've hit or exceeded the max output length, should stop
        length_stop = (len(input_ids) >= len(prompt_input_ids) + output_len)
        # If draft and target have same outputs, should stop. Normally target should return 1 more token.
        # If they are the same length, they should differ at the last token
        target_draft_equal = draft_output_ids is not None and np.array_equal(
            draft_output_ids, target_output_ids)
        # If tokens no longer change, should stop, means we have hit early stopping
        last_current_equal = np.array_equal(last_input_ids, input_ids)
        # Need to check if stop words was encountered
        hit_stop_words = encountered_stop_words(input_ids, stop_words_ids[0])

        if verbose:
            print("length_stop:", length_stop)
            print("target_draft_equal:", target_draft_equal)
            print("last_current_equal:", last_current_equal)
            print("hit_stop_words:", hit_stop_words)

        if (length_stop or target_draft_equal or last_current_equal
                or hit_stop_words):
            break

    # Call the postprocessor
    postprocessor_inputs = get_postprocessor_inputs(input_ids, cum_log_probs,
                                                    output_log_probs,
                                                    context_logits,
                                                    generation_logits)
    postprocessor_result = client_target.infer(postprocessor_model_name,
                                               postprocessor_inputs,
                                               request_id=request_id)
    check_result(postprocessor_result, postprocessor_model_name)
    output = postprocessor_result.as_numpy("OUTPUT")
    return output[0].decode("utf8")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')

    parser.add_argument('--url-target',
                        type=str,
                        required=True,
                        help='Inference server URL for the target model')

    parser.add_argument('--url-draft',
                        type=str,
                        required=False,
                        help='Inference server URL for the draft model')

    parser.add_argument(
        '--preprocessor-model-name',
        type=str,
        required=False,
        default="preprocessing",
        help='Name of the preprocessor model (should be hosted at url-draft)')

    parser.add_argument(
        '--postprocessor-model-name',
        type=str,
        required=False,
        default="postprocessing",
        help='Name of the postprocessor model (should be hosted at url-target)')

    parser.add_argument(
        '--draft-tensorrt-llm-model-name',
        type=str,
        required=False,
        default="tensorrt_llm",
        help='Name of the tensorrt_llm draft model (hosted at url-draft)')

    parser.add_argument(
        '--target-tensorrt-llm-model-name',
        type=str,
        required=False,
        default="tensorrt_llm",
        help='Name of the tensorrt_llm draft model (hosted at url-target)')

    parser.add_argument('-p',
                        '--prompt',
                        type=str,
                        required=True,
                        help='Input prompt.')

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=None,
        help="The repetition penalty value",
    )

    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )

    parser.add_argument(
        "--frequency-penalty",
        type=float,
        required=False,
        default=None,
        help="The frequency penalty value",
    )

    parser.add_argument('-o',
                        '--output-len',
                        type=int,
                        default=100,
                        required=False,
                        help='Specify output length')

    parser.add_argument(
        '--num-draft-tokens',
        type=int,
        default=5,
        required=False,
        help=
        'Specify the number of speculative tokens for the draft model to generate per lookahead.'
    )

    parser.add_argument('--end-id',
                        type=int,
                        default=None,
                        required=False,
                        help='The end if token')

    parser.add_argument('--pad-id',
                        type=int,
                        default=None,
                        required=False,
                        help='The pad if token')

    parser.add_argument('--request-id',
                        type=str,
                        default='1',
                        required=False,
                        help='The request_id for the stop request')

    parser.add_argument('--stop-words',
                        nargs='+',
                        default=[],
                        help='The stop words')

    parser.add_argument('--bad-words',
                        nargs='+',
                        default=[],
                        help='The bad words')

    parser.add_argument(
        "--return-draft-model-draft-logits",
        action="store_true",
        required=False,
        default=False,
        help=
        "Return draft model's draft tokens' logits, require to enable `gather_generation_logits` when build engine"
    )

    parser.add_argument(
        "--return-target-model-accepted-token-logits",
        action="store_true",
        required=False,
        default=False,
        help=
        "Return target model's accepted token logits, require to enable `gather_generation_logits` when build engine",
    )

    FLAGS = parser.parse_args()
    if not FLAGS.url_target:
        FLAGS.url_target = "localhost:8001"

    if not FLAGS.url_draft:
        FLAGS.url_draft = FLAGS.url_target

    try:
        client_target = grpcclient.InferenceServerClient(url=FLAGS.url_target)
        client_draft = grpcclient.InferenceServerClient(
            url=FLAGS.url_draft) if (FLAGS.url_target
                                     != FLAGS.url_draft) else client_target
    except Exception as e:
        print("client creation failed: " + str(e))
        sys.exit(1)

    if (FLAGS.beam_width > 1):
        raise Exception(
            'Beam width > 1 is not yet supported with speculative decoding')

    output_text = run_speculative_inference(
        client_draft, client_target, FLAGS.prompt, FLAGS.output_len,
        FLAGS.num_draft_tokens, FLAGS.request_id, FLAGS.repetition_penalty,
        FLAGS.presence_penalty, FLAGS.frequency_penalty, FLAGS.temperature,
        FLAGS.stop_words, FLAGS.bad_words, FLAGS.end_id, FLAGS.pad_id,
        FLAGS.beam_width, FLAGS.preprocessor_model_name,
        FLAGS.draft_tensorrt_llm_model_name,
        FLAGS.target_tensorrt_llm_model_name, FLAGS.postprocessor_model_name,
        FLAGS.return_draft_model_draft_logits,
        FLAGS.return_target_model_accepted_token_logits, FLAGS.verbose)

    # Print the final text
    print("Final text:\n", output_text)
