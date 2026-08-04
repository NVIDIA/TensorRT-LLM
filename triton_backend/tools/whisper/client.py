#!/usr/bin/python
"""
This script supports to load audio files and sends it to the server
for decoding, in parallel.

Usage:
# For offlien whisper server
python3 client.py \
    --server-addr localhost \
    --model-name whisper_bls \
    --num-tasks $num_task \
    --whisper-prompt "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>" \
    --audio-path $audio_path
"""

import argparse
import asyncio
import queue
import sys
import time
import types

import numpy as np
import soundfile
import tritonclient
import tritonclient.grpc.aio as grpcclient
from tritonclient.grpc import InferenceServerException
from tritonclient.utils import np_to_triton_dtype


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--server-addr",
        type=str,
        default="localhost",
        help="Address of the server",
    )

    parser.add_argument(
        "--server-port",
        type=int,
        default=8001,
        help="Grpc port of the triton server, default is 8001",
    )

    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help=
        "Path to a single audio file. It can't be specified at the same time with --manifest-dir",
    )

    parser.add_argument(
        "--whisper-prompt",
        type=str,
        default="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
        help=
        "e.g. <|startofprev|>My hot words<|startoftranscript|><|en|><|transcribe|><|notimestamps|>, please check https://arxiv.org/pdf/2305.11095.pdf also.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="whisper_bls",
        choices=[
            "whisper_bls",
            "whisper_bls_async",
            "whisper",
        ],
        help="triton model_repo module name to request",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        default=False,
        help="""True for decopuled mode, False for synchronous mode""",
    )

    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of parallel tasks",
    )

    return parser.parse_args()


def load_audio(wav_path):
    waveform, sample_rate = soundfile.read(wav_path)
    assert sample_rate == 16000, f"Only support 16k sample rate, but got {sample_rate}"
    return waveform, sample_rate


async def send_whisper(
    dps: list,
    name: str,
    triton_client: tritonclient.grpc.aio.InferenceServerClient,
    protocol_client: types.ModuleType,
    log_interval: int,
    model_name: str,
    padding_duration: int = 10,
    whisper_prompt:
    str = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
    streaming: bool = False,
):
    total_duration = 0.0
    results = []
    latency_data = []
    task_id = int(name[5:])
    for i, dp in enumerate(dps):
        if i % log_interval == 0:
            print(f"{name}: {i}/{len(dps)}")

        waveform, sample_rate = load_audio(dp["audio_filepath"])
        duration = int(len(waveform) / sample_rate)

        # padding to nearest 10 seconds
        samples = np.zeros(
            (
                1,
                padding_duration * sample_rate *
                ((duration // padding_duration) + 1),
            ),
            dtype=np.float32,
        )

        samples[0, :len(waveform)] = waveform

        lengths = np.array([[len(waveform)]], dtype=np.int32)

        # Prepare inputs and outputs
        inputs = [
            protocol_client.InferInput("WAV", samples.shape,
                                       np_to_triton_dtype(samples.dtype)),
            protocol_client.InferInput("WAV_LENS", lengths.shape,
                                       np_to_triton_dtype(lengths.dtype)),
            protocol_client.InferInput("TEXT_PREFIX", [1, 1], "BYTES"),
        ]
        inputs[0].set_data_from_numpy(samples)
        inputs[1].set_data_from_numpy(lengths)
        input_data_numpy = np.array([whisper_prompt], dtype=object)
        input_data_numpy = input_data_numpy.reshape((1, 1))
        inputs[2].set_data_from_numpy(input_data_numpy)
        outputs = [protocol_client.InferRequestedOutput("TRANSCRIPTS")]

        # Send request
        sequence_id = 100000000 + i + task_id * 10
        start = time.time()
        if streaming:
            user_data = UserData()

            async def async_request_iterator():
                yield {
                    "model_name": model_name,
                    "inputs": inputs,
                    "outputs": outputs,
                    "request_id": str(sequence_id)
                }

            try:
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=async_request_iterator(),
                    stream_timeout=None,
                )
                async for response in response_iterator:
                    result, error = response
                    if error:
                        print(error)
                        user_data._completed_requests.put(error)
                    else:
                        user_data._completed_requests.put(result)
            except InferenceServerException as error:
                print(error)
                sys.exit(1)
            results = []
            while True:
                try:
                    data_item = user_data._completed_requests.get(block=False)
                    if type(data_item) == InferenceServerException:
                        sys.exit(1)
                    else:

                        decoding_results = data_item.as_numpy("TRANSCRIPTS")[0]
                        if type(decoding_results) == np.ndarray:
                            decoding_results = b" ".join(
                                decoding_results).decode("utf-8")
                        else:
                            decoding_results = decoding_results.decode("utf-8")
                        results.append(decoding_results)
                except Exception:
                    break
            decoding_results = results[-1]
        else:
            response = await triton_client.infer(model_name,
                                                 inputs,
                                                 request_id=str(sequence_id),
                                                 outputs=outputs)
            decoding_results = response.as_numpy("TRANSCRIPTS")[0]
            if type(decoding_results) == np.ndarray:
                decoding_results = b" ".join(decoding_results).decode("utf-8")
            else:
                decoding_results = decoding_results.decode("utf-8")
        end = time.time() - start
        latency_data.append((end, duration))
        total_duration += duration
        results.append((
            dp["id"],
            dp["text"].split(),
            decoding_results.split(),
        ))
        print(results[-1])

    return total_duration, results, latency_data


async def main():
    args = get_args()
    dps_list = [[{
        "audio_filepath": args.audio_path,
        "text": "foo",
        "id": 0,
    }]] * args.num_tasks

    url = f"{args.server_addr}:{args.server_port}"

    triton_client = grpcclient.InferenceServerClient(url=url, verbose=False)
    protocol_client = grpcclient

    tasks = []
    start_time = time.time()
    for i in range(args.num_tasks):
        task = asyncio.create_task(
            send_whisper(
                dps=dps_list[i],
                name=f"task-{i}",
                triton_client=triton_client,
                protocol_client=protocol_client,
                log_interval=1,
                model_name=args.model_name,
                whisper_prompt=args.whisper_prompt,
                streaming=args.streaming,
            ))
        tasks.append(task)

    answer_list = await asyncio.gather(*tasks)

    end_time = time.time()
    elapsed = end_time - start_time

    results = []
    total_duration = 0.0
    latency_data = []
    for answer in answer_list:
        total_duration += answer[0]
        results += answer[1]
        latency_data += answer[2]

    rtf = elapsed / total_duration

    s = f"RTF: {rtf:.4f}\n"
    s += f"total_duration: {total_duration:.3f} seconds\n"
    s += f"({total_duration/3600:.2f} hours)\n"
    s += f"processing time: {elapsed:.3f} seconds " f"({elapsed/3600:.2f} hours)\n"

    latency_list = [chunk_end for (chunk_end, chunk_duration) in latency_data]
    latency_ms = sum(latency_list) / float(len(latency_list)) * 1000.0
    latency_variance = np.var(latency_list, dtype=np.float64) * 1000.0
    s += f"latency_variance: {latency_variance:.2f}\n"
    s += f"latency_50_percentile_ms: {np.percentile(latency_list, 50) * 1000.0:.2f}\n"
    s += f"latency_90_percentile_ms: {np.percentile(latency_list, 90) * 1000.0:.2f}\n"
    s += f"latency_95_percentile_ms: {np.percentile(latency_list, 95) * 1000.0:.2f}\n"
    s += f"latency_99_percentile_ms: {np.percentile(latency_list, 99) * 1000.0:.2f}\n"
    s += f"average_latency_ms: {latency_ms:.2f}\n"

    print(s)


if __name__ == "__main__":
    asyncio.run(main())
