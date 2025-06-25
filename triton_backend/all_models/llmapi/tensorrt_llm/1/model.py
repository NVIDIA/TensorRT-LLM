# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import gc
import json
import os
import queue
import threading
from contextlib import asynccontextmanager

import numpy as np
import triton_python_backend_utils as pb_utils
import yaml
from helpers import (get_input_tensor_by_name, get_output_config_from_request,
                     get_sampling_params_from_request,
                     get_streaming_from_request)
from mpi4py.futures import MPICommExecutor
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm import LLM as PyTorchLLM
from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm._utils import global_mpi_rank, global_mpi_size
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict


def get_model_config(filename, include_keys=None, exclude_keys=None):
    engine_args_filepath = os.path.join(pb_utils.get_model_dir(), filename)
    engine_config = None
    if os.path.isfile(engine_args_filepath):
        try:
            with open(engine_args_filepath) as file:
                engine_config = yaml.safe_load(file)
        except Exception as e:
            raise pb_utils.TritonModelException(
                f"Failed to parse YAML engine config: {e}")

    assert engine_config is not None, f"'{filename}' containing TRT-LLM engine args not found in '{pb_utils.get_model_dir()}'"

    if include_keys:
        engine_config = {
            k: v
            for k, v in engine_config.items() if k in include_keys
        }
    if exclude_keys:
        engine_config = {
            k: v
            for k, v in engine_config.items() if k not in exclude_keys
        }
    return engine_config


class TritonPythonModel:

    @classmethod
    def auto_complete_config(cls, auto_complete_model_config):
        """
        Set triton_config values in model.yaml to auto_complete_model_config

        Args:
            auto_complete_model_config: Default configurations loaded from config.pbtxt

        Returns:
            auto_complete_model_config: Updated triton server configurations loading triton_config from model.yaml

        Notes:
            - This function is called when Triton server starts.
            - It combines the default configurations in config.pbtxt with the triton_config values in model.yaml
        """
        triton_config = get_model_config(os.environ.get('LLM_CONFIG_PATH',
                                                        'model.yaml'),
                                         include_keys=["triton_config"
                                                       ])["triton_config"]
        auto_complete_model_config.set_model_transaction_policy(
            dict(decoupled=bool(triton_config["decoupled"])))
        auto_complete_model_config.set_max_batch_size(
            int(triton_config["max_batch_size"]))

        return auto_complete_model_config

    def initialize(self, args):
        """
        Function allows the model to initialize any state associated with it.

        Args:
            args: triton configurations loaded from config.pbtxt and extended by auto_complete_config
        Note:
            - `initialize` is called only once when the model is being loaded.
            - Implementing `initialize` function is optional.
        """
        from tensorrt_llm.llmapi import MpiCommSession

        self.model_config = json.loads(args["model_config"])
        triton_config = get_model_config(os.environ.get('LLM_CONFIG_PATH',
                                                        'model.yaml'),
                                         include_keys=["triton_config"
                                                       ])["triton_config"]
        self.decoupled = bool(triton_config["decoupled"])
        self.params = self.model_config['parameters']
        self.logger = pb_utils.Logger

        text_output_config = pb_utils.get_output_config_by_name(
            self.model_config, "text_output")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            text_output_config["data_type"])
        if global_mpi_rank() == 0:
            # Initialize engine arguments
            self.llm_engine_args = update_llm_args_with_extra_dict(
                {},
                get_model_config(os.environ.get('LLM_CONFIG_PATH',
                                                'model.yaml'),
                                 exclude_keys=["triton_config"]),
            )
            self.logger.log_info(
                f"[trtllm] rank{global_mpi_rank()} is starting trtllm engine with args: {self.llm_engine_args}"
            )

            if global_mpi_size() > 1:
                mpi_session = MpiCommSession(comm=COMM_WORLD,
                                             n_workers=COMM_WORLD.Get_size())
                self.llm_engine_args["_mpi_session"] = mpi_session

            # Starting the TRT-LLM engine with LLM API and its event thread running the AsyncIO event loop.
            self._init_engine()

            # Starting the response thread. It allows TRT-LLM to keep making progress while
            # response sender(s) are sending responses to server frontend.
            self._response_queue = queue.Queue()
            self._response_thread = threading.Thread(target=self._response_loop)
            self._response_thread.start()
        else:
            self.logger.log_info(
                f"[trtllm] rank{global_mpi_rank()} is waiting for the leader node..."
            )
            with MPICommExecutor(COMM_WORLD) as executor:
                if executor is not None:
                    raise RuntimeError(
                        f"[trtllm] rank{COMM_WORLD.rank} should not have executor"
                    )
            return

    def _init_engine(self):
        """
        Initialize the LLM engine in a separate thread running the AsyncIO event loop.
        """
        self._llm_engine = None
        self._llm_engine_start_cv = threading.Condition()
        self._llm_engine_shutdown_event = asyncio.Event()
        self._event_thread = threading.Thread(target=asyncio.run,
                                              args=(self._run_llm_engine(), ))
        self._event_thread.start()
        with self._llm_engine_start_cv:
            while self._llm_engine is None:
                self._llm_engine_start_cv.wait()

        # The 'threading.Thread()' will not raise the exception here should the engine
        # failed to start, so the exception is passed back via the engine variable.
        if isinstance(self._llm_engine, Exception):
            e = self._llm_engine
            self.logger.log_error(f"[trtllm] Failed to start engine: {e}")
            if self._event_thread is not None:
                self._event_thread.join()
                self._event_thread = None
            raise e

    async def _run_llm_engine(self):
        """
        Run the LLM engine in an asynchronous context.
        """
        # Counter to keep track of ongoing request counts.
        self._ongoing_request_count = 0

        @asynccontextmanager
        async def async_llm_wrapper():
            # Create LLM in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            try:
                backend = self.llm_engine_args.get("backend", None)
                # Update LLM engine args with disaggregated config if present
                if backend == "pytorch":
                    llm = await loop.run_in_executor(
                        None, lambda: PyTorchLLM(**self.llm_engine_args))
                else:
                    self.llm_engine_args["pytorch_backend_config"] = None
                    llm = await loop.run_in_executor(
                        None, lambda: LLM(**self.llm_engine_args))
                yield llm
            finally:
                if 'llm' in locals():
                    # Run shutdown in a thread to avoid blocking
                    await loop.run_in_executor(None, llm.shutdown)

        try:
            async with async_llm_wrapper() as engine:
                # Capture the engine event loop and make it visible to other threads.
                self._event_loop = asyncio.get_running_loop()

                # Signal the engine is started and make it visible to other threads.
                with self._llm_engine_start_cv:
                    self._llm_engine = engine
                    self._llm_engine_start_cv.notify_all()

                # Wait for the engine shutdown signal.
                await self._llm_engine_shutdown_event.wait()

                # Wait for the ongoing requests to complete.
                while self._ongoing_request_count > 0:
                    self.logger.log_info(
                        "[trtllm] Awaiting remaining {} requests".format(
                            self._ongoing_request_count))
                    await asyncio.sleep(1)

                # Cancel all tasks in the event loop.
                for task in asyncio.all_tasks(loop=self._event_loop):
                    if task is not asyncio.current_task():
                        task.cancel()

        except Exception as e:
            # Signal and pass the exception back via the engine variable if the engine
            # failed to start. If the engine has started, re-raise the exception.
            with self._llm_engine_start_cv:
                if self._llm_engine is None:
                    self._llm_engine = e
                    self._llm_engine_start_cv.notify_all()
                    return
            raise e

        self._llm_engine = None
        self.logger.log_info("[trtllm] Shutdown complete")

    def _response_loop(self):
        """
        Helper function to process responses from the response queue when streaming is enabled.
        """
        while True:
            item = self._response_queue.get()
            # To signal shutdown a None item will be added to the queue.
            if item is None:
                break
            response_state, response, response_flag = item
            response_sender = response_state["response_sender"]
            try:
                response_sender.send(response, response_flag)
                # Stop checking for cancellation if the last response is generated.
                if not response_state["last_response_generated"]:
                    response_state[
                        "is_cancelled"] = response_sender.is_cancelled()
            except Exception as e:
                self.logger.log_error(
                    f"An error occurred while sending a response: {e}")
            finally:
                if response_flag == pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL:
                    self._ongoing_request_count -= 1

    def execute(self, requests):
        """
        Function is called by Triton server when a new request is received.

        Args:
            requests: a list of pb_utils.InferenceRequest

        Notes:
            - `execute` must be implemented in every Triton Python model.
        """
        # TODO: [JIRA-4040] Add health check here
        for request in requests:
            # TODO : [JIRA-4040] Verify Lora
            if request is not None:
                assert (
                    self._llm_engine_shutdown_event.is_set() is False
                ), "Cannot create tasks after shutdown has been requested"
                coro = self._execute_single_request(request)
                asyncio.run_coroutine_threadsafe(coro, self._event_loop)

        return None

    async def _execute_single_request(self, request):
        """
        Execute a single inference request asynchronously.
        """
        response_sender = request.get_response_sender()
        response_state = {
            "response_sender": response_sender,
            "is_cancelled": False,
            "last_response_generated":
            False,  # last response ready but not yet sent
        }
        self._ongoing_request_count += 1
        decrement_ongoing_request_count = True
        try:
            # TODO: [JIRA-4496] Implement when request contains batched prompts
            (prompt, sampling_params, streaming,
             output_config) = self._convert_request(request)
            if streaming and not self.decoupled:
                raise pb_utils.TritonModelException(
                    "Streaming is only supported in decoupled mode.")
            # Generate the response.
            response_iterator = self._llm_engine.generate_async(
                prompt, SamplingParams(**sampling_params), streaming)

            async for request_output in response_iterator:
                # TODO: [JIRA-4040] Add request cancellation check here
                # Send each response if streaming.
                if streaming:
                    response = self._create_response(
                        request_output=request_output,
                        output_config=output_config)
                    flags = 0
                    if request_output.finished:
                        response_state["last_response_generated"] = True
                        flags = pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        # with streaming, self._response_loop will decrement self._ongoing_request_count
                        decrement_ongoing_request_count = False
                    self._response_queue.put_nowait(
                        (response_state, response, flags))

            # Send the last response which contains all the outputs if not streaming.
            if not streaming:
                response_sender.send(
                    self._create_response(request_output=request_output,
                                          output_config=output_config),
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                )

        except Exception as e:
            self.logger.log_error(f"[trtllm] Error generating request: {e}")
            error = pb_utils.TritonError(f"Error generating request: {e}")
            text_output_tensor = pb_utils.Tensor(
                "text_output", np.asarray(["N/A"], dtype=self.output_dtype))
            response = pb_utils.InferenceResponse(
                output_tensors=[text_output_tensor], error=error)
            response_sender.send(
                response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            raise e

        finally:
            if decrement_ongoing_request_count:
                self._ongoing_request_count -= 1

    def _convert_request(self, request):
        """Helper function to convert the request into a prompt for LLM.generate_async

        Args:
            request: Triton Server request

        Returns:
            prompt: A LLM PromptInputs object

        Notes:
            - The current implementation only supports text_input being a 1D tensor(a single prompt).
        """
        text_input = get_input_tensor_by_name(request, 'text_input')
        if text_input is None:
            raise pb_utils.TritonModelException(
                f"text_input is missing from the request")
        if len(text_input.shape) > 1:
            raise pb_utils.TritonModelException(
                f"The current implementation only supports text_input being a 1D tensor."
            )

        prompt = text_input[0]

        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        sampling_params = get_sampling_params_from_request(request)
        output_config = get_output_config_from_request(request)
        streaming = get_streaming_from_request(request)
        return prompt, sampling_params, streaming, output_config

    def _create_response(self, request_output, output_config):
        """Process the generated request_output and create the client response.

        Args:
            request_output (tensorrt_llm.llmapi.RequestOutput): Inferred results running the LLM engine and input prompt.
            Parameters:
                request_id (int): The unique ID of the request.
                prompt (str, optional): The prompt string of the request.
                prompt_token_ids (List[int]): The token ids of the prompt.
                outputs (List[CompletionOutput]): The output sequences of the request.
                    Args:
                        index (int): The index of the output in the request.
                        text (str): The generated output text.
                        token_ids (List[int], optional): The token ids of the generated output text.
                        cumulative_logprob (float, optional): The cumulative log probability of the generated output text.
                        logprobs (List[float], optional): The log probabilities of the top probability words at each position if the logprobs are requested.
                        finish_reason (Literal['stop', 'length', 'timeout', 'cancelled'], optional): The reason why the sequence is finished.
                        stop_reason (int, str, optional): The stop string or token id that caused the completion to stop, None if the completion finished for some other reason.
                        generation_logits (torch.Tensor, optional): The logits on the generated output token ids.
                        disaggregated_params (tensorrt_llm.disaggregated_params.DisaggregatedParams, optional): Parameters needed for disaggregated serving.
                context_logits (torch.Tensor, optional): The logits on the prompt token ids.
                finished (bool): Whether the whole request is finished.

        Returns:
            pb_utils.InferenceResponse: Converted output response
                The arguments are defined in config.pbtxt output
                triton_config:output_config in model.yaml controls which output to send besides text_output
        """
        # TODO: [JIRA-4040] Check if request_output has_error and handle it
        response = []
        text_output = [
            output.text.encode("utf-8") for output in request_output.outputs
        ]

        response.append(
            pb_utils.Tensor("text_output",
                            np.asarray(text_output, dtype=self.output_dtype)))

        # Extract and add configurable output fields
        # The output_config loads related input from request
        output_fields = {
            "return_finish_reason":
            ("finish_reason", lambda output: output.finish_reason),
            "return_stop_reason":
            ("stop_reason", lambda output: output.stop_reason),
            "return_cumulative_logprob":
            ("cumulative_logprob", lambda output: output.cumulative_logprob)
        }

        for config_key, (output_name, extractor) in output_fields.items():
            if output_config[config_key]:
                tensor_data = [
                    str(extractor(output)) for output in request_output.outputs
                ]
                response.append(
                    pb_utils.Tensor(output_name,
                                    np.asarray(tensor_data, dtype=np.object_)))

        return pb_utils.InferenceResponse(output_tensors=response)

    def finalize(self):
        """
        Function is called by Triton server before exiting.

        Notes:
            - `finalize` is called only once when the model is being unloaded.
            - Implementing `finalize` function is optional.
        """
        self.logger.log_info("[trtllm] Issuing finalize to trtllm backend")
        self._event_loop.call_soon_threadsafe(
            self._llm_engine_shutdown_event.set)

        # Shutdown the event thread.
        if self._event_thread is not None:
            self._event_thread.join()
            self._event_thread = None

        # # Shutdown the response thread.
        self._response_queue.put(None)
        if self._response_thread is not None:
            self._response_thread.join()
            self._response_thread = None

        # When using parallel tensors, the stub process may not shutdown due to
        # unreleased references, so manually run the garbage collector once.
        self.logger.log_info(
            "[trtllm] Running Garbage Collector on finalize...")
        gc.collect()
        self.logger.log_info("[trtllm] Garbage Collector on finalize... done")
