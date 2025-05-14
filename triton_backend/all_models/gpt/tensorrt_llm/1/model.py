import json

import torch
import triton_python_backend_utils as pb_utils
from torch import from_numpy

from tensorrt_llm.runtime import ModelRunner, SamplingConfig


def mpi_comm():
    from mpi4py import MPI
    return MPI.COMM_WORLD


def mpi_rank():
    return mpi_comm().Get_rank()


def get_engine_name(model, dtype, tp_size, rank):
    return '{}_{}_tp{}_rank{}.engine'.format(model, dtype, tp_size, rank)


def get_input_tensor_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> PyTorch tensor
        return from_numpy(tensor.as_numpy())
    else:
        return tensor


def get_input_scalar_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is not None:
        # Triton tensor -> numpy tensor -> first scalar
        tensor = tensor.as_numpy()
        return tensor.reshape((tensor.size, ))[0]
    else:
        return tensor


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        model_config = json.loads(args['model_config'])
        engine_dir = model_config['parameters']['engine_dir']['string_value']
        self.comm = mpi_comm()
        self.rank = mpi_rank()
        self.runner = ModelRunner.from_dir(engine_dir=engine_dir,
                                           rank=self.rank)
        if self.rank != 0:
            while (True):
                self.execute([None])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            inputs = {}
            if self.rank == 0:
                inputs['input_ids'] = get_input_tensor_by_name(
                    request, 'input_ids')
                inputs['input_lengths'] = get_input_tensor_by_name(
                    request, 'input_lengths')
                inputs['request_output_len'] = get_input_scalar_by_name(
                    request, 'request_output_len')
                inputs['end_id'] = get_input_scalar_by_name(request, 'end_id')
                inputs['pad_id'] = get_input_scalar_by_name(request, 'pad_id')
                inputs['beam_width'] = get_input_scalar_by_name(
                    request, 'beam_width')
                inputs['temperature'] = get_input_scalar_by_name(
                    request, 'temperature')
                inputs['runtime_top_k'] = get_input_scalar_by_name(
                    request, 'runtime_top_k')
                inputs['runtime_top_p'] = get_input_scalar_by_name(
                    request, 'runtime_top_p')
                inputs['len_penalty'] = get_input_scalar_by_name(
                    request, 'len_penalty')
                inputs['repetition_penalty'] = get_input_scalar_by_name(
                    request, 'repetition_penalty')
                inputs['min_tokens'] = get_input_scalar_by_name(
                    request, 'min_tokens')
                inputs['presence_penalty'] = get_input_scalar_by_name(
                    request, 'presence_penalty')
                inputs['frequency_penalty'] = get_input_scalar_by_name(
                    request, 'frequency_penalty')
                inputs['seed'] = get_input_scalar_by_name(request, 'seed')
                inputs['output_log_probs'] = get_input_scalar_by_name(
                    request, 'output_log_probs')

            # Broadcast requests to other clients
            inputs = self.comm.bcast(inputs, root=0)
            input_ids = inputs['input_ids'].cuda()
            end_id = inputs['end_id']
            pad_id = inputs['pad_id']

            sampling_config = SamplingConfig(end_id=end_id, pad_id=pad_id)
            if inputs['beam_width'] is not None:
                sampling_config.num_beams = inputs['beam_width']
            if inputs['temperature'] is not None:
                sampling_config.temperature = inputs['temperature']
            if inputs['runtime_top_k'] is not None:
                sampling_config.top_k = inputs['runtime_top_k']
            if inputs['runtime_top_p'] is not None:
                sampling_config.top_p = inputs['runtime_top_p']
            if inputs['len_penalty'] is not None:
                sampling_config.length_penalty = inputs['len_penalty']
            if inputs['repetition_penalty'] is not None:
                sampling_config.repetition_penalty = inputs[
                    'repetition_penalty']
            if inputs['min_tokens'] is not None:
                sampling_config.min_tokens = inputs['min_tokens']
            if inputs['presence_penalty'] is not None:
                sampling_config.presence_penalty = inputs['presence_penalty']
            if inputs['frequency_penalty'] is not None:
                sampling_config.frequency_penalty = inputs['frequency_penalty']
            sampling_config.seed = inputs['seed']
            sampling_config.output_log_probs = inputs['output_log_probs']
            sampling_config.return_dict = True

            outputs = self.runner.generate(input_ids, None, sampling_config)
            output_ids = outputs["output_ids"]

            if self.rank == 0:
                # Create output tensors. You need pb_utils.Tensor
                # objects to create pb_utils.InferenceResponse.
                torch.cuda.synchronize()
                output_tensors = [
                    pb_utils.Tensor("output_ids",
                                    output_ids.cpu().numpy())
                ]

                if sampling_config.output_log_probs:
                    # [max_seq_len, batch_size, num_beams] -> [batch_size, max_seq_len, num_beams]
                    log_probs = self.runner.session.log_probs_tiled.transpose(
                        0, 1).cpu().numpy()
                    output_tensors.append(
                        pb_utils.Tensor("log_probs", log_probs))

                # Create InferenceResponse. You can set an error here in case
                # there was a problem with handling this inference request.
                # Below is an example of how you can set errors in inference
                # response:
                #
                # pb_utils.InferenceResponse(
                #    output_tensors=..., TritonError("An error occurred"))

                inference_response = pb_utils.InferenceResponse(output_tensors)
            else:
                inference_response = pb_utils.InferenceResponse([])
            responses.append(inference_response)

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        return
