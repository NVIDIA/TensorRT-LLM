import json

import triton_python_backend_utils as pb_utils


def read_parameter_as_type(value, name, pytype=str):
    if value == "":
        return None
    if value.startswith("${") and value.endswith("}"):
        return None
    if pytype is bool:
        return value.lower() in ["1", "true"]
    try:
        result = pytype(value)
        return result
    except:
        pb_utils.Logger.log_warning(
            f"Could not read parameter '{name}' with value '{value}', will use default."
        )
        return None


def get_parameter(model_config, name, pytype=str):
    if name not in model_config['parameters']:
        return None
    return read_parameter_as_type(
        model_config['parameters'][name]['string_value'], name, pytype)


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
        self.context_model_name = get_parameter(model_config,
                                                "context_model_name")
        self.generation_model_name = get_parameter(model_config,
                                                   "generation_model_name")
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config)

    def create_context_request(self, request):
        inputs = request.inputs()
        triton_request = pb_utils.InferenceRequest(
            model_name=self.context_model_name,
            inputs=inputs,
            parameters={"request_type": "context_only"},
            requested_output_names=[])
        return triton_request

    def create_generation_request(self, request, context_response):
        inputs = request.inputs()
        context_phase_params = pb_utils.get_output_tensor_by_name(
            context_response, "context_phase_params")
        if context_phase_params is None:
            raise pb_utils.TritonModelException(
                "Context response must have an output named context phase params"
            )
        inputs.append(context_phase_params)
        triton_request = pb_utils.InferenceRequest(
            model_name=self.generation_model_name,
            inputs=inputs,
            parameters={"request_type": "generation_only"},
            requested_output_names=[])
        return triton_request

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
        for request in requests:
            context_request = self.create_context_request(request)
            context_responses = context_request.exec(decoupled=self.decoupled)
            if self.decoupled:
                context_responses = list(context_responses)
                assert len(
                    context_responses) == 1, "Expected 1 context response"

            if self.decoupled:
                context_response = context_responses[0]
            else:
                context_response = context_responses
            if context_response.has_error():
                raise pb_utils.TritonModelException(
                    f"Context model {self.context_model_name} failed with error: {context_response.error().message()}"
                )
            generation_request = self.create_generation_request(
                request, context_response)

            # TODO(itabrizian): Send the context response to reduce TTFT in decoupled case.
            # It requires adding the generated token to the generation request
            # to avoid sending the first token multiple times.
            responses = generation_request.exec(decoupled=self.decoupled)

            if self.decoupled:
                for response in responses:
                    if response.has_error():
                        raise pb_utils.TritonModelException(
                            f"Generation model {self.generation_model_name} failed with error: {response.error().message()}"
                        )
                    request.get_response_sender().send(response)

                request.get_response_sender().send(
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
            else:
                request.get_response_sender().send(
                    responses,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
