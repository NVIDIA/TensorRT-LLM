# -*- coding: utf-8 -*-
import json
from typing import List

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


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
        # Parse model configs
        model_config = json.loads(args['model_config'])
        tokenizer_dir = model_config['parameters']['tokenizer_dir'][
            'string_value']

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir,
                                                       padding_side='left',
                                                       legacy=False,
                                                       trust_remote_code=True)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token,
                                            add_special_tokens=False)[0]

        # Parse model output configs and convert Triton types to numpy types
        input_names = [
            "INPUT_ID", "REQUEST_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS"
        ]
        for input_name in input_names:
            setattr(
                self,
                input_name.lower() + "_dtype",
                pb_utils.triton_string_to_numpy(
                    pb_utils.get_output_config_by_name(
                        model_config, input_name)['data_type']))

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
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

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request,
                                                      'QUERY').as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(
                request, 'REQUEST_OUTPUT_LEN').as_numpy()

            bad_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'BAD_WORDS_DICT').as_numpy()
            stop_words_dict = pb_utils.get_input_tensor_by_name(
                request, 'STOP_WORDS_DICT').as_numpy()

            # Preprocessing input data.
            input_id, request_input_len = self._create_request(query)
            bad_words = self._to_word_list_format(bad_words_dict)
            stop_words = self._to_word_list_format(stop_words_dict)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID',
                np.array(input_id).astype(self.input_id_dtype))
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                np.array(request_input_len).astype(
                    self.request_input_len_dtype))
            request_output_len_tensor = pb_utils.Tensor('REQUEST_OUTPUT_LEN',
                                                        request_output_len)
            bad_words_ids_tensor = pb_utils.Tensor('BAD_WORDS_IDS', bad_words)
            stop_words_ids_tensor = pb_utils.Tensor('STOP_WORDS_IDS',
                                                    stop_words)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor, bad_words_ids_tensor, stop_words_ids_tensor,
                request_input_len_tensor, request_output_len_tensor
            ])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _create_request(self, query):
        """
            query : batch string (2D numpy array)
        """
        start_ids = [
            torch.IntTensor(self.tokenizer.encode(s[0].decode())) for s in query
        ]
        start_lengths = torch.IntTensor([[len(ids)] for ids in start_ids])

        start_ids = pad_sequence(start_ids,
                                 batch_first=True,
                                 padding_value=self.pad_id)
        # input_len = min(start_lengths)
        #attn_mask = torch.ones((batch_size, input_len, input_len)).tril()

        return start_ids, start_lengths

    def _to_word_list_format(self, word_lists: List[List[str | bytes]]):
        '''
        word_lists format:
            len(word_lists) == batch_size
            word_lists[i] means the words associated to batch item i. A "word" may actually be any string. Like "lorem" or "lorem ipsum".
        '''
        assert self.tokenizer != None, "need to set tokenizer"

        if word_lists is None:
            # Return an empty array of shape (1,2,0)
            return np.empty([1, 2, 0], dtype="int32")

        flat_ids = []
        offsets = []
        for word_list in word_lists:
            item_flat_ids = []
            item_offsets = []

            for word in word_list:
                if isinstance(word, bytes):
                    word = word.decode()

                ids = self.tokenizer.encode(word, add_special_tokens=False)

                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)),
                                constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
