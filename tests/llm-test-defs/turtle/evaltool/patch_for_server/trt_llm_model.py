import ast
from copy import deepcopy
from typing import List

import torch
from transformers import AutoTokenizer

import tensorrt_llm
from tensorrt_llm.runtime import ModelRunnerCpp


class TRTLLMModel:

    def __init__(
        self,
        engine_path: str,
        tokenizer_path: str,
        max_output_len: int,
        lookahead_config: str = None,
    ):
        self.runtime_rank = tensorrt_llm.mpi_rank()

        self._tokenizer = self._init_tokenizer(tokenizer_path)
        self.max_output_len = max_output_len
        self.engine_path = engine_path

        self.lookahead_config = None
        if lookahead_config is not None:
            self.lookahead_config = ast.literal_eval(lookahead_config)
            assert len(
                self.lookahead_config
            ) == 3, "Lookahead needs [max_window_size, max_ngram_size, max_verification_set_size]"

        runner_cls = ModelRunnerCpp
        language_model_cfg = dict(engine_dir=engine_path,
                                  rank=self.runtime_rank,
                                  lookahead_config=self.lookahead_config)
        self.runner = runner_cls.from_dir(**language_model_cfg)

    def _init_tokenizer(self, tokenizer_path):
        """Returns tokenizer object based on input config"""
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="left",
            truncation_side="left",
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def generate_text(
        self,
        context: List[str],
        max_output_len: int = None,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 0.0,
        echo: bool = False,
        stop: list = [],
    ) -> List[str]:
        """Run inference on batch of inputs"""
        if max_output_len is None:
            max_output_len = self.max_output_len
        else:
            max_output_len = min(max_output_len, self.max_output_len)

        max_input_len = self.runner.max_seq_len - max_output_len

        batch_input_ids = [
            self._tokenizer.encode(
                c,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_input_len,
            ).squeeze(0) for c in context
        ]

        batch_size = len(batch_input_ids)
        input_lengths = [x.size(0) for x in batch_input_ids]

        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids,
                end_id=self._tokenizer.eos_token_id,
                pad_id=self._tokenizer.pad_token_id,
                stop_words_list=None,
                output_log_probs=True,
                return_dict=True,
                max_new_tokens=max_output_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            torch.cuda.synchronize()

        output_beams_list = [
            self._tokenizer.batch_decode(
                outputs["output_ids"][batch_idx, :, input_lengths[batch_idx]:],
                skip_special_tokens=True,
            ) for batch_idx in range(batch_size)
        ]
        assert all([len(e) == 1 for e in output_beams_list])  # one beam

        token_ids, tokens, predictions, logprobs = [], [], [], []
        for batch_idx in range(batch_size):
            # stop words
            prediction = output_beams_list[batch_idx][0]
            for _stop in stop:
                prediction = prediction.split(_stop)[0]
            _, n_tokens = self._tokenizer.encode(
                prediction,
                return_tensors="pt",
                add_special_tokens=False,
            ).shape

            batch_output_ids = outputs["output_ids"][batch_idx, 0, :]
            batch_output_ids = batch_output_ids[:input_lengths[batch_idx] +
                                                n_tokens]

            batch_log_probs = outputs["log_probs"][batch_idx, 0, :]
            batch_log_probs = batch_log_probs[:input_lengths[batch_idx] +
                                              n_tokens]

            # echo prompt back in completion
            if echo:
                start_token_idx = 0
            else:
                start_token_idx = input_lengths[batch_idx]

            # token ids
            token_ids.append(
                deepcopy(batch_output_ids[start_token_idx:].tolist()))
            try:
                idx = token_ids[batch_idx].index(self._tokenizer.pad_token_id)
                token_ids[batch_idx] = token_ids[batch_idx][:idx]
            except ValueError:
                pass

            # tokens
            tokens += [
                self._tokenizer.convert_ids_to_tokens(
                    token_ids[batch_idx],
                    skip_special_tokens=False,
                )
            ]

            # predictions
            predictions.append(
                self._tokenizer.decode(
                    token_ids[batch_idx],
                    skip_special_tokens=True,
                ))

            # logprobs
            if self.runner.gather_context_logits:
                full_logprobs = deepcopy(batch_log_probs)
                if echo:
                    batch_context_logits = outputs["context_logits"][batch_idx]
                    batch_context_logits = batch_context_logits[:input_lengths[
                        batch_idx] + n_tokens, :]

                    batch_log_softmax = torch.nn.functional.log_softmax(
                        batch_context_logits, dim=-1)
                    batch_context_logprobs = torch.concatenate((
                        torch.tensor([torch.nan]).to(batch_log_softmax.device),
                        batch_log_softmax[:-1,
                                          batch_input_ids[batch_idx][1:]].diag(
                                          ),
                    ))

                    full_logprobs[:batch_context_logprobs.
                                  shape[0]] = batch_context_logprobs
                logprobs += [
                    full_logprobs[start_token_idx:start_token_idx +
                                  len(tokens[batch_idx])].tolist()
                ]

        if self.runner.gather_context_logits:
            return {
                "inputs": context,
                "predictions": predictions,
                "tokens": tokens,
                "logprobs": logprobs,
            }
        else:
            return {
                "inputs": context,
                "predictions": predictions,
                "tokens": tokens,
            }
