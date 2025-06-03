from contextlib import contextmanager
from typing import Dict, List, Tuple

# isort: off
import torch
# isort: on

import os
from pathlib import Path
from typing import Optional

import datasets
from datasets import load_dataset

from transformers import BertConfig, BertPreTrainedModel, BertForQuestionAnswering, BertForSequenceClassification, BertModel  # isort:skip
from transformers import RobertaConfig, RobertaPreTrainedModel, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaModel  # isort:skip


# NOTE: This routine is copied from from tests/unittests/utils/llm_data.py
def llm_models_root(check=False) -> Optional[Path]:
    root = Path("/home/scratch.trt_llm_data/llm-models/")

    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    if check:
        assert root.exists(), \
        "You shall set LLM_MODELS_ROOT env or be able to access /home/scratch.trt_llm_data to run this test"

    return root if root.exists() else None


def llm_datasets_root() -> Path:
    return llm_models_root(check=True) / "datasets"


def prepare_text_inputs(model_name, batch_size=8):
    print(
        f"HF_DATASETS_OFFLINE inside function: {datasets.config.HF_DATASETS_OFFLINE}"
    )
    if model_name == "BertForQuestionAnswering" or model_name == "RobertaForQuestionAnswering":
        squad_dataset_root = str(llm_datasets_root(
        )) + "/" if datasets.config.HF_DATASETS_OFFLINE else ""
        squad_dataset_path = squad_dataset_root + "squad_v2"
        squad_dataset = load_dataset(squad_dataset_path, trust_remote_code=True)
        val_dataset = squad_dataset["validation"]
        samples = val_dataset.select(range(batch_size))

        qa_real_test_inputs = {
            'text': samples["question"],
            'text_pair': samples["context"]
        }
        return qa_real_test_inputs
    elif model_name == "BertForSequenceClassification" or model_name == "RobertaForSequenceClassification":
        yelp_dataset_root = str(llm_datasets_root(
        )) + "/" if datasets.config.HF_DATASETS_OFFLINE else "fancyzhx/"
        yelp_dataset_path = yelp_dataset_root + "yelp_polarity"
        yelp_dataset = load_dataset(yelp_dataset_path, trust_remote_code=True)
        val_dataset = yelp_dataset["test"]
        samples = val_dataset.select(range(batch_size))

        seqcls_real_test_inputs = {'text': samples['text']}
        return seqcls_real_test_inputs
    elif model_name == "BertModel" or model_name == "RobertaModel":
        #NOTE: For BertModel, it is used as an encoder, so we use dummy input here,
        #       you can choose whatevert you like, but the numerical accuracy might vary.
        test_input = 'To be or not to be: that is the question'
        input_strings = [test_input for _ in range(batch_size)]
        base_real_test_inputs = {'text': input_strings}
        return base_real_test_inputs

    else:
        raise NotImplementedError(f"Unknown model {model_name}")


def get_engine_name(rank):
    return 'rank{}.engine'.format(rank)


def decode_bertqa_output(inputs_text, hf_tokenizer,
                         start_logits: Tuple[torch.Tensor],
                         end_logits: Tuple[torch.Tensor]):
    question, context = inputs_text['text'], inputs_text['text_pair']
    assert len(context) == len(question)
    batch_size = len(context)

    # regenerate inputs_ids because it is flatten for remove_input_padding=True
    inputs = hf_tokenizer(**inputs_text, padding=True, return_tensors='pt')
    inputs_ids = inputs['input_ids']
    answer_start_index = [logit.argmax(dim=0) for logit in start_logits]
    answer_end_index = [logit.argmax(dim=0) for logit in end_logits]
    decode_answer = []
    for i in range(batch_size):
        predict_answer_tokens = inputs_ids[
            i, answer_start_index[i]:answer_end_index[i] + 1]
        predict_text = hf_tokenizer.decode(predict_answer_tokens,
                                           skip_special_tokens=True)
        decode_answer.append(predict_text)
    return decode_answer


def compare_bertqa_result(inputs_text, res_answers, ref_answers):
    from difflib import SequenceMatcher
    question, context = inputs_text['text'], inputs_text['text_pair']
    assert len(res_answers) == len(ref_answers)
    batch_size = len(res_answers)
    for i in range(batch_size):
        print(f"Context: {context[i]}\nQuestion: {question[i]}")
        print(f"Ref Answer: {ref_answers[i]}")
        print(f"Res Answer: {res_answers[i]}")
        match_rate = SequenceMatcher(None, "\n".join(res_answers[i]),
                                     "\n".join(ref_answers[i])).ratio()
        assert match_rate > 0.95
        print(
            f"TRT-LLM results match HF results with literal match rate {match_rate}"
        )


def decode_bertcls_output(logits: torch.Tensor, hf_model_config, inputs_text):
    text = inputs_text['text']
    id2label = hf_model_config.id2label
    class_ids = logits.argmax(dim=1)
    decode_answer = []
    batch_size = len(text)
    for i in range(batch_size):
        predicted_class_id = class_ids[i].item()
        predicted_label = id2label[predicted_class_id]
        decode_answer.append(predicted_label)
    return decode_answer


def compare_bertcls_result(inputs_text, res_answers, ref_answers):
    from difflib import SequenceMatcher
    text = inputs_text['text']
    batch_size = len(text)
    for i in range(batch_size):
        print(f"Context: {text[i]}")
        print(f"Ref Label: {ref_answers[i]}")
        print(f"Res Label: {res_answers[i]}")
        match_rate = SequenceMatcher(None, "\n".join(res_answers[i]),
                                     "\n".join(ref_answers[i])).ratio()
        assert match_rate > 0.95
        print(
            f"TRT-LLM results match HF results with literal match rate {match_rate}"
        )


def process_input(input_ids_list: List[torch.Tensor],
                  token_type_ids_list: List[torch.Tensor],
                  is_roberta=False,
                  padding_idx=1):
    input_lengths = []
    position_ids_list = []
    max_input_length = 0
    for i, input_ids in enumerate(input_ids_list):
        input_len = len(input_ids)
        assert input_len == len(token_type_ids_list[i]), f"sample {i}: len(input_ids)={len(input_ids)}, " \
                                                         f"len(token_type_ids)={len(token_type_ids_list[i])}, not equal"
        input_lengths.append(input_len)
        position_ids = torch.arange(0, input_len, dtype=torch.int32)
        if is_roberta:
            position_ids = position_ids + 1 + padding_idx

        position_ids_list.append(position_ids)
        max_input_length = max(max_input_length, input_len)

    # [num_tokens]
    input_ids = torch.concat(input_ids_list).int().cuda()
    token_type_ids = torch.concat(token_type_ids_list).int().cuda()
    position_ids = torch.concat(position_ids_list).int().cuda()

    input_lengths = torch.tensor(input_lengths).int().cuda()  # [batch_size]
    max_input_length = torch.empty((max_input_length, )).int().cuda()
    return input_ids, input_lengths, token_type_ids, position_ids, max_input_length


def intermediate_check(tllm_inter: Dict, hf_ref: Tuple[torch.Tensor], attn_mask,
                       logger):

    def apply_mask(x):
        return x * attn_mask

    # minus one because there is an embedding output
    num_layers = len(hf_ref) - 1

    res = tllm_inter['embedding_output']
    res = apply_mask(res)
    ref = hf_ref[0]
    ref = apply_mask(ref)
    torch.testing.assert_close(actual=res, expected=ref, rtol=1e-2, atol=1e-2)
    logger.debug("Embedding are all close")

    for i in range(num_layers - 1):
        res = tllm_inter[f'layer_{i}_output']
        res = apply_mask(res)
        ref = hf_ref[i + 1]
        ref = apply_mask(ref)
        is_close = torch.allclose(res, ref, rtol=1e-2, atol=1e-2)
        logger.debug(f'BertEncoderLayer_{i}_output is close: {is_close}')


@contextmanager
def temporary_datasets_config(**kwargs):
    # Save original settings
    original_settings = {}
    for key, value in kwargs.items():
        original_settings[key] = getattr(datasets.config, key)
        setattr(datasets.config, key, value)
    try:
        yield
    finally:
        # Restore original settings
        for key, value in original_settings.items():
            setattr(datasets.config, key, value)
