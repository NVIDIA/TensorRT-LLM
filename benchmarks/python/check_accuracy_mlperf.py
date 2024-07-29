import json
import os
from enum import Enum

import evaluate
import nltk
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, LlamaTokenizerFast

nltk.download("punkt", quiet=False)
import argparse


class Model(Enum):
    Llama_v2_70B = 1
    GPT_J = 2


ACCURACY_TARGETS = {
    Model.Llama_v2_70B: {
        "rouge1": 44.4312 * 0.999,
        "rouge2": 22.0352 * 0.999,
        "rougeL": 28.6162 * 0.999,
        "tokens_per_sample": 294.45 * 0.9
    },
    Model.GPT_J: {
        "rouge1": 42.9435135,
        "rouge2": 20.1033765,
        "rougeL": 29.9581119,
        # "tokens_per_sample": ??
    }
}


def get_reference_df(processed_dataset_file):
    data = pd.read_pickle(processed_dataset_file)
    return data["output"].tolist()


def get_reference_json(cnn_dailymail_valset):
    # Load from CNN dailymail
    with open(cnn_dailymail_valset, 'r') as fh:
        list_data_dict = json.load(fh)

    targets = [f"{example['output']}" for example in list_data_dict]

    print(f"Loaded {len(targets)} samples from {cnn_dailymail_valset}")
    return targets


def get_responses_json(response_file):
    f = open(response_file)
    responses = json.load(f)
    ordered_responses = sorted(responses, key=lambda x: int(x['response_id']))
    return ordered_responses


def postprocess_text(preds, targets):
    # Post-process output texts for ROUGE evaluation
    preds = [pred.strip() for pred in preds]
    targets = [target.strip() for target in targets]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    targets = ["\n".join(nltk.sent_tokenize(target)) for target in targets]

    return preds, targets


def strip_eos(pred_toks, eos_id):
    while len(pred_toks) > 0 and pred_toks[-1] == eos_id:
        pred_toks.pop()
    if len(pred_toks) == 0:
        raise RuntimeError("Empty output sequence detected with EOS")
    return pred_toks


def calculate_toks_per_sample(preds, eos_id):
    preds = [strip_eos(pred, eos_id) for pred in preds]
    avg_len = sum(len(pred) for pred in preds)
    num_samples = len(preds)
    return avg_len / num_samples


def calculate_rouge_score(preds, targets, rouge_dir=None):
    print("Calculating ROUGE scores...")
    rouge_dir = rouge_dir if rouge_dir and os.path.exists(
        rouge_dir) else "rouge"
    metric = evaluate.load(rouge_dir)
    preds, targets = postprocess_text(preds, targets[0:len(preds)])
    result = metric.compute(predictions=preds,
                            references=targets,
                            use_stemmer=True,
                            use_aggregator=False)
    result = {k: round(np.mean(v) * 100, 4) for k, v in result.items()}
    prediction_lens = [len(pred) for pred in preds]
    result["gen_len"] = np.sum(prediction_lens)
    result["gen_num"] = len(preds)

    return result


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help=
        "Path to the reference dataset against which the responses are evaluated for accuracy. MLPerf uses open-orca (pkl) and cnn-dailymail (np) for Llama2-70B and GPT-J respectively."
    )
    parser.add_argument(
        "--responses",
        type=str,
        help="Path to the json file holding the responses from our benchmark run"
    )
    parser.add_argument("--base_model",
                        type=str,
                        help="Location of the model used (to create tokenizer)")

    parser.add_argument(
        '--rouge_dir',
        default=None,
        type=str,
        help=
        "evaluate.load('rouge') will attempt to pull rouge package from HF. Use cached rouge can avoid network outage of host or HF."
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()

    if args.dataset.lower().endswith(".pkl"):
        target_texts = get_reference_df(args.dataset)
        model = Model.Llama_v2_70B
        tokenizer = LlamaTokenizerFast.from_pretrained(args.base_model)
        relaxing_factor = 1.0
    elif args.dataset.lower().endswith(".json"):
        target_texts = get_reference_json(args.dataset)
        model = Model.GPT_J
        tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                                  model_max_length=2047,
                                                  padding_side="left",
                                                  use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        relaxing_factor = 0.93
    else:
        raise RuntimeError(
            "Dataset expected to be pkl (open-orca) or json (cnn-dailymail)")

    pred_out = get_responses_json(args.responses)
    pred_toks = [x['response_tokens'] for x in pred_out]

    tps_score = calculate_toks_per_sample(pred_toks, tokenizer.eos_token)

    pred_texts = tokenizer.batch_decode(pred_toks, skip_special_tokens=True)
    achieved_scores = calculate_rouge_score(pred_texts, target_texts,
                                            args.rouge_dir)

    achieved_scores['tokens_per_sample'] = tps_score
    targets = ACCURACY_TARGETS[model]

    print("Achieved rouge scores: ", achieved_scores)
    print("Tokens per sample: ", tps_score)
    print("Targets: ", targets)

    for k, _ in targets.items():
        assert targets[k] * relaxing_factor <= achieved_scores[k]


if __name__ == "__main__":
    main()
