import os
import sys
from argparse import ArgumentParser

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tensorrt_llm.logger import logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.utils import make_context

parser = ArgumentParser()
parser.add_argument(
    "--hf_model_dir",
    type=str,
    default=None,
)
parser.add_argument('--tokenizer_dir',
                    type=str,
                    default=None,
                    help="Directory containing the tokenizer.model.")
parser.add_argument(
    "--quant_ckpt_path",
    type=str,
    default=None,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    choices=["cuda", "cpu"],
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=512,
)

args = parser.parse_args()
logger.set_level('info')

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir,
                                          use_fast=True,
                                          trust_remote_code=True)

dataset_cnn = load_dataset("ccdv/cnn_dailymail", "3.0.0")
dataset = dataset_cnn["test"]

num_samples = min(args.num_samples, len(dataset))
examples = []
for i in tqdm(range(num_samples), desc="tokenizing datasets"):
    line = dataset[i]["article"]
    line = line + ' TL;DR: '
    line = line.strip()
    line = line.replace(" n't", "n't")
    # use make_content to generate prompt
    raw_text, _ = make_context(
        tokenizer=tokenizer,
        query=line,
        history=[],
    )
    example = tokenizer(raw_text)
    examples.append(example)

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=
    False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    true_sequential=True,
)

logger.info(f"Loading model from {args.hf_model_dir}")
model = (AutoGPTQForCausalLM.from_pretrained(args.hf_model_dir,
                                             quantize_config,
                                             trust_remote_code=True,
                                             use_flash_attn=False,
                                             fp16=True).eval())
if args.device == "cuda":
    model.cuda()
else:
    logger.warning(
        "using cpu only support on Qwen 7b v1.0, not support on Qwen 7b v1.1 / Qwen 14b"
    )

logger.info("loading model to run gptq, may need few minute...")
# quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
model.quantize(examples, cache_examples_on_gpu=False)
logger.info("quantized ok!")

# save quantized model
model.save_quantized(args.quant_ckpt_path, use_safetensors=True)
