import os
import subprocess
import sys

PROMPT = "Tell a story"
LLAMA_MODEL_DIR = sys.argv[1]
EXAMPLES_ROOT = sys.argv[2] if len(sys.argv) > 2 else ""
LLM_EXAMPLES = os.path.join(EXAMPLES_ROOT, 'llm_examples.py')

run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_with_quantization",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}",
    "--quant_type=int4_awq"
]
subprocess.run(run_cmd, check=True)

run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_with_quantization",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}",
    "--quant_type=fp8"
]
subprocess.run(run_cmd, check=True)
