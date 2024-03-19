import os
import subprocess
import sys

PROMPT = "Tell a story"
LLAMA_MODEL_DIR = sys.argv[1]
TMP_ENGINE_DIR = sys.argv[2] if len(sys.argv) > 2 else "./tllm.engine.example"
EXAMPLES_ROOT = sys.argv[3] if len(sys.argv) > 3 else ""
LLM_EXAMPLES = os.path.join(EXAMPLES_ROOT, 'llm_examples.py')

run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_from_huggingface_model",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}",
    f"--dump_engine_dir={TMP_ENGINE_DIR}"
]
subprocess.run(run_cmd, check=True)

# TP enabled
run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_from_huggingface_model",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}", "--tp_size=2"
]
subprocess.run(run_cmd, check=True)

run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_from_tllm_engine",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}",
    f"--dump_engine_dir={TMP_ENGINE_DIR}"
]
subprocess.run(run_cmd, check=True)

run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_generate_async_example",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}"
]
subprocess.run(run_cmd, check=True)

# Both TP and streaming enabled
run_cmd = [
    sys.executable, LLM_EXAMPLES, "--task=run_llm_generate_async_example",
    f"--prompt={PROMPT}", f"--hf_model_dir={LLAMA_MODEL_DIR}", "--streaming",
    "--tp_size=2"
]
subprocess.run(run_cmd, check=True)
