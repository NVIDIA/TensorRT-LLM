import pynvml

DEFAULT_HF_MODEL_DIRS = {
    'BaichuanForCausalLM': 'baichuan-inc/Baichuan-13B-Chat',
    'BloomForCausalLM': 'bigscience/bloom-560m',
    'GLMModel': 'THUDM/glm-10b',
    'ChatGLMModel': 'THUDM/chatglm3-6b',
    'ChatGLMForCausalLM': 'THUDM/chatglm3-6b',
    'FalconForCausalLM': 'tiiuae/falcon-rw-1b',
    'GPTForCausalLM': 'gpt2-medium',
    'GPTJForCausalLM': 'EleutherAI/gpt-j-6b',
    'GPTNeoXForCausalLM': 'EleutherAI/gpt-neox-20b',
    'InternLMForCausalLM': 'internlm/internlm-chat-7b',
    'InternLM2ForCausalLM': 'internlm/internlm2-chat-7b',
    'LlamaForCausalLM': 'meta-llama/Llama-2-7b-hf',
    'MPTForCausalLM': 'mosaicml/mpt-7b',
    'PhiForCausalLM': 'microsoft/phi-2',
    'OPTForCausalLM': 'facebook/opt-350m',
    'QWenLMHeadModel': 'Qwen/Qwen-7B',
    'QWenForCausalLM': 'Qwen/Qwen-7B',
    'Qwen2ForCausalLM': 'Qwen/Qwen1.5-7B',
    'Qwen2MoeForCausalLM': 'Qwen/Qwen1.5-MoE-A2.7B',
    'RecurrentGemmaForCausalLM': 'google/recurrentgemma-2b',
}


def get_device_memory():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total / (1024**3)
    pynvml.nvmlShutdown()

    return total_memory
