from transformers import AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", trust_remote_code=True)

tokens = torch.tensor([278, 278, 278, 278, 278, 278, 278, 278])
print(tokenizer.decode(tokens))