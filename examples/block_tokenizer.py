from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True)

tokens = torch.tensor([30258,  19526,  78778,  15776,  28871, 116712,  54572,  22518,  56361, 102652])
print(tokenizer.decode(tokens))