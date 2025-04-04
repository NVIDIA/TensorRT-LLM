from transformers import AutoTokenizer, Gemma3ForCausalLM

model_path = "/home/bbuddharaju/scratch/random/hf_models/gemma-3-1b-it/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
input_text = "The main cities in Italy are (Write a blog post)"
input_tokens = tokenizer(input_text, return_tensors="pt")

model = Gemma3ForCausalLM.from_pretrained(model_path).eval()

output_tokens = model.generate(input_tokens["input_ids"], max_length=14)
output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
print(output_text)
