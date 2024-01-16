from transformers import LlamaForCausalLM
from peft import get_peft_model, IA3Config, TaskType
from datasets import load_dataset
from trl import SFTTrainer, AutoModelForCausalLMWithValueHead


peft_config = IA3Config(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["k_proj", "v_proj", "down_proj"], 
    feedforward_modules=["down_proj"]
)

model = LlamaForCausalLM.from_pretrained(
    "llama-2/7b", 
    load_in_8bit=True
)
model = get_peft_model(model, peft_config)


dataset = load_dataset("imdb", split="train[10:20]")
trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=256,
    dataset_batch_size=1
)
trainer.train()

model.save_pretrained('llama-2/7b-ia3')