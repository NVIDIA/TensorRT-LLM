from tensorrt_llm.runtime import ModelRunnerCpp
import torch
from transformers import AutoTokenizer


def main():

    runner = ModelRunnerCpp.from_dir(
        engine_dir='models/multi_vocab/llama1b_engine/',
        max_input_len=512,
        rank=0,
    )

    # Load tokenizer for encoding the input text
    tokenizer = AutoTokenizer.from_pretrained('models/multi_vocab/llama1b')
    
    # Create a complex question that will require a lengthy answer
    random_text = """hi, how are things?"""
    
    # Tokenize the input text
    tokens = tokenizer.encode(random_text)

    input_ids = torch.tensor(tokens, dtype=torch.int32)
    print(f"Input tokens [{input_ids.shape}]: {str(input_ids)}")

    # introduce multiple vocabs
    input_ids = input_ids.unsqueeze(-1)  # [nTok, 1]
    input_ids = input_ids.repeat(1, 2)  # [nTok, 2]
    input_ids[:, 1] += 128256
    input_ids = input_ids.flatten()  # [nTok * 2]

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=[input_ids],
            max_new_tokens=20,
            end_id=1,
            pad_id=0,
            streaming=False,
        )
        torch.cuda.synchronize()

    output_ids = outputs.cpu().numpy().tolist()
    print(f"Output tokens: {str(output_ids)}")
    decoded_output = tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
    print(f"Decoded output: {decoded_output}")


if __name__ == "__main__":
    main()
