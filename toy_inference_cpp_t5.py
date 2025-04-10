from tensorrt_llm.runtime import ModelRunnerCpp
import torch
from transformers import T5Tokenizer


def main():

    runner = ModelRunnerCpp.from_dir(
        engine_dir='my_models/t5base_engine/',
        is_enc_dec=True,
        max_input_len=512,
        cross_kv_cache_fraction=0.5,
        rank=0,
    )

    # Load tokenizer for encoding the input text
    tokenizer = T5Tokenizer.from_pretrained('my_models/t5base')
    
    # Create a complex question that will require a lengthy answer
    random_text = """
    Cafeteria has 100 apples, each of 20 students ate one, how many apples left after each student ate one?
    """
    # Prepend the instruction
    input_text = "Answer the following question step by step: " + random_text
    
    # Tokenize the input text
    encoder_tokens = tokenizer.encode(input_text)
    
    # Create batch_size=512 by repeating the same input
    batch_size = 512
    
    # Use [0] as decoder input (decoder start token) and repeat for batch_size
    decoder_input_ids = [
        torch.tensor([0], dtype=torch.int32)
        for _ in range(batch_size)
    ]
    
    # Use the tokenized input as encoder input and repeat for batch_size
    encoder_input_ids = [
        torch.tensor(encoder_tokens, dtype=torch.int32)
        for _ in range(batch_size)
    ]
    print(f"Encoder input length: {len(encoder_tokens)} tokens")
    print(f"Decoder input length: {decoder_input_ids[0].shape} tokens")
    print(f"Batch size: {len(encoder_input_ids)}")

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_ids=encoder_input_ids,
            max_new_tokens=1024,
            end_id=1,
            pad_id=0,
            streaming=False,
        )
        torch.cuda.synchronize()

    output_ids = outputs.cpu().numpy().tolist()
    print(output_ids)
    print("Output tokens:")
    print(len([x for x in output_ids[0][0] if x != 1]))

    decoded_output = tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
    print(f"Decoded output: {decoded_output}")


if __name__ == "__main__":
    main()
