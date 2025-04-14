from tensorrt_llm.runtime import ModelRunnerCpp
import torch
from transformers import T5Tokenizer


def main():

    runner = ModelRunnerCpp.from_dir(
        engine_dir='models/multi_vocab/t5base_2vocab_engine/',
        is_enc_dec=True,
        max_input_len=512,
        cross_kv_cache_fraction=0.5,
        rank=0,
    )

    # Load tokenizer for encoding the input text
    tokenizer = T5Tokenizer.from_pretrained('models/single_vocab/t5base')
    
    # Create a complex question that will require a lengthy answer
    input_text = "hi, how are things?"
    # Prepend the instruction
    #input_text = "Answer the following question step by step: " + input_text
    
    # Tokenize the input text
    encoder_tokens = tokenizer.encode(input_text)
    encoder_tokens = torch.tensor(encoder_tokens, dtype=torch.int32)
    print(f"Encoder input tokens: {str(encoder_tokens)}")
    
    # Create batch_size=512 by repeating the same input
    batch_size = 1
    
    # Use [0] as decoder input (decoder start token) and repeat for batch_size
    decoder_input_ids = [
        torch.tensor([0, 32128], dtype=torch.int32)
        for _ in range(batch_size)
    ]
    
    # Use the tokenized input as encoder input and repeat for batch_size
    encoder_input_ids = [
        encoder_tokens
        for _ in range(batch_size)
    ]

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids,
            encoder_input_ids=encoder_input_ids,
            max_new_tokens=20,
            end_id=1,
            pad_id=0,
            streaming=False,
        )
        torch.cuda.synchronize()

    output_ids = outputs.cpu().numpy()[0][0][:20]
    print(f"Output tokens (len {len(output_ids)}):")
    print(output_ids)

    output_ids = output_ids.reshape(-1, 2)
    print(f"Output tokens (vocab 0): {str(output_ids[:, 0])}")
    print(f"Output tokens (vocab 1): {str(output_ids[:, 1])}")
    print(f"Output tokens (vocab 1): {str(output_ids[:, 1] - 32128)}")
    decoded_output = tokenizer.decode(output_ids[:, 0].tolist(), skip_special_tokens=True)
    print(f"Decoded output (vocab 0): {decoded_output}")


if __name__ == "__main__":
    main()
