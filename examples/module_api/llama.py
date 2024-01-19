import sys

from tensorrt_llm.models import LLaMAForCausalLM


def read_input():
    while (True):
        input_text = input("<")
        if input_text in ("q", "quit"):
            break
        yield input_text


def main():
    assert len(sys.argv) == 2
    hf_model_dir = sys.argv[1]
    tokenizer_dir = hf_model_dir

    llama = LLaMAForCausalLM.from_hugging_face(hf_model_dir)
    max_batch_size, max_isl, max_osl = 1, 256, 20
    llama.to_trt(max_batch_size, max_isl, max_osl)

    for (inp, output) in llama._generate(read_input(),
                                         max_osl,
                                         tokenizer_dir=tokenizer_dir):
        print(f">{output}")


main()
