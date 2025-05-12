# Interpret the output_ids into text

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "/work/models/repo/TinyLlama-1.1B-Chat-v1.0/", trust_remote_code=True)

output_ids = [ \
    [ ],
              ]

for i in range(len(output_ids)):
    output = output_ids[i]
    count = 0 - 1  # one "2" at least
    for j in range(len(output)):
        if output[j] == 2:
            count += 1
    output_ids[i] = output[:len(output) - count]

for output in output_ids:
    print(f"--------------------length={len(output)}")
    print(tokenizer.decode(output))
    print()
