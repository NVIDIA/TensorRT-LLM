Usage:

# Single GPU
```bash
python ./llama.py <hf llama dir>
```

The bot read one sentence at a time and generate at max 20 tokens for you.
Type "q" or "quit" to stop chatting.


# Multi-GPU

Using multi GPU tensor parallel to build and run llama, and then generate on pre-defined dataset.
Note that multi GPU can also support the chat scenario, need to add additional code to read input from the root process, and broadcast the tokens to all worker processes.
The example only targets to demonstrate the TRT-LLM API usage here, so it uses pre-defined dataset for simplicity.

```
python ./llama_multi_gpu.py <hf llama dir> <tp size>
```
