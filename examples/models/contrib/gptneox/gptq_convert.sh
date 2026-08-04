git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git GPTQ-for-LLaMa

pip install -r ./GPTQ-for-LLaMa/requirements.txt

CUDA_VISIBLE_DEVICES=0 python3 GPTQ-for-LLaMa/neox.py ./gptneox_model \
wikitext2 \
--wbits 4 \
--groupsize 128 \
--save_safetensors ./gptneox_model/gptneox-20b-4bit-gs128.safetensors
