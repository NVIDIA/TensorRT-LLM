
rm -rf gptneox_model
git clone https://huggingface.co/EleutherAI/gpt-neox-20b gptneox_model

rm -f gptneox_model/model-*.safetensors
rm -f gptneox_model/model.safetensors.index.json
wget -q https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/model.safetensors.index.json --directory-prefix gptneox_model

for i in $(seq -f %05g 46)
do
  echo -n "Downloading $i of 00046..."
  wget -q https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/model-$i-of-00046.safetensors --directory-prefix gptneox_model
  echo "Done"
done
