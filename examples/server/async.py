import argparse
from asyncio import run
from pathlib import Path

from tensorrt_llm.engine import AsyncLLMEngine


async def main(model_dir: Path):
    engine = AsyncLLMEngine.from_hf_dir(model_dir)
    text = "deep learning is"
    async for response in engine.generate(prompt=text, max_new_tokens=16):
        text += response.text
        print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    run(main(args.model_dir))
