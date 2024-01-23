import argparse
from asyncio import run
from pathlib import Path

from executor import GenerationExecutor


async def main(model_dir: Path, tokenizer: Path | str):
    engine = GenerationExecutor(model_dir, tokenizer)
    text = "deep learning is"
    async for response in engine.generate(prompt=text, max_new_tokens=16):
        print(response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    run(main(args.model_dir, args.tokenizer))
