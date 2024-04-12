#! /usr/bin/env python3
import argparse
import code
import readline  # NOQA
from argparse import ArgumentParser
from pathlib import Path

from tensorrt_llm.executor import GenerationExecutorWorker


class LLMChat(code.InteractiveConsole):

    def __init__(self, executor):
        super().__init__()
        self.executor = executor
        self.generation_kwargs = {
            "max_new_tokens": 100,
            "repetition_penalty": 1.05,
        }
        self.parser = ArgumentParser(prefix_chars="!")
        self.parser.add_argument("!!max_new_tokens", type=int)
        self.parser.add_argument("!!repetition_penalty", type=float)

    def runsource(self,
                  source: str,
                  filename: str = "<input>",
                  symbol: str = "single") -> bool:
        del filename, symbol

        if source.startswith("!"):
            args = self.parser.parse_args(source.split(" "))
            for k, v in vars(args).items():
                if v is not None:
                    self.generation_kwargs[k] = v
            return False

        future = self.executor.generate_async(source,
                                              streaming=True,
                                              **self.generation_kwargs)
        for partial_result in future:
            print(partial_result.text_diff, end="")
        print("")

        return False


def main(model_dir: Path, tokenizer: Path | str):

    with GenerationExecutorWorker(model_dir, tokenizer, 1) as executor:
        executor.block_subordinates()
        repl = LLMChat(executor)
        repl.interact(banner="", exitmsg="")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("tokenizer", type=Path)
    args = parser.parse_args()
    main(args.model_dir, args.tokenizer)
