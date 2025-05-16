import argparse

from tensorrt_llm.tools.multimodal_builder import (MultimodalEngineBuilder,
                                                   add_multimodal_arguments)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_multimodal_arguments(parser)
    args = parser.parse_args()

    builder = MultimodalEngineBuilder(args)
    builder.build()
