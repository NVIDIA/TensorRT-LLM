# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sample post-processing hooks for the trtllm-serve hook integration test.

These are deliberately stateless and deterministic so the test can assert the
client-visible effect regardless of the (non-deterministic) model output. Each
class is a top-level, no-arg-constructible, importable callable so it can be
supplied to ``trtllm-serve --post_processor_hook`` and reconstructed by reference in
the post-processing worker process.
"""

from tensorrt_llm.executor.postprocessor_hook import (
    PostProcessorHookChunk,
    PostProcessorHookVerdict,
    emit,
    suppress,
    terminate,
)


class UppercaseHook:
    """Rewrite every chunk's text to upper case."""

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return emit(chunk.text_diff.upper())


class SuppressHook:
    """Withhold all output (every chunk is suppressed)."""

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return suppress()


class TerminateHook:
    """Terminate the stream immediately on the first chunk seen."""

    def __call__(self, chunk: PostProcessorHookChunk) -> PostProcessorHookVerdict:
        return terminate("test_policy")
