r"""KV-cache hit-rate analysis for scaffolding traces.

Public entry point: :func:`compute_cache_hit_upper_bound` — give it a parsed
trace and it returns a JSON-serializable hit-rate record under the
infinite-capacity prefix-cache assumption, with all distinct
``system_prompt_id`` templates pre-loaded into the radix tree before any
request is scored. Branched / parallel sub-agent traces (Tree of Thought,
Open Deep Research) are supported under the shared global-cache policy and
emit per-(branch root / branch depth / system prompt) rollups.
"""

from .aggregation import DATASET_SCHEMA, aggregate_dataset_record
from .annotate import ANNOTATION_FIELDS, build_annotated_trace
from .blocks import BlockPrefixCache, full_blocks, reusable_token_len, validate_tokens_per_block
from .branch_summary import compute_branch_rollups, merge_rollup_arrays
from .cache_hit import SCHEMA, compute_cache_hit_upper_bound
from .io import (
    ANNOTATED_TRACE_SUFFIX,
    OUTPUT_SUFFIX,
    default_annotated_trace_path,
    default_dataset_output_path,
    default_output_path,
    load_trace,
    resolve_input_trace_files,
    resolve_trace_file,
    write_json,
)
from .streams import ConversationSegments, SystemPromptRegistry, TokenIdAllocator

__all__ = [
    "ANNOTATED_TRACE_SUFFIX",
    "ANNOTATION_FIELDS",
    "BlockPrefixCache",
    "ConversationSegments",
    "DATASET_SCHEMA",
    "OUTPUT_SUFFIX",
    "SCHEMA",
    "SystemPromptRegistry",
    "TokenIdAllocator",
    "aggregate_dataset_record",
    "build_annotated_trace",
    "compute_branch_rollups",
    "compute_cache_hit_upper_bound",
    "default_annotated_trace_path",
    "default_dataset_output_path",
    "default_output_path",
    "full_blocks",
    "load_trace",
    "merge_rollup_arrays",
    "resolve_input_trace_files",
    "resolve_trace_file",
    "reusable_token_len",
    "validate_tokens_per_block",
    "write_json",
]
