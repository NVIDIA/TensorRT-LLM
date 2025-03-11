"""Attention Interface to handle various attention operators and cache operations.

This module provides an interface between the high-level runtime and cache management system and
the low-level functional attention operators. The interface is designed to provide a homogeneous
object-oriented interface to the high-level runtime via the SequenceInfo dataclass. The SequenceInfo
is also responsible for functionalizing information about the sequence and pass it on the the
various attention interface. The AttentionDescriptor is the main interface to the attention operator
and operates on a purely functional paradigm that is compatible with the torch custom op system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple, Type, Union

import torch
from torch.export import Dim


@dataclass
class PositionalEmbeddingConfig:
    """A dataclass to hold positional embedding information."""

    mode: Optional[Literal["rope"]] = None
    rope_theta: float = 10000.0
    rope_scale: float = 1.0

    def __post_init__(self):
        assert self.mode in [None, "rope"], f"Invalid mode: {self.mode}."
        if self.mode == "rope":
            assert self.rope_theta > 0, f"Invalid rope theta: {self.rope_theta}."


@dataclass
class CacheConfig:
    """A dataclass to hold information how to configure the cache."""

    dtype: Optional[torch.dtype] = None


@dataclass
class AttentionInfo:
    """Information about the attention op.

    This is the dataclass collected by the kvcache transformation and passed in to the
    AttentionDescriptor methods to inform the attention op about the attention configuration.
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    cache_config: CacheConfig
    pos_embd_config: PositionalEmbeddingConfig


@dataclass
class SequenceInfo:
    """A dataclass to hold information about how the sequence is laid out and stored in cache.

    We assume the sequence + cache is laid out in the following way:

    - input_ids: [id_0, ..., id_{s_total-1}]
      flattened sequence of [b, 1] or [1, s_total]. We use [b, 1] to denote generate-only batches.
    - seq_len: [s_0, s_1, ..., s_{b-1}] such that s_total = sum(s_i)
      Describes how long each sequence is. For example,
      input_ids[:s_0] will correspond to sequence 0 in the batch and input_ids[s_0:s_1] will
      correspond to sequence 1 in the batch.
    - input_pos: [pos_0, ..., pos_{b-1}]
      Corresponds to the total number of tokens that has been already been cached for each sequence
      in the batch.
    - cache_loc: [c0, ...., c_{np-1}] where np is total number of pages allocated to describe all
      sequences in the batch.
    - pages_per_seq: [ps_0, ps_1, ..., ps_{b-1}] where ps_i is the number of pages allocated for
      sequence i. Note that, for example, cache_loc[p_0:p_1] will correspond to the pages associated
      with sequence 1 in the batch.

    Here are a couple of notes to emphasize this notation:

    - The total number of allocated token space for sequence i is given by ps_i * page_size. This is
      the total number of tokens that can be cached for each sequence.

    - NOTE: It must hold that pos_i + s_i <= ps_i * page_size for all i in [0, b-1]. Moreover, it is
      the responsibility of the cache manager and/or runtime to ensure sufficient page allocation
      for each sequence.

    """

    ## USE TO INITIALIZE DATA CLASS  ###############################################################
    max_seq_len: int = 1
    max_batch_size: int = 1
    page_size: int = 0

    ## [UPDATE WITH CARE] TENSOR FIELDS THAT WILL BE PASSED TO PREPARE_METADATA OP #################
    # input_ids MUST ALWAYS BE THE FIRST FIELD
    input_ids: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, dtype=torch.int))
    seq_len: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
    input_pos: torch.Tensor = field(default_factory=lambda: torch.zeros(1, dtype=torch.int))
    cache_loc: torch.Tensor = field(default_factory=lambda: torch.arange(1, dtype=torch.int))
    pages_per_seq: torch.Tensor = field(default_factory=lambda: torch.ones(1, dtype=torch.int))
    ################################################################################################

    ## PRIVATE FIELDS ##############################################################################
    _sequence_lengths: List[int] = field(default_factory=list)
    _num_pages: int = 1

    def __post_init__(self):
        if self.page_size < 1:
            self.page_size = self.max_seq_len
        total_tokens = self.max_batch_size * self.max_seq_len
        self._num_pages = (total_tokens) // self.page_size + (total_tokens % self.page_size > 0)

        self.input_ids = torch.ones(self.max_batch_size, 1, dtype=torch.int)
        self.seq_len = torch.empty(self.max_batch_size, dtype=torch.int)
        self.input_pos = torch.empty_like(self.seq_len)
        self.cache_loc = torch.empty(self.num_pages, dtype=torch.int)
        self.pages_per_seq = torch.empty_like(self.seq_len)

        # dynamic shape descriptors for tensor args
        self._dynamic_shapes: Optional[Tuple[Dict[str, Dim]]] = None

        # keep a list-like object of sequence lengths for simplicity as well
        self._sequence_lengths = [0] * self.max_batch_size

        # call reset once to initialize the tensors
        self.reset()

    @property
    def device(self) -> torch.device:
        return self.input_pos.device

    @property
    def args(self) -> List[torch.Tensor]:
        args = []
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                args.append(val)
        return args

    @property
    def extra_arg_names(self) -> List[str]:
        """Return extra arg names for the prepare_metadata op beyond input_ids."""
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), torch.Tensor)][1:]

    @property
    def dynamic_shapes(self) -> Tuple[Dict[str, Dim]]:
        """Return dynamic shapes of sequence info tensors.

        NOTE: will be lazily initialized since the Dim object is not picklable for multi-processing.
        """
        if self._dynamic_shapes is None:
            dynamic_shapes = ({},)
            if self.max_batch_size > 1:
                dynamic_shapes[0][0] = Dim("batch_size", max=self.max_batch_size)
            dynamic_shapes[0][1] = Dim("seq_len", max=self.max_seq_len)
            dynamic_shapes += ({},) * len(self.extra_arg_names)
            self._dynamic_shapes = dynamic_shapes
        return self._dynamic_shapes

    @property
    def num_sequences(self) -> int:
        return len(self._sequence_lengths)

    @property
    def sequence_lengths(self) -> List[int]:
        return self._sequence_lengths

    @property
    def input_positions(self) -> List[int]:
        return self.input_pos[: self.num_sequences].tolist()

    @property
    def is_generate(self) -> bool:
        return all(sl == 1 for sl in self.sequence_lengths)

    @property
    def num_pages(self) -> int:
        return self._num_pages

    @property
    def is_paged(self) -> bool:
        return self.page_size < self.max_seq_len

    @property
    def max_num_tokens(self) -> int:
        return self.max_batch_size * self.max_seq_len

    @property
    def page_assignments(self) -> List[List[int]]:
        """Return the page assignments for each sequence."""
        pages_per_seq = self.pages_per_seq[: self.num_sequences].tolist()
        return [
            c_loc_one_seq.tolist()
            for c_loc_one_seq in torch.split(self.cache_loc[: sum(pages_per_seq)], pages_per_seq)
        ]

    @classmethod
    def _get_sanitized_seq_len(cls, input_ids: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
        """Sanitize sequence lengths.

        We want to cover the following scenarios with this function:

        1. Pre-fill:
            input_ids: [1, s_total, ...]
            seq_len: [s_0, s_1, ..., s_{b-1}, 0, 0, ..., 0]
            ---> returns [s_0, s_1, ..., s_{b-1}]
        2. Decode:
            input_ids: [b, 1, ...]
            seq_len: [1, 1, ..., 1, 0, 0, ..., ..., ..., ..., 0]
                     |---- b ----|--- (max_batch_size - b) ---|
            --> returns [1,] * b
        3. Decode in Cudagraph:
            input_ids: [b_cudagraph, 1, ...]
            seq_len: [1, 1, ..., 1, 0, 0, ..., ..., ..., ..., 0]
                     |---- b ----|--- (max_batch_size - b) ---|

            --> returns [1,] * b_cudagraph
            Here b <= b_cudagraph. We want to make sure that the seq_len is one-padded to
            b_cudagraph.

            # TODO (lliebenwein): I could see one possible issue with this approach in the future.
            # If we have b < b_cudagraph we now one-pad. However, we don't pad the cache location
            # information. What could happen is that the for the padded sequences the cache location
            # tensors point to allocated pages. This could lead to a situation where we write into
            # allocated cache pages polluting the cache of other sequences. Now this is not an issue
            # if we write the dummy sequences into unallocated cache pages... One fix could be to
            # pad not only the seq len but also pad the cache locations by just repeating the last
            # valid cache location in the batch. This would ensure that the dummy sequences just
            # repeats valid computation...
        """
        _, s = input_ids.shape[:2]
        num_seq = cls._get_sanitized_num_sequences(input_ids, seq_len)
        if s > 1:
            return seq_len[:num_seq].detach().clone()
        else:
            return torch.ones(num_seq, dtype=seq_len.dtype, device=seq_len.device)

    @staticmethod
    def _get_sanitized_num_sequences(input_ids: torch.Tensor, seq_len: torch.Tensor) -> int:
        """Get number of sequences.

        We makes sure that this function is compatible with both torch graph capture and cudagraph.
        Both can be a bit temparamental when trying to extract the number of sequences from a tensor
        with max_batch_size or max_batch_size*max_seq_len.
        """
        b, s = input_ids.shape[:2]
        if s > 1:
            num_seq = torch.sum(seq_len > 0)
            assert seq_len[num_seq:].sum() == 0, "seq_len should be zero-padded"
        else:
            num_seq = b
        return num_seq

    def to(self, *args, **kwargs) -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            if isinstance(val, torch.Tensor):
                setattr(self, f.name, val.to(*args, **kwargs))

    def sync(self, other: "SequenceInfo") -> None:
        for f in fields(self):
            val = getattr(self, f.name)
            val_other = getattr(other, f.name)
            if f.name == "input_ids":
                setattr(self, f.name, val_other.to(self.device))
            elif f.name == "_sequence_lengths":
                self._sequence_lengths = val_other
            elif isinstance(val, torch.Tensor):
                val[: len(val_other)] = val_other.to(self.device)
            else:
                assert val == val_other, f"Field {f.name} mismatch: {val} != {val_other}."

    def reset(self) -> None:
        """Reset the sequence information.

        After reset the sequence information should correspond to a "generate-only" batch of
        sequences (b, s==1) without cache history.
        """
        # set a dummy sequence corresponding to a generate-only batch
        self.nest_sequences(torch.zeros(self.max_batch_size, 1, dtype=torch.int))

        # reset cache information
        self.input_pos.zero_()
        self.cache_loc[:] = torch.arange(self.num_pages, dtype=torch.int, device=self.device)
        self.pages_per_seq.fill_(1)

    def _set_example_sequence(self) -> None:
        """Set an example sequence for export purposes."""
        self.reset()
        input_ids = torch.ones(
            min(2, self.max_batch_size),
            min(4, self.max_seq_len),
            dtype=torch.int,
            device=self.device,
        )
        self.nest_sequences(input_ids)
        self.input_ids = input_ids

    def nest_sequences(self, input_ids: Sequence[Sequence[int]]) -> None:
        """Create and store a flattened list of input_ids from the provided list of sequences.

        This i/f will also update any relevant sequence information.
        """
        # set new sequence lengths
        seq_lens = [len(ids) for ids in input_ids]
        self.seq_len.zero_()
        self.seq_len[: len(seq_lens)] = torch.tensor(seq_lens, device=self.device)

        # set new input_ids as new tensor from flattened input_ids
        ids_tnsr_list = [
            lst.detach().to(self.device)
            if isinstance(lst, torch.Tensor)
            else torch.tensor(lst, dtype=torch.int, device=self.device)
            for lst in input_ids
        ]
        self.input_ids = torch.cat(ids_tnsr_list, dim=0)

        # set derivative properties
        self._sequence_lengths = seq_lens

        # use [b,1] shape to indicate generate-only batch, otherwise use [1,total_len]
        if self.is_generate:
            self.input_ids = self.input_ids.view(-1, 1, *self.input_ids.shape[1:])
        else:
            self.input_ids = self.input_ids.view(1, -1, *self.input_ids.shape[1:])

    def unnest_sequences(self, t_nested: torch.Tensor) -> List[torch.Tensor]:
        t_squeezed = t_nested.squeeze(1) if self.is_generate else t_nested.squeeze(0)
        return list(torch.split(t_squeezed, self.sequence_lengths))

    def update_pos(self, seq_len: Union[torch.Tensor, List[int], int], reset: bool = False) -> None:
        """Update the starting position for each sequence in the cache.

        If ``reset=True`, ``input_pos`` will be reset to zero before updating.
        """
        if not isinstance(seq_len, torch.Tensor):
            seq_len = torch.tensor(seq_len, dtype=torch.int)
        bs = len(seq_len) if seq_len.dim() > 0 else self.max_batch_size

        if reset:
            self.input_pos[:bs] = seq_len.to(self.device)
        else:
            self.input_pos[:bs] += seq_len.to(self.device)

    def assign_cache_loc(self, page_assignments: Sequence[Sequence[int]]) -> None:
        """Set the cache location and pages_per_seq tensors from page assignments."""
        cache_loc_flat = torch.tensor(
            [p_idx for pages in page_assignments for p_idx in pages], dtype=torch.int
        )
        self.cache_loc[: len(cache_loc_flat)] = cache_loc_flat.to(self.device)

        pages_per_seq = torch.tensor([len(p) for p in page_assignments], dtype=torch.int)
        self.pages_per_seq[: len(pages_per_seq)] = pages_per_seq.to(self.device)


Constant = Union[int, float, str, None]


class MHACallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *metadata_and_caches: Union[torch.Tensor, Constant],
    ) -> torch.Tensor: ...


class PrepareMetadataCallable(Protocol):
    def __call__(
        self,
        input_ids: torch.Tensor,
        seq_len: torch.Tensor,
        input_pos: torch.Tensor,
        cache_loc: torch.Tensor,
        pages_per_seq: torch.Tensor,
        page_size: int,
    ) -> List[torch.Tensor]: ...


class GetCacheCallable(Protocol):
    def __call__(self, sequence_info: SequenceInfo) -> torch.Tensor: ...


class GetBufferCallable(GetCacheCallable):
    pass


class GetAttentionInfo(Protocol):
    def __call__() -> AttentionInfo: ...


CacheInitializerDict = Dict[str, GetCacheCallable]
BufferInitializerDict = Dict[str, GetBufferCallable]


class AttentionDescriptor(ABC):
    """An interface to define a functional attention operator.

    The main logic is contained with the actual attention op as well as the prepare_metadata op. The
    prepare_metadata op is responsible for converting the standardized sequence info into metadata
    specific to the attention op.
    """

    @classmethod
    @abstractmethod
    def is_paged(cls) -> bool:
        """Return if the attention op is paged or not."""

    @classmethod
    def get_attention_op(cls) -> MHACallable:
        """Get the attention op.

        The attention_op should follow the below signature:

        ```
        def attention_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *metadata,  # global info about the sequences as returned by the prepare_metadata op
            *caches,    # contains layer-specific caches per provided cache initializers
            *buffers,   # global buffers used by the attention op as provided by buffer initializers
            *constants, # basic arguments (int, float, str, None) added as CONSTANTS in the graph
        ) -> torch.Tensor: ...
        ```

        **Note that the attention op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_prepare_metadata_op(cls) -> Tuple[PrepareMetadataCallable, int]:
        """Get the prepare_metadata op.

        The prepare_metadata op should follow the below signature:

        ```
        def prepare_metadata(
            input_ids: torch.Tensor,
            seq_len: torch.Tensor,
            input_pos: torch.Tensor,
            cache_loc: torch.Tensor,
        ) -> List[torch.Tensor]: ...
        ```
        The metadata should contain all necessary global information required for the underlying
        attention op to process the input sequence and the returned list of tensors will be passed
        on to each invocation of the attention op in the graph.

        prepare_metadata is called once at the beginning of the forward pass.

        **Note that the prepare_metadata op should be a valid torch custom op, which comes with
        restrictions on the supported types in the signature.**
        """
        return NotImplementedError

    @classmethod
    @abstractmethod
    def get_cache_initializers(cls, get_info: GetAttentionInfo) -> CacheInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize the caches.

        The key corresponds to the argument name used in the attention op signature. The function
        key doesn't need to be unique across multiple attention nodes in the graph. The key used to
        describe the cache in the graph will be patched with the attention node index to ensure
        uniqueness.

        ``get_cache_initializers`` will be called *once* after the model initialization and before
        the initial forward pass for each attention op detected in the graph. The caches will be
        managed by the global CacheManager and passed back to the attention op during the forward
        pass.

        If the cache initializer requires information about the attention op, the ``get_info``
        function can be called **inside** the cache initializer to retrieve the necessary
        information.
        """
        raise NotImplementedError

    @classmethod
    def get_global_buffer_initializers(cls, get_info: GetAttentionInfo) -> BufferInitializerDict:
        """Provide a dictionary of function pointers that can be used to initialize buffers.

        The key corresponds to the buffer name used in the graph module and will **not**
        be patched unlike a cache key. Hence, it is a **global** key that is shared across all
        attention ops in the model much like a regular buffer in an nn.Module. That means if this
        i/f is called for multiple attention ops, the same buffer will be shared across all of them
        if this function provides the same key multiple times.

        Buffers are initialize *once* after the model initialization and before the initial forward
        pass for each attention op detected in the graph. The buffer will be managed by the global
        CacheManager and passed back to the attention op during the forward pass.

        If the buffer initializer requires information about the attention op, the ``get_info``
        function can be called **inside** the buffer initializer to retrieve the necessary
        information.
        """
        return {}

    @classmethod
    def get_constants(cls, attention_info: AttentionInfo) -> List[Constant]:
        """Provide a list of constant arguments to be passed to the attention op.

        The constant arguments are passed to the attention op as additional arguments after the
        caches and buffers. The constants are expected to be of type int, float, str, or None.
        """
        return []


class AttentionRegistry:
    """A simple registry to look up different attention implementations."""

    _attention_registry: Dict[str, Type["AttentionDescriptor"]] = {}

    @classmethod
    def register(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        def decorator(attention_cls: Type["AttentionDescriptor"]):
            assert kernel_source not in cls._attention_registry, (
                f"Attention source {kernel_source} already registered."
            )
            cls._attention_registry[kernel_source] = attention_cls
            return attention_cls

        return decorator

    @classmethod
    def get(cls, kernel_source: str) -> Type["AttentionDescriptor"]:
        assert cls.has(kernel_source), f"Attention source {kernel_source} not registered."
        return cls._attention_registry[kernel_source]

    @classmethod
    def has(cls, kernel_source: str) -> bool:
        return kernel_source in cls._attention_registry
