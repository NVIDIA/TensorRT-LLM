import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.transformer.attention import AttentionCallable, MaskedAttentionCallable

# Mirrors the kernel's DEFAULT_BARRIER_TIMEOUT_SECONDS (configs.cuh), which All2All converts to
# cycles via the device peak SM clock. Stored so the timeout can be read back to reset after a raise.
_DEFAULT_ALL2ALL_TIMEOUT_SECONDS = 10.0


class AttentionManager:
    """Owns sequence-parallel attention state for an aggregate token capacity."""

    def __init__(
        self,
        max_tokens: int,
        num_heads: int,
        head_dim: int,
        tensor_dtype: torch.dtype,
        group: torch.distributed.ProcessGroup,
        copy_out_: bool = False,
    ) -> None:
        # Lazy: ltx_kernels is an optional GPU-only dep, and this constructor already
        # requires CUDA -- so importing it here (not at module scope) keeps the multigpu
        # modules importable without the kernels installed (e.g. CPU CI test collection).
        from ltx_kernels import All2All  # noqa: PLC0415

        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.max_tokens = max_tokens
        hidden_dim = num_heads * head_dim
        num_sms = torch.cuda.get_device_properties(self.rank).multi_processor_count
        self.copy_out = copy_out_
        max_rank_tokens = (max_tokens + self.world_size - 1) // self.world_size
        self.all2all_heads, self.all2all_q = (
            All2All(
                rank=self.rank,
                world_size=self.world_size,
                seqlen=max_rank_tokens,
                hidden_dim=hidden_dim,
                num_sms=num_sms,
                tensor_dtype=tensor_dtype,
                group=group,
            )
            for _ in range(2)
        )
        self.all2all_k, self.all2all_v = (
            All2All(
                rank=self.rank,
                world_size=self.world_size,
                seqlen=max_rank_tokens,
                hidden_dim=hidden_dim,
                num_sms=num_sms,
                tensor_dtype=tensor_dtype,
                group=group,
            )
            if not self.copy_out
            else self.all2all_q
            for _ in range(2)
        )
        self.group = group
        self._all2all_timeout_seconds = _DEFAULT_ALL2ALL_TIMEOUT_SECONDS
        # Padded tokens in the current forward's video sequence, one at the end of each of the last
        # `key_padding` shards (see `tile_modality_for_rank`), which the redistributions cut from the
        # gathered keys. Set by the SP model wrapper per forward; host state the block loop branches
        # on, so Dynamo guards on it and the wrapper adds it to the CUDA-graph capture key.
        self.key_padding = 0

    @property
    def all2all_timeout_seconds(self) -> float:
        """The all2all barrier (deadlock-detection) timeout, in seconds, applied to every instance."""
        return self._all2all_timeout_seconds

    @all2all_timeout_seconds.setter
    def all2all_timeout_seconds(self, seconds: float) -> None:
        # Raise it for the first ``torch.compile`` forward -- where one rank's recompile can delay its
        # all2all kernel launch past the steady-state timeout, tripping the barrier -- then reset to
        # the prior value. ``all2all_k``/``all2all_v`` may alias ``all2all_q`` (copy-out path);
        # setting twice is idempotent. Fan out first (it validates) so a rejected value leaves the
        # stored steady-state value untouched.
        for a2a in (self.all2all_q, self.all2all_k, self.all2all_v, self.all2all_heads):
            a2a.set_timeout_seconds(seconds)
        self._all2all_timeout_seconds = seconds

    def send_recv_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_q = self.all2all_q.send_recv_heads(q, copy_out=self.copy_out)
        t_k = self.all2all_k.send_recv_heads(k, copy_out=self.copy_out)
        t_v = self.all2all_v.send_recv_heads(v, copy_out=self.copy_out)
        return t_q, t_k, t_v

    def gather_heads(self, heads_local: torch.Tensor) -> torch.Tensor:
        out = self.all2all_heads.gather_heads(heads_local, copy_out=self.copy_out)
        return out


def drop_padding_keys(t: torch.Tensor, world_size: int, padding: int, dim: int = 1) -> torch.Tensor:
    """Drop the padded tokens from a gathered sequence in shard order along ``dim``.
    Each of the last ``padding`` of the ``world_size`` equal shards ends with one padded token (the
    layout of :func:`~ltx_core.multigpu.transformer.sequence_parallel.tile_modality_for_rank`). With
    one, it is the last token and the result is a view.
    """
    if padding == 0:
        return t
    dim %= t.ndim
    tokens = t.shape[dim]
    if padding == 1:
        return t.narrow(dim, 0, tokens - 1)
    shard = tokens // world_size
    full = (world_size - padding) * shard
    short = t.narrow(dim, full, tokens - full).unflatten(dim, (padding, shard)).narrow(dim + 1, 0, shard - 1)
    return torch.cat([t.narrow(dim, 0, full), short.flatten(dim, dim + 1)], dim=dim)


class _All2AllRedistribute:
    """Shared redistribute/gather pipeline for self-attention SP wrappers.
    Folds the head dim view-and-shuffle so the masked and unmasked variants only
    have to choose how to invoke the inner attention (with or without the mask
    kwarg) -- the rest of the SP plumbing is identical.
    """

    def __init__(self, manager: AttentionManager) -> None:
        self.manager = manager

    def redistribute(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if heads % self.manager.world_size != 0:
            raise ValueError(f"heads ({heads}) must be divisible by world_size ({self.manager.world_size})")

        head_dim = q.shape[-1] // heads
        q = q.view(q.shape[0], q.shape[1], heads, head_dim)
        k = k.view(k.shape[0], k.shape[1], heads, head_dim)
        v = v.view(v.shape[0], v.shape[1], heads, head_dim)

        t_q, t_k, t_v = self.manager.send_recv_qkv(q, k, v)
        local_heads = heads // self.manager.world_size
        # The gathered keys hold the sharding's padded tokens; cut them here, so attention only ever
        # sees real keys. Queries keep them: their rows are dropped when the outputs are gathered.
        t_k = drop_padding_keys(t_k, self.manager.world_size, self.manager.key_padding)
        t_v = drop_padding_keys(t_v, self.manager.world_size, self.manager.key_padding)

        # `flatten` / `unflatten` collapse only the head dims, avoiding a `-1` in the
        # seq position -- that would otherwise be ambiguous if the seq is 0 for a
        # zero-token modality.
        t_q = t_q.flatten(-2)
        t_k = t_k.flatten(-2)
        t_v = t_v.flatten(-2)
        return t_q, t_k, t_v, local_heads, head_dim

    def gather(self, hidden_states: torch.Tensor, local_heads: int, head_dim: int) -> torch.Tensor:
        hidden_states = hidden_states.unflatten(-1, (local_heads, head_dim))
        hidden_states = self.manager.gather_heads(hidden_states)
        return hidden_states.flatten(-2)


class All2AllAttention(AttentionCallable):
    def __init__(self, manager: AttentionManager, original_attention: AttentionCallable):
        self._sp = _All2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        t_q, t_k, t_v, local_heads, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=local_heads)
        return self._sp.gather(hidden_states, local_heads, head_dim)


class MaskedAll2AllAttention(MaskedAttentionCallable):
    """Masked SP attention, for a caller's own mask.
    The mask is in the shard order the keys are gathered in
    (:func:`~ltx_core.multigpu.transformer.sequence_parallel.tile_modality_for_rank`); its columns for
    the padded keys the redistribution cuts are cut with them.
    """

    def __init__(self, manager: AttentionManager, original_attention: MaskedAttentionCallable):
        self._sp = _All2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_q, t_k, t_v, local_heads, head_dim = self._sp.redistribute(q, k, v, heads)
        mask = drop_padding_keys(mask, self._sp.manager.world_size, self._sp.manager.key_padding, dim=-1)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=local_heads, mask=mask)
        return self._sp.gather(hidden_states, local_heads, head_dim)


class _AudioAll2AllRedistribute:
    """Shared redistribute/gather pipeline for audio cross-attention SP wrappers.
    Q is sliced locally per rank (no cross-rank shuffle on Q because the audio
    sequence length is small enough to replicate); K/V are redistributed across
    ranks via ``send_recv_heads``; outputs are gathered via
    ``all_gather_into_tensor`` along the head dimension. The masked and unmasked
    variants share this plumbing and only differ in how they invoke the inner
    attention.
    """

    def __init__(self, manager: AttentionManager) -> None:
        self.manager = manager

    def redistribute(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        if heads % self.manager.world_size != 0:
            raise ValueError(f"heads ({heads}) must be divisible by world_size ({self.manager.world_size})")

        head_dim = q.shape[-1] // heads
        heads_per_rank = heads // self.manager.world_size
        rank = self.manager.rank

        q = q.view(q.shape[0], q.shape[1], heads, head_dim)
        k = k.view(k.shape[0], k.shape[1], heads, head_dim)
        v = v.view(v.shape[0], v.shape[1], heads, head_dim)

        t_q = q[:, :, heads_per_rank * rank : heads_per_rank * (rank + 1), :].clone()
        t_k = self.manager.all2all_k.send_recv_heads(k, copy_out=self.manager.copy_out)
        t_v = self.manager.all2all_v.send_recv_heads(v, copy_out=self.manager.copy_out)
        # The keys are the sharded video sequence: cut its padded tokens, as for self-attention.
        t_k = drop_padding_keys(t_k, self.manager.world_size, self.manager.key_padding)
        t_v = drop_padding_keys(t_v, self.manager.world_size, self.manager.key_padding)

        # `flatten` / `unflatten` collapse only the head dims, avoiding a `-1` in the
        # seq position -- that would otherwise be ambiguous if the seq is 0 for a
        # zero-token modality.
        t_q = t_q.flatten(-2)
        t_k = t_k.flatten(-2)
        t_v = t_v.flatten(-2)
        return t_q, t_k, t_v, heads_per_rank, head_dim

    def gather(self, hidden_states: torch.Tensor, heads_per_rank: int, head_dim: int) -> torch.Tensor:
        # (B, S, heads_per_rank, head_dim). Move head dim to dim 0 so all_gather_into_tensor
        # gathers along it; permute back after the collective.
        hidden_states = hidden_states.unflatten(-1, (heads_per_rank, head_dim)).permute(2, 0, 1, 3).contiguous()
        gathered = torch.empty(
            (heads_per_rank * self.manager.world_size, *hidden_states.shape[1:]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        dist.all_gather_into_tensor(gathered, hidden_states, group=self.manager.group)
        # (heads, B, S, head_dim) -> (B, S, heads, head_dim) -> (B, S, heads * head_dim)
        return gathered.permute(1, 2, 0, 3).flatten(-2)


class AudioAll2AllAttention(AttentionCallable):
    """All2All attention for audio cross-attention (video_to_audio).
    Q is sliced locally per rank, K/V are redistributed via send_recv_heads,
    then outputs are gathered via all_gather across the head dimension.
    """

    def __init__(self, manager: AttentionManager, original_attention: AttentionCallable):
        self._sp = _AudioAll2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        t_q, t_k, t_v, heads_per_rank, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=heads_per_rank)
        return self._sp.gather(hidden_states, heads_per_rank, head_dim)


class MaskedAudioAll2AllAttention(MaskedAttentionCallable):
    """Masked counterpart to :class:`AudioAll2AllAttention`.
    No current caller invokes A2V / V2A cross-attention with a mask, so the SP
    mutator pre-installs an unmasked-only :class:`AudioAll2AllAttention` and the
    masked slot stays at the model default. Defined now so adding masked audio
    cross-attention later is just an SP-mutator change, not a missing-piece
    discovery.
    """

    def __init__(self, manager: AttentionManager, original_attention: MaskedAttentionCallable):
        self._sp = _AudioAll2AllRedistribute(manager)
        self.original_attention = original_attention

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        t_q, t_k, t_v, heads_per_rank, head_dim = self._sp.redistribute(q, k, v, heads)
        hidden_states = self.original_attention(q=t_q, k=t_k, v=t_v, heads=heads_per_rank, mask=mask)
        return self._sp.gather(hidden_states, heads_per_rank, head_dim)
