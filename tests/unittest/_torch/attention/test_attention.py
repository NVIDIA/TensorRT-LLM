import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import flashinfer
import pytest
import torch
from utils.util import getSMVersion

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import (AttentionBackend,
                                                   FlashInferAttention,
                                                   VanillaAttention)
from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


atol = 1e-2
rtol = 1e-3
fp8_atol = 5e-2


@dataclass(kw_only=True, frozen=True)
class Scenario:
    dtype: torch.dtype = torch.float16
    kvcache_dtype: torch.dtype = torch.float16
    num_layers: int
    num_heads: int = 64
    num_kv_heads: int = 16
    head_dim: int = 128
    page_size: int = 256
    """flash-attention requires `page_size` to be a multiple of 256"""
    num_pages: int = 4
    qo_len: int = 32
    """setting kv_len to non-zero to test cross attention"""
    kv_len: int = 0
    causal: bool = True
    batch_size: int = 7

    @property
    def cross(self) -> bool:
        return self.kv_len != 0

    @property
    def kv_len_resolved(self) -> int:
        return self.kv_len or self.qo_len

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def kv_cache_len(self) -> int:
        return self.page_size * self.num_pages

    @property
    def past_kv_len(self) -> int:
        return self.kv_cache_len - self.kv_len_resolved

    @property
    def max_num_pages(self) -> int:
        return self.batch_size * self.num_pages

    @property
    def nnz_qo(self):
        return self.batch_size * self.qo_len

    @property
    def nnz_kv(self):
        return self.batch_size * self.kv_len_resolved

    def __post_init__(self) -> None:
        assert self.kv_len <= self.kv_cache_len, "KV len larger than cache len"
        assert self.kv_len != 0 or self.qo_len <= self.kv_cache_len, "Seq len larger than cache len"
        assert not (self.cross
                    and self.causal), "Cross attention cannot be causal"


@dataclass(kw_only=True, frozen=True)
class PagedScenario(Scenario):
    num_generations: int

    @property
    def num_contexts(self) -> int:
        return self.batch_size - self.num_generations

    @property
    def num_ctx_q_tokens(self) -> int:
        return self.num_contexts * self.qo_len

    @property
    def num_ctx_kv_tokens(self) -> int:
        return self.num_contexts * self.kv_len_resolved

    @property
    def nnz_qo(self) -> int:
        return self.num_ctx_q_tokens + self.num_generations

    @property
    def nnz_kv(self) -> int:
        n = self.num_ctx_kv_tokens
        if not self.cross:
            n += self.num_generations
        return n


paged_backends = {
    VanillaAttention: False,
    FlashInferAttention: True,
}


def kv_cache_manager_from(Attention: type[AttentionBackend], s: Scenario,
                          kv_cache: torch.Tensor) -> KVCacheManager:
    paged = paged_backends[Attention]

    num_blocks = s.max_num_pages if paged else s.batch_size
    tokens_per_block = s.page_size if paged else s.kv_cache_len
    num_layers = s.num_layers
    num_kv_heads = s.num_kv_heads
    head_dim = s.head_dim
    max_seq_len = num_blocks * tokens_per_block
    batch_size = s.batch_size

    if s.kvcache_dtype == torch.float16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.HALF
    elif s.kvcache_dtype == torch.bfloat16:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.BF16
    elif s.kvcache_dtype == torch.float8_e4m3fn:
        kv_cache_dtype = tensorrt_llm.bindings.DataType.FP8
    else:
        raise ValueError("Invalid dtype for unit test")

    kv_cache_config = KvCacheConfig(max_tokens=num_blocks * tokens_per_block)

    mapping = Mapping(world_size=1, tp_size=1, rank=0)

    cache_type = tensorrt_llm.bindings.internal.batch_manager.CacheType.CROSS if s.cross else tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF

    result = KVCacheManager(
        kv_cache_config,
        cache_type,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
        mapping=mapping,
        dtype=kv_cache_dtype,
    )

    for i in range(s.num_layers):
        result.get_buffers(i).view_as(kv_cache[i]).copy_(kv_cache[i])
    return result


def produce_outputs(
    Attention: type[AttentionBackend],
    q_at_layer: torch.Tensor,
    kv: Optional[torch.Tensor],
    s: Scenario,
    *,
    kv_cache: torch.Tensor,
    num_cached_tokens: Callable[[int], int] | int,
    num_contexts: int | None = None,
    seq_lens: torch.Tensor,
    seq_lens_kv: Optional[torch.Tensor] = None,
    quant_config: Optional[QuantConfig] = None,
) -> list[torch.Tensor]:
    num_cached_tokens_per_seq = [
        num_cached_tokens
        if isinstance(num_cached_tokens, int) else num_cached_tokens(i)
        for i in range(s.batch_size)
    ]

    kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=num_cached_tokens_per_seq)
    kv_cache_manager = kv_cache_manager_from(Attention, s, kv_cache)
    request_ids = list(range(s.batch_size))
    seq_lens_append = seq_lens_kv if seq_lens_kv is not None else seq_lens
    token_nums = (torch.tensor(num_cached_tokens_per_seq) +
                  seq_lens_append).tolist()
    kv_cache_manager.add_dummy_requests(request_ids, token_nums)

    metadata = Attention.Metadata(
        num_contexts=num_contexts if num_contexts is not None else s.batch_size,
        kv_cache_params=kv_cache_params,
        seq_lens=seq_lens,
        seq_lens_kv=seq_lens_kv,
        max_num_requests=s.batch_size,
        max_num_tokens=8192,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        prompt_lens=token_nums,
    )
    metadata.prepare()
    mask = PredefinedAttentionMask.CAUSAL if s.causal else PredefinedAttentionMask.FULL
    outputs = []
    for i in range(s.num_layers):
        q = q_at_layer[i]
        if kv is not None:
            k, v = kv[i][0], kv[i][1]
        else:
            k, v = None, None
        attention = Attention(
            layer_idx=i,
            num_heads=s.num_heads,
            num_kv_heads=s.num_kv_heads,
            head_dim=s.head_dim,
            quant_config=quant_config,
        )
        o = attention.forward(q, k, v, metadata, attention_mask=mask)
        assert list(o.shape) == [s.nnz_qo, s.num_heads * s.head_dim]
        outputs.append(o)
    kv_cache_manager.shutdown()
    return outputs


def allclose(ref: Sequence[torch.Tensor],
             impls: dict[str, Sequence[torch.Tensor]],
             *,
             layer=0,
             atol=atol,
             rtol=rtol):
    for name, outputs in impls.items():
        print(f"{name} output: ", float(outputs[layer].abs().mean()))
    print("ref outputs: ", float(ref[layer].abs().mean()))
    for name, outputs in impls.items():
        print(f"{name} & ref diff: ",
              float((ref[layer] - outputs[layer]).abs().mean()))
    for name, outputs in impls.items():
        torch.testing.assert_close(outputs[layer],
                                   ref[layer],
                                   atol=atol,
                                   rtol=rtol,
                                   msg=f"Allclose failed: ref<->{name}"),


def test_flashinfer_prefill():
    s = Scenario(num_layers=1)
    dtype = s.dtype
    num_layers = s.num_layers
    num_qo_heads = s.num_heads
    num_kv_heads = s.num_kv_heads
    num_kv_groups = s.num_kv_groups
    head_dim = s.head_dim
    page_size = s.page_size
    num_pages = s.num_pages
    kv_cache_len = s.kv_cache_len
    qo_len = s.qo_len
    past_kv_len = s.past_kv_len
    batch_size = s.batch_size
    nnz_qo = s.nnz_qo
    max_num_pages = s.max_num_pages

    # allocate 128MB workspace buffer
    workspace_buffer = torch.empty(128 * 1024 * 1024,
                                   dtype=torch.uint8,
                                   device="cuda")
    paged_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2")
    ragged_wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2")
    paged_kv_indices = torch.arange(max_num_pages).int().cuda()
    paged_kv_indptr = torch.arange(0, batch_size + 1).int().cuda() * num_pages
    # 1 <= paged_kv_last_page_len <= page_size
    paged_kv_last_page_len = torch.full((batch_size, ), page_size).int().cuda()
    qo_indptr = torch.arange(0, batch_size + 1).int().to("cuda") * qo_len
    kv_indptr = torch.arange(0, batch_size + 1).int().to("cuda") * kv_cache_len

    q_at_layer = torch.randn(num_layers,
                             nnz_qo,
                             num_qo_heads,
                             head_dim,
                             device="cuda").to(dtype).cuda()
    kv_cache_at_layer = torch.randn(num_layers,
                                    max_num_pages,
                                    2,
                                    page_size,
                                    num_kv_heads,
                                    head_dim,
                                    device="cuda").to(s.kvcache_dtype)
    kv_data = kv_cache_at_layer.transpose(1, 2).contiguous().view(
        num_layers, 2, batch_size, kv_cache_len, num_kv_heads, head_dim)

    causal_mask = torch.full((qo_len, kv_cache_len),
                             fill_value=torch.finfo(dtype).min,
                             dtype=dtype,
                             device="cuda")
    cache_position = torch.arange(past_kv_len, kv_cache_len).cuda()
    bool_causal_mask = torch.arange(
        kv_cache_len).cuda() <= cache_position.reshape(-1, 1)
    causal_mask *= ~bool_causal_mask
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    # create auxiliary data structures for batch prefill attention
    paged_wrapper.plan(
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=True,
    )
    ragged_wrapper.plan(
        qo_indptr,
        kv_indptr,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        causal=True,
    )
    flashinfer_outputs = []
    for i in range(num_layers):
        q = q_at_layer[i]
        kv_cache = kv_cache_at_layer[i]
        o = paged_wrapper.run(q, kv_cache)
        k = kv_data[i][0]
        v = kv_data[i][1]
        k = k.view(-1, num_kv_heads, head_dim)
        v = v.view(-1, num_kv_heads, head_dim)
        ragged_o = ragged_wrapper.run(q, k, v)
        assert list(o.shape) == [nnz_qo, num_qo_heads, head_dim]
        print("paged output: ", float(o.abs().mean()))
        print("ragged output: ", float(ragged_o.abs().mean()))
        print("paged & ragged diff: ", float((ragged_o - o).abs().mean()))
        assert torch.allclose(o, ragged_o, atol=atol, rtol=rtol)
        flashinfer_outputs.append(o)

    sdpa_outputs = []
    for i in range(num_layers):
        q = q_at_layer[i]
        k = kv_data[i][0]
        v = kv_data[i][1]
        q = q.view(batch_size, qo_len, num_qo_heads, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)
        o = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=causal_mask,
        )
        o = o.transpose(1, 2).contiguous().view(nnz_qo, num_qo_heads, head_dim)
        sdpa_outputs.append(o)

    ref_outputs = []
    for i in range(num_layers):
        q = q_at_layer[i]
        k = kv_data[i][0]
        v = kv_data[i][1]
        q = q.view(batch_size, qo_len, num_qo_heads, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       q.dtype)
        o = torch.matmul(attn_weights, v)

        o = o.transpose(1, 2).contiguous().view(nnz_qo, num_qo_heads, head_dim)
        ref_outputs.append(o)

    allclose(
        ref_outputs,
        {
            "flashinfer": flashinfer_outputs,
            "sdpa": sdpa_outputs,
        },
    )


@pytest.mark.parametrize(
    "s", [
        Scenario(num_layers=1),
        Scenario(num_layers=1, causal=False),
        Scenario(num_layers=1, qo_len=32, kv_len=32, causal=False),
        Scenario(num_layers=1, qo_len=32, kv_len=64, causal=False)
    ],
    ids=["typical", "non-causal", "cross", "cross-diff-kv-len"])
def test_attention_backend(s: Scenario):
    dtype = s.dtype
    num_layers = s.num_layers
    num_heads = s.num_heads
    num_kv_heads = s.num_kv_heads
    num_kv_groups = s.num_kv_groups
    head_dim = s.head_dim
    page_size = s.page_size
    kv_cache_len = s.kv_cache_len
    qo_len = s.qo_len
    kv_len = s.kv_len_resolved
    past_kv_len = s.past_kv_len
    batch_size = s.batch_size
    nnz_qo = s.nnz_qo
    nnz_kv = s.nnz_kv
    causal = s.causal

    q_at_layer = torch.randn(num_layers,
                             nnz_qo,
                             num_heads * head_dim,
                             device="cuda").to(dtype)
    flashinfer_kv_cache = torch.randn(num_layers,
                                      s.max_num_pages,
                                      2,
                                      page_size,
                                      num_kv_heads,
                                      head_dim,
                                      device="cuda").to(s.kvcache_dtype)
    ref_kv_cache = flashinfer_kv_cache.transpose(1, 2).contiguous().view(
        num_layers, 2, batch_size, kv_cache_len, num_kv_heads, head_dim)
    kv = torch.randn(num_layers,
                     2,
                     nnz_kv,
                     num_kv_heads * head_dim,
                     device="cuda").to(dtype)

    def produce(Attention: type[AttentionBackend], kv_cache: torch.Tensor):
        return produce_outputs(
            Attention,
            q_at_layer,
            kv,
            s,
            kv_cache=kv_cache,
            num_cached_tokens=past_kv_len,
            seq_lens=torch.full((batch_size, ), qo_len).int(),
            seq_lens_kv=torch.full(
                (batch_size, ), kv_len).int() if s.cross else None,
        )

    flashinfer_outputs = produce(FlashInferAttention, flashinfer_kv_cache)
    sdpa_outputs = produce(VanillaAttention, ref_kv_cache.transpose(1, 2))

    # Test reference attention
    if causal:
        causal_mask = torch.full((qo_len, kv_cache_len),
                                 fill_value=torch.finfo(dtype).min,
                                 dtype=dtype,
                                 device="cuda")
        cache_position = torch.arange(past_kv_len, kv_cache_len).cuda()
        bool_causal_mask = torch.arange(
            kv_cache_len).cuda() <= cache_position.reshape(-1, 1)
        causal_mask *= ~bool_causal_mask
        causal_mask = causal_mask[None,
                                  None, :, :].expand(batch_size, 1, -1, -1)
    else:
        causal_mask = 0

    ref_outputs = []
    for i in range(num_layers):
        q = q_at_layer[i]
        ref_kv_cache[i][:, :, past_kv_len:kv_cache_len] = kv[i].view(
            2, batch_size, kv_len, num_kv_heads, head_dim)
        k = ref_kv_cache[i][0]
        v = ref_kv_cache[i][1]
        q = q.view(batch_size, qo_len, num_heads, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       q.dtype)
        o = torch.matmul(attn_weights, v)

        o = o.transpose(1, 2).contiguous().view(nnz_qo, num_heads * head_dim)
        ref_outputs.append(o)

    allclose(
        ref_outputs,
        {
            "flashinfer": flashinfer_outputs,
            "sdpa": sdpa_outputs,
        },
    )

    del flashinfer_kv_cache
    del ref_kv_cache
    torch.cuda.empty_cache()


def generate_causal_mask(seq_lens, qo_lens, batch_size, dtype):
    causal_masks = []
    max_seq_len = int(seq_lens.max())
    max_qo_len = int(qo_lens.max())
    for i in range(batch_size):
        kv_len = seq_lens[i]
        qo_len = qo_lens[i]
        past_kv_len = kv_len - qo_len
        causal_mask = torch.full((qo_len, kv_len),
                                 fill_value=torch.finfo(dtype).min,
                                 dtype=dtype,
                                 device="cuda")
        cache_position = torch.arange(past_kv_len, kv_len).cuda()
        causal_mask *= torch.arange(kv_len).cuda() > cache_position.reshape(
            -1, 1)
        causal_mask = torch.nn.functional.pad(
            causal_mask, (0, max_seq_len - kv_len, 0, max_qo_len - qo_len),
            'constant',
            torch.finfo(dtype).min)
        causal_masks.append(causal_mask)
    causal_mask = torch.stack(causal_masks).view(batch_size, 1, max_qo_len,
                                                 max_seq_len)
    return causal_mask


@pytest.mark.parametrize("s", [
    PagedScenario(num_layers=4, num_generations=5),
    PagedScenario(num_layers=4, num_generations=5, kv_len=64, causal=False),
    PagedScenario(
        num_layers=4, num_generations=5, kvcache_dtype=torch.float8_e4m3fn),
    PagedScenario(num_layers=4,
                  num_generations=5,
                  kv_len=64,
                  causal=False,
                  kvcache_dtype=torch.float8_e4m3fn),
],
                         ids=["fp16", "fp16-cross", "fp8", "fp8-cross"])
def test_attention_backend_ifb(s: PagedScenario):
    dtype = s.dtype
    is_fp8 = s.kvcache_dtype == torch.float8_e4m3fn
    if is_fp8 and getSMVersion() < 89:
        pytest.skip("This test is not supported in pre-Ada architecture.")
    torch.manual_seed(0)
    num_layers = s.num_layers
    num_heads = s.num_heads
    num_kv_heads = s.num_kv_heads
    num_kv_groups = s.num_kv_groups
    head_dim = s.head_dim
    page_size = s.page_size
    kv_cache_len = s.kv_cache_len
    past_kv_len = s.past_kv_len
    qo_len = s.qo_len
    kv_len = s.kv_len_resolved
    batch_size = s.batch_size
    num_generations = s.num_generations
    num_contexts = s.num_contexts
    num_ctx_q_tokens = s.num_ctx_q_tokens
    num_ctx_kv_tokens = s.num_ctx_kv_tokens
    nnz_qo = s.nnz_qo
    nnz_kv = s.nnz_kv
    cross = s.cross

    q_at_layer = torch.randn(num_layers, nnz_qo,
                             num_heads * head_dim).half().cuda()
    flashinfer_kv_cache = torch.randn(num_layers,
                                      s.max_num_pages,
                                      2,
                                      page_size,
                                      num_kv_heads,
                                      head_dim,
                                      device="cuda").to(s.kvcache_dtype)
    ref_kv_cache = flashinfer_kv_cache.transpose(1, 2).contiguous().view(
        num_layers, 2, batch_size, kv_cache_len, num_kv_heads, head_dim)
    vanilla_kv_cache = ref_kv_cache.transpose(1, 2).contiguous()
    kv = torch.randn(num_layers,
                     2,
                     nnz_kv,
                     num_kv_heads * head_dim,
                     device="cuda").to(dtype)

    # Test flashinfer attention

    context_lens = torch.full((num_contexts, ), qo_len).int()
    qo_lens = torch.concat([context_lens, torch.ones(num_generations).int()])

    if cross:
        context_lens_kv = torch.full((num_contexts, ), kv_len).int()
        seq_lens_kv = torch.concat(
            [context_lens_kv,
             torch.zeros(num_generations).int()])
    else:
        seq_lens_kv = None

    num_cached_tokens_prefill = past_kv_len
    num_cached_tokens_decode = kv_cache_len - (0 if cross else 1)

    def produce(Attention: type[AttentionBackend], kv_cache: torch.Tensor):
        return produce_outputs(
            Attention,
            q_at_layer,
            kv,
            s,
            kv_cache=kv_cache,
            num_cached_tokens=lambda i: num_cached_tokens_prefill
            if i < num_contexts else num_cached_tokens_decode,
            seq_lens=qo_lens,
            seq_lens_kv=seq_lens_kv,
            num_contexts=num_contexts,
            quant_config=QuantConfig(
                quant_algo=QuantAlgo.FP8,
                kv_cache_quant_algo=QuantAlgo.FP8,
            ) if is_fp8 else None)

    flashinfer_outputs = produce(FlashInferAttention, flashinfer_kv_cache)
    vanilla_outputs = produce(VanillaAttention, vanilla_kv_cache)

    # Test reference attention

    kv_lens = torch.full((batch_size, ), kv_cache_len).int().cuda()
    causal_mask = generate_causal_mask(kv_lens, qo_lens, batch_size,
                                       dtype) if s.causal else 0

    ref_outputs = []
    for i in range(num_layers):
        q = q_at_layer[i]
        ref_kv_cache[i][:, :num_contexts, past_kv_len:kv_cache_len] = kv[
            i][:, :num_ctx_kv_tokens].view(2, num_contexts, kv_len,
                                           num_kv_heads, head_dim)
        if not cross:
            ref_kv_cache[i][:, num_contexts:,
                            -1:] = kv[i][:, num_ctx_kv_tokens:].view(
                                2, num_generations, 1, num_kv_heads, head_dim)
        k = ref_kv_cache[i][0]
        v = ref_kv_cache[i][1]
        ctx_q, gen_q = q.split([num_ctx_q_tokens, num_generations])
        gen_q = torch.nn.functional.pad(gen_q.unsqueeze(1),
                                        (0, 0, 0, qo_len - 1), 'constant',
                                        0).view(num_generations * qo_len,
                                                num_heads * head_dim)
        q = torch.cat([ctx_q, gen_q], dim=0)
        q = q.view(batch_size, qo_len, num_heads, head_dim)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous().to(dtype)
        v = v.transpose(1, 2).contiguous().to(dtype)
        k = repeat_kv(k, num_kv_groups)
        v = repeat_kv(v, num_kv_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
        attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = torch.nn.functional.softmax(attn_weights,
                                                   dim=-1,
                                                   dtype=torch.float32).to(
                                                       q.dtype)
        o = torch.matmul(attn_weights, v)

        o = o.transpose(1, 2).contiguous()
        o = o.view(batch_size, qo_len, -1)
        ctx_o, gen_o = o.split([num_contexts, num_generations], dim=0)
        gen_o = gen_o[:, 0].view(num_generations, num_heads * head_dim)
        ctx_o = ctx_o.view(num_ctx_q_tokens, num_heads * head_dim)
        o = torch.cat([ctx_o, gen_o], dim=0)
        assert list(o.shape) == [nnz_qo, num_heads * head_dim]
        ref_outputs.append(o)

    for i in range(num_layers):
        print(f"validate accuracy for layer {i}")

        allclose(ref_outputs, {
            "flashinfer": flashinfer_outputs,
            "vanilla": vanilla_outputs
        },
                 layer=i,
                 atol=fp8_atol if is_fp8 else atol)
        assert torch.allclose(flashinfer_outputs[i],
                              vanilla_outputs[i],
                              atol=fp8_atol if is_fp8 else atol,
                              rtol=rtol)

    del flashinfer_kv_cache
    del ref_kv_cache
    del vanilla_kv_cache
    torch.cuda.empty_cache()
