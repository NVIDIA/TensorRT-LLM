Call chain:
```

tensorrt_llm/_torch/disaggregation/transceiver.py
respond_and_send_async
->
tensorrt_llm/_torch/disaggregation/transceiver.py
KvCacheTransceiverV2::_create_kv_slice 
->
tensorrt_llm/_torch/disaggregation/resource/cache_reuse.py
_CacheReuseAdapterV1::get_block_ids
->
tensorrt_llm/_torch/pyexecutor/resource_manager.py
KVCacheManager::get_batch_cache_indices

resource_manager.KVCacheManager.get_batch_cache_indices → self.impl.get_batch_cache_block_ids(...) → C++ KVCacheManager::getBatchCacheBlockIds → sequence.getCacheBlockIds(windowSize).
```

cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp KVCacheManager
```
    def get_batch_cache_indices(
        self,
        request_ids: List[int],
        layer_idx: Optional[int] = None,
    ) -> List[List[int]]:
        if layer_idx is None:
            if len(self.max_attention_window_vec) > 1:
                raise ValueError("layer_idx must be provided for VSWA")
            window_size = self.max_attention_window_vec[0]
        else:
            layer_offset = self.layer_offsets[layer_idx]
            window_size = self.max_attention_window_vec[layer_offset % len(
                self.max_attention_window_vec)]

        result = self.impl.get_batch_cache_block_ids(request_ids, window_size)
        for i in range(len(result)):
            assert (len(result[i])) == 1
            result[i] = result[i][0]
        return result

# Downstream uses include attention metadata (block_ids_per_seq), get_block_ids_per_seq (pads into a tensor), and disagg CacheReuseAdapterV1.get_block_ids.
```

need to modify get_batch_cache_indices so that it doesn't flatten the beams. Need to modify all functions in this call stack:

```
tensorrt_llm/_torch/disaggregation/transceiver.py
KvCacheTransceiverV2::_create_kv_slice 
->
tensorrt_llm/_torch/disaggregation/resource/cache_reuse.py
_CacheReuseAdapterV1::get_block_ids
->
tensorrt_llm/_torch/pyexecutor/resource_manager.py
KVCacheManager::get_batch_cache_indices
```
should I make beamWidth accessible to the transceiver? It's in cpp/tensorrt_llm/batch_manager/kvCacheManager.cpp so maybe it makes sense? 

beam search terminology:
beam_width=best_of=4: “Search with 4 live hypotheses every decode step.”
n=2: “When done, give me the 2 best sequences,” not all 4.
max_beam_width=8 → memory and tensors allow up to 8 beams per slot

max_beam_width is specified during Server / model / executor setup (e.g. BaseLlmArgs, build config) while beam_width is specified per request. this way, one server can handle many request types (not all requests have the same beam_width i.e. not all requests use beam search)


end to end picture: we already have support for sending first_gen_log_probs and first_gen_logits
```
Context worker
  ├─ KV transceiver  → GPU KV blocks (+ gen-first: aux int32 tokens/draft)
  └─ HTTP response   → DisaggregatedParams:
                         first_gen_tokens, first_gen_log_probs, first_gen_logits, …

Disagg orchestrator
  └─ Copies disaggregated_params into generation_only HTTP request

Gen worker
  ├─ KV transceiver  → receive KV
  ├─ add_new_token(first_gen_tokens)     ← token ID (not re-sampled)
  └─ append_log_probs / append_generation_logits  ← from py_disaggregated_params, not transceiver
  ```

No. Beam search does not read req.py_result to pick the best beams each step. Selection uses separate GPU/decoder state; py_result is mainly for what you return (and for fixing token/logprob histories after beams reorder).

Where beam scoring actually happens (PyTorch / IFB)
Each decode step, TorchSampler passes beam_search_store.cum_log_probs into beam_search_sampling_batch via BeamSearchMetadata:
```
metadata = BeamSearchMetadata(
    cache_indirection=beam_search_store.cache_indirection,
    cache_indirection_buffer=beam_search_store.cache_indirection_buffer,
    cum_log_probs=beam_search_store.cum_log_probs,
```
So the running “best beam” scores live in BeamSearchStore.cum_log_probs, keyed by py_seq_slot, not in py_result.

Wait but request.py_result log probs come from BeamSearchStore? So surely we can just convert again right? Or just store BeamSearchStore in the DisaggregatedParams instead of py_result in the first place instead of converting and reverting.
```
request.py_result.set_log_probs(
    gen_log_probs_list, cum_log_probs=beam_history.cum_logprobs.tolist()
```

If conditional_disagg_config and context server is skipped then this whole thing proceeds like IFB beam search: this is already supported without any changes to make disaggregated serving support beam search.


result[request][beam][block]

result
[
1, 2, 3, 4, 5
1, 2, 3, 4, 6
1, 2, 3, 4, 7
]

option 1:
result
[
1, 2, 3, 4, 5
6, 7
]

option 2:
transceiver: check for duplicate indices, dont send duplicates

0  1 2  3 4 5 
{}{}{}{}{}{x}{}{}{}

I have modified get_batch_cache_indices to never flatten the beam_width dimension. Read tensorrt_llm/_torch/disaggregation/transceiver.py and tensorrt_llm/_torch/pyexecutor/resource_manager.py and then modify _create_kv_slice to handle the new dimension correctly. Update _CacheReuseAdapterV2 to take beam_width


## initial implementation

debug messages
```
ctx:

[05/19/2026-14:25:08] [TRT-LLM] [I] [_torch  ] get_batch_cache_indices raw block_ids: request_ids=[353039620337664] layer_idx=0 window_size=2048 beam_width=4 tokens_per_block=32 result=[tensorrt_llm.bindings.internal.batch_manager.CacheBlockIds([[22183, 22184, 22185], [22183, 22184, 22186], [22183, 22184, 22187], [22183, 22184, 22188]])] token_layout=[{'request_id': 353039620337664, 'sequence_num_tokens': 91, 'beams': [{'block_ids': [22183, 22184, 22185], 'num_tokens': 91, 'tokens_per_block': 32, 'tokens_in_each_block': [32, 32, 27]}, {'block_ids': [22183, 22184, 22186], 'num_tokens': 91, 'tokens_per_block': 32, 'tokens_in_each_block': [32, 32, 27]}, {'block_ids': [22183, 22184, 22187], 'num_tokens': 91, 'tokens_per_block': 32, 'tokens_in_each_block': [32, 32, 27]}, {'block_ids': [22183, 22184, 22188], 'num_tokens': 91, 'tokens_per_block': 32, 'tokens_in_each_block': [32, 32, 27]}]}]
[05/19/2026-14:25:08] [TRT-LLM] [I] [_torch  ] get_batch_cache_indices flattened block_ids: request_ids=[353039620337664] result=[[22183, 22184, 22185, 22186, 22187, 22188]] token_layout=[{'request_id': 353039620337664, 'sequence_num_tokens': 91, 'tokens_per_block': 32, 'block_ids': [22183, 22184, 22185, 22186, 22187, 22188], 'tokens_in_each_block': [32, 32, 32, 32, 32, -69]}]
[05/19/2026-14:25:08] [TRT-LLM] [I] [_torch  ] create_kv_slice block_ids: py_request_id=353039620337664 group_idx=0 block_ids=[22183, 22184, 22185, 22186, 22187, 22188]

proxy:
[05/19/2026-14:25:00] [TRT-LLM] [I] [serve   ] server is ready with info: {'disaggregated_params': {'ctx_dp_rank': 0, 'ctx_info_endpoint': ['tcp://10.117.11.164:37771']}}
[05/19/2026-14:25:00] [TRT-LLM] [I] [serve   ] All servers are ready
INFO:     Application startup complete.
[05/19/2026-14:25:08] [TRT-LLM] [E] [serve   ] Internal server error:  Traceback (most recent call last):
  File "/home/scratch.itabrizian_sw/athenac/TensorRT-LLM/tensorrt_llm/serve/openai_disagg_server.py", line 216, in wrapper
    response_or_generator = await entry_point(req, hooks)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.itabrizian_sw/athenac/TensorRT-LLM/tensorrt_llm/serve/openai_disagg_service.py", line 115, in openai_completion
    return await self._send_disagg_request(request, hooks)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/scratch.itabrizian_sw/athenac/TensorRT-LLM/tensorrt_llm/serve/openai_disagg_service.py", line 148, in _send_disagg_request_ctx_first
    await self._verify_ctx_response(ctx_response)
  File "/home/scratch.itabrizian_sw/athenac/TensorRT-LLM/tensorrt_llm/serve/openai_disagg_service.py", line 511, in _verify_ctx_response
    raise ValueError(
ValueError: Context server returned 4 choices, expecting 1.

INFO:     127.0.0.1:44784 - "POST /v1/completions HTTP/1.1" 500 Internal Server Error
```