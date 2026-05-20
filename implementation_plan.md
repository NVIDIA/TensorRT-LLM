Context: we want to add support for beam search decoding in disaggregated serving using KVCacheManager v1 and KVCacheTransceiver v2.
The case of running in conditional_disagg_config mode and context server is skipped should be covered by IFB beam search support.

1. 
Modify the functions in this stack:
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
I think there is only one request throughout this whole stack.
Make beam_width accessible in _create_kv_slice and pass it to get_block_ids and get_batch_cache_indices. Remove beam dimension flattening and beam demension == 1 assertion in get_batch_cache_indices. Modify function indexing to support beam_width > 1 (this will be an extra dimension in the block indices tensor).

2.
Context server needs to include log probabilities of beam_search_store in DisaggregatedParams.first_gen_log_probs. Generation server will receive DisaggregatedParams and append the log probabilities to its initial beam_search_store. Possible cleanup opportunity: remove the current handling of req.py_result.append_log_probs. Do the same for logits if they exist in beam_search_store?


context beam_search_store.log_probs -> DisaggregatedParams.first_gen_log_probs
decode DisaggregatedParams.first_gen_log_probs -> beam_search_store

3. 
Generation server needs to allocate extra blocks for the extra beams (check if this is already happening)











Caveats to be aware of (not part of this fix)
The downstream transfer pipeline (Sender._build_kv_write_meta, _align_kv_blocks, RecvReqInfo.to_bytes/from_bytes) treats each block_ids_per_layer_groups[i] as a 1-D suffix where src_start = (total_blocks − size) * tpb. With beam_width > 1, block_ids.size = K + (beam_width − 1) so src_start ends up negative-ish (total_blocks − K − (beam_width − 1)), and _align_kv_blocks will transfer the per-beam tail block IDs as if they belonged to the prompt-block suffix — which is wrong for the receiver side, which still allocates only beam-0-sized prompt blocks via the V1 path. So this fix unblocks the immediate FIXME but the sender's src_start/dst_start math and the receiver's RecvReqInfo construction need a parallel update (either also append per-beam tails on dst, or move the per-beam tail handling into a dedicated channel rather than reusing the same block_ids array). Worth flagging before sending a request with beam_width > 1 end-to-end.