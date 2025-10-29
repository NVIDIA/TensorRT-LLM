### Get KV Cache Events

from tensorrt_llm import SamplingParams
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm.llmapi import KvCacheConfig


def main():

    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
              tensor_parallel_size=2,
              enable_autotuner=False,
              kv_cache_dtype='auto',
              kv_cache_config=KvCacheConfig(enable_block_reuse=True,
                                            event_buffer_max_size=1024),
              backend="pytorch")

    # Sample prompts having a common prefix.
    common_prefix = (
        "After the ghost's departure, Barnardo notes Horatio's pale appearance and asks if he's okay. "
        "Horatio concedes that he's shaken and confesses that, without witnessing the ghost himself, he wouldn't have believed it existed. "
        "He's also disturbed by the ghost's striking resemblance to the king. It even seems to be wearing the former king's armor. "
        "Horatio thinks the ghost's presence foretells that something is about to go wrong in Denmark. "
        "Marcellus concurs with Horatio, as he and the other guards have observed that their schedules have become more rigorous and have also noticed the preparations taking place within Elsinore, including the building of cannons, the storing of weapons, and the preparation of ships."
    )
    prompts = [
        common_prefix, common_prefix + " Marcellus also notes that the king's"
    ]

    # Create a sampling params.
    sampling_params = SamplingParams(temperature=0.001,
                                     top_p=0.001,
                                     max_tokens=5)

    for output in llm.generate(prompts, sampling_params=sampling_params):
        print(
            f"Prompt: {output.prompt!r}, Generated text: {output.outputs[0].text!r}"
        )

    kv_events = llm.get_kv_cache_events(10)
    print(kv_events)

    # Got output like follows:
    # [{'event_id': 0, 'data': {'type': 'created', 'num_blocks_per_cache_level': [101230, 0]}},
    #  {'event_id': 1, 'data': {'type': 'stored', 'parent_hash': None, 'blocks': [{'type': 'stored_block', 'block_hash': 4203099703668305365, 'tokens': [{'type': 'unique_token', 'token_id': 1, 'token_extra_id': 0}, ...


if __name__ == '__main__':
    main()
