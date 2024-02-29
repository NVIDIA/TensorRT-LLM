from pathlib import Path

from tensorrt_llm.executor import GenerationExecutor, GenerationRequest


def test_sync_generation():
    model_path = Path(
        "cpp/tests/resources/models/rt_engine/gpt2/fp16-plugin-packed-paged/tp1-pp1-gpu/"
    )
    tokenizer = "gpt2"
    prompt = "deep learning"
    max_new_tokens = 4
    executor = GenerationExecutor(model_path, tokenizer)

    # Simple generations (synchronous)
    result = executor.generate(prompt, max_new_tokens=max_new_tokens)
    print(result.text)

    results = executor.generate(
        [prompt, prompt], max_new_tokens=[max_new_tokens, 2 * max_new_tokens])
    for result in results:
        print(result.text)

    # Simple generations (asynchronous)
    #
    # Iterate the partial results when streaming
    future = executor.generate_async(prompt,
                                     streaming=True,
                                     max_new_tokens=max_new_tokens)
    for partial_result in future:
        print(partial_result.text)

    # Iterate the partial results when streaming
    # Streaming results in nested loop
    futures = executor.generate_async(
        [prompt, prompt],
        streaming=True,
        max_new_tokens=[max_new_tokens, 2 * max_new_tokens])
    for future in futures:
        for partial_result in future:
            print(partial_result.text)

    # Low-level api with .submit
    # Submit a batch of requests
    futures = []
    for _ in range(5):
        futures.append(
            executor.submit(
                GenerationRequest(prompt, max_new_tokens=[max_new_tokens])))

    print("We have sent the requests: ", [id(f) for f in futures])
    for future in executor.wait_first_completed(futures):
        print(
            f"Request {id(future)} has finished: {future.wait_completion(timeout=0).text}"
        )
