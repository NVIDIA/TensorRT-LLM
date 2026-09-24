"""Baseline: Vanilla vs Static Speculative Decoding (vLLM)"""
import time, json, gc, torch
from pathlib import Path
from vllm import LLM, SamplingParams

PROMPTS = [
    "Write a Python function that returns the factorial of a number:",
    "List the first 20 elements of the periodic table:",
    "Write a SQL query to select all users older than 25:",
    "Translate 'Hello, how are you?' to French, Spanish, and German:",
    "Explain how backpropagation works in simple terms:",
    "Compare TCP and UDP for a junior developer:",
    "Write a short poem about debugging code at 3am:",
    "As an alien visiting Earth, describe a coffee shop:",
    "Pitch a startup that sells dreams:",
    "Argue why 1+1 might not equal 2:",
]

TARGET = "/teamspace/studios/this_studio/models/qwen-2.5-3b"
DRAFT = "/teamspace/studios/this_studio/models/qwen-2.5-0.5b"

def run_bench(name, llm, max_tokens=200):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    sampling = SamplingParams(max_tokens=max_tokens, temperature=0.7)
    results = []
    for i, prompt in enumerate(PROMPTS):
        start = time.perf_counter()
        output = llm.generate([prompt], sampling_params=sampling)
        elapsed = time.perf_counter() - start
        ntok = len(output[0].outputs[0].token_ids)
        tps = ntok / elapsed
        ptype = "easy" if i < 4 else ("medium" if i < 6 else "hard")
        results.append({"id": i, "type": ptype, "tokens": ntok,
                       "elapsed": round(elapsed,3), "tok_per_sec": round(tps,2)})
        print(f"  [{ptype:6s}] {i}: {ntok} tok in {elapsed:.2f}s = {tps:.1f} tok/s")
    return results

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    # VANILLA
    print("Loading vanilla model...")
    llm = LLM(model=TARGET, gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=True)
    vanilla = run_bench("VANILLA", llm)
    del llm; free_gpu()

    # SPEC k=3
    print("\nLoading spec k=3...")
    llm = LLM(model=TARGET, spec_model=DRAFT,
              spec_tokens=3, gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=True)
    s3 = run_bench("SPECULATIVE k=3", llm)
    del llm; free_gpu()

    # SPEC k=5
    print("\nLoading spec k=5...")
    llm = LLM(model=TARGET, spec_model=DRAFT,
              spec_tokens=5, gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=True)
    s5 = run_bench("SPECULATIVE k=5", llm)
    del llm; free_gpu()

    # SPEC k=7
    print("\nLoading spec k=7...")
    llm = LLM(model=TARGET, spec_model=DRAFT,
              spec_tokens=7, gpu_memory_utilization=0.90,
              max_model_len=2048, enforce_eager=True)
    s7 = run_bench("SPECULATIVE k=7", llm)
    del llm; free_gpu()

    # Save
    out = {"vanilla": vanilla, "spec_k3": s3, "spec_k5": s5, "spec_k7": s7,
           "meta": {"target": TARGET, "draft": DRAFT, "engine": "vllm",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}}
    outpath = Path("results/baseline.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for name, res in [("Vanilla",vanilla),("Spec k=3",s3),("Spec k=5",s5),("Spec k=7",s7)]:
        avg = sum(r["tok_per_sec"] for r in res) / len(res)
        easy = sum(r["tok_per_sec"] for r in res if r["type"]=="easy") / 4
        hard = sum(r["tok_per_sec"] for r in res if r["type"]=="hard") / 4
        print(f"  {name:12s} | Avg: {avg:6.1f} tok/s | Easy: {easy:6.1f} | Hard: {hard:6.1f}")
    print(f"\n✅ Saved to {outpath}")

if __name__ == "__main__":
    main()
