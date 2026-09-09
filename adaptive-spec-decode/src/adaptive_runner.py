"""
Adaptive Speculative Decoding Runner.
Wraps TensorRT-LLM with our adaptive controller.
"""
import time, json
from pathlib import Path
from dataclasses import asdict
from .acceptance_monitor import AcceptanceMonitor
from .draft_controller import AdaptiveDraftController

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

class AdaptiveSpeculativeRunner:
    def __init__(self, target_path, draft_path, ema_alpha=0.3, max_tokens=200):
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_tokens = max_tokens
        self.monitor = AcceptanceMonitor(ema_alpha=ema_alpha)
        self.controller = AdaptiveDraftController(self.monitor)
        self._trace = []

    def generate(self, prompt, prompt_id=0):
        from tensorrt_llm.llmapi import LLM, SamplingParams, DraftTargetDecodingConfig
        
        decision = self.controller.decide()
        sampling = SamplingParams(max_tokens=self.max_tokens, temperature=0.7)

        if decision.draft_length == 0:
            llm = LLM(model=self.target_path)
            start = time.perf_counter()
            output = llm.generate([prompt], sampling_params=sampling)
            elapsed = time.perf_counter() - start
            ntok = len(output[0].outputs[0].token_ids)
            self.monitor.update(drafted=0, accepted=0)
            del llm
        else:
            spec_cfg = DraftTargetDecodingConfig(
                max_draft_len=decision.draft_length,
                speculative_model=self.draft_path,
            )
            llm = LLM(model=self.target_path, speculative_config=spec_cfg)
            start = time.perf_counter()
            output = llm.generate([prompt], sampling_params=sampling)
            elapsed = time.perf_counter() - start
            ntok = len(output[0].outputs[0].token_ids)
            # Estimate acceptance from throughput
            estimated_accept = min(0.95, max(0.1, ntok / (elapsed * 50)))
            n_cycles = max(1, ntok // max(decision.draft_length, 1))
            self.monitor.update(
                drafted=decision.draft_length * n_cycles,
                accepted=int(estimated_accept * decision.draft_length * n_cycles)
            )
            del llm

        tps = ntok / elapsed
        ptype = "easy" if prompt_id < 4 else ("medium" if prompt_id < 6 else "hard")
        
        result = {
            "id": prompt_id, "type": ptype, "tokens": ntok,
            "elapsed": round(elapsed, 3), "tok_per_sec": round(tps, 2),
            "strategy": decision.strategy, "draft_k": decision.draft_length,
            "acceptance_rate": round(self.monitor.acceptance_rate, 3),
            "trend": self.monitor.acceptance_trend,
            "reason": decision.reason,
        }
        self._trace.append(result)
        print(f"  [{ptype:6s}] Prompt {prompt_id}: {ntok} tok @ {tps:.1f} tok/s "
              f"| strategy={decision.strategy} k={decision.draft_length} "
              f"accept={self.monitor.acceptance_rate:.2f}")
        return result

    def run_all(self, prompts):
        print("\n" + "="*60 + "\nADAPTIVE SPECULATIVE DECODING\n" + "="*60)
        results = []
        for i, prompt in enumerate(prompts):
            results.append(self.generate(prompt, i))
        return {
            "results": results,
            "controller_report": self.controller.get_report(),
        }
