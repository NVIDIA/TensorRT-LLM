import json
import os
import matplotlib.pyplot as plt
import numpy as np

base_path = os.path.expanduser("~/adaptive-spec-decode/results/baseline.json")
adapt_path = os.path.expanduser("~/adaptive-spec-decode/results/adaptive.json")

with open(base_path) as f:
    baseline = json.load(f)

with open(adapt_path) as f:
    adaptive = json.load(f)

categories = ["Easy", "Medium", "Hard", "Overall"]

def get_avgs(dataset):
    easy = sum(r["tok_per_sec"] for r in dataset if r["type"] == "easy") / 4
    med = sum(r["tok_per_sec"] for r in dataset if r["type"] == "medium") / 2
    hard = sum(r["tok_per_sec"] for r in dataset if r["type"] == "hard") / 4
    overall = sum(r["tok_per_sec"] for r in dataset) / len(dataset)
    return [easy, med, hard, overall]

vanilla_scores = get_avgs(baseline["vanilla"])
k3_scores = get_avgs(baseline["spec_k3"])
k5_scores = get_avgs(baseline["spec_k5"])
k7_scores = get_avgs(baseline["spec_k7"])
adapt_scores = get_avgs(adaptive["adaptive_results"])

x = np.arange(len(categories))
width = 0.15

plt.figure(figsize=(10, 6))
plt.bar(x - 2*width, vanilla_scores, width, label="Vanilla", color="#4A5568")
plt.bar(x - width, k3_scores, width, label="Static k=3", color="#CBD5E0")
plt.bar(x, k5_scores, width, label="Static k=5", color="#A0AEC0")
plt.bar(x + width, k7_scores, width, label="Static k=7", color="#E53E3E")
plt.bar(x + 2*width, adapt_scores, width, label="Adaptive Engine", color="#319795")

plt.ylabel("Tokens Per Second")
plt.title("Workload-Aware Adaptive Speculative Decoding Performance")
plt.xticks(x, categories)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()

chart_path = os.path.expanduser("~/adaptive-spec-decode/results/throughput_comparison.png")
plt.savefig(chart_path, dpi=300)
print(f"Saved chart to {chart_path}")
