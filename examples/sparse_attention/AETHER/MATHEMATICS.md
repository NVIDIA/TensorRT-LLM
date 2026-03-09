# AETHER: Mathematical Formulation

This document provides the formal mathematical foundation for the AETHER (Adaptive Event-driven Threshold Hybrid Entangled Rendering) sparse attention mechanism.

## 1. Geometric Abstraction: Cosine-Aware Spaces

The core efficiency of AETHER comes from abstracting groups of Key vectors into directional clusters, recognizing that attention is heavily dependent on cosine similarity (direction) rather than just Euclidean magnitude.

Let the Key matrix $K \in \mathbb{R}^{S \times D}$ be partitioned into $N$ blocks of size $B$. For each block $i \in \{1, ..., N\}$, we compute a unit-normalized centroid $\hat{\mathbf{\mu}}_i$ and a maximum angular error $\theta_i$.

First, we normalize all key vectors to unit length to operate purely on direction:
$$ \hat{\mathbf{k}}_j = \frac{\mathbf{k}_j}{\|\mathbf{k}_j\|} $$

The centroid $\hat{\mathbf{\mu}}_i$ for block $i$ is the unit-normalized mean of the keys in that block:
$$ \mathbf{\mu}_{raw, i} = \frac{1}{B} \sum_{j \in \text{Block}_i} \hat{\mathbf{k}}_j $$
$$ \hat{\mathbf{\mu}}_i = \frac{\mathbf{\mu}_{raw, i}}{\|\mathbf{\mu}_{raw, i}\|} $$

Next, instead of a Euclidean radius, we calculate the maximum angular error (or spread) of the block, $\theta_i$:
$$ \cos(\theta_i) = \min_{j \in \text{Block}_i} (\hat{\mathbf{k}}_j \cdot \hat{\mathbf{\mu}}_i) $$

Thus, every key vector in block $i$ is guaranteed to lie within an angular cone of size $\theta_i$ around the centroid $\hat{\mathbf{\mu}}_i$.

## 2. Derivation of Angular Upper-Bounds

The objective is to strictly bound the maximum possible attention score between a query vector $\mathbf{q}$ and any key in block $i$.

The attention score is proportional to $\mathbf{q} \cdot \mathbf{k}_j = \|\mathbf{q}\| \|\mathbf{k}_j\| \cos(\phi_{q, k_j})$.
Assuming keys are normalized or their magnitudes are bounded, the primary driver is the angle $\phi_{q, k_j}$ between the query and the key.

By triangle inequality of angles on a hypersphere, the angle between the query and any key in the block is bounded by:
$$ \phi_{q, k_j} \ge \max(0, \phi_{q, \mu_i} - \theta_i) $$

Therefore, the maximum possible cosine similarity is:
$$ \cos(\phi_{max}) = \cos(\max(0, \phi_{q, \mu_i} - \theta_i)) $$

Using trigonometric expansion, if the angular error is small, the maximum dot product score is bounded as:
$$ \max_{j \in \text{Block}_i} (\mathbf{q} \cdot \mathbf{k}_j) \leq \|\mathbf{q}\| \left( \hat{\mathbf{q}} \cdot \hat{\mathbf{\mu}}_i + \sin(\theta_i) \right) $$

In practice, a simpler **Cosine-Aware Pruning** heuristic operates directly on the angular distance. We pre-calculate the cosine similarity between the query head $\hat{\mathbf{q}}$ and the block centroid $\hat{\mathbf{\mu}}_i$. If the angular distance exceeds a threshold, the block is pruned:
$$ \text{If } \arccos(\hat{\mathbf{q}} \cdot \hat{\mathbf{\mu}}_i) > \tau_{angle}, \text{ then prune block } i. $$

This approach leverages the fact that trained LLM KV-caches cluster strongly by semantic meaning, and angular distance separates these clusters much more tightly than Euclidean norms.

## 3. High-Order Corrections

To refine the bounds and reduce false positives (blocks selected but with low actual attention), AETHER introduces statistical penalties.

### Variance Penalty
Blocks with high variance (diffuse clusters) are penalized, prioritizing tight clusters where the mean direction is highly representative.
$$ U'_i = U_i \cdot \frac{1}{1 + \sigma^2_i} $$
Where $\sigma^2_i$ is the mean squared distance of keys from the centroid.

### Concentration Factor
The concentration $\kappa_i$ measures the average cosine alignment of keys with the centroid:
$$ \kappa_i = \frac{1}{B} \sum_{j \in \text{Block}_i} (\hat{\mathbf{k}}_j \cdot \mathbf{\mu}_i) $$
This acts as a confidence score for the block's directionality.

## 4. Complexity Analysis

Standard attention has a time complexity of $O(L^2 D)$. AETHER reduces this by decoupling the scoring and retrieval phases.

1.  **Metadata (Offline/Linear)**: $O(L \cdot D)$ to compute means and radii. Done once per generated token or precomputed for prefix.
2.  **Scoring (Sub-quadratic)**: $O(\frac{L}{B} \cdot D)$ per query token. We score $N = L/B$ blocks instead of $L$ tokens.
3.  **sparse Attention**: $O(k \cdot B \cdot D)$ where $k$ is the number of selected blocks ($k \ll N$).

Total complexity per step: $O(\frac{L}{B}D + kBD)$.
For a target sparsity $S$ (e.g., 90%), $k \approx (1-S) \frac{L}{B}$, yielding a theoretical speedup approaching $\frac{1}{1-S}$.
