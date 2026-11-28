# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Engram n-gram hash of DeepSeek-V4.1 (every target's): the compressed vocabulary, the prime bucket sizes of each
Engram layer's table and the look-back hash of a step's positions. It must match the checkpoint's tables bit for bit."""

import os

import numpy as np
import torch

from .limits import (
    ENGRAM_RING,  # noqa: F401  (the targets' look-back ring; routing declines longer drafts)
)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _next_prime(start: int, seen: set) -> int:
    c = start + 1
    while not _is_prime(c) or c in seen:
        c += 1
    return c


def engram_buckets(cfg):
    """-> (primes [layers][max_n-1][heads], flat [layers][cols]): the bucket sizes of every engram layer's table."""
    max_n, n_heads = cfg.engram_max_ngram_size, cfg.engram_n_heads
    # The engine's config carries the bucket base per n-gram size (2-gram .. max_n-gram);
    # the checkpoint's own config.json has one number for all of them.
    base = cfg.engram_vocab_size
    bases = list(base) if isinstance(base, (list, tuple)) else [base] * (max_n - 1)
    assert len(bases) == max_n - 1, (base, max_n)
    primes, seen = [], set()
    for _ in cfg.engram_layer_ids:
        per_n = []
        for ni in range(max_n - 1):
            sizes, cur = [], int(bases[ni]) - 1
            for _ in range(n_heads):
                cur = _next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per_n.append(sizes)
        primes.append(per_n)
    flat = [[p for per_n in layer for p in per_n] for layer in primes]
    # Every layer's table is exactly its buckets laid end to end: a strong check on the
    # prime sequence (and therefore on every hash id this produces).
    for slot, sizes in enumerate(flat):
        assert sum(sizes) == cfg.engram_num_embeddings[slot], (
            slot, sum(sizes), cfg.engram_num_embeddings[slot], bases,
        )  # fmt: skip
    return primes, flat


class EngramHash:
    """Maps positions to the hash ids of the n-grams ending there, per request.

    Token ids go through a compressed vocabulary (tokens that normalize alike
    collapse), each position is hashed with the `max_ngram_size - 1` tokens
    before it, and every (n-gram size, head) pair owns a prime-sized bucket
    range. The look-back tokens come from the request itself (see `__call__`).
    """

    def __init__(self, cfg, checkpoint_dir: str, device):
        from tokenizers import Regex, Tokenizer, normalizers

        self.max_n = cfg.engram_max_ngram_size
        self.n_heads = cfg.engram_n_heads
        self.layer_ids = tuple(cfg.engram_layer_ids)

        tok = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
        sentinel = ""
        norm = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), sentinel),
                normalizers.Strip(),
                normalizers.Replace(sentinel, " "),
            ]
        )
        key_to_new: dict = {}
        lookup = [0] * tok.get_vocab_size(with_added_tokens=True)
        for tid in range(len(lookup)):
            text = tok.decode([tid], skip_special_tokens=False)
            if "�" in text:
                key = tok.id_to_token(tid)
            else:
                normalized = norm.normalize_str(text)
                key = normalized if normalized else text
            new = key_to_new.get(key)
            if new is None:
                new = len(key_to_new)
                key_to_new[key] = new
            lookup[tid] = new
        vocab = len(key_to_new)
        assert vocab == cfg.engram_compressed_vocab_size, (vocab, cfg.engram_compressed_vocab_size)
        self.pad_id = lookup[cfg.engram_pad_token_id]
        image = getattr(cfg, "_v41_image_token_id", None)
        if image is not None:
            # Image positions break the n-grams (stock hashes them with a dead id that turns that look-back and every
            # longer one into the pad id). An image span is at least 4 tokens (`rows * (columns + 1) + 2`, both >= 1),
            # longer than the look-back, so every look-back position past the first image token is an image token too:
            # hashing the image token as the pad id gives stock's ids on every text row, here, in the ring the later
            # steps read (`engram_lookback_hash` included), and in the look-back a new request is seeded with (a
            # generation server's included). This text target refuses a context request with image input
            # (`refuse_image_request`), so image tokens reach it only in a generation-only request of an image prompt.
            assert self.max_n - 1 < 4, "the look-back must stay inside the shortest image span"
            lookup[image] = self.pad_id
        self.token_map = torch.tensor(lookup, dtype=torch.int64, device=device)

        primes, flat = engram_buckets(cfg)
        bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
        mults = []
        for lid in self.layer_ids:
            rng = np.random.default_rng(10007 * lid)
            mults.append(rng.integers(0, bound, size=(self.max_n,), dtype=np.int64) * 2 + 1)
        # [layers, max_n-1, heads] bucket moduli; [layers, max_n]
        self.primes = torch.tensor(primes, dtype=torch.int64, device=device)
        self.mults = torch.tensor(np.stack(mults), dtype=torch.int64, device=device)

    def lookback(self, input_ids, positions, slot, hist) -> torch.Tensor:
        """-> [n, n_engram_layers, (max_n - 1) * n_heads] int64 table rows.

        Graph-safe and host-free. `hist` is a per-request-slot ring of the compressed ids written
        at each absolute position. A look-back position inside this step (a request's tokens are
        contiguous rows of consecutive positions) is read from the step's own row; an earlier one
        from the ring. Then only each request's last `ring` tokens of the step are written (the
        others go to the dummy slot): a context chunk longer than the ring would otherwise write
        colliding positions in an unspecified order. A rejected draft token is simply overwritten
        when the accepted token is processed at its position, so speculative rollback, prefix
        reuse and a generation server that never ran the prompt (seeded by `seed`) all hash the
        same way. Positions before 0 read the pad id.
        """
        ring = hist.shape[1]
        # out-of-range ids read the map's ends, as the hash kernels and the engine's decode hash clamp them
        comp = self.token_map[input_ids.long().clamp(0, self.token_map.shape[0] - 1)]
        pos = positions.long()
        n = comp.shape[0]
        rows = torch.arange(n, device=comp.device)
        cols = [comp]
        for back in range(1, self.max_n):
            pb = pos - back
            j = (rows - back).clamp_min(0)
            inside = (rows >= back) & (slot[j] == slot) & (pos[j] == pb)
            v = torch.where(inside, comp[j], hist[slot, pb % ring].long())
            cols.append(torch.where(pb < 0, torch.full_like(v, self.pad_id), v))
        later = (rows + ring).clamp_max(max(n - 1, 0))
        last = (rows + ring >= n) | (slot[later] != slot) | (pos[later] != pos + ring)
        hist[torch.where(last, slot, hist.shape[0] - 1), pos % ring] = comp.to(hist.dtype)
        toks = torch.stack(cols, dim=-1)  # [n, max_n]
        prods = toks.unsqueeze(1) * self.mults  # [n, layers, max_n]
        rolling, hashes = prods[..., 0], []
        for i in range(1, self.max_n):
            rolling = torch.bitwise_xor(rolling, prods[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1)  # bucket-local: the table adds each head's offset
