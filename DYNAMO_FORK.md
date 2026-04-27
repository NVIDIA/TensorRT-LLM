# Dynamo fork of TensorRT-LLM

This repository is a fork of [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
used by [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo) to land TensorRT-LLM
patches needed for Dynamo — most notably GPU Memory Service (GMS) integration,
shadow-engine failover hooks, and MPI worker bootstrap.

## Ownership

Currently hosted under `galletas1712/TensorRT-LLM` (personal account). The original
plan was to host this under the `ai-dynamo` GitHub org, but:
1. `NVIDIA/TensorRT-LLM` is blocked by NVIDIA org policy from being forked directly
   into `ai-dynamo`.
2. Direct transfer into `ai-dynamo` requires repo-create permissions that the
   current maintainer does not have in the org.

Ownership may be re-homed to `ai-dynamo` later if/when the policy/permissions allow.
Until then, treat this repository as the canonical Dynamo-side TensorRT-LLM fork.

## Branch layout

- `main` — tracks upstream `NVIDIA/TensorRT-LLM:main` unmodified. Sync with
  `git fetch upstream && git push origin upstream/main:main`.
- `dynamo/main` — the patched Dynamo branch. All Dynamo-specific PRs land here.
  Branched from upstream tag `v1.3.0rc11` (commit `4e69c14f732`).
- `dynamo/release/1.3` — release-series branch pinned to the 1.3 line. Consumer
  images in `ai-dynamo/dynamo` pin a specific commit on this branch.

## Opening a PR

- Target `dynamo/main` on this repository (`galletas1712/TensorRT-LLM`).
- Keep upstream-friendly patches small and reviewable — they should be easy to
  port to a PR against `NVIDIA/TensorRT-LLM:main` when NVIDIA is ready to accept
  them.

## How Dynamo consumes this fork

The Dynamo container build pins a specific TensorRT-LLM wheel/commit. The swap
points are in `ai-dynamo/dynamo` at `container/context.yaml`:

```yaml
trtllm:
  pip_wheel: tensorrt-llm==<version>            # PyPI pin (upstream)
  trtllm_wheel_image: <wheel image>             # Release image (upstream)
  github_trtllm_commit: <sha-or-tag>            # Used by install_tensorrt.sh
```

To consume this fork, one of those is swapped to either a fork-built wheel image
or a `pip install git+https://github.com/galletas1712/TensorRT-LLM@<sha>`.
Details and options live in `gms/trtllm-fork-infrastructure-plan.md` in the
Dynamo repo.

## License

Apache-2.0, inherited from upstream NVIDIA/TensorRT-LLM. See `LICENSE`.
