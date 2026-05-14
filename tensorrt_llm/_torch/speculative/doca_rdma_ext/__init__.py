"""Packaged DOCA RDMA extension artifacts."""

from pathlib import Path


def build_dir() -> Path:
    return Path(__file__).resolve().parent
