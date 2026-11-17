"""Fetch the UCI Adult zip, extract adult.data and verify its SHA-256."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import requests


class ChecksumError(RuntimeError):
    """Downloaded file does not match the pinned SHA-256."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_member(url: str, member: str, timeout: int = 60) -> bytes:
    """Download a zip from `url` and return the bytes of `member` inside it."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        return zf.read(member)


def verify(data: bytes, expected_sha256: str) -> str:
    """Return the actual hash; raise ChecksumError if a pinned hash does not match."""
    actual = sha256_bytes(data)
    if expected_sha256 and actual != expected_sha256:
        raise ChecksumError(f"SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def download(url: str, member: str, expected_sha256: str, out_path: Path) -> str:
    data = fetch_member(url, member)
    actual = verify(data, expected_sha256)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return actual
