#!/usr/bin/env python3
"""Basic RAW-to-DNG conversion using the dnglab_py native API.

Converts a RAW file to DNG entirely in-memory via the native Python
bindings — no subprocess, no temp files.

Usage:
    python convert.py INPUT.ARW [OUTPUT.dng]
"""

import sys
import time
from pathlib import Path

import dnglab_py


def convert(raw_path: str, out_path: str | None = None) -> None:
    raw = Path(raw_path)
    if not raw.exists():
        print(f"Error: {raw} not found", file=sys.stderr)
        sys.exit(1)

    if not dnglab_py.is_supported(raw_path):
        print(f"Error: {raw.suffix} is not a supported RAW format", file=sys.stderr)
        sys.exit(1)

    if out_path is None:
        out_path = str(raw.with_suffix(".dng"))

    print(f"Converting {raw} -> {out_path}")
    t0 = time.perf_counter()

    # Minimal output: no embedded raw, no preview, no thumbnail
    dng_bytes: bytes = dnglab_py.convert_to_dng(
        raw_path,
        embed_raw=False,
        preview=False,
        thumbnail=False,
    )

    elapsed = time.perf_counter() - t0
    Path(out_path).write_bytes(dng_bytes)
    print(f"Done: {len(dng_bytes):,} bytes in {elapsed:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} INPUT.RAW [OUTPUT.dng]")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
