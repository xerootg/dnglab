#!/usr/bin/env python3
"""FastAPI server that converts RAW files to DNG on-the-fly with a disk cache.

Exposes a REST endpoint that accepts a path to a RAW file, converts it to
DNG via the native dnglab_py API, and streams the result back to the client.
Includes ETag / If-None-Match support and a persistent on-disk file cache
so repeated requests for the same file are served instantly.

Run:
    pipenv run server
    # or: uvicorn serve:app --reload --port 8800

Try:
    curl -o photo.dng http://localhost:8800/convert?path=/path/to/IMG_1234.ARW
"""

import asyncio
import hashlib
import io
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

import dnglab_py

CACHE_DIR = Path(os.environ.get("DNG_CACHE_DIR", "/tmp/dnglab_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Optional DCP profile directory
DCP_DIR: str | None = None
for env_key in ("CUSTOM_DCP_DIR", "DCP_DIR"):
    d = os.environ.get(env_key)
    if d and os.path.isdir(d):
        DCP_DIR = d
        break

CHUNK_SIZE = 256 * 1024  # 256 KiB streaming chunks

app = FastAPI(title="dnglab conversion server")


def _cache_key(raw_path: str) -> str:
    """Deterministic cache key from the absolute path + mtime."""
    p = Path(raw_path)
    stat = p.stat()
    blob = f"{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    return hashlib.sha256(blob).hexdigest()


def _etag(cache_key: str) -> str:
    return f'"dng-{cache_key[:16]}"'


def _stream_file(path: Path):
    """Yield file contents in chunks for StreamingResponse."""
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            yield chunk


def _stream_bytes(data: bytes):
    """Yield an in-memory buffer in chunks for StreamingResponse."""
    view = memoryview(data)
    for offset in range(0, len(view), CHUNK_SIZE):
        yield bytes(view[offset : offset + CHUNK_SIZE])


@app.get("/convert")
async def convert_raw(
    request: Request,
    path: str = Query(..., description="Absolute path to the RAW file"),
    embed_raw: bool = Query(False),
    preview: bool = Query(False),
    thumbnail: bool = Query(False),
    compression: str = Query("lossless"),
    crop: str = Query("best"),
    force: bool = Query(False, description="Bypass cache"),
):
    """Convert a RAW file to DNG and stream the result.

    Implements HTTP ETag / If-None-Match caching plus a persistent
    disk cache so identical requests are served instantly.
    """
    raw = Path(path)
    if not raw.is_file():
        raise HTTPException(404, f"File not found: {path}")
    if not dnglab_py.is_supported(path):
        raise HTTPException(
            415, f"Unsupported format: {raw.suffix}"
        )

    key = _cache_key(path)
    etag = _etag(key)

    # --- HTTP 304 shortcut ------------------------------------------------
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(status_code=304)

    headers = {
        "ETag": etag,
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": f'inline; filename="{raw.stem}.dng"',
    }

    # --- Disk-cache hit ----------------------------------------------------
    cached = CACHE_DIR / f"{key}.dng"
    if cached.exists() and not force:
        headers["X-Cache"] = "HIT"
        return StreamingResponse(
            _stream_file(cached),
            media_type="image/x-adobe-dng",
            headers=headers,
        )

    # --- Convert -----------------------------------------------------------
    t0 = time.perf_counter()
    try:
        dng_bytes: bytes = dnglab_py.convert_to_dng(
            path,
            embed_raw=embed_raw,
            preview=preview,
            thumbnail=thumbnail,
            compression=compression,
            crop=crop,
            dcp_dir=DCP_DIR,
        )
    except RuntimeError as exc:
        raise HTTPException(500, f"Conversion failed: {exc}") from exc

    elapsed = time.perf_counter() - t0

    # Write to disk cache (atomic rename to avoid partial reads)
    tmp = cached.with_suffix(".tmp")
    tmp.write_bytes(dng_bytes)
    tmp.rename(cached)

    headers["X-Cache"] = "MISS"
    headers["X-Convert-Time"] = f"{elapsed:.3f}s"

    return StreamingResponse(
        _stream_bytes(dng_bytes),
        media_type="image/x-adobe-dng",
        headers=headers,
    )


@app.get("/supported/cameras")
async def list_cameras():
    """Return the list of supported camera models."""
    return {"cameras": sorted(dnglab_py.supported_cameras())}


@app.get("/supported/extensions")
async def list_extensions():
    """Return the list of supported RAW file extensions."""
    return {"extensions": sorted(dnglab_py.supported_extensions())}


@app.get("/metadata")
async def get_metadata(
    path: str = Query(..., description="Absolute path to the RAW file"),
):
    """Extract EXIF / camera metadata from a RAW file.

    Returns make, model, lens info, and the full EXIF block (exposure,
    ISO, dates, orientation, GPS, etc.) without needing exiftool.
    """
    raw = Path(path)
    if not raw.is_file():
        raise HTTPException(404, f"File not found: {path}")
    if not dnglab_py.is_supported(path):
        raise HTTPException(415, f"Unsupported format: {raw.suffix}")

    try:
        return dnglab_py.raw_metadata(path)
    except RuntimeError as exc:
        raise HTTPException(500, f"Metadata extraction failed: {exc}") from exc


@app.get("/preview")
async def get_preview(
    request: Request,
    path: str = Query(..., description="Absolute path to the RAW file"),
    max_dimension: int | None = Query(None, description="Resize longest side to this many pixels"),
    quality: int = Query(85, ge=1, le=100, description="JPEG quality (1-100)"),
    force: bool = Query(False, description="Bypass cache"),
):
    """Return a JPEG preview of a RAW file.

    Fast path: returns the camera-embedded JPEG when no resize is needed.
    Slow path: decodes the embedded preview (or generates one), optionally
    resizes, and encodes at the requested JPEG quality.  Results are
    cached on disk keyed by (file identity + params).
    """
    raw = Path(path)
    if not raw.is_file():
        raise HTTPException(404, f"File not found: {path}")
    if not dnglab_py.is_supported(path):
        raise HTTPException(415, f"Unsupported format: {raw.suffix}")

    # Build a cache key that includes the resize / quality params.
    file_key = _cache_key(path)
    param_tag = f"d{max_dimension or 0}_q{quality}"
    preview_key = hashlib.sha256(f"{file_key}:{param_tag}".encode()).hexdigest()
    etag = f'"prev-{preview_key[:16]}"'

    # HTTP 304 shortcut
    if_none_match = request.headers.get("if-none-match")
    if if_none_match and if_none_match == etag:
        return Response(status_code=304)

    headers = {
        "ETag": etag,
        "Cache-Control": "private, max-age=3600",
        "Content-Disposition": f'inline; filename="{raw.stem}_preview.jpg"',
    }

    # Disk-cache hit
    cached = CACHE_DIR / f"{preview_key}.jpg"
    if cached.exists() and not force:
        headers["X-Cache"] = "HIT"
        return StreamingResponse(
            _stream_file(cached),
            media_type="image/jpeg",
            headers=headers,
        )

    # Extract preview (GIL is released inside the Rust call).
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        result = await loop.run_in_executor(
            None,
            dnglab_py.extract_preview,
            path,
            max_dimension,
            quality,
        )
    except RuntimeError as exc:
        raise HTTPException(500, f"Preview extraction failed: {exc}") from exc

    if result is None:
        raise HTTPException(
            404, "No embedded preview available and fallback generation failed"
        )

    jpeg_bytes, width, height = result
    elapsed = time.perf_counter() - t0

    # Write to disk cache (atomic)
    tmp = cached.with_suffix(".tmp")
    tmp.write_bytes(jpeg_bytes)
    tmp.rename(cached)

    headers["X-Cache"] = "MISS"
    headers["X-Convert-Time"] = f"{elapsed:.3f}s"
    headers["X-Image-Width"] = str(width)
    headers["X-Image-Height"] = str(height)

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers=headers,
    )


@app.get("/cache/stats")
async def cache_stats():
    """Report disk cache usage."""
    files = list(CACHE_DIR.glob("*.dng"))
    total = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "total_bytes": total,
        "cache_dir": str(CACHE_DIR),
    }


@app.delete("/cache")
async def clear_cache():
    """Purge all cached DNG files."""
    removed = 0
    for f in CACHE_DIR.glob("*.dng"):
        f.unlink(missing_ok=True)
        removed += 1
    return {"removed": removed}
