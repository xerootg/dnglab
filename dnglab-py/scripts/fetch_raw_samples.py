#!/usr/bin/env python3
"""Fetch RAW camera sample files from raw.pixls.us for testing.

Downloads one CC0-licensed sample per camera make, picking from a curated
list of popular makes to keep the download size manageable.  Files are
saved into a ``test_samples/`` directory next to this script.

Usage
-----
    python scripts/fetch_raw_samples.py [--output-dir DIR] [--max-per-make N] [--makes MAKE,...]

The JSON index is fetched from https://raw.pixls.us/json/getrepository.php?set=all
and individual files are downloaded via the ``getfile.php`` links embedded in
each record.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

INDEX_URL = "https://raw.pixls.us/json/getrepository.php?set=all"

# Camera makes to target — covers the most common brands.
DEFAULT_MAKES = [
    "Canon",
    "Nikon",
    "Sony",
    "FUJIFILM",
    "Olympus",
    "OM Digital Solutions",
    "Panasonic",
    "Pentax",
    "Leica",
    "Samsung",
    "Apple",
    "DJI",
    "GoPro",
    "Hasselblad",
    "Sigma",
    "Ricoh",
]


def fetch_index() -> list[list]:
    """Return the ``data`` array from the raw.pixls.us JSON index."""
    print(f"Fetching index from {INDEX_URL} ...")
    req = urllib.request.Request(INDEX_URL, headers={"User-Agent": "dnglab-test-fetcher/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    entries = body.get("data", [])
    print(f"  {len(entries)} entries in index")
    return entries


def parse_entry(entry: list) -> dict | None:
    """Parse a single JSON array entry into a usable dict.

    Returns None if the entry has no downloadable file or is not CC0.
    """
    if len(entry) < 8:
        return None

    make = html.unescape(entry[0]).strip()
    model = html.unescape(entry[1]).strip()
    mode = html.unescape(entry[2]).strip() if entry[2] else ""
    license_html = entry[5] or ""

    # Only take CC0 / public-domain samples.
    if "CC0" not in license_html and "Public Domain" not in license_html:
        return None

    dl_html = entry[7] or ""
    # Extract href from the download link HTML.
    m = re.search(r"href=['\"]([^'\"]+)['\"]", dl_html)
    if not m:
        return None
    url = html.unescape(m.group(1))
    if not url.startswith("http"):
        url = "https://raw.pixls.us/" + url.lstrip("/")

    # Extract filename from URL or link text.
    fname_match = re.search(r"/nice/(.+?)(?:\?|$)", url)
    if fname_match:
        filename = fname_match.group(1)
    else:
        # Fallback: use link text
        text_match = re.search(r">([^<]+)<", dl_html)
        filename = text_match.group(1).strip() if text_match else None

    if not filename:
        return None

    # Extract SHA256 if present.
    sha_match = re.search(r"SHA256:\s*([0-9a-fA-F]{64})", dl_html)
    sha256 = sha_match.group(1).lower() if sha_match else None

    return {
        "make": make,
        "model": model,
        "mode": mode,
        "url": url,
        "filename": filename,
        "sha256": sha256,
    }


def pick_samples(
    entries: list[list],
    makes: list[str],
    max_per_make: int,
) -> list[dict]:
    """Select up to *max_per_make* samples per camera make."""
    parsed = []
    for e in entries:
        p = parse_entry(e)
        if p:
            parsed.append(p)

    make_lower = {m.lower(): m for m in makes}
    selected: dict[str, list[dict]] = {m: [] for m in makes}

    for p in parsed:
        key = p["make"].lower()
        if key not in make_lower:
            continue
        canonical = make_lower[key]
        if len(selected[canonical]) < max_per_make:
            selected[canonical].append(p)

    result = []
    for m in makes:
        result.extend(selected[m])
    return result


def download_file(url: str, dest: Path, sha256: str | None = None) -> bool:
    """Download *url* to *dest*, verifying SHA256 if provided."""
    if dest.exists():
        if sha256:
            h = hashlib.sha256(dest.read_bytes()).hexdigest()
            if h == sha256:
                print(f"  [skip] {dest.name} (SHA256 matches)")
                return True
            else:
                print(f"  [re-download] {dest.name} (SHA256 mismatch)")
        else:
            print(f"  [skip] {dest.name} (already exists)")
            return True

    # Percent-encode spaces and special characters in the URL path.
    parts = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parts.path, safe="/:@!$&'()*+,;=-._~")
    encoded_url = urllib.parse.urlunsplit(parts._replace(path=encoded_path))

    print(f"  [download] {encoded_url}")
    req = urllib.request.Request(encoded_url, headers={"User-Agent": "dnglab-test-fetcher/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
    except Exception as exc:
        print(f"    FAILED: {exc}")
        return False

    if sha256:
        h = hashlib.sha256(data).hexdigest()
        if h != sha256:
            print(f"    WARNING: SHA256 mismatch (expected {sha256}, got {h})")

    dest.write_bytes(data)
    print(f"    saved {len(data) / 1024 / 1024:.1f} MB")
    return True


def write_manifest(samples: list[dict], output_dir: Path) -> None:
    """Write a manifest.json describing all downloaded samples."""
    manifest = []
    for s in samples:
        safe_name = sanitize_filename(s["filename"])
        path = output_dir / s["make"] / safe_name
        if path.exists():
            manifest.append(
                {
                    "make": s["make"],
                    "model": s["model"],
                    "mode": s["mode"],
                    "filename": safe_name,
                    "path": str(path.relative_to(output_dir)),
                    "sha256": s.get("sha256"),
                }
            )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest written to {manifest_path} ({len(manifest)} files)")


def sanitize_filename(name: str) -> str:
    """Replace characters that are problematic on some filesystems."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RAW samples from raw.pixls.us")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "test_samples",
        help="Directory to save samples into (default: dnglab-py/test_samples/)",
    )
    parser.add_argument(
        "--max-per-make",
        type=int,
        default=2,
        help="Maximum samples to download per camera make (default: 2)",
    )
    parser.add_argument(
        "--makes",
        type=str,
        default=None,
        help="Comma-separated list of makes to fetch (default: built-in list)",
    )
    args = parser.parse_args()

    makes = args.makes.split(",") if args.makes else DEFAULT_MAKES
    output_dir: Path = args.output_dir

    entries = fetch_index()
    samples = pick_samples(entries, makes, args.max_per_make)

    if not samples:
        print("No matching samples found!")
        sys.exit(1)

    print(f"\nSelected {len(samples)} samples across {len(set(s['make'] for s in samples))} makes:\n")
    for s in samples:
        print(f"  {s['make']:20s} {s['model']:30s} {s['filename']}")

    print(f"\nDownloading to {output_dir} ...\n")

    ok = 0
    for s in samples:
        make_dir = output_dir / s["make"]
        make_dir.mkdir(parents=True, exist_ok=True)
        safe_name = sanitize_filename(s["filename"])
        dest = make_dir / safe_name
        if download_file(s["url"], dest, s.get("sha256")):
            ok += 1

    print(f"\nDownloaded {ok}/{len(samples)} files successfully.")
    write_manifest(samples, output_dir)


if __name__ == "__main__":
    main()
