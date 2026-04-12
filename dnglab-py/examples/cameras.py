#!/usr/bin/env python3
"""List supported cameras and file extensions.

Usage:
    python cameras.py
    python cameras.py --extensions
    python cameras.py --search nikon
"""

import argparse

import dnglab_py


def main() -> None:
    parser = argparse.ArgumentParser(description="Query dnglab supported cameras/formats")
    parser.add_argument("--extensions", action="store_true", help="List file extensions instead of cameras")
    parser.add_argument("--search", "-s", type=str, default=None, help="Case-insensitive filter")
    args = parser.parse_args()

    if args.extensions:
        exts = sorted(dnglab_py.supported_extensions())
        if args.search:
            exts = [e for e in exts if args.search.upper() in e]
        print(f"{len(exts)} extensions:")
        for e in exts:
            print(f"  .{e}")
    else:
        cameras = sorted(dnglab_py.supported_cameras())
        if args.search:
            needle = args.search.lower()
            cameras = [c for c in cameras if needle in c.lower()]
        print(f"{len(cameras)} cameras:")
        for c in cameras:
            print(f"  {c}")


if __name__ == "__main__":
    main()
