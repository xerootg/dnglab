#!/usr/bin/env python3
"""
Extract ForwardMatrix calibration data from Adobe DCP profiles and apply
to dnglab camera TOML files.

Requires: exiftool (https://exiftool.org/)

Usage:
  # Scan and show what would be updated (dry-run):
  python3 scripts/extract_dcp_profiles.py --dcp-dir /path/to/dcp/files --dry-run

  # Apply ForwardMatrix data to camera TOMLs:
  python3 scripts/extract_dcp_profiles.py --dcp-dir /path/to/dcp/files

  # Process a single DCP file against a specific TOML:
  python3 scripts/extract_dcp_profiles.py --single profile.dcp --toml rawler/data/cameras/nikon/d850.toml

See ADDING_PROFILES.md for full instructions.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CAMERAS_DIR = Path(__file__).resolve().parent.parent / "rawler" / "data" / "cameras"

# Default DCP location (Adobe DNG Converter under Wine)
DEFAULT_DCP_DIR = os.path.expanduser(
    "~/.wine/drive_c/ProgramData/Adobe/CameraRaw/CameraProfiles/Adobe Standard/"
)

ILLUMINANT_MAP = {
    "Standard Light A": "A",
    "Tungsten (Incandescent)": "A",
    "D65": "D65",
    "D50": "D50",
    "D55": "D55",
    "D75": "D75",
    "Flash": "Flash",
}

# ── DCP extraction ──────────────────────────────────────────────────────────


def extract_dcps(dcp_dir: str) -> dict:
    """Extract calibration data from all DCP files via exiftool."""
    dcp_files = sorted(
        str(Path(dcp_dir) / f)
        for f in os.listdir(dcp_dir)
        if f.lower().endswith(".dcp")
    )
    if not dcp_files:
        print(f"No .dcp files found in {dcp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(dcp_files)} DCP files in {dcp_dir}", file=sys.stderr)

    all_data: dict = {}
    batch_size = 50
    for i in range(0, len(dcp_files), batch_size):
        batch = dcp_files[i : i + batch_size]
        result = subprocess.run(
            [
                "exiftool", "-j",
                "-UniqueCameraModel",
                "-ForwardMatrix1", "-ForwardMatrix2",
                "-ColorMatrix1", "-ColorMatrix2",
                "-CalibrationIlluminant1", "-CalibrationIlluminant2",
            ]
            + batch,
            capture_output=True,
            text=True,
            timeout=120,
        )
        try:
            records = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"  Warning: failed to parse batch {i}–{i + batch_size}", file=sys.stderr)
            continue

        for rec in records:
            model = rec.get("UniqueCameraModel", "")
            if model and model not in all_data:
                all_data[model] = rec

        print(
            f"  Processed {min(i + batch_size, len(dcp_files))}/{len(dcp_files)} DCPs, "
            f"{len(all_data)} unique models",
            file=sys.stderr,
        )

    return all_data


def extract_single_dcp(dcp_path: str) -> dict:
    """Extract calibration data from a single DCP file."""
    result = subprocess.run(
        [
            "exiftool", "-j",
            "-UniqueCameraModel",
            "-ForwardMatrix1", "-ForwardMatrix2",
            "-ColorMatrix1", "-ColorMatrix2",
            "-CalibrationIlluminant1", "-CalibrationIlluminant2",
            dcp_path,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    records = json.loads(result.stdout)
    if not records:
        print(f"No data extracted from {dcp_path}", file=sys.stderr)
        sys.exit(1)
    return {records[0].get("UniqueCameraModel", "unknown"): records[0]}


# ── TOML parsing ────────────────────────────────────────────────────────────


def parse_toml_models(cameras_dir: Path) -> dict:
    """Parse all camera TOML files, returning {path: info_dict}."""
    tomls = {}
    for toml_file in sorted(cameras_dir.rglob("*.toml")):
        content = toml_file.read_text()

        model_m = re.search(r'^model\s*=\s*"([^"]+)"', content, re.MULTILINE)
        make_m = re.search(r'^make\s*=\s*"([^"]+)"', content, re.MULTILINE)
        clean_make_m = re.search(r'^clean_make\s*=\s*"([^"]+)"', content, re.MULTILINE)
        clean_model_m = re.search(r'^clean_model\s*=\s*"([^"]+)"', content, re.MULTILINE)

        if not model_m:
            continue

        tomls[toml_file] = {
            "model": model_m.group(1),
            "make": make_m.group(1) if make_m else "",
            "clean_make": clean_make_m.group(1) if clean_make_m else "",
            "clean_model": clean_model_m.group(1) if clean_model_m else "",
            "has_forward_matrix": "forward_matrix" in content,
            "has_color_matrix": "color_matrix" in content,
            "path": str(toml_file),
            "manufacturer": toml_file.parent.name,
        }
    return tomls


# ── Matching ────────────────────────────────────────────────────────────────


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def match_tomls_to_dcps(tomls: dict, dcps: dict) -> tuple[list, list]:
    """Match TOML cameras to DCP profiles using progressively looser strategies."""
    dcp_lookup = {_norm(k): v for k, v in dcps.items()}

    matches = []
    unmatched = []

    for toml_path, info in tomls.items():
        model = info["model"]
        make = info["make"]
        clean_make = info["clean_make"]

        # Try each candidate key until one matches
        candidates = [
            model,                                  # direct match
            f"{clean_make} {model}" if clean_make else None,  # clean_make + model
            f"{make} {model}" if make else None,              # make + model
        ]

        # Brand-specific fixups
        brand_prefixes = {
            "Nikon":      ["Nikon "],
            "Fujifilm":   ["Fujifilm ", "FUJIFILM "],
            "Olympus":    ["Olympus ", "OLYMPUS "],
            "Pentax":     ["PENTAX ", "Pentax "],
            "Panasonic":  ["Panasonic ", "PANASONIC "],
            "Samsung":    ["Samsung "],
            "Leica":      ["Leica ", "LEICA "],
            "Phase One":  ["Phase One ", "PHASE ONE "],
            "Hasselblad": ["Hasselblad "],
            "Minolta":    ["Minolta "],
            "Epson":      ["Epson ", "EPSON "],
            "Mamiya":     ["Mamiya "],
            "OM Digital Solutions": ["OM Digital Solutions ", "OM System "],
        }
        for brand, prefixes in brand_prefixes.items():
            if clean_make == brand or brand.upper() in make.upper():
                for pfx in prefixes:
                    candidates.append(f"{pfx}{model}")

        matched = False
        for cand in candidates:
            if cand is None:
                continue
            key = _norm(cand)
            if key in dcp_lookup:
                matches.append((toml_path, info, dcp_lookup[key]))
                matched = True
                break

        if not matched:
            unmatched.append(info)

    return matches, unmatched


# ── Matrix helpers ──────────────────────────────────────────────────────────


def parse_matrix_string(s: str) -> list[float] | None:
    if not s:
        return None
    return [float(x) for x in s.split()]


def illuminant_key(s: str) -> str:
    return ILLUMINANT_MAP.get(s, s)


def format_matrix(values: list[float]) -> str:
    """Format matrix values as a TOML array, e.g. [0.4627, 0.3896, 0.1121, ...]."""
    parts = []
    for v in values:
        s = f"{v:.4f}"
        if "." in s:
            s = s.rstrip("0")
            if s.endswith("."):
                s += "0"
        parts.append(s)
    return "[" + ", ".join(parts) + "]"


# ── TOML update ─────────────────────────────────────────────────────────────


def inject_forward_matrix(toml_path: str, fm1: list, fm2: list, illu1: str, illu2: str, dry_run: bool = False) -> tuple[bool, str]:
    """
    Insert a [cameras.forward_matrix] section after [cameras.color_matrix].
    Returns (success, message).
    """
    content = Path(toml_path).read_text()

    if "forward_matrix" in content:
        return False, "already has forward_matrix"

    if "[cameras.color_matrix]" not in content:
        return False, "no [cameras.color_matrix] section (DNG-native camera?)"

    fm_section = "\n[cameras.forward_matrix]\n"
    fm_section += f"{illu1} = {format_matrix(fm1)}\n"
    fm_section += f"{illu2} = {format_matrix(fm2)}\n"

    # Find end of color_matrix block (next section or EOF)
    cm_match = re.search(
        r"(\[cameras\.color_matrix\]\s*\n(?:.*\n)*?)"
        r"(?=\n\[|\n#?\[|\Z)",
        content,
    )
    if not cm_match:
        return False, "could not locate end of [cameras.color_matrix]"

    if dry_run:
        return True, f"would add FM ({illu1}/{illu2})"

    new_content = content[: cm_match.end()] + fm_section + content[cm_match.end() :]
    Path(toml_path).write_text(new_content)
    return True, "ok"


# ── CLI ─────────────────────────────────────────────────────────────────────


def cmd_batch(args):
    """Process all DCPs against all camera TOMLs."""
    dcp_dir = args.dcp_dir or DEFAULT_DCP_DIR
    if not os.path.isdir(dcp_dir):
        print(f"DCP directory not found: {dcp_dir}", file=sys.stderr)
        print("Use --dcp-dir to specify the path to Adobe DCP profiles.", file=sys.stderr)
        sys.exit(1)

    dcps = extract_dcps(dcp_dir)
    print(f"Extracted {len(dcps)} unique DCP models\n", file=sys.stderr)

    tomls = parse_toml_models(CAMERAS_DIR)
    print(f"Found {len(tomls)} camera TOML files\n", file=sys.stderr)

    matches, unmatched = match_tomls_to_dcps(tomls, dcps)

    # Filter to cameras that need ForwardMatrix and have DCP data for it
    actionable = []
    for toml_path, info, dcp_data in matches:
        if info["has_forward_matrix"]:
            continue
        fm1 = parse_matrix_string(dcp_data.get("ForwardMatrix1", ""))
        fm2 = parse_matrix_string(dcp_data.get("ForwardMatrix2", ""))
        if fm1 and fm2:
            actionable.append((toml_path, info, dcp_data, fm1, fm2))

    print(f"Matched:   {len(matches)}/{len(tomls)} cameras")
    print(f"Unmatched: {len(unmatched)}")
    print(f"Can add ForwardMatrix: {len(actionable)}")

    if unmatched:
        print(f"\nUnmatched cameras:")
        for info in sorted(unmatched, key=lambda x: x["model"]):
            print(f"  {info['manufacturer']:15s} {info['model']}")

    if not actionable:
        print("\nNothing to do — all matched cameras already have ForwardMatrix.")
        return

    # Group by manufacturer for display
    by_mfg: dict[str, list] = {}
    for toml_path, info, dcp_data, fm1, fm2 in actionable:
        mfg = info["manufacturer"]
        by_mfg.setdefault(mfg, []).append((toml_path, info, dcp_data, fm1, fm2))

    print(f"\nCameras to update:")
    for mfg in sorted(by_mfg):
        items = by_mfg[mfg]
        print(f"  {mfg}: {len(items)} cameras")

    # Apply updates
    success = skip = fail = 0
    for toml_path, info, dcp_data, fm1, fm2 in actionable:
        illu1 = illuminant_key(dcp_data.get("CalibrationIlluminant1", ""))
        illu2 = illuminant_key(dcp_data.get("CalibrationIlluminant2", ""))
        ok, msg = inject_forward_matrix(str(toml_path), fm1, fm2, illu1, illu2, dry_run=args.dry_run)
        if ok:
            success += 1
        else:
            skip += 1
            if "already" not in msg:
                print(f"  Skip {info['model']}: {msg}", file=sys.stderr)

    verb = "Would update" if args.dry_run else "Updated"
    print(f"\n{verb}: {success}  Skipped: {skip}  Failed: {fail}")


def cmd_single(args):
    """Process a single DCP file against a specific TOML."""
    if not os.path.isfile(args.single):
        print(f"DCP file not found: {args.single}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.toml):
        print(f"TOML file not found: {args.toml}", file=sys.stderr)
        sys.exit(1)

    # Extract DCP
    result = subprocess.run(
        [
            "exiftool", "-j",
            "-UniqueCameraModel",
            "-ForwardMatrix1", "-ForwardMatrix2",
            "-ColorMatrix1", "-ColorMatrix2",
            "-CalibrationIlluminant1", "-CalibrationIlluminant2",
        ]
        + [args.single],
        capture_output=True,
        text=True,
        timeout=30,
    )
    records = json.loads(result.stdout)
    if not records:
        print(f"No data extracted from {args.single}", file=sys.stderr)
        sys.exit(1)

    rec = records[0]
    model = rec.get("UniqueCameraModel", "unknown")
    fm1 = parse_matrix_string(rec.get("ForwardMatrix1", ""))
    fm2 = parse_matrix_string(rec.get("ForwardMatrix2", ""))
    illu1 = illuminant_key(rec.get("CalibrationIlluminant1", ""))
    illu2 = illuminant_key(rec.get("CalibrationIlluminant2", ""))

    print(f"DCP model:     {model}")
    print(f"Illuminants:   {illu1} / {illu2}")

    if rec.get("ColorMatrix1"):
        cm1 = parse_matrix_string(rec["ColorMatrix1"])
        print(f"ColorMatrix1:  {format_matrix(cm1)}")
    if rec.get("ColorMatrix2"):
        cm2 = parse_matrix_string(rec["ColorMatrix2"])
        print(f"ColorMatrix2:  {format_matrix(cm2)}")

    if fm1 and fm2:
        print(f"ForwardMatrix1: {format_matrix(fm1)}")
        print(f"ForwardMatrix2: {format_matrix(fm2)}")
    else:
        print("No ForwardMatrix data in this DCP.")
        sys.exit(1)

    ok, msg = inject_forward_matrix(args.toml, fm1, fm2, illu1, illu2, dry_run=args.dry_run)
    if ok:
        verb = "Would update" if args.dry_run else "Updated"
        print(f"\n{verb} {args.toml}")
    else:
        print(f"\nSkipped: {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract ForwardMatrix data from Adobe DCP profiles and add to dnglab camera TOMLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run against all Adobe DCP profiles:
  %(prog)s --dcp-dir ~/.wine/drive_c/.../Adobe\\ Standard/ --dry-run

  # Apply ForwardMatrix to all matching cameras:
  %(prog)s --dcp-dir /path/to/dcps

  # Update a single camera from a specific DCP:
  %(prog)s --single "Adobe Standard/Nikon Z f.dcp" --toml rawler/data/cameras/nikon/z_f.toml
""",
    )
    parser.add_argument(
        "--dcp-dir",
        help=f"Directory containing Adobe DCP profile files (default: {DEFAULT_DCP_DIR})",
    )
    parser.add_argument(
        "--single",
        help="Path to a single DCP file (use with --toml)",
    )
    parser.add_argument(
        "--toml",
        help="Path to a specific camera TOML file (use with --single)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying any files",
    )

    args = parser.parse_args()

    # Validate argument combos
    if args.single:
        if not args.toml:
            parser.error("--single requires --toml")
        cmd_single(args)
    else:
        cmd_batch(args)


if __name__ == "__main__":
    main()
