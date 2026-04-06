#!/usr/bin/env python3
"""Extract Adobe LCP lens correction profiles into TOML for dnglab embedding.

Usage:
    python3 scripts/extract_lcp_profiles.py /path/to/LensProfiles/1.0/ rawler/data/lens_profiles/

Reads all .lcp files under the given directory and produces per-lens TOML
files suitable for embedding in rawler via include_str!().

Each LCP file contains calibration data for a camera+lens combination.
We group by lens name (ignoring camera body) and for each lens:
  - For each focal length, pick entries at the LONGEST focus distance
  - Store per-aperture calibration points (distortion + vignetting)
  - This gives DNG-ready data indexed by (focal_length, aperture)

The TOML format stores Adobe's raw coefficients + FocalLengthX normalization.
Conversion to DNG WarpRectilinear / FixVignetteRadial happens at runtime in
rawler, since it depends on the output DNG image dimensions.

Adobe PerspectiveModel v2 normalises radius by (width * FocalLengthX).
DNG WarpRectilinear normalises by max(width, height).
Conversion: kr_dng = k_adobe / flx^(2*order)  (for landscape images where max_dim = width)
"""

import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

NS = "http://ns.adobe.com/photoshop/1.0/camera-profile"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"


def nskey(ns, name):
    return f"{{{ns}}}{name}"


def get_prop(parent, name):
    """Get a property value from an XMP element, handling both serialization styles.
    Style 1: child element with text content (D90 format)
    Style 2: attribute on rdf:Description (Nikon 1 format)
    """
    key = nskey(NS, name)
    # Try as child element text first
    elem = parent.find(key)
    if elem is not None and elem.text and elem.text.strip():
        return elem.text.strip()
    # Try as attribute
    val = parent.attrib.get(key)
    if val:
        return val
    # Try on rdf:Description child
    desc = parent.find(nskey(RDF, "Description"))
    if desc is not None:
        val = desc.attrib.get(key)
        if val:
            return val
        elem = desc.find(key)
        if elem is not None and elem.text and elem.text.strip():
            return elem.text.strip()
    return None


def get_float(parent, name, default=0.0):
    v = get_prop(parent, name)
    return float(v) if v else default


def get_sub_element(parent, name):
    """Find a sub-element, handling both serialization styles."""
    key = nskey(NS, name)
    # Direct child
    elem = parent.find(key)
    if elem is not None:
        return elem
    # Under rdf:Description
    desc = parent.find(nskey(RDF, "Description"))
    if desc is not None:
        elem = desc.find(key)
        if elem is not None:
            return elem
    return None


def parse_lcp(path):
    """Parse one LCP file, returning a list of calibration entries."""
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return []

    entries = []
    for li in tree.getroot().iter(nskey(RDF, "li")):
        # The li may contain data directly or via rdf:Description child
        # get_prop handles both styles
        lens_name = get_prop(li, "LensPrettyName")
        if not lens_name:
            continue

        is_raw = (get_prop(li, "CameraRawProfile") or "").lower() == "true"
        if not is_raw:
            continue

        image_width = get_float(li, "ImageWidth")
        image_height = get_float(li, "ImageLength")
        if image_width == 0 or image_height == 0:
            continue

        lens_id_str = get_prop(li, "LensID")

        entry = {
            "lens_name": lens_name,
            "camera_name": get_prop(li, "CameraPrettyName") or "",
            "lens_id": lens_id_str,
            "image_width": image_width,
            "image_height": image_height,
            "focal_length": get_float(li, "FocalLength"),
            "aperture_value": get_float(li, "ApertureValue"),
            "focus_distance": get_float(li, "FocusDistance"),
            "crop_factor": get_float(li, "SensorFormatFactor", 1.0),
        }

        # Parse PerspectiveModel
        pm = get_sub_element(li, "PerspectiveModel")
        if pm is None:
            # Also check under rdf:Description
            desc = li.find(nskey(RDF, "Description"))
            if desc is not None:
                pm = desc.find(nskey(NS, "PerspectiveModel"))

        if pm is None:
            continue

        flx = get_float(pm, "FocalLengthX")
        if flx == 0:
            continue

        entry["flx"] = flx
        entry["cx"] = get_float(pm, "ImageXCenter", 0.5)
        entry["cy"] = get_float(pm, "ImageYCenter", 0.5)
        entry["k1"] = get_float(pm, "RadialDistortParam1")
        entry["k2"] = get_float(pm, "RadialDistortParam2")
        entry["k3"] = get_float(pm, "RadialDistortParam3")

        # Vignette sub-model (inside PerspectiveModel)
        vig = get_sub_element(pm, "VignetteModel")
        if vig is not None:
            entry["v1"] = get_float(vig, "VignetteModelParam1")
            entry["v2"] = get_float(vig, "VignetteModelParam2")
            entry["v3"] = get_float(vig, "VignetteModelParam3")
            # Vignette may have its own center/flx
            vig_flx = get_float(vig, "FocalLengthX")
            if vig_flx > 0:
                entry["vig_flx"] = vig_flx
            vig_cx = get_float(vig, "ImageXCenter", -1)
            if vig_cx >= 0:
                entry["vig_cx"] = vig_cx
                entry["vig_cy"] = get_float(vig, "ImageYCenter", 0.5)

        # CA sub-models
        for color, prefix in [("red", "ChromaticRedGreenModel"), ("blue", "ChromaticBlueGreenModel")]:
            ca = get_sub_element(pm, prefix)
            if ca is not None:
                entry[f"ca_{color}_scale"] = get_float(ca, "ScaleFactor", 1.0)

        entries.append(entry)
    return entries


def fnumber_from_apex(av):
    """Convert APEX ApertureValue to f-number."""
    if av <= 0:
        return 0.0
    return round(2.0 ** (av / 2.0), 1)


def build_lens_profiles(lcp_dir):
    """Scan all LCPs and build per-lens correction profiles."""
    lcp_dir = Path(lcp_dir)
    lens_data = defaultdict(list)
    file_count = 0

    for lcp_path in sorted(lcp_dir.rglob("*.lcp")):
        file_count += 1
        if file_count % 500 == 0:
            print(f"  Parsed {file_count} LCP files...", flush=True)
        if file_count >= 2500:
            print(f"  Processing: {lcp_path.name}", flush=True)
        try:
            entries = parse_lcp(lcp_path)
        except Exception as ex:
            print(f"  WARNING: failed to parse {lcp_path}: {ex}", flush=True)
            continue
        for e in entries:
            lens_data[e["lens_name"]].append(e)

    print(f"  Parsed {file_count} LCP files total")

    profiles = {}
    for lens_name, all_entries in sorted(lens_data.items()):
        # Pick the camera with the most entries (best calibration)
        by_camera = defaultdict(list)
        for e in all_entries:
            by_camera[e["camera_name"]].append(e)
        # Use the camera body that has the most calibration points
        best_camera = max(by_camera, key=lambda c: len(by_camera[c]))
        entries = by_camera[best_camera]

        sample = entries[0]
        crop_factor = sample["crop_factor"]
        image_width = sample["image_width"]
        image_height = sample["image_height"]
        lens_id = sample.get("lens_id")
        try:
            lens_id_int = int(lens_id) if lens_id and " " not in lens_id else None
        except (ValueError, TypeError):
            lens_id_int = None

        # Group by focal length
        by_focal = defaultdict(list)
        for e in entries:
            by_focal[e["focal_length"]].append(e)

        calibration_points = []
        for focal in sorted(by_focal.keys()):
            focal_entries = by_focal[focal]

            # Group by aperture
            by_aperture = defaultdict(list)
            for e in focal_entries:
                fnum = fnumber_from_apex(e["aperture_value"])
                by_aperture[fnum].append(e)

            for fnum in sorted(by_aperture.keys()):
                candidates = by_aperture[fnum]
                # Pick the entry with the longest focus distance (most "infinity-like")
                best = max(candidates, key=lambda x: x.get("focus_distance", 0))

                point = {
                    "focal": focal,
                    "aperture": fnum,
                    "k1": best["k1"],
                    "k2": best["k2"],
                    "k3": best["k3"],
                    "cx": best["cx"],
                    "cy": best["cy"],
                    "flx": best["flx"],
                }

                if "v1" in best:
                    point["v1"] = best["v1"]
                    point["v2"] = best["v2"]
                    point["v3"] = best["v3"]
                    # Use vignette-specific normalization if different
                    if "vig_flx" in best:
                        point["vig_flx"] = best["vig_flx"]
                    if "vig_cx" in best:
                        point["vig_cx"] = best["vig_cx"]
                        point["vig_cy"] = best["vig_cy"]

                calibration_points.append(point)

        if not calibration_points:
            continue

        profiles[lens_name] = {
            "lens_name": lens_name,
            "lens_id": lens_id_int,
            "crop_factor": crop_factor,
            "image_width": image_width,
            "image_height": image_height,
            "camera_name": best_camera,
            "calibration": calibration_points,
        }

    return profiles


def fmt_f64(v):
    """Format float for TOML (enough precision, no trailing zeros)."""
    s = f"{v:.8f}".rstrip("0").rstrip(".")
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def escape_toml_string(s):
    """Escape a string for TOML double-quoted format."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def profile_to_toml(prof):
    """Convert a profile dict to a TOML [[profiles]] block."""
    lines = ["[[profiles]]"]
    lines.append(f'lens_name = "{escape_toml_string(prof["lens_name"])}"')
    if prof["lens_id"] is not None:
        lines.append(f"lens_id = {prof['lens_id']}")
    lines.append(f"crop_factor = {fmt_f64(prof['crop_factor'])}")
    lines.append(f"image_width = {fmt_f64(prof['image_width'])}")
    lines.append(f"image_height = {fmt_f64(prof['image_height'])}")

    for pt in prof["calibration"]:
        lines.append("")
        lines.append("[[profiles.calibration]]")
        lines.append(f"focal = {fmt_f64(pt['focal'])}")
        lines.append(f"aperture = {fmt_f64(pt['aperture'])}")
        lines.append(f"flx = {fmt_f64(pt['flx'])}")
        lines.append(f"cx = {fmt_f64(pt['cx'])}")
        lines.append(f"cy = {fmt_f64(pt['cy'])}")
        lines.append(f"k1 = {fmt_f64(pt['k1'])}")
        lines.append(f"k2 = {fmt_f64(pt['k2'])}")
        lines.append(f"k3 = {fmt_f64(pt['k3'])}")

        if "v1" in pt:
            lines.append(f"v1 = {fmt_f64(pt['v1'])}")
            lines.append(f"v2 = {fmt_f64(pt['v2'])}")
            lines.append(f"v3 = {fmt_f64(pt['v3'])}")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <lcp_dir> <output_dir>")
        sys.exit(1)

    lcp_dir = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {lcp_dir} for LCP files...")
    profiles = build_lens_profiles(lcp_dir)
    print(f"Found {len(profiles)} unique lens profiles")

    all_toml = []
    for lens_name in sorted(profiles.keys()):
        all_toml.append(profile_to_toml(profiles[lens_name]))

    out_file = out_dir / "adobe_lcp.toml"
    content = "# Adobe LCP lens correction profiles\n"
    content += "# Extracted from Adobe Camera Raw lens profiles\n"
    content += f"# Total: {len(profiles)} lenses\n"
    content += "#\n"
    content += "# Coefficients are in Adobe PerspectiveModel v2 space.\n"
    content += "# Conversion to DNG WarpRectilinear happens at runtime:\n"
    content += "#   kr_dng = k_adobe / flx^(2*order)\n"
    content += "#   (for landscape images where max_dim = width)\n\n"
    content += "\n\n".join(all_toml) + "\n"

    out_file.write_text(content)
    print(f"Wrote {out_file} ({len(content)} bytes, {len(profiles)} profiles)")

    # Print some stats
    total_cal = sum(len(p["calibration"]) for p in profiles.values())
    with_vig = sum(1 for p in profiles.values() for c in p["calibration"] if "v1" in c)
    print(f"Total calibration points: {total_cal} ({with_vig} with vignetting)")


if __name__ == "__main__":
    main()
