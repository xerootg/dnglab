# Adding Camera Profiles

This document explains how to add or update color calibration data (ForwardMatrix) for cameras in dnglab, using DCP (Digital Camera Profile) files as a source.

## Background

Each camera TOML in `rawler/data/cameras/` can contain:

- **ColorMatrix** — Maps camera RGB to XYZ under a reference illuminant. Required for RAW conversion.
- **ForwardMatrix** — Maps white-balanced camera RGB to XYZ. Improves color accuracy when present.

DCP files contain ForwardMatrix data sometimes. The `extract_dcp_profiles.py` script extracts ForwardMatrix data from these profiles and injects it into existing camera TOMLs.

## Prerequisites

1. **exiftool** — Install via your package manager:
   ```sh
   # Arch/Manjaro
   sudo pacman -S perl-image-exiftool
   # Debian/Ubuntu
   sudo apt install libimage-exiftool-perl
   # macOS
   brew install exiftool
   ```

## Usage

### Batch: Update all cameras

Preview what would change (dry-run):

```sh
python3 scripts/extract_dcp_profiles.py \
    --dcp-dir ~/some_folder_of_dcp_files \
    --dry-run
```

Apply the changes:

```sh
python3 scripts/extract_dcp_profiles.py \
    --dcp-dir ~/some_folder_of_dcp_files
```

### Single camera

If you have a DCP file for a specific camera:

```sh
python3 scripts/extract_dcp_profiles.py \
    --single "~/some_folder_of_dcp_files/Nikon Z f.dcp" \
    --toml rawler/data/cameras/nikon/z_f.toml
```

Add `--dry-run` to preview without writing.

### What the script does

1. Extracts calibration data from DCP files using exiftool
2. Parses all camera TOMLs in `rawler/data/cameras/`
3. Matches cameras by model name (handles manufacturer naming differences)
4. For each match that lacks ForwardMatrix, inserts a `[cameras.forward_matrix]` section after `[cameras.color_matrix]`

### What it does NOT do

- It **never modifies** existing ColorMatrix values
- It **skips** cameras that already have a `[cameras.forward_matrix]` section
- It **skips** DNG-native cameras (e.g. some Leica models) that have no `[cameras.color_matrix]`

## TOML format

A camera TOML with full calibration data looks like:

```toml
make = "NIKON CORPORATION"
model = "NIKON D850"
clean_make = "Nikon"
clean_model = "D850"
color_pattern = "RGGB"
active_area = [0, 0, 0, 0]

[cameras.color_matrix]
A = [1.1284, -0.4911, -0.0608, -0.6062, 1.4228, 0.196, -0.0877, 0.1605, 0.8117]
D65 = [1.0405, -0.3755, -0.127, -0.5461, 1.3787, 0.1793, -0.104, 0.2015, 0.6785]

[cameras.forward_matrix]
A = [0.5418, 0.3063, 0.1162, 0.3344, 0.6292, 0.0364, 0.1822, 0.0005, 0.6424]
D65 = [0.4451, 0.3492, 0.17, 0.2292, 0.7192, 0.0516, 0.0794, 0.0031, 0.7426]

[[cameras.modes]]
mode = "14bit"
whitepoint = 15520
```

The illuminant keys are typically `A` (tungsten, ~2856K) and `D65` (daylight, ~6500K). Less common illuminants (`D50`, `D55`, `Flash`) are preserved as-is.

## Manually adding a profile

If you don't have a DCP file but have matrix values from another source (e.g. dcraw, Adobe DNG SDK):

1. Open the camera's TOML in `rawler/data/cameras/<manufacturer>/<model>.toml`
2. Add a `[cameras.forward_matrix]` section after `[cameras.color_matrix]`:
   ```toml
   [cameras.forward_matrix]
   A = [v1, v2, v3, v4, v5, v6, v7, v8, v9]
   D65 = [v1, v2, v3, v4, v5, v6, v7, v8, v9]
   ```
3. Values are 3x3 matrix in row-major order, stored as a flat 9-element array
4. Each row of the ForwardMatrix should sum to approximately 1.0

## Verifying

After updating TOMLs, rebuild and test:

```sh
cargo build --release -p dnglab
dnglab convert input.RAW output.dng
exiftool -ForwardMatrix1 -ForwardMatrix2 -ColorMatrix1 -ColorMatrix2 output.dng
```

All four matrices should be present in the output DNG.

## Troubleshooting

**Camera not matched:** The script uses model name matching with brand-specific fixups. If a camera isn't matched, check how its TOML `model` field compares to the DCP's `UniqueCameraModel` field:

```sh
exiftool -UniqueCameraModel /path/to/camera.dcp
```

**exiftool not found:** Make sure it's installed and on your PATH.

