# dnglab-ffi

C ABI over [rawler](../rawler) for embedding dnglab's RAW engine in native
applications. The first consumer is the Android app
[RawImageSnapseedBridge](https://github.com/xerootg/RawImageSnapseedBridge),
which links `libdnglab_ffi.a` into its JNI library.

## What it offers

| Function | Purpose |
|----------|---------|
| `dnglab_convert_to_dng` | RAW → DNG (rawler's writer: lossless JPEG-92, previews, EXIF/GPS, XMP, camera profile) |
| `dnglab_convert_to_jpeg` | RAW → JPEG. Reuses the camera's embedded full-resolution JPEG (policy: never / full-resolution only / always) with fresh EXIF, otherwise demosaics and renders |
| `dnglab_extract_preview` | JPEG preview with a longest-edge limit, rotated upright; embedded JPEG → decoder preview → full develop |
| `dnglab_metadata_json` | Flat, backend-neutral metadata document plus the complete rawler metadata under `details` |
| `dnglab_supported_extensions`, `dnglab_is_supported_extension`, `dnglab_version` | Self-description |
| `dnglab_init_logging` | Routes `log` output to logcat on Android |

See [`include/dnglab_ffi.h`](include/dnglab_ffi.h) for the declarations and
[`src/api.rs`](src/api.rs) for the safe Rust API the C functions wrap.

Every entry point is panic-safe (`catch_unwind`), thread-safe and never
unwinds across the FFI boundary. Strings handed to the caller are freed with
`dnglab_string_free`.

## Embedded JPEG policy

Most RAW formats carry the full-size JPEG the camera rendered (NEF, CR2, CR3,
most ORF bodies, newer PEF, DNG previews). Some embed only a reduced preview
(ARW, RW2, older ORF/PEF, many RAF). `dnglab_jpeg_options.embedded_policy`
decides what to do:

- `DNGLAB_EMBEDDED_FULL_RESOLUTION_ONLY` (default): copy the embedded JPEG when
  both edges are at least `full_resolution_threshold` (0.9) of the sensor's
  default crop, otherwise render.
- `DNGLAB_EMBEDDED_ALWAYS`: copy whatever is embedded.
- `DNGLAB_EMBEDDED_NEVER`: always render.

Rendering: rescale → demosaic (PPG for Bayer, bilinear for X-Trans/4-colour)
→ Fuji rotation → active-area crop → white balance → camera matrix → default
crop → dcraw-style auto brightness (or DNG BaselineExposure) → sRGB gamma →
rotated upright → `jpeg-encoder` with quality, chroma subsampling and
optimized Huffman tables. EXIF (incl. GPS, lens, orientation) is written in
both paths; MakerNotes are stripped.

## Building

Host (library, example CLI, tests):

```bash
cargo build -p dnglab-ffi
cargo test  -p dnglab-ffi
cargo run   -p dnglab-ffi --example dnglab_ffi_cli -- jpeg IMG_0001.NEF out.jpg 1
```

Android static libraries (one per ABI) using the NDK's clang as linker:

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi i686-linux-android x86_64-linux-android
dnglab-ffi/scripts/build-android.sh --ndk $ANDROID_NDK_HOME --abi arm64-v8a --api 24
# → target/aarch64-linux-android/android/libdnglab_ffi.a
```

The `android` cargo profile (workspace `Cargo.toml`) is release + thin LTO +
stripped debug info.

## Metadata JSON schema

Flat keys (omitted when unknown): `backend`, `backend_version`, `format`,
`make`, `model`, `body_serial`, `lens_make`, `lens_model`, `lens_serial`,
`min_focal`, `max_focal`, `focal_length`, `focal_length_35mm`, `aperture`,
`shutter` (`"1/250"`), `shutter_raw` (seconds), `iso`, `exposure_bias`,
`exposure_program`, `metering_mode`, `flash`, `white_balance`, `timestamp`
(`YYYY-MM-DD HH:MM:SS`), `timestamp_raw` (unix seconds), `width`, `height`
(default crop), `raw_width`, `raw_height`, `orientation` (EXIF 1..8),
`colors`, `bayer_pattern`, `bits_per_sample`, `clean_make`, `clean_model`,
`baseline_exposure`, `has_gps`, `gps_lat_deg/min/sec`, `gps_lat_ref`,
`gps_lon_deg/min/sec`, `gps_lon_ref`, `gps_altitude`, `gps_alt_ref`,
`gps_time_hour/min/sec`, `description`, `artist`, `copyright`,
`user_comment`, `rating`.

Nested: `preview` (`embedded_jpeg`, `width`, `height`, `bytes`,
`full_resolution`) and `details` (serialized `RawMetadata` without
MakerNotes).
