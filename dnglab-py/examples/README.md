# dnglab-py Examples

These examples show how to use the `dnglab_py` native Python API for
RAW-to-DNG conversion without shelling out to the `dnglab` CLI binary.

## Setup

### 1. Build the native module

From the repo root (requires Rust ≥ 1.88 and [maturin](https://www.maturin.rs/)):

```bash
cd dnglab-py
pip install maturin
maturin develop --release    # installs dnglab_py into current env
```

### 2. Install example dependencies

```bash
cd examples
pip install pipenv
pipenv install
```

## Examples

### `convert.py` — Basic conversion

Convert a single RAW file to DNG (equivalent to
`dnglab convert --embed-raw false --dng-preview false --dng-thumbnail false`):

```bash
pipenv run python convert.py /photos/IMG_1234.ARW
pipenv run python convert.py /photos/IMG_1234.ARW /tmp/output.dng
```

### `cameras.py` — List supported hardware

```bash
pipenv run python cameras.py                    # all cameras
pipenv run python cameras.py --search nikon     # filter
pipenv run python cameras.py --extensions       # file formats
```

### `serve.py` — FastAPI conversion server with disk cache

REST endpoint that converts RAW files to DNG on demand.
Streams DNG bytes back to the client with ETag caching and persistent disk cache.

```bash
pipenv run server
# or: pipenv run uvicorn serve:app --reload --port 8800
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/convert?path=...` | Convert RAW → DNG (streamed, cached) |
| GET | `/preview?path=...` | JPEG preview (embedded or generated, cached) |
| GET | `/metadata?path=...` | Extract EXIF / camera metadata (no exiftool) |
| GET | `/supported/cameras` | List supported cameras |
| GET | `/supported/extensions` | List supported file extensions |
| GET | `/cache/stats` | Cache size / entry count |
| DELETE | `/cache` | Purge all cached DNG files |

**Query parameters for `/convert`:**

| Param | Default | Description |
|-------|---------|-------------|
| `path` | *(required)* | Absolute path to the RAW file |
| `embed_raw` | `false` | Embed original RAW in DNG |
| `preview` | `false` | Generate JPEG preview |
| `thumbnail` | `false` | Generate thumbnail |
| `compression` | `lossless` | `lossless` or `uncompressed` |
| `crop` | `best` | `best`, `activearea`, or `none` |
| `force` | `false` | Bypass disk cache |

**Environment variables:**

| Var | Default | Description |
|-----|---------|-------------|
| `DNG_CACHE_DIR` | `/tmp/dnglab_cache` | Where to store cached `.dng` files |
| `CUSTOM_DCP_DIR` / `DCP_DIR` | *(none)* | Directory of `.dcp` color profiles |

**Example client call:**

```bash
curl -o photo.dng 'http://localhost:8800/convert?path=/photos/IMG_1234.ARW'
```

## Subprocess vs native API

Typical subprocess approach:

```python
result = subprocess.run(
    ["dnglab", "convert",
     "--embed-raw", "false",
     "--dng-preview", "false",
     "--dng-thumbnail", "false",
     input_path, output_path],
    capture_output=True, timeout=60,
)
dng_bytes = Path(output_path).read_bytes()
```

With `dnglab_py`:

```python
dng_bytes = dnglab_py.convert_to_dng(
    input_path,
    embed_raw=False,
    preview=False,
    thumbnail=False,
)
```

No temp files, no process spawn, no PATH lookup — just bytes back in memory.
The GIL is released during conversion, so other async handlers keep running.
