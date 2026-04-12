"""Tests for dnglab_py.convert_to_dng() across many camera makes/models."""

from __future__ import annotations

import struct
from io import BytesIO

import pytest
from PIL import Image

import dnglab_py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_tiff(data: bytes) -> bool:
    """Check that *data* starts with a valid TIFF header (DNG is TIFF-based)."""
    if len(data) < 8:
        return False
    byte_order = data[:2]
    if byte_order == b"II":  # little-endian
        magic = struct.unpack_from("<H", data, 2)[0]
    elif byte_order == b"MM":  # big-endian
        magic = struct.unpack_from(">H", data, 2)[0]
    else:
        return False
    return magic == 42  # TIFF magic number


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------

class TestConvertToDng:
    """Test RAW → DNG conversion for each sample."""

    def test_basic_conversion(self, raw_sample):
        """convert_to_dng() returns valid DNG bytes with default params."""
        dng = dnglab_py.convert_to_dng(str(raw_sample["abs_path"]))
        assert isinstance(dng, bytes)
        assert len(dng) > 1000, "DNG output suspiciously small"
        assert _is_valid_tiff(dng), "Output does not have a valid TIFF/DNG header"

    def test_conversion_uncompressed(self, raw_sample):
        """Uncompressed conversion produces valid output."""
        dng = dnglab_py.convert_to_dng(
            str(raw_sample["abs_path"]),
            compression="uncompressed",
        )
        assert isinstance(dng, bytes)
        assert _is_valid_tiff(dng)

    def test_conversion_with_thumbnail(self, raw_sample):
        """Conversion with thumbnail=True produces larger output."""
        dng_no_thumb = dnglab_py.convert_to_dng(str(raw_sample["abs_path"]))
        dng_thumb = dnglab_py.convert_to_dng(
            str(raw_sample["abs_path"]),
            thumbnail=True,
        )
        assert _is_valid_tiff(dng_thumb)
        # Thumbnail adds data, so output should be at least as large.
        assert len(dng_thumb) >= len(dng_no_thumb)

    def test_conversion_with_preview(self, raw_sample):
        """Conversion with preview=True produces larger output."""
        dng_no_preview = dnglab_py.convert_to_dng(str(raw_sample["abs_path"]))
        dng_preview = dnglab_py.convert_to_dng(
            str(raw_sample["abs_path"]),
            preview=True,
        )
        assert _is_valid_tiff(dng_preview)
        assert len(dng_preview) >= len(dng_no_preview)

    def test_conversion_with_embed_raw(self, raw_sample):
        """Embedding the original RAW inside the DNG works."""
        dng = dnglab_py.convert_to_dng(
            str(raw_sample["abs_path"]),
            embed_raw=True,
        )
        assert _is_valid_tiff(dng)
        # Embedded RAW means the DNG should be at least as large as the original.
        original_size = raw_sample["abs_path"].stat().st_size
        assert len(dng) >= original_size * 0.9, (
            "DNG with embedded RAW should be at least ~original size"
        )

    def test_conversion_all_options(self, raw_sample):
        """Conversion with all bells and whistles enabled."""
        dng = dnglab_py.convert_to_dng(
            str(raw_sample["abs_path"]),
            embed_raw=True,
            preview=True,
            thumbnail=True,
            compression="lossless",
            crop="best",
            artist="Test Artist",
        )
        assert _is_valid_tiff(dng)
        assert len(dng) > 1000

    def test_crop_modes(self, raw_sample):
        """All crop modes produce valid DNG output."""
        path = str(raw_sample["abs_path"])
        for crop in ("best", "activearea", "none"):
            dng = dnglab_py.convert_to_dng(path, crop=crop)
            assert _is_valid_tiff(dng), f"crop={crop!r} produced invalid DNG"

    def test_lossless_smaller_than_uncompressed(self, raw_sample):
        """Lossless compression should produce smaller output than uncompressed."""
        path = str(raw_sample["abs_path"])
        dng_lossless = dnglab_py.convert_to_dng(path, compression="lossless")
        dng_uncompressed = dnglab_py.convert_to_dng(path, compression="uncompressed")
        assert len(dng_lossless) < len(dng_uncompressed), (
            f"Lossless ({len(dng_lossless)} B) should be smaller than "
            f"uncompressed ({len(dng_uncompressed)} B)"
        )


# ---------------------------------------------------------------------------
# DNG content accuracy — verify the DNG faithfully represents the source RAW
# ---------------------------------------------------------------------------

_dng_md_cache: dict[str, dict] = {}


@pytest.fixture(autouse=True, scope="session")
def _cleanup_dng_md_cache():
    """Free the DNG metadata cache after all tests complete."""
    yield
    _dng_md_cache.clear()


def _cached_dng_metadata(raw_path: str, tmp_path) -> tuple[dict, dict]:
    """Return (raw_metadata, dng_metadata) for a default-options DNG.

    Converts and writes to disk once per sample, reads metadata, then
    immediately deletes the temp file.
    """
    if raw_path not in _dng_md_cache:
        dng_bytes = dnglab_py.convert_to_dng(raw_path)
        dng_file = tmp_path / "cache.dng"
        dng_file.write_bytes(dng_bytes)
        raw_md = dnglab_py.raw_metadata(raw_path)
        dng_md = dnglab_py.raw_metadata(str(dng_file))
        dng_file.unlink()
        _dng_md_cache[raw_path] = (raw_md, dng_md)
    return _dng_md_cache[raw_path]


class TestDngAccuracy:
    """Round-trip tests: convert RAW → DNG, then read the DNG back and
    verify its metadata and preview match the original RAW."""

    def test_dng_metadata_matches_raw(self, raw_sample, tmp_path):
        """Make, model, and key EXIF fields survive RAW → DNG conversion."""
        raw_md, dng_md = _cached_dng_metadata(
            str(raw_sample["abs_path"]), tmp_path,
        )
        assert dng_md["make"] == raw_md["make"], (
            f"DNG make {dng_md['make']!r} != RAW make {raw_md['make']!r}"
        )
        assert dng_md["model"] == raw_md["model"], (
            f"DNG model {dng_md['model']!r} != RAW model {raw_md['model']!r}"
        )

    def test_dng_exif_fields_preserved(self, raw_sample, tmp_path):
        """Critical EXIF fields are identical between RAW and DNG."""
        raw_md, dng_md = _cached_dng_metadata(
            str(raw_sample["abs_path"]), tmp_path,
        )
        raw_exif = raw_md["exif"]
        dng_exif = dng_md["exif"]

        critical_keys = [
            "fnumber",
            "exposure_time",
            "focal_length",
            "orientation",
            "date_time_original",
        ]
        for key in critical_keys:
            raw_val = raw_exif.get(key)
            if raw_val is None:
                continue
            dng_val = dng_exif.get(key)
            assert dng_val == raw_val, (
                f"EXIF {key}: RAW={raw_val!r}, DNG={dng_val!r}"
            )

    def test_dng_artist_tag_written(self, raw_sample, tmp_path):
        """The artist parameter is written into the DNG EXIF."""
        raw_path = str(raw_sample["abs_path"])
        dng_bytes = dnglab_py.convert_to_dng(raw_path, artist="Test Artist")
        dng_file = tmp_path / "artist.dng"
        dng_file.write_bytes(dng_bytes)
        dng_md = dnglab_py.raw_metadata(str(dng_file))
        dng_file.unlink()

        dng_artist = dng_md.get("exif", {}).get("artist")
        assert dng_artist == "Test Artist", (
            f"Expected artist='Test Artist', got {dng_artist!r}"
        )

    def test_dng_preview_is_valid_jpeg(self, raw_sample, tmp_path):
        """A DNG converted with preview=True contains an extractable JPEG preview."""
        raw_path = str(raw_sample["abs_path"])
        dng_bytes = dnglab_py.convert_to_dng(raw_path, preview=True)
        dng_file = tmp_path / "preview.dng"
        dng_file.write_bytes(dng_bytes)

        result = dnglab_py.extract_preview(str(dng_file))
        dng_file.unlink()

        assert result is not None, "DNG with preview=True should have an extractable preview"
        jpeg_bytes, w, h = result
        assert jpeg_bytes[:2] == b"\xff\xd8", "DNG preview is not valid JPEG"
        assert w > 0 and h > 0

        img = Image.open(BytesIO(jpeg_bytes))
        assert img.size == (w, h)

    def test_dng_preview_resembles_raw_preview(self, raw_sample, tmp_path):
        """The DNG preview should be visually similar to the RAW preview."""
        raw_path = str(raw_sample["abs_path"])

        raw_preview = dnglab_py.extract_preview(raw_path, max_dimension=200)
        if raw_preview is None:
            pytest.xfail(
                f"extract_preview() returned None for "
                f"{raw_sample['make']} {raw_sample['model']} — "
                f"rawler preview extraction not implemented for this format"
            )

        dng_bytes = dnglab_py.convert_to_dng(raw_path, preview=True)
        dng_file = tmp_path / "preview.dng"
        dng_file.write_bytes(dng_bytes)
        dng_preview = dnglab_py.extract_preview(str(dng_file), max_dimension=200)
        dng_file.unlink()

        assert dng_preview is not None, "DNG preview missing"

        raw_img = Image.open(BytesIO(raw_preview[0])).convert("RGB")
        dng_img = Image.open(BytesIO(dng_preview[0])).convert("RGB")

        raw_thumb = raw_img.resize((8, 8)).tobytes()
        dng_thumb = dng_img.resize((8, 8)).tobytes()

        diffs = [abs(raw_thumb[i] - dng_thumb[i]) for i in range(len(raw_thumb))]
        avg_diff = sum(diffs) / len(diffs)

        assert avg_diff < 15, (
            f"DNG preview differs too much from RAW preview "
            f"(avg pixel diff = {avg_diff:.1f})"
        )

    def test_dng_without_preview_has_no_preview(self, raw_sample, tmp_path):
        """A DNG converted without preview=True should not have a JPEG preview."""
        raw_path = str(raw_sample["abs_path"])
        dng_bytes = dnglab_py.convert_to_dng(raw_path, preview=False)
        dng_file = tmp_path / "nopreview.dng"
        dng_file.write_bytes(dng_bytes)
        result = dnglab_py.extract_preview(str(dng_file))
        dng_file.unlink()

        assert result is None, (
            "DNG converted without preview=True should not contain a JPEG preview"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestConvertErrors:
    """Verify error paths for convert_to_dng()."""

    def test_nonexistent_file(self):
        with pytest.raises(RuntimeError, match="File not found"):
            dnglab_py.convert_to_dng("/nonexistent/raw_file.arw")

    def test_invalid_compression(self):
        """Compression string is validated before the file is touched."""
        with pytest.raises(RuntimeError, match="Unknown compression"):
            dnglab_py.convert_to_dng(
                "/nonexistent/raw_file.arw",
                compression="bogus",
            )

    def test_invalid_crop(self):
        """Crop string is validated before the file is touched."""
        with pytest.raises(RuntimeError, match="Unknown crop mode"):
            dnglab_py.convert_to_dng(
                "/nonexistent/raw_file.arw",
                crop="bogus",
            )
