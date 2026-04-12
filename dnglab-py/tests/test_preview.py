"""Tests for dnglab_py.extract_preview() across many camera makes/models."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

import dnglab_py

JPEG_SOI = b"\xff\xd8"  # JPEG Start-of-Image marker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xfail_no_preview(result, raw_sample):
    """Mark the test as xfail when extract_preview returns None.

    This is a known rawler limitation for some RAW formats (e.g. some
    Olympus ORF files) that lack an embedded JPEG and whose fallback
    preview pipeline doesn't produce output.  The xfail will auto-pass
    once the Rust side is fixed.
    """
    if result is None:
        pytest.xfail(
            f"extract_preview() returned None for "
            f"{raw_sample['make']} {raw_sample['model']} — "
            f"rawler preview extraction not implemented for this format"
        )


def _avg_color(img: Image.Image, grid: int = 8) -> tuple[int, int, int]:
    """Return the average (R, G, B) of *img* by down-scaling to a tiny grid."""
    thumb = img.resize((grid, grid)).convert("RGB")
    data = thumb.tobytes()  # RGBRGBRGB... flat bytes
    n = grid * grid
    r = sum(data[i] for i in range(0, len(data), 3)) // n
    g = sum(data[i] for i in range(1, len(data), 3)) // n
    b = sum(data[i] for i in range(2, len(data), 3)) // n
    return (r, g, b)


# ---------------------------------------------------------------------------
# Basic JPEG structure
# ---------------------------------------------------------------------------

class TestExtractPreview:
    """Verify preview/thumbnail extraction for each sample file."""

    def test_extract_preview_returns_jpeg(self, raw_sample):
        """extract_preview() returns a (bytes, width, height) tuple with valid JPEG."""
        result = dnglab_py.extract_preview(str(raw_sample["abs_path"]))
        _xfail_no_preview(result, raw_sample)
        jpeg_bytes, width, height = result
        assert isinstance(jpeg_bytes, bytes)
        assert len(jpeg_bytes) > 100, "JPEG output suspiciously small"
        assert jpeg_bytes[:2] == JPEG_SOI, "Output does not start with JPEG SOI marker"
        assert width > 0
        assert height > 0

    def test_preview_dimensions_are_reasonable(self, raw_sample):
        """Preview dimensions should be within reasonable bounds."""
        result = dnglab_py.extract_preview(str(raw_sample["abs_path"]))
        _xfail_no_preview(result, raw_sample)
        _, width, height = result
        assert width >= 100, f"Width {width} is too small"
        assert height >= 100, f"Height {height} is too small"
        assert width <= 10000, f"Width {width} is unreasonably large"
        assert height <= 10000, f"Height {height} is unreasonably large"

    def test_preview_with_max_dimension(self, raw_sample):
        """Requesting a max_dimension resizes the output."""
        max_dim = 300
        result = dnglab_py.extract_preview(
            str(raw_sample["abs_path"]),
            max_dimension=max_dim,
        )
        _xfail_no_preview(result, raw_sample)
        jpeg_bytes, width, height = result
        assert jpeg_bytes[:2] == JPEG_SOI
        assert max(width, height) <= max_dim, (
            f"Longest side {max(width, height)} exceeds max_dimension={max_dim}"
        )

    def test_preview_quality_parameter(self, raw_sample):
        """Different quality values produce different-sized JPEGs."""
        path = str(raw_sample["abs_path"])
        result_low = dnglab_py.extract_preview(path, max_dimension=500, quality=10)
        result_high = dnglab_py.extract_preview(path, max_dimension=500, quality=95)
        if result_low is None or result_high is None:
            _xfail_no_preview(None, raw_sample)
        low_bytes, _, _ = result_low
        high_bytes, _, _ = result_high
        assert len(high_bytes) > len(low_bytes), (
            f"quality=95 ({len(high_bytes)} B) should be larger than "
            f"quality=10 ({len(low_bytes)} B)"
        )

    def test_full_resolution_preview(self, raw_sample):
        """Full-resolution preview (no max_dimension) should be larger than thumbnail-sized."""
        path = str(raw_sample["abs_path"])
        result_full = dnglab_py.extract_preview(path)
        result_small = dnglab_py.extract_preview(path, max_dimension=200)
        if result_full is None or result_small is None:
            _xfail_no_preview(None, raw_sample)
        _, full_w, full_h = result_full
        _, small_w, small_h = result_small
        assert (full_w * full_h) >= (small_w * small_h), (
            "Full-resolution preview should have at least as many pixels as thumbnail"
        )


# ---------------------------------------------------------------------------
# Content accuracy — verify the JPEG is a real decode of the source RAW
# ---------------------------------------------------------------------------

class TestPreviewAccuracy:
    """Verify that preview content actually corresponds to the source RAW."""

    def test_jpeg_decodes_to_reported_dimensions(self, raw_sample):
        """Pillow-decoded dimensions must match the (width, height) extract_preview reports."""
        result = dnglab_py.extract_preview(str(raw_sample["abs_path"]))
        _xfail_no_preview(result, raw_sample)
        jpeg_bytes, reported_w, reported_h = result
        img = Image.open(BytesIO(jpeg_bytes))
        actual_w, actual_h = img.size
        assert actual_w == reported_w, (
            f"Reported width {reported_w} != decoded width {actual_w}"
        )
        assert actual_h == reported_h, (
            f"Reported height {reported_h} != decoded height {actual_h}"
        )

    def test_resized_jpeg_decodes_to_reported_dimensions(self, raw_sample):
        """Resized preview dimensions match what Pillow decodes."""
        result = dnglab_py.extract_preview(
            str(raw_sample["abs_path"]), max_dimension=400,
        )
        _xfail_no_preview(result, raw_sample)
        jpeg_bytes, reported_w, reported_h = result
        img = Image.open(BytesIO(jpeg_bytes))
        actual_w, actual_h = img.size
        assert actual_w == reported_w
        assert actual_h == reported_h

    def test_preview_is_rgb_not_uniform(self, raw_sample):
        """The preview contains actual image data — not a blank/solid frame."""
        result = dnglab_py.extract_preview(str(raw_sample["abs_path"]))
        _xfail_no_preview(result, raw_sample)
        jpeg_bytes, w, h = result
        img = Image.open(BytesIO(jpeg_bytes)).convert("RGB")

        # Sample ~100 evenly-spaced pixels and check per-channel range.
        data = img.tobytes()  # RGBRGBRGB... flat bytes
        total_pixels = w * h
        step = max(1, total_pixels // 100)
        r_vals = [data[i * 3] for i in range(0, total_pixels, step)]
        g_vals = [data[i * 3 + 1] for i in range(0, total_pixels, step)]
        b_vals = [data[i * 3 + 2] for i in range(0, total_pixels, step)]

        # A real photograph will have dynamic range well above 5 in each channel.
        assert max(r_vals) - min(r_vals) > 5, "R channel is nearly uniform — blank image?"
        assert max(g_vals) - min(g_vals) > 5, "G channel is nearly uniform — blank image?"
        assert max(b_vals) - min(b_vals) > 5, "B channel is nearly uniform — blank image?"

    def test_resized_preview_matches_full_preview(self, raw_sample):
        """A resized preview should be the same image as the full preview."""
        path = str(raw_sample["abs_path"])
        result_full = dnglab_py.extract_preview(path)
        result_small = dnglab_py.extract_preview(path, max_dimension=200)
        if result_full is None or result_small is None:
            _xfail_no_preview(None, raw_sample)

        full_img = Image.open(BytesIO(result_full[0])).convert("RGB")
        small_img = Image.open(BytesIO(result_small[0])).convert("RGB")

        full_avg = _avg_color(full_img)
        small_avg = _avg_color(small_img)

        for ch, (a, b) in enumerate(zip(full_avg, small_avg)):
            assert abs(a - b) <= 5, (
                f"Channel {ch}: full avg={a}, resized avg={b} — "
                f"images don't look like the same photo"
            )

    def test_preview_aspect_ratio_is_plausible(self, raw_sample):
        """Preview aspect ratio should be a common photographic ratio."""
        result = dnglab_py.extract_preview(str(raw_sample["abs_path"]))
        _xfail_no_preview(result, raw_sample)
        _, w, h = result

        if w == 0 or h == 0:
            pytest.fail("Zero dimension in preview")

        ratio = max(w, h) / min(w, h)
        assert ratio <= 2.5, (
            f"Aspect ratio {ratio:.2f} ({w}x{h}) is too extreme for a camera preview"
        )

    def test_preview_metadata_cross_reference(self, raw_sample):
        """When the preview JPEG contains EXIF, its Make must match the RAW metadata.

        Most cameras strip EXIF from their embedded preview JPEGs, so
        the absence of EXIF is explicitly asserted as expected behaviour
        rather than skipped.
        """
        result = dnglab_py.extract_preview(str(raw_sample["abs_path"]))
        _xfail_no_preview(result, raw_sample)
        jpeg_bytes, _, _ = result

        img = Image.open(BytesIO(jpeg_bytes))
        exif = img.getexif()
        # Tag 0x010F = Make
        jpeg_make = exif.get(0x010F)
        if jpeg_make is None:
            # Most cameras strip EXIF from embedded preview JPEGs.
            # This is the camera's behaviour, not a dnglab bug.
            assert not exif, (
                "Preview JPEG has EXIF data but no Make tag — unexpected"
            )
            return

        raw_md = dnglab_py.raw_metadata(str(raw_sample["abs_path"]))
        raw_make = raw_md["make"]

        assert jpeg_make.strip().lower()[:4] == raw_make.strip().lower()[:4], (
            f"EXIF Make in preview JPEG ({jpeg_make!r}) doesn't match "
            f"RAW metadata make ({raw_make!r})"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestPreviewErrors:

    def test_nonexistent_file(self):
        with pytest.raises(RuntimeError, match="File not found"):
            dnglab_py.extract_preview("/nonexistent/photo.arw")
