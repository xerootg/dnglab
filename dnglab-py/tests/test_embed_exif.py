"""Tests for dnglab_py.embed_exif() — EXIF embedding into JPEG bytes."""

from __future__ import annotations

import struct
from io import BytesIO

import pytest
from PIL import Image

import dnglab_py

JPEG_SOI = b"\xff\xd8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_jpeg(width: int = 8, height: int = 8) -> bytes:
    """Create a minimal valid JPEG using Pillow."""
    img = Image.new("RGB", (width, height), (128, 128, 128))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _find_app1_exif_segments(data: bytes) -> list[tuple[int, int]]:
    """Find all APP1 EXIF segments.  Returns list of (offset, total_len)."""
    segments = []
    pos = 2  # skip SOI
    while pos + 4 <= len(data) and data[pos] == 0xFF:
        marker = data[pos + 1]
        seg_len = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
        if marker == 0xE1 and pos + 4 + 4 <= len(data) and data[pos + 4 : pos + 8] == b"Exif":
            segments.append((pos, seg_len + 2))
        pos += 2 + seg_len
    return segments


def _get_ifd0_exif(jpeg_bytes: bytes) -> dict:
    """Parse IFD0 EXIF tags from JPEG bytes using Pillow."""
    img = Image.open(BytesIO(jpeg_bytes))
    return dict(img.getexif())


def _get_exif_subifd(jpeg_bytes: bytes) -> dict:
    """Parse Exif sub-IFD (0x8769) tags from JPEG bytes using Pillow."""
    img = Image.open(BytesIO(jpeg_bytes))
    exif = img.getexif()
    return dict(exif.get_ifd(0x8769))


def _get_gps_ifd(jpeg_bytes: bytes) -> dict:
    """Parse GPS IFD (0x8825) tags from JPEG bytes using Pillow."""
    img = Image.open(BytesIO(jpeg_bytes))
    exif = img.getexif()
    return dict(exif.get_ifd(0x8825))


# ---------------------------------------------------------------------------
# Basic structure tests
# ---------------------------------------------------------------------------

class TestEmbedExifStructure:
    """Verify that embed_exif produces structurally valid output."""

    def test_returns_bytes(self, raw_sample):
        """embed_exif() returns a bytes object."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        assert isinstance(result, bytes)

    def test_output_starts_with_soi(self, raw_sample):
        """Output JPEG starts with the SOI marker."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        assert result[:2] == JPEG_SOI, "Output doesn't start with JPEG SOI"

    def test_output_contains_app1_exif(self, raw_sample):
        """Output contains exactly one APP1 EXIF segment."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        segments = _find_app1_exif_segments(result)
        assert len(segments) == 1, f"Expected 1 APP1 EXIF segment, found {len(segments)}"

    def test_output_is_valid_jpeg(self, raw_sample):
        """Output is loadable by Pillow as a valid JPEG image."""
        jpeg_in = _make_minimal_jpeg(64, 48)
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        img = Image.open(BytesIO(result))
        assert img.format == "JPEG"
        assert img.size == (64, 48), "Image dimensions changed during EXIF embedding"


# ---------------------------------------------------------------------------
# IFD0 / root tag tests
# ---------------------------------------------------------------------------

class TestEmbedExifRootTags:
    """Verify IFD0 tags (Make, Model, Orientation) are embedded correctly."""

    def test_make_is_present(self, raw_sample):
        """Camera Make tag (0x010F) should be present in IFD0."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        ifd0 = _get_ifd0_exif(result)
        assert 0x010F in ifd0, "Make tag (0x010F) missing from IFD0"

    def test_make_matches_raw_metadata(self, raw_sample):
        """Embedded Make must match what raw_metadata() reports."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        ifd0 = _get_ifd0_exif(result)
        assert ifd0.get(0x010F, "").strip() == md["make"].strip()

    def test_model_is_present(self, raw_sample):
        """Camera Model tag (0x0110) should be present in IFD0."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        ifd0 = _get_ifd0_exif(result)
        assert 0x0110 in ifd0, "Model tag (0x0110) missing from IFD0"

    def test_model_matches_raw_metadata(self, raw_sample):
        """Embedded Model must match what raw_metadata() reports."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        ifd0 = _get_ifd0_exif(result)
        assert ifd0.get(0x0110, "").strip() == md["model"].strip()

    def test_default_orientation_is_1(self, raw_sample):
        """Default orientation should be 1 (normal)."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        ifd0 = _get_ifd0_exif(result)
        assert ifd0.get(0x0112) == 1, f"Orientation {ifd0.get(0x0112)} != 1"

    def test_orientation_override(self, raw_sample):
        """Orientation parameter should override the embedded value."""
        path = str(raw_sample["abs_path"])
        jpeg_in = _make_minimal_jpeg()
        for orient in (1, 3, 6, 8):
            result = dnglab_py.embed_exif(path, jpeg_in, orientation=orient)
            ifd0 = _get_ifd0_exif(result)
            assert ifd0.get(0x0112) == orient, (
                f"orientation={orient}: got {ifd0.get(0x0112)}"
            )


# ---------------------------------------------------------------------------
# Exif sub-IFD tag tests
# ---------------------------------------------------------------------------

class TestEmbedExifSubIFD:
    """Verify Exif sub-IFD tags match the RAW source metadata."""

    def test_exif_subifd_is_populated(self, raw_sample):
        """The Exif sub-IFD should contain at least some tags."""
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(str(raw_sample["abs_path"]), jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert len(exif_ifd) >= 3, f"Exif sub-IFD only has {len(exif_ifd)} tags"

    def test_iso_matches(self, raw_sample):
        """ISOSpeedRatings (0x8827) should match raw_metadata()."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_iso = md.get("exif", {}).get("iso_speed_ratings")
        if raw_iso is None:
            pytest.skip("No ISO in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert exif_ifd.get(0x8827) == raw_iso, (
            f"ISO mismatch: embedded {exif_ifd.get(0x8827)} vs raw {raw_iso}"
        )

    def test_date_time_original_matches(self, raw_sample):
        """DateTimeOriginal (0x9003) should match raw_metadata()."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_dto = md.get("exif", {}).get("date_time_original")
        if raw_dto is None:
            pytest.skip("No DateTimeOriginal in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert exif_ifd.get(0x9003) == raw_dto

    def test_focal_length_present(self, raw_sample):
        """FocalLength (0x920A) should be present when the RAW has it."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_fl = md.get("exif", {}).get("focal_length")
        if raw_fl is None:
            pytest.skip("No FocalLength in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert 0x920A in exif_ifd, "FocalLength (0x920A) missing from Exif sub-IFD"

    def test_exposure_time_present(self, raw_sample):
        """ExposureTime (0x829A) should be present when the RAW has it."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_et = md.get("exif", {}).get("exposure_time")
        if raw_et is None:
            pytest.skip("No ExposureTime in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert 0x829A in exif_ifd, "ExposureTime (0x829A) missing from Exif sub-IFD"

    def test_fnumber_present(self, raw_sample):
        """FNumber (0x829D) should be present when the RAW has it."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_fn = md.get("exif", {}).get("fnumber")
        if raw_fn is None:
            pytest.skip("No FNumber in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert 0x829D in exif_ifd, "FNumber (0x829D) missing from Exif sub-IFD"

    def test_lens_model_matches(self, raw_sample):
        """LensModel (0xA434) should match raw_metadata() when present."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_lens = md.get("exif", {}).get("lens_model")
        if raw_lens is None:
            pytest.skip("No LensModel in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        exif_ifd = _get_exif_subifd(result)
        assert exif_ifd.get(0xA434, "").strip() == raw_lens.strip()


# ---------------------------------------------------------------------------
# GPS IFD tests
# ---------------------------------------------------------------------------

class TestEmbedExifGPS:
    """Verify GPS metadata embedding when the RAW contains GPS data."""

    def test_gps_embedded_when_present(self, raw_sample):
        """If the RAW has GPS data, the embedded JPEG should too."""
        path = str(raw_sample["abs_path"])
        md = dnglab_py.raw_metadata(path)
        raw_gps = md.get("exif", {}).get("gps")
        if not raw_gps:
            pytest.skip("No GPS data in RAW metadata for this sample")
        jpeg_in = _make_minimal_jpeg()
        result = dnglab_py.embed_exif(path, jpeg_in)
        gps_ifd = _get_gps_ifd(result)
        # At minimum the GPS IFD should exist with at least GPSVersionID.
        assert len(gps_ifd) >= 1, "GPS IFD is empty despite RAW having GPS data"


# ---------------------------------------------------------------------------
# Idempotency and replacement
# ---------------------------------------------------------------------------

class TestEmbedExifIdempotency:
    """Verify that re-embedding doesn't corrupt or duplicate."""

    def test_replaces_existing_exif(self, raw_sample):
        """Re-embedding should replace, not duplicate, the APP1 segment."""
        path = str(raw_sample["abs_path"])
        jpeg_in = _make_minimal_jpeg()
        first = dnglab_py.embed_exif(path, jpeg_in)
        second = dnglab_py.embed_exif(path, first)
        segments = _find_app1_exif_segments(second)
        assert len(segments) == 1, (
            f"Expected 1 APP1 EXIF after re-embed, found {len(segments)}"
        )

    def test_double_embed_preserves_tags(self, raw_sample):
        """Tags should be identical after a second embed pass."""
        path = str(raw_sample["abs_path"])
        jpeg_in = _make_minimal_jpeg()
        first = dnglab_py.embed_exif(path, jpeg_in)
        second = dnglab_py.embed_exif(path, first)
        ifd0_first = _get_ifd0_exif(first)
        ifd0_second = _get_ifd0_exif(second)
        assert ifd0_first.get(0x010F) == ifd0_second.get(0x010F), "Make changed"
        assert ifd0_first.get(0x0110) == ifd0_second.get(0x0110), "Model changed"
        assert ifd0_first.get(0x0112) == ifd0_second.get(0x0112), "Orientation changed"

    def test_image_data_preserved(self, raw_sample):
        """The image pixel data should be unchanged after embedding."""
        path = str(raw_sample["abs_path"])
        jpeg_in = _make_minimal_jpeg(32, 32)
        result = dnglab_py.embed_exif(path, jpeg_in)
        img_in = Image.open(BytesIO(jpeg_in)).convert("RGB")
        img_out = Image.open(BytesIO(result)).convert("RGB")
        assert img_in.size == img_out.size, "Image dimensions changed"
        # Pixel data should be bitwise identical (we splice EXIF, not re-encode).
        assert img_in.tobytes() == img_out.tobytes(), "Pixel data changed"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestEmbedExifErrors:

    def test_nonexistent_raw_file(self):
        """Should raise RuntimeError for a missing RAW file."""
        jpeg_in = _make_minimal_jpeg()
        with pytest.raises(RuntimeError, match="File not found"):
            dnglab_py.embed_exif("/nonexistent/photo.nef", jpeg_in)

    def test_invalid_jpeg_bytes(self, raw_sample):
        """Should raise RuntimeError for non-JPEG input."""
        with pytest.raises(RuntimeError, match="not a valid JPEG"):
            dnglab_py.embed_exif(str(raw_sample["abs_path"]), b"not a jpeg")

    def test_empty_jpeg_bytes(self, raw_sample):
        """Should raise RuntimeError for empty input."""
        with pytest.raises(RuntimeError, match="not a valid JPEG"):
            dnglab_py.embed_exif(str(raw_sample["abs_path"]), b"")

    def test_truncated_jpeg(self, raw_sample):
        """A truncated JPEG (just SOI) should still get EXIF embedded."""
        # SOI + APP0 is minimal; just SOI alone is technically valid enough
        # for our splice logic (we insert after SOI).
        result = dnglab_py.embed_exif(
            str(raw_sample["abs_path"]), b"\xff\xd8"
        )
        assert result[:2] == JPEG_SOI
        assert len(result) > 2, "No EXIF was added to the truncated JPEG"

    def test_unsupported_raw_format(self, tmp_path):
        """A non-RAW file as source should raise RuntimeError."""
        fake = tmp_path / "fake.txt"
        fake.write_bytes(b"hello world")
        jpeg_in = _make_minimal_jpeg()
        with pytest.raises(RuntimeError):
            dnglab_py.embed_exif(str(fake), jpeg_in)
