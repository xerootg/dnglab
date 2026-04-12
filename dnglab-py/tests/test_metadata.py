"""Tests for dnglab_py.raw_metadata() across many camera makes/models."""

from __future__ import annotations

import pytest

import dnglab_py


class TestRawMetadata:
    """Verify metadata extraction for each sample file."""

    def test_returns_dict(self, raw_sample):
        """raw_metadata() returns a dict."""
        md = dnglab_py.raw_metadata(str(raw_sample["abs_path"]))
        assert isinstance(md, dict)

    def test_has_make_and_model(self, raw_sample):
        """Metadata contains make and model fields."""
        md = dnglab_py.raw_metadata(str(raw_sample["abs_path"]))
        assert "make" in md, "Missing 'make' in metadata"
        assert "model" in md, "Missing 'model' in metadata"
        assert isinstance(md["make"], str)
        assert isinstance(md["model"], str)
        assert len(md["make"]) > 0
        assert len(md["model"]) > 0

    def test_make_matches_expected(self, raw_sample):
        """Extracted make should match what the manifest says."""
        md = dnglab_py.raw_metadata(str(raw_sample["abs_path"]))
        # Manufacturers sometimes differ in casing/abbreviation, so just
        # check the first few characters case-insensitively.
        expected = raw_sample["make"].lower()[:4]
        actual = md["make"].lower()[:4]
        assert actual == expected or expected in md["make"].lower(), (
            f"Make mismatch: expected ~{raw_sample['make']!r}, got {md['make']!r}"
        )

    def test_has_exif_section(self, raw_sample):
        """Metadata contains an exif sub-dict."""
        md = dnglab_py.raw_metadata(str(raw_sample["abs_path"]))
        assert "exif" in md, "Missing 'exif' key in metadata"
        assert isinstance(md["exif"], dict)

    def test_exif_has_common_fields(self, raw_sample):
        """EXIF sub-dict contains at least some standard fields.

        Not all cameras populate every field, so we check that at least
        one of the common fields is present.
        """
        md = dnglab_py.raw_metadata(str(raw_sample["abs_path"]))
        exif = md.get("exif", {})
        common_keys = {
            "exposure_time",
            "f_number",
            "iso_speed_ratings",
            "focal_length",
            "iso",
            "date_time_original",
            "orientation",
        }
        found = common_keys & set(exif.keys())
        assert len(found) > 0, (
            f"No common EXIF fields found. Available keys: {sorted(exif.keys())}"
        )


class TestMetadataErrors:

    def test_nonexistent_file(self):
        with pytest.raises(RuntimeError, match="File not found"):
            dnglab_py.raw_metadata("/nonexistent/photo.nef")
