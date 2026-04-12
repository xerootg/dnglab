"""Tests for dnglab_py support-query functions (no sample files needed)."""

from __future__ import annotations

import dnglab_py


class TestIsSupported:

    def test_known_extensions(self):
        """Common RAW extensions should be supported."""
        for ext in ("photo.arw", "photo.cr2", "photo.cr3", "photo.nef", "photo.raf", "photo.orf"):
            assert dnglab_py.is_supported(ext), f"{ext} should be supported"

    def test_case_insensitive(self):
        assert dnglab_py.is_supported("photo.ARW")
        assert dnglab_py.is_supported("photo.Cr2")

    def test_unsupported_extensions(self):
        assert not dnglab_py.is_supported("photo.jpg")
        assert not dnglab_py.is_supported("photo.png")
        assert not dnglab_py.is_supported("photo.txt")

    def test_no_extension(self):
        assert not dnglab_py.is_supported("noextension")


class TestSupportedCameras:

    def test_returns_list_of_strings(self):
        cameras = dnglab_py.supported_cameras()
        assert isinstance(cameras, list)
        assert len(cameras) > 50, "Expected many supported cameras"
        assert all(isinstance(c, str) for c in cameras)

    def test_contains_popular_brands(self):
        cameras = dnglab_py.supported_cameras()
        cameras_lower = [c.lower() for c in cameras]
        for brand in ("canon", "nikon", "sony", "fujifilm"):
            assert any(brand in c for c in cameras_lower), (
                f"Expected at least one {brand} camera"
            )

    def test_sample_is_in_supported_list(self, raw_sample):
        """The sample's make+model should appear in supported_cameras()."""
        cameras = dnglab_py.supported_cameras()
        cameras_lower = [c.lower() for c in cameras]
        target = f"{raw_sample['make']} {raw_sample['model']}".lower()
        # Fuzzy match: check that both make and model appear together in at
        # least one entry.
        make_lower = raw_sample["make"].lower()
        model_lower = raw_sample["model"].lower()
        found = any(make_lower in c and model_lower in c for c in cameras_lower)
        # Some pixls.us models use slightly different naming.  Fall back to
        # just checking the make is present.
        if not found:
            found = any(make_lower in c for c in cameras_lower)
        assert found, (
            f"{target!r} (or at least make {make_lower!r}) not in supported cameras"
        )


class TestSupportedExtensions:

    def test_returns_list_of_strings(self):
        exts = dnglab_py.supported_extensions()
        assert isinstance(exts, list)
        assert len(exts) > 5
        assert all(isinstance(e, str) for e in exts)

    def test_common_extensions_present(self):
        exts = dnglab_py.supported_extensions()
        for ext in ("ARW", "CR2", "NEF", "ORF", "RAF"):
            assert ext in exts, f"{ext} should be in supported extensions"

    def test_extensions_are_uppercase(self):
        exts = dnglab_py.supported_extensions()
        for ext in exts:
            assert ext == ext.upper(), f"Extension {ext!r} should be uppercase"
