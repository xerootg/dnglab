"""Regression test: repeated convert_to_dng calls must produce consistent BlackLevel.

A buffer-reuse bug in the TIFF writer caused BlackLevel tag overflow data to be
overwritten with zeros after the first few conversions in the same process.  This
resulted in incorrect colour rendering (pink/magenta cast) in downstream viewers.

The bug manifests after 2–4 calls to convert_to_dng in the same Python process.
"""

from __future__ import annotations

import struct

import pytest
import dnglab_py


def _extract_blacklevel(dng_bytes: bytes) -> list[int] | None:
    """Parse the DNG TIFF structure and extract BlackLevel (tag 50714) u16 values."""
    if len(dng_bytes) < 8:
        return None
    is_le = dng_bytes[0] == ord("I")
    bo = "<" if is_le else ">"

    def r16(o: int) -> int:
        return struct.unpack_from(f"{bo}H", dng_bytes, o)[0]

    def r32(o: int) -> int:
        return struct.unpack_from(f"{bo}I", dng_bytes, o)[0]

    # Walk IFD chain
    ifd_off = r32(4)
    while 0 < ifd_off < len(dng_bytes) - 2:
        num = r16(ifd_off)
        for i in range(num):
            eoff = ifd_off + 2 + i * 12
            if eoff + 12 > len(dng_bytes):
                break
            tag = r16(eoff)
            if tag == 50714:  # BlackLevel
                dtype = r16(eoff + 2)
                count = r32(eoff + 4)
                if dtype != 3:  # expect SHORT
                    return None
                total = count * 2
                voff = eoff + 8 if total <= 4 else r32(eoff + 8)
                if voff + total > len(dng_bytes):
                    return None
                return [r16(voff + j * 2) for j in range(count)]
        # Next IFD
        next_pos = ifd_off + 2 + num * 12
        if next_pos + 4 > len(dng_bytes):
            break
        ifd_off = r32(next_pos)

    return None


OM5_ORF = "/home/xero/Downloads/7215093.ORF"


class TestBlackLevelOM5:
    """Reproduce the pink-rendering bug with the specific OM-5 ORF file."""

    @pytest.fixture(autouse=True)
    def _skip_if_missing(self):
        if not __import__("os").path.exists(OM5_ORF):
            pytest.skip(f"Test file not found: {OM5_ORF}")

    def test_blacklevel_stable_om5(self):
        """OM-5 ORF: every conversion must produce BlackLevel=[256,254,254,255].

        A memory corruption bug in the TIFF writer causes the BlackLevel tag's
        overflow data to be overwritten with zeros.  The corruption is
        non-deterministic — it can affect the first call or later calls
        depending on process memory state.
        """
        expected = [256, 254, 254, 255]
        for i in range(6):
            dng = dnglab_py.convert_to_dng(OM5_ORF)
            bl = _extract_blacklevel(bytes(dng))
            assert bl == expected, (
                f"Conversion {i}: BlackLevel is {bl}, expected {expected}"
            )


class TestBlackLevelStability:
    """BlackLevel must be identical across repeated conversions of the same file."""

    def test_blacklevel_stable_across_repeated_conversions(self, raw_sample):
        """Convert the same RAW file 6 times; all must have identical BlackLevel."""
        path = str(raw_sample["abs_path"])
        results = []
        for _ in range(6):
            dng = dnglab_py.convert_to_dng(path)
            bl = _extract_blacklevel(bytes(dng))
            results.append(bl)

        first = results[0]
        assert first is not None, "BlackLevel not found in first DNG conversion"

        for i, bl in enumerate(results[1:], start=1):
            assert bl == first, (
                f"BlackLevel changed between conversion 0 ({first}) and {i} ({bl}) "
                f"for {raw_sample['rel_path']}"
            )

    def test_blacklevel_nonzero_when_expected(self, raw_sample):
        """BlackLevel should not be all-zero for cameras with nonzero optical black."""
        path = str(raw_sample["abs_path"])
        dng = dnglab_py.convert_to_dng(path)
        bl = _extract_blacklevel(bytes(dng))
        if bl is None:
            pytest.skip("BlackLevel tag not found in DNG")
        # Many cameras have nonzero black level; a few (e.g. some Phase One)
        # legitimately have BL=0.  This test flags unexpected zeros as a
        # warning rather than a hard failure.
        if all(v == 0 for v in bl):
            import warnings
            warnings.warn(
                f"BlackLevel is all-zero {bl} for {raw_sample.get('path', '?')} — "
                "verify this is expected for this camera",
                stacklevel=1,
            )

    def test_blacklevel_stable_across_many_same_file_conversions(self, raw_sample):
        """Convert the SAME file 10 times; BlackLevel must not degrade over time.

        This specifically catches buffer-reuse bugs where the TIFF writer's
        internal state leaks between calls, corrupting tag overflow data.
        """
        path = str(raw_sample["abs_path"])
        first_bl = None
        for i in range(10):
            dng = dnglab_py.convert_to_dng(path)
            bl = _extract_blacklevel(bytes(dng))
            if i == 0:
                first_bl = bl
                assert first_bl is not None, "BlackLevel not found in first DNG"
            else:
                assert bl == first_bl, (
                    f"BlackLevel degraded at conversion {i}: was {first_bl}, now {bl} "
                    f"for {raw_sample.get('path', '?')}"
                )
