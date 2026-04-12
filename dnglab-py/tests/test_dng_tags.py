"""Verify DNG tag fidelity against exiftool-reported RAW tags.

Each test class covers a "feature" — a set of DNG tags that must be
present (or must match the source RAW) when certain conditions hold.
Both the RAW file and its converted DNG are inspected via ``exiftool -j -G -n``
so the ground truth is the same tool photographers actually use.

Requires: ``exiftool`` on PATH.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path

import pytest

import dnglab_py


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    shutil.which("exiftool") is None,
    reason="exiftool not found on PATH",
)


def _exiftool_tags(path: str) -> dict[str, object]:
    """Return the full exiftool tag dict (``-G -n`` format) for *path*."""
    out = subprocess.check_output(
        ["exiftool", "-j", "-G", "-n", path],
        text=True,
        timeout=30,
    )
    return json.loads(out)[0]


def _write_dng(raw_path: str, dest: Path, **kwargs) -> str:
    """Convert *raw_path* → DNG, write to *dest* dir, return the DNG path."""
    dng_bytes = dnglab_py.convert_to_dng(raw_path, **kwargs)
    dng_path = str(dest / "output.dng")
    Path(dng_path).write_bytes(dng_bytes)
    return dng_path


# ---------------------------------------------------------------------------
# Cached DNG fixtures — each sample is converted once and reused across
# many tests to avoid redundant conversions and /tmp exhaustion.
# ---------------------------------------------------------------------------

_base_dng_cache: dict[str, dict] = {}


@pytest.fixture()
def base_dng(raw_sample, tmp_path):
    """Default-options DNG tag dicts for the current sample.

    On the first call per sample, converts to DNG, writes a temp file,
    runs exiftool on both RAW and DNG, then *deletes* the temp file
    immediately.  Only the tag dicts are cached — no DNG bytes are
    kept in memory or on disk across tests.
    """
    key = str(raw_sample["abs_path"])
    if key not in _base_dng_cache:
        dng_bytes = dnglab_py.convert_to_dng(key)
        raw_tags = _exiftool_tags(key)
        # Write temporarily, call exiftool, then remove.
        dng_path = tmp_path / "base.dng"
        dng_path.write_bytes(dng_bytes)
        dng_tags = _exiftool_tags(str(dng_path))
        dng_path.unlink()
        _base_dng_cache[key] = {
            "raw_path": key,
            "raw_tags": raw_tags,
            "dng_tags": dng_tags,
        }

    return {**_base_dng_cache[key], "raw_sample": raw_sample}


@pytest.fixture(autouse=True, scope="session")
def _cleanup_base_dng_cache():
    """Free the tag cache after all tests complete."""
    yield
    _base_dng_cache.clear()


# ---------------------------------------------------------------------------
# DNG structural tags — must be present in every converted DNG
# ---------------------------------------------------------------------------

# Tags that every well-formed DNG must contain regardless of options.
DNG_REQUIRED_TAGS = [
    "EXIF:DNGVersion",
    "EXIF:DNGBackwardVersion",
    "EXIF:UniqueCameraModel",
    "EXIF:ColorMatrix1",
    "EXIF:CalibrationIlluminant1",
    "EXIF:AsShotNeutral",
    "EXIF:BaselineExposure",
    "EXIF:BaselineNoise",
    "EXIF:BaselineSharpness",
    "EXIF:BlackLevel",
    "EXIF:WhiteLevel",
    "EXIF:DefaultCropOrigin",
    "EXIF:DefaultCropSize",
    "EXIF:DefaultScale",
    "EXIF:ColorimetricReference",
    "EXIF:BestQualityScale",
]


class TestDngStructure:
    """Every DNG must contain core DNG-spec structural tags."""

    def test_required_dng_tags_present(self, base_dng):
        missing = [t for t in DNG_REQUIRED_TAGS if t not in base_dng["dng_tags"]]
        assert not missing, f"Missing required DNG tags: {missing}"

    def test_dng_version_is_present(self, base_dng):
        ver = base_dng["dng_tags"].get("EXIF:DNGVersion")
        assert ver is not None and ver != 0, f"DNGVersion missing or zero: {ver!r}"


# ---------------------------------------------------------------------------
# Color calibration feature — matrices and illuminants
# ---------------------------------------------------------------------------

class TestColorCalibration:
    """DNG must contain colour-science tags for accurate rendering."""

    def test_has_color_matrix1(self, base_dng):
        assert "EXIF:ColorMatrix1" in base_dng["dng_tags"]

    def test_has_color_matrix2_when_dual_illuminant(self, base_dng):
        """If CalibrationIlluminant2 is present, ColorMatrix2 must be too."""
        tags = base_dng["dng_tags"]
        if "EXIF:CalibrationIlluminant2" not in tags:
            pytest.skip("Single-illuminant DNG")
        assert "EXIF:ColorMatrix2" in tags, (
            "CalibrationIlluminant2 present but ColorMatrix2 missing"
        )

    def test_forward_matrices_present(self, base_dng):
        """ForwardMatrix1/2 are recommended for accurate colour rendering.

        Per the DNG spec these are optional, so this test warns rather
        than hard-failing when they are absent in a dual-illuminant DNG.
        """
        tags = base_dng["dng_tags"]
        if "EXIF:CalibrationIlluminant2" not in tags:
            pytest.skip("Single-illuminant DNG")
        has_fm1 = "EXIF:ForwardMatrix1" in tags
        has_fm2 = "EXIF:ForwardMatrix2" in tags
        if not (has_fm1 and has_fm2):
            import warnings
            warnings.warn(
                f"ForwardMatrix1={has_fm1} ForwardMatrix2={has_fm2} — "
                "optional but recommended for dual-illuminant DNG",
                stacklevel=1,
            )


# ---------------------------------------------------------------------------
# Shooting parameter preservation feature
# ---------------------------------------------------------------------------

# EXIF tags that carry photographic exposure data.  When present in the
# RAW, they must survive conversion unchanged.
SHOOTING_PARAM_TAGS = [
    "EXIF:ExposureTime",
    "EXIF:FNumber",
    "EXIF:ISOSpeedRatings",
    "EXIF:FocalLength",
    "EXIF:ExposureProgram",
    "EXIF:MeteringMode",
    "EXIF:Flash",
    "EXIF:WhiteBalance",
    "EXIF:FocalLengthIn35mmFormat",
    "EXIF:ExposureCompensation",
    "EXIF:MaxApertureValue",
    "EXIF:SceneCaptureType",
    "EXIF:Contrast",
    "EXIF:Saturation",
    "EXIF:Sharpness",
]


class TestShootingParams:
    """Photographic exposure parameters must survive RAW → DNG."""

    def test_shooting_params_preserved(self, base_dng):
        raw_tags = base_dng["raw_tags"]
        dng_tags = base_dng["dng_tags"]

        mismatches = []
        for tag in SHOOTING_PARAM_TAGS:
            raw_val = raw_tags.get(tag)
            if raw_val is None:
                continue  # Tag not in this RAW — nothing to preserve.
            dng_val = dng_tags.get(tag)
            if dng_val != raw_val:
                mismatches.append(f"{tag}: raw={raw_val!r} dng={dng_val!r}")

        assert not mismatches, (
            "Shooting parameters differ between RAW and DNG:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# Date/time preservation feature
# ---------------------------------------------------------------------------

class TestDatePreservation:
    """Original capture timestamps must survive; ModifyDate may change."""

    def test_date_time_original_preserved(self, base_dng):
        raw_dto = base_dng["raw_tags"].get("EXIF:DateTimeOriginal")
        if raw_dto is None:
            pytest.skip("RAW has no DateTimeOriginal")
        assert base_dng["dng_tags"].get("EXIF:DateTimeOriginal") == raw_dto

    def test_create_date_preserved(self, base_dng):
        raw_cd = base_dng["raw_tags"].get("EXIF:CreateDate")
        if raw_cd is None:
            pytest.skip("RAW has no CreateDate")
        assert base_dng["dng_tags"].get("EXIF:CreateDate") == raw_cd


# ---------------------------------------------------------------------------
# Orientation feature
# ---------------------------------------------------------------------------

class TestOrientation:
    """EXIF Orientation must be preserved exactly."""

    def test_orientation_preserved(self, base_dng):
        raw_orient = base_dng["raw_tags"].get("EXIF:Orientation")
        if raw_orient is None:
            pytest.skip("RAW has no Orientation tag")
        assert base_dng["dng_tags"].get("EXIF:Orientation") == raw_orient


# ---------------------------------------------------------------------------
# Lens info feature — DNG lens tags when RAW has lens data
# ---------------------------------------------------------------------------

class TestLensInfo:
    """When the RAW contains lens metadata, the DNG must too."""

    def test_dng_has_lens_model_when_raw_does(self, base_dng):
        raw_md = dnglab_py.raw_metadata(base_dng["raw_path"])
        raw_lens_model = raw_md.get("exif", {}).get("lens_model")
        if not raw_lens_model:
            # API has no lens_model — verify DNG also has none.
            assert "EXIF:LensModel" not in base_dng["dng_tags"], (
                "API has no lens_model but DNG has EXIF:LensModel"
            )
            return
        assert "EXIF:LensModel" in base_dng["dng_tags"], (
            f"RAW has lens_model={raw_lens_model!r} but DNG is missing EXIF:LensModel"
        )

    def test_dng_has_dng_lens_info_when_raw_has_lens_spec(self, base_dng):
        raw_md = dnglab_py.raw_metadata(base_dng["raw_path"])
        raw_lens_spec = raw_md.get("exif", {}).get("lens_spec")
        tags = base_dng["dng_tags"]
        has_lens_info = "EXIF:DNGLensInfo" in tags or "EXIF:LensInfo" in tags
        if not raw_lens_spec:
            # API has no lens_spec.  If the camera has a known lens model,
            # this is a gap in rawler's lens_spec extraction.
            raw_lens_model = raw_md.get("exif", {}).get("lens_model")
            if raw_lens_model and has_lens_info:
                # DNG has lens info derived from another source — fine.
                return
            if raw_lens_model and not has_lens_info:
                pytest.xfail(
                    f"API has lens_model={raw_lens_model!r} but no "
                    f"lens_spec — rawler lens_spec extraction gap"
                )
            # No lens info at all — nothing to check.
            return
        assert has_lens_info, (
            f"RAW has lens_spec={raw_lens_spec!r} but DNG has neither "
            "EXIF:DNGLensInfo nor EXIF:LensInfo"
        )


# ---------------------------------------------------------------------------
# Serial number feature
# ---------------------------------------------------------------------------

class TestSerialNumber:
    """When the RAW has a camera serial number, the DNG should carry it.

    When the API has no serial, we verify the DNG also has none.  If
    exiftool's MakerNotes *do* contain a serial that rawler didn't
    extract, this surfaces as an xfail.
    """

    def test_dng_has_serial_when_raw_does(self, base_dng):
        raw_md = dnglab_py.raw_metadata(base_dng["raw_path"])
        raw_serial = raw_md.get("exif", {}).get("serial_number")
        tags = base_dng["dng_tags"]
        has_dng_serial = (
            "EXIF:CameraSerialNumber" in tags or "EXIF:SerialNumber" in tags
        )
        if not raw_serial:
            # API reports no serial.  Verify DNG also has none.
            if has_dng_serial:
                pytest.fail(
                    "API has no serial_number but DNG has "
                    f"CameraSerialNumber or SerialNumber"
                )
            # Check if exiftool MakerNotes had a serial we missed.
            mn_serial = base_dng["raw_tags"].get("MakerNotes:SerialNumber")
            if mn_serial is not None and str(mn_serial).strip():
                pytest.xfail(
                    f"API has no serial_number but RAW MakerNotes has "
                    f"SerialNumber={mn_serial!r} — extraction gap"
                )
            return
        assert has_dng_serial, (
            f"RAW has serial_number={raw_serial!r} but DNG has neither "
            "EXIF:CameraSerialNumber nor EXIF:SerialNumber"
        )


# ---------------------------------------------------------------------------
# Preview feature — preview=True adds JPEG preview tags
# ---------------------------------------------------------------------------

class TestPreviewFeature:
    """DNG tags that appear only when ``preview=True``."""

    def test_preview_tags_present_when_enabled(self, raw_sample, tmp_path):
        dng_path = _write_dng(
            str(raw_sample["abs_path"]), tmp_path, preview=True,
        )
        tags = _exiftool_tags(dng_path)
        Path(dng_path).unlink()
        assert "EXIF:PreviewImage" in tags, "preview=True but no EXIF:PreviewImage"
        assert "EXIF:PreviewColorSpace" in tags

    def test_preview_tags_absent_when_disabled(self, base_dng):
        """Default (preview=False) DNG should not have preview tags."""
        assert "EXIF:PreviewImage" not in base_dng["dng_tags"], (
            "preview=False but EXIF:PreviewImage is present"
        )


# ---------------------------------------------------------------------------
# Thumbnail feature — thumbnail=True adds TIFF thumbnail
# ---------------------------------------------------------------------------

class TestThumbnailFeature:
    """DNG tags that appear only when ``thumbnail=True``."""

    def test_thumbnail_present_when_enabled(self, raw_sample, tmp_path):
        dng_path = _write_dng(
            str(raw_sample["abs_path"]), tmp_path, thumbnail=True,
        )
        tags = _exiftool_tags(dng_path)
        Path(dng_path).unlink()
        assert "EXIF:ThumbnailTIFF" in tags, (
            "thumbnail=True but no EXIF:ThumbnailTIFF"
        )

    def test_thumbnail_absent_when_disabled(self, base_dng):
        """Default (thumbnail=False) DNG should not have thumbnail."""
        assert "EXIF:ThumbnailTIFF" not in base_dng["dng_tags"]


# ---------------------------------------------------------------------------
# Embed-raw feature — embed_raw=True embeds the original file
# ---------------------------------------------------------------------------

class TestEmbedRawFeature:
    """DNG tags that appear only when ``embed_raw=True``."""

    def test_original_raw_tags_present_when_embedded(self, raw_sample, tmp_path):
        dng_path = _write_dng(
            str(raw_sample["abs_path"]), tmp_path, embed_raw=True,
        )
        tags = _exiftool_tags(dng_path)
        Path(dng_path).unlink()
        assert "EXIF:OriginalRawFileName" in tags
        assert "EXIF:OriginalRawFileDigest" in tags

    def test_original_raw_tags_absent_when_not_embedded(self, base_dng):
        """Default (embed_raw=False) DNG should not have original RAW tags."""
        assert "EXIF:OriginalRawFileName" not in base_dng["dng_tags"]
        assert "EXIF:OriginalRawFileDigest" not in base_dng["dng_tags"]


# ---------------------------------------------------------------------------
# Artist feature — artist param overrides the Artist tag
# ---------------------------------------------------------------------------

class TestArtistFeature:

    def test_artist_written_to_dng(self, raw_sample, tmp_path):
        dng_path = _write_dng(
            str(raw_sample["abs_path"]), tmp_path, artist="Jane Doe",
        )
        tags = _exiftool_tags(dng_path)
        Path(dng_path).unlink()
        assert tags.get("EXIF:Artist") == "Jane Doe"

    def test_artist_preserved_or_absent_when_not_overridden(self, base_dng):
        """When artist param is omitted, the RAW's artist (if any) is preserved.
        If the RAW has no artist, the DNG should have none either."""
        raw_artist = base_dng["raw_tags"].get("EXIF:Artist")
        dng_artist = base_dng["dng_tags"].get("EXIF:Artist")
        if raw_artist is not None:
            # RAW has an artist — verify it survives conversion.
            assert dng_artist is not None, (
                f"RAW has Artist={raw_artist!r} but DNG has no Artist"
            )
            assert dng_artist.strip() == raw_artist.strip(), (
                f"RAW Artist={raw_artist!r} but DNG Artist={dng_artist!r}"
            )
        else:
            # RAW has no artist — DNG should not fabricate one.
            assert dng_artist is None, (
                f"RAW has no Artist but DNG has Artist={dng_artist!r}"
            )


# ---------------------------------------------------------------------------
# Software tag — dnglab-py identifies itself
# ---------------------------------------------------------------------------

class TestSoftwareTag:

    def test_software_is_dnglab_py(self, base_dng):
        assert base_dng["dng_tags"].get("EXIF:Software") == "dnglab-py"


# ---------------------------------------------------------------------------
# Make/Model feature — normalised but recognisable
# ---------------------------------------------------------------------------

class TestMakeModel:
    """Make and Model may be normalised but must still identify the camera."""

    def test_make_present_and_nonempty(self, base_dng):
        assert base_dng["dng_tags"].get("EXIF:Make"), "DNG is missing EXIF:Make"

    def test_model_present_and_nonempty(self, base_dng):
        assert base_dng["dng_tags"].get("EXIF:Model"), "DNG is missing EXIF:Model"

    def test_unique_camera_model_present(self, base_dng):
        assert base_dng["dng_tags"].get("EXIF:UniqueCameraModel"), (
            "DNG is missing EXIF:UniqueCameraModel"
        )

    def test_make_resembles_raw_make(self, base_dng):
        """DNG Make should at least share a prefix with the RAW Make."""
        raw_make = (base_dng["raw_tags"].get("EXIF:Make") or "").strip().lower()
        dng_make = (base_dng["dng_tags"].get("EXIF:Make") or "").strip().lower()
        # Normalisation may shorten "NIKON CORPORATION" → "Nikon", so
        # check the shorter is a prefix of the longer (case-insensitive).
        shorter, longer = sorted([raw_make, dng_make], key=len)
        assert longer.startswith(shorter[:4]), (
            f"DNG Make {dng_make!r} doesn't resemble RAW Make {raw_make!r}"
        )
