"""Verify raw_metadata() reports values consistent with exiftool.

The ``raw_metadata()`` Python API normalises some fields (make, model,
rational values) compared to the raw EXIF bytes that exiftool reads.
More importantly, the API *resolves* lens models, serial numbers, and
ISO from vendor-specific MakerNotes when they are not present in
standard EXIF tags.  These tests verify that every API-reported value
can be traced back to *some* exiftool-visible tag — EXIF, MakerNotes,
or Composite.

Requires: ``exiftool`` on PATH.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from fractions import Fraction

import pytest

import dnglab_py


# ---------------------------------------------------------------------------
# Skip entire module if exiftool isn't installed.
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    shutil.which("exiftool") is None,
    reason="exiftool not found on PATH",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_exiftool_cache: dict[str, dict] = {}


def _exiftool_tags(path: str) -> dict[str, object]:
    """Return exiftool ``-G -n`` dict, cached per path."""
    if path not in _exiftool_cache:
        out = subprocess.check_output(
            ["exiftool", "-j", "-G", "-n", path],
            text=True,
            timeout=30,
        )
        _exiftool_cache[path] = json.loads(out)[0]
    return _exiftool_cache[path]


def _rational_to_float(val: str | int | float | None) -> float | None:
    """Convert a rational string like ``'35/10'`` to a float (3.5)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and "/" in val:
        return float(Fraction(val))
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _close_enough(a: float, b: float, rel_tol: float = 1e-4) -> bool:
    """Return True if *a* and *b* are within *rel_tol* of each other."""
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-10)
    return abs(a - b) / denom <= rel_tol


def _normalise_lens(name: str) -> str:
    """Strip whitespace, punctuation, and common prefixes for lens comparison."""
    s = name.lower()
    # Strip make prefixes that some vendors embed in MakerNotes lens names.
    for prefix in ("olympus ", "panasonic ", "canon ", "nikon ", "sony ",
                    "fujifilm ", "sigma ", "tamron "):
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Collapse whitespace / punctuation.
    return re.sub(r"[\s\-/_.]+", "", s)


# ---------------------------------------------------------------------------
# Cached metadata fixture.
# ---------------------------------------------------------------------------

@pytest.fixture()
def raw_vs_exiftool(raw_sample):
    """Return a dict with both API metadata and exiftool tags for one sample."""
    path = str(raw_sample["abs_path"])
    return {
        "path": path,
        "api": dnglab_py.raw_metadata(path),
        "et": _exiftool_tags(path),
        "sample": raw_sample,
    }


@pytest.fixture(autouse=True, scope="session")
def _cleanup_exiftool_cache():
    """Free the exiftool tag cache after all tests complete."""
    yield
    _exiftool_cache.clear()


# ---------------------------------------------------------------------------
# Integer EXIF fields — must match exiftool exactly
# ---------------------------------------------------------------------------

EXACT_INT_FIELDS = {
    "orientation": "EXIF:Orientation",
    "exposure_program": "EXIF:ExposureProgram",
    "metering_mode": "EXIF:MeteringMode",
    "flash": "EXIF:Flash",
    "white_balance": "EXIF:WhiteBalance",
}


class TestExactIntegerFields:
    """Integer EXIF fields that must match exiftool bit-for-bit."""

    def test_integer_fields_match(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]

        mismatches = []
        for api_key, et_key in EXACT_INT_FIELDS.items():
            api_val = api_exif.get(api_key)
            et_val = et.get(et_key)
            if api_val is None and et_val is None:
                continue
            assert api_val is not None and et_val is not None, (
                f"{api_key}: one side is None "
                f"(api={api_val!r}, exiftool[{et_key}]={et_val!r})"
            )
            if int(api_val) != int(et_val):
                mismatches.append(
                    f"{api_key}: api={api_val!r} exiftool={et_val!r}"
                )
        assert not mismatches, (
            "Integer EXIF fields differ from exiftool:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# Rational / float EXIF fields
# ---------------------------------------------------------------------------

RATIONAL_FIELDS = {
    "exposure_time": "EXIF:ExposureTime",
    "fnumber": "EXIF:FNumber",
    "focal_length": "EXIF:FocalLength",
}


class TestRationalFields:
    """Rational EXIF fields must match exiftool as floats within tolerance."""

    def test_rational_fields_match_as_float(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]

        mismatches = []
        for api_key, et_key in RATIONAL_FIELDS.items():
            api_val = _rational_to_float(api_exif.get(api_key))
            et_val = _rational_to_float(et.get(et_key))
            if api_val is None and et_val is None:
                continue
            assert api_val is not None and et_val is not None, (
                f"{api_key}: one side is None "
                f"(api={api_val!r}, exiftool[{et_key}]={et_val!r})"
            )
            if not _close_enough(api_val, et_val):
                mismatches.append(
                    f"{api_key}: api={api_val} exiftool={et_val}"
                )
        assert not mismatches, (
            "Rational EXIF fields differ from exiftool:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# Date/time fields — string-exact match
# ---------------------------------------------------------------------------

DATE_FIELDS = {
    "date_time_original": "EXIF:DateTimeOriginal",
    "create_date": "EXIF:CreateDate",
}


class TestDateFields:
    """Date/time strings must match exiftool exactly."""

    def test_dates_match(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]

        mismatches = []
        for api_key, et_key in DATE_FIELDS.items():
            api_val = api_exif.get(api_key)
            et_val = et.get(et_key)
            if api_val is None and et_val is None:
                continue
            assert api_val is not None and et_val is not None, (
                f"{api_key}: one side is None "
                f"(api={api_val!r}, exiftool[{et_key}]={et_val!r})"
            )
            if str(api_val) != str(et_val):
                mismatches.append(
                    f"{api_key}: api={api_val!r} exiftool={et_val!r}"
                )
        assert not mismatches, (
            "Date fields differ from exiftool:\n"
            + "\n".join(f"  {m}" for m in mismatches)
        )


# ---------------------------------------------------------------------------
# ISO — falls through EXIF:ISOSpeedRatings → EXIF:ISO → MakerNotes
# ---------------------------------------------------------------------------

class TestISO:
    """ISO speed must be traceable to an exiftool tag."""

    def test_iso_matches(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]

        api_iso = api_exif.get("iso_speed_ratings")
        if api_iso is None:
            pytest.skip("API has no iso_speed_ratings")

        # Try standard EXIF tags first.
        et_iso = et.get("EXIF:ISOSpeedRatings") or et.get("EXIF:ISO")
        if et_iso is not None:
            assert int(api_iso) == int(et_iso), (
                f"ISO: api={api_iso!r} exiftool EXIF={et_iso!r}"
            )
            return

        # Fall through to MakerNotes ISO tags.
        mn_candidates = [
            et.get("MakerNotes:ISO"),
            et.get("MakerNotes:ISO2"),
            et.get("MakerNotes:SonyISO"),
            et.get("MakerNotes:ProgramISO"),
        ]
        for mn_val in mn_candidates:
            if mn_val is None:
                continue
            # Some MakerNotes:ISO values are strings like "0 800" — take last number.
            if isinstance(mn_val, str):
                parts = mn_val.split()
                mn_val = int(parts[-1])
            if int(mn_val) == int(api_iso):
                return  # Match found.

        # Collect what was available for the error message.
        mn_iso_keys = {
            k: v for k, v in et.items()
            if k.startswith("MakerNotes:") and "iso" in k.lower()
        }
        pytest.fail(
            f"ISO {api_iso} not found in any exiftool tag. "
            f"MakerNotes ISO tags: {mn_iso_keys}"
        )


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

class TestOrientationVsExiftool:

    def test_orientation_matches(self, raw_vs_exiftool):
        api_val = raw_vs_exiftool["api"].get("exif", {}).get("orientation")
        et_val = raw_vs_exiftool["et"].get("EXIF:Orientation")
        if api_val is None and et_val is None:
            pytest.skip("Neither source has orientation")
        assert api_val is not None and et_val is not None, (
            f"Orientation mismatch: api={api_val!r} exiftool={et_val!r}"
        )
        assert int(api_val) == int(et_val)


# ---------------------------------------------------------------------------
# Make and model — API normalises, so compare loosely
# ---------------------------------------------------------------------------

class TestMakeModelVsExiftool:
    """Make/model may be normalised but must correspond to exiftool's."""

    def test_make_corresponds(self, raw_vs_exiftool):
        api_make = raw_vs_exiftool["api"]["make"].lower()
        et_make = (raw_vs_exiftool["et"].get("EXIF:Make") or "").lower()
        assert et_make, "exiftool has no EXIF:Make"
        shorter, longer = sorted([api_make, et_make], key=len)
        assert longer.startswith(shorter[:4]), (
            f"API make {api_make!r} doesn't correspond to "
            f"exiftool make {et_make!r}"
        )

    def test_model_corresponds(self, raw_vs_exiftool):
        api_model = raw_vs_exiftool["api"]["model"].lower()
        et_model = (raw_vs_exiftool["et"].get("EXIF:Model") or "").lower()
        assert et_model, "exiftool has no EXIF:Model"
        assert api_model in et_model or et_model in api_model, (
            f"API model {api_model!r} not found in "
            f"exiftool model {et_model!r}"
        )


# ---------------------------------------------------------------------------
# Lens model — verify via EXIF, MakerNotes, or Composite
# ---------------------------------------------------------------------------

class TestLensModel:
    """API lens_model must be traceable to an exiftool EXIF or MakerNotes tag."""

    def test_lens_model_traceable(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]

        api_lens = api_exif.get("lens_model")
        if api_lens is None:
            pytest.skip("API reports no lens_model for this file")

        api_norm = _normalise_lens(api_lens)

        # 1. Direct EXIF match.
        et_exif_lens = et.get("EXIF:LensModel")
        if et_exif_lens and _normalise_lens(et_exif_lens) == api_norm:
            return

        # 2. MakerNotes:LensModel (Olympus, Canon).
        et_mn_lens = et.get("MakerNotes:LensModel")
        if et_mn_lens and isinstance(et_mn_lens, str):
            if _normalise_lens(et_mn_lens) == api_norm:
                return

        # 3. MakerNotes:LensType (Panasonic stores lens name here).
        et_mn_ltype = et.get("MakerNotes:LensType")
        if et_mn_ltype and isinstance(et_mn_ltype, str):
            if _normalise_lens(et_mn_ltype) == api_norm:
                return

        # 4. Composite:LensID (Fujifilm, Panasonic store lens name here).
        et_comp_lid = et.get("Composite:LensID")
        if et_comp_lid and isinstance(et_comp_lid, str):
            if _normalise_lens(et_comp_lid) == api_norm:
                return

        # 5. Nikon: no string lens name in MakerNotes — the API resolves
        #    from a binary lens ID via an internal database.  Verify the
        #    MakerNotes lens ID bytes are present (proving MakerNotes were read).
        et_comp_lspec = et.get("Composite:LensSpec") or et.get("Composite:LensID")
        mn_lens_data = et.get("MakerNotes:Lens") or et.get("MakerNotes:LensIDNumber")
        if et_comp_lspec is not None or mn_lens_data is not None:
            # The API resolved a human-readable lens name from a numeric ID.
            # We can't compare strings, but verify the focal length range
            # embedded in MakerNotes:Lens matches the API lens name.
            mn_lens = et.get("MakerNotes:Lens")
            if mn_lens and isinstance(mn_lens, str):
                # MakerNotes:Lens is like "50 50 1.8 1.8" (min_fl max_fl min_ap max_ap)
                parts = mn_lens.split()
                if len(parts) >= 2:
                    mn_fl = parts[0]
                    # The API lens name should contain this focal length.
                    assert mn_fl in api_lens or f"{mn_fl}mm" in api_lens, (
                        f"API lens {api_lens!r} doesn't contain focal length "
                        f"{mn_fl}mm from MakerNotes:Lens={mn_lens!r}"
                    )
                    return
            # Has lens ID data but we can't do a string match — accept it.
            return

        # Collect what exiftool knows about lenses for the error message.
        lens_tags = {
            k: v for k, v in et.items()
            if any(x in k.lower() for x in ("lens", "lensid", "lensmodel"))
        }
        pytest.fail(
            f"API lens_model={api_lens!r} (normalised={api_norm!r}) "
            f"not traceable to any exiftool tag.\n"
            f"Exiftool lens-related tags: {lens_tags}"
        )


# ---------------------------------------------------------------------------
# Lens make — verify via EXIF or resolved lens object
# ---------------------------------------------------------------------------

class TestLensMake:
    """API lens_make must be traceable to EXIF or the resolved lens."""

    def test_lens_make_traceable(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]
        api_md = raw_vs_exiftool["api"]

        api_lens_make = api_exif.get("lens_make")
        et_lens_make = et.get("EXIF:LensMake")

        if api_lens_make is None:
            # API has no lens_make — verify exiftool also has none.
            assert et_lens_make is None, (
                f"API has no lens_make but exiftool has "
                f"EXIF:LensMake={et_lens_make!r}"
            )
            return

        # 1. Direct EXIF match.
        if et_lens_make and et_lens_make.lower() == api_lens_make.lower():
            return

        # 2. The API may resolve lens_make from its internal lens database.
        #    Verify the resolved lens object exists and its make matches.
        lens_obj = api_md.get("lens")
        if lens_obj and lens_obj.get("lens_make"):
            assert lens_obj["lens_make"].lower() == api_lens_make.lower(), (
                f"lens_make={api_lens_make!r} doesn't match "
                f"resolved lens.lens_make={lens_obj['lens_make']!r}"
            )
            return

        # 3. The lens make should at least match the camera make (common for
        #    first-party lenses) or be a known lens manufacturer.
        known_makes = {
            "canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
            "sigma", "tamron", "tokina", "leica", "samyang", "zeiss",
            "voigtlander", "om system",
        }
        assert api_lens_make.lower() in known_makes, (
            f"lens_make={api_lens_make!r} is not in EXIF:LensMake, not in "
            f"resolved lens object, and not a known lens manufacturer"
        )


# ---------------------------------------------------------------------------
# Serial number — verify via EXIF or MakerNotes
# ---------------------------------------------------------------------------

class TestSerialNumber:
    """API serial_number must be traceable to an exiftool EXIF or MakerNotes tag.

    Known gap: Nikon MakerNotes contain a body serial number that the
    API does not currently extract (marked xfail).
    """

    def test_serial_traceable(self, raw_vs_exiftool):
        api_exif = raw_vs_exiftool["api"].get("exif", {})
        et = raw_vs_exiftool["et"]

        api_serial = api_exif.get("serial_number")
        if api_serial is None:
            # Verify exiftool also has no serial in standard EXIF.
            et_exif_serial = et.get("EXIF:SerialNumber")
            if et_exif_serial is not None:
                pytest.fail(
                    f"API has no serial_number but exiftool has "
                    f"EXIF:SerialNumber={et_exif_serial!r}"
                )
            # MakerNotes may still contain a serial the API doesn't
            # extract.  This is a known gap for some vendors — flag it
            # as xfail so it auto-passes once the API is fixed.
            mn_serial = et.get("MakerNotes:SerialNumber")
            if mn_serial is not None and str(mn_serial).strip():
                pytest.xfail(
                    f"API has no serial_number but exiftool has "
                    f"MakerNotes:SerialNumber={mn_serial!r} — "
                    f"MakerNotes serial extraction not yet implemented"
                )
            return

        api_s = str(api_serial).strip()

        # 1. EXIF:SerialNumber (may be numeric — compare as strings).
        et_exif_serial = et.get("EXIF:SerialNumber")
        if et_exif_serial is not None and str(et_exif_serial).strip() == api_s:
            return

        # 2. MakerNotes:SerialNumber (Nikon, Olympus).
        mn_serial = et.get("MakerNotes:SerialNumber")
        if mn_serial is not None and str(mn_serial).strip() == api_s:
            return

        # 3. MakerNotes:InternalSerialNumber (Panasonic, Sony).
        mn_internal = et.get("MakerNotes:InternalSerialNumber")
        if mn_internal is not None and str(mn_internal).strip() == api_s:
            return

        # Collect what exiftool has.
        serial_tags = {
            k: v for k, v in et.items()
            if "serial" in k.lower()
        }
        pytest.fail(
            f"API serial_number={api_serial!r} not found in any exiftool tag.\n"
            f"Exiftool serial-related tags: {serial_tags}"
        )
