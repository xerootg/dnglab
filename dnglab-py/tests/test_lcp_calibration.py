"""Parity pins for ``dnglab_py.lcp_dng_calibration``.

This is the canonical Adobe LCP → DNG WarpRectilinear / FixVignetteRadial
conversion that the backend's ``LcpDB.get_calibration`` delegates to (the math
used to be duplicated in pure Python in ``backend/utils/lcp_db.py``). The golden
values below were captured from that original Python implementation before it
was deleted, so this test guards against drift in the folded rawler code.

No RAW sample files are needed, so these run in any environment with the
``dnglab_py`` wheel installed.
"""

from __future__ import annotations

import json
import math

import pytest

dnglab_py = pytest.importorskip("dnglab_py")


def _isclose(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def _assert_dict_close(got: dict, expected: dict) -> None:
    assert got is not None
    assert set(got.keys()) == set(expected.keys()), (got.keys(), expected.keys())
    for key, exp in expected.items():
        val = got[key]
        if isinstance(exp, dict):
            assert set(val.keys()) == set(exp.keys()), (key, val.keys(), exp.keys())
            for sub, sexp in exp.items():
                if isinstance(sexp, list):
                    assert len(val[sub]) == len(sexp), (key, sub)
                    for a, b in zip(val[sub], sexp):
                        assert _isclose(a, b), (key, sub, a, b)
                else:
                    assert _isclose(val[sub], sexp), (key, sub, val[sub], sexp)
        else:
            assert _isclose(val, exp), (key, val, exp)


SAMPLE1_POINTS = [
    {"focal": 50.0, "aperture": 2.8, "flx": 1.0, "cx": 0.5, "cy": 0.5,
     "k1": 0.1, "k2": -0.05, "k3": 0.02},
]
SAMPLE1_EXPECTED = {
    "distortion": {
        "k": [1.0, 0.03611111111111112, -0.0065200617283950645, 0.0009417866941015095],
        "kt": [0.0, 0.0],
        "cx": 0.5,
        "cy": 0.5,
    },
}

SAMPLE2_POINTS = [
    {"focal": 24.0, "aperture": 4.0, "flx": 1.2, "cx": 0.5, "cy": 0.5,
     "k1": -0.2, "k2": 0.1, "k3": -0.03,
     "v1": -1.5, "v2": 0.8, "v3": -0.2,
     "ca_red_scale": 1.0002, "ca_blue_scale": 0.9995},
]
SAMPLE2_EXPECTED = {
    "distortion": {
        "k": [1.0, -0.054253472222222224, 0.007358598120418597, -0.0005988442480809405],
        "kt": [0.0, 0.0],
        "cx": 0.5,
        "cy": 0.5,
    },
    "tca": {"kr": 1.0002, "kb": 0.9995},
    "vignetting": {
        "k": [-0.4069010416666667, 0.058868784963348776, -0.00399229498720627, 0.0, 0.0],
        "cx": 0.5,
        "cy": 0.5,
    },
}

# Canon EF 100-400mm f/4.5-5.6L IS II USM, focal=100 aperture=5.6, 5640x3752
SAMPLE3_POINTS = [
    {"focal": 100.0, "aperture": 5.6, "flx": 2.987735, "cx": 0.5, "cy": 0.5,
     "k1": -0.291151, "k2": -1.1918, "k3": 1.114907,
     "v1": -2.182246, "v2": -101.904841, "v3": 1062.806708},
]
SAMPLE3_EXPECTED = {
    "distortion": {
        "k": [1.0, -0.011762688252819233, -0.0019452704203300967, 7.351966932995161e-05],
        "kt": [0.0, 0.0],
        "cx": 0.5,
        "cy": 0.5,
    },
    "vignetting": {
        "k": [-0.088164146401564, -0.16633031791050654, 0.07008404982102942, 0.0, 0.0],
        "cx": 0.5,
        "cy": 0.5,
    },
}


def test_sample1_distortion_only():
    got = dnglab_py.lcp_dng_calibration(json.dumps(SAMPLE1_POINTS), 50.0, 2.8, 6000, 4000)
    _assert_dict_close(got, SAMPLE1_EXPECTED)
    assert "tca" not in got
    assert "vignetting" not in got


def test_sample2_distortion_tca_vignette():
    got = dnglab_py.lcp_dng_calibration(json.dumps(SAMPLE2_POINTS), 24.0, 4.0, 4000, 3000)
    _assert_dict_close(got, SAMPLE2_EXPECTED)


def test_sample3_canon_real_point():
    got = dnglab_py.lcp_dng_calibration(json.dumps(SAMPLE3_POINTS), 100.0, 5.6, 5640, 3752)
    _assert_dict_close(got, SAMPLE3_EXPECTED)
    assert "tca" not in got


def test_none_focal_uses_first_point():
    # Sample 2 has a single point; None focal/aperture must use it verbatim.
    expected = {
        "distortion": {
            "k": [1.0, -0.05015432098765433, 0.006288639784331658, -0.00047310368747865484],
            "kt": [0.0, 0.0],
            "cx": 0.5,
            "cy": 0.5,
        },
        "tca": {"kr": 1.0002, "kb": 0.9995},
        "vignetting": {
            "k": [-0.37615740740740744, 0.050309118274653265, -0.003154024583191033, 0.0, 0.0],
            "cx": 0.5,
            "cy": 0.5,
        },
    }
    got = dnglab_py.lcp_dng_calibration(json.dumps(SAMPLE2_POINTS), None, None, 6000, 4000)
    _assert_dict_close(got, expected)


def test_tca_emitted_at_identity_scale():
    # A present-but-identity CA scale must still produce a tca key (the API
    # contract the frontend uniform compiler relies on).
    points = [dict(SAMPLE1_POINTS[0], ca_red_scale=1.0)]
    got = dnglab_py.lcp_dng_calibration(json.dumps(points), 50.0, 2.8, 6000, 4000)
    assert got["tca"] == {"kr": 1.0, "kb": 1.0}


def test_guard_cases_return_none():
    assert dnglab_py.lcp_dng_calibration(json.dumps(SAMPLE1_POINTS), 50.0, 2.8, 0, 4000) is None
    assert dnglab_py.lcp_dng_calibration(json.dumps(SAMPLE1_POINTS), 50.0, 2.8, 6000, 0) is None
    assert dnglab_py.lcp_dng_calibration("[]", 50.0, 2.8, 6000, 4000) is None
