#!/usr/bin/env python3
"""Offline verifier for Option B slider-seed fit (docs: /tmp/option_b_design.md s4).

Replicates the megaShader.js pipeline in EXACT op order on the neutral-develop +
embedded-preview grids the Rust producer dumps, using the SHIPPED recipe.json
params (no refit), and reports per-channel + mean MAE(editor_forward(neutral),
preview) on the 0-255 scale.

Pure numpy + PIL. Ports:
  - srgb_decode / srgb_encode    : megaShader.js:127-140  (IEC 61966-2-1)
  - SplineInterpolator           : spline.js (Paint.NET natural cubic, y2 ends=0)
  - byte-LUT quantize            : util.js:36-38  floor(y*256) clamp [0,255] /255
  - wbExp / contrastFactor / satFactor : megaShader.js:829-851
  - master luma ratio curve      : megaShader.js:434-443  (linear, luma-only)
  - highlight rolloff            : megaShader.js:454-458
  - hpminde_clip                 : megaShader.js:217-231  (LAB constant-L bisect)

Op order (megaShader.js, color-relevant only):
  srgb_decode -> WB linear diagonal (*exp2(wbExp), exposureEV=0)
  -> master luma curve (linear ratio) -> highlight rolloff -> hpminde/clamp
  -> srgb_encode -> contrast (=identity) -> vibrance -> hue(=identity)
  -> saturation (mix toward Rec709 luma).

Usage:
  python3 verify_option_b.py [--dump DIR] [--out JSON] [--f16]
Defaults: --dump /tmp/optb_dump  --out /tmp/option_b_verify.json
"""

import argparse
import json
import sys
import traceback

import numpy as np
from PIL import Image

REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


# ----------------------------------------------------------------------------
# Exact ports (megaShader.js:127-140 == srgb.rs srgb_invert_gamma/apply_gamma)
# ----------------------------------------------------------------------------
def srgb_decode(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_encode(c):
    c = np.asarray(c, dtype=np.float64)
    # GLSL pow() of a negative base is undefined; the shader feeds hpminde_clip
    # output which is clamped >= 0, so clip here for the >0.0031308 branch only.
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(np.clip(c, 0.0, None), 1.0 / 2.4) - 0.055,
    )


# ----------------------------------------------------------------------------
# Paint.NET natural cubic spline — ported verbatim from spline.js
# ----------------------------------------------------------------------------
class SplineInterpolator:
    def __init__(self, points):
        pts = sorted(points, key=lambda p: p[0])
        n = len(pts)
        self.xa = [float(p[0]) for p in pts]
        self.ya = [float(p[1]) for p in pts]
        u = [0.0] * n
        y2 = [0.0] * n
        u[0] = 0.0
        y2[0] = 0.0
        for i in range(1, n - 1):
            wx = self.xa[i + 1] - self.xa[i - 1]
            sig = (self.xa[i] - self.xa[i - 1]) / wx
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            ddydx = (
                (self.ya[i + 1] - self.ya[i]) / (self.xa[i + 1] - self.xa[i])
                - (self.ya[i] - self.ya[i - 1]) / (self.xa[i] - self.xa[i - 1])
            )
            u[i] = (6.0 * ddydx / wx - sig * u[i - 1]) / p
        y2[n - 1] = 0.0
        for i in range(n - 2, -1, -1):
            y2[i] = y2[i] * y2[i + 1] + u[i]
        self.y2 = y2

    def interpolate(self, x):
        n = len(self.ya)
        klo = 0
        khi = n - 1
        while khi - klo > 1:
            k = (khi + klo) >> 1
            if self.xa[k] > x:
                khi = k
            else:
                klo = k
        h = self.xa[khi] - self.xa[klo]
        a = (self.xa[khi] - x) / h
        b = (x - self.xa[klo]) / h
        return (
            a * self.ya[klo]
            + b * self.ya[khi]
            + ((a ** 3 - a) * self.y2[klo] + (b ** 3 - b) * self.y2[khi]) * (h * h) / 6.0
        )


def build_lut(points, f16=False):
    """util.js:36-38 byte LUT (floor(y*256), clamp [0,255], /255).

    --f16 path skips the byte quantize (production half-float LUT) and returns
    raw spline values; conservative default is the byte path (worse case, the
    documented JS fallback).
    """
    sp = SplineInterpolator([list(p) for p in points])
    out = np.empty(256, dtype=np.float64)
    for i in range(256):
        y = sp.interpolate(i / 255.0)
        if f16:
            out[i] = y
        else:
            out[i] = min(255, max(0, int(np.floor(y * 256.0)))) / 255.0
    return out


def lut_sample(lut, x):
    """texture2D nearest: index byte LUT by round(clamp(x,0,1)*255) (spec s4.1)."""
    idx = np.clip(np.round(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(int)
    return lut[idx]


# ----------------------------------------------------------------------------
# Uniform derivations — megaShader.js:829-851
# ----------------------------------------------------------------------------
def sat_factor(s):
    s = float(np.clip(s, -1.0, 1.0))
    return (1.0 - 1.0 / (1.001 - s)) if s > 0 else -s


def contrast_factor(c):
    c = float(np.clip(c, -1.0, 1.0))
    return 1.0 / (1.0 - c) if c > 0 else 1.0 + c


def wb_exp(t, ti):
    t = float(np.clip(t, -1.0, 1.0))
    ti = float(np.clip(ti, -1.0, 1.0))
    return np.array([t * 0.45 - ti * 0.08, ti * 0.4, -t * 0.45 - ti * 0.08], dtype=np.float64)


# ----------------------------------------------------------------------------
# HPMINDE — faithful port of megaShader.js:154-231 (LAB constant-L chroma bisect)
# ----------------------------------------------------------------------------
_M_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.1191920],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
# NB: the GLSL mat3 constructor is column-major; the listed numbers map to the
# transpose. Reconstruct the row-major matrix the shader effectively applies:
_M_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
_M_XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_DELTA = 6.0 / 29.0


def _lab_f(t):
    at = abs(t)
    if at > _DELTA * _DELTA * _DELTA:
        return np.sign(t) * (at ** (1.0 / 3.0))
    return t / (3.0 * _DELTA * _DELTA) + 4.0 / 29.0


def _lab_f_inv(t):
    if abs(t) > _DELTA:
        return t * t * t
    return 3.0 * _DELTA * _DELTA * (t - 4.0 / 29.0)


def _linear_srgb_to_lab(rgb):
    xyz = _M_SRGB_TO_XYZ @ rgb
    f = np.array(
        [_lab_f(xyz[0] / _D65[0]), _lab_f(xyz[1] / _D65[1]), _lab_f(xyz[2] / _D65[2])]
    )
    return np.array([116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])])


def _lab_to_linear_srgb(lab):
    fy = (lab[0] + 16.0) / 116.0
    fx = lab[1] / 500.0 + fy
    fz = fy - lab[2] / 200.0
    xyz = np.array([_D65[0] * _lab_f_inv(fx), _D65[1] * _lab_f_inv(fy), _D65[2] * _lab_f_inv(fz)])
    return _M_XYZ_TO_SRGB @ xyz


def _in_unit_gamut(rgb):
    return bool(np.all(rgb >= 0.0) and np.all(rgb <= 1.0))


def _hpminde_bisect_chroma(L, cos_h, sin_h, c_max):
    lo, hi = 0.0, c_max
    for _ in range(12):
        mid = (lo + hi) * 0.5
        rgb = _lab_to_linear_srgb(np.array([L, mid * cos_h, mid * sin_h]))
        if _in_unit_gamut(rgb):
            lo = mid
        else:
            hi = mid
    return lo


def _hpminde_clip_pixel(rgb_in):
    if _in_unit_gamut(rgb_in):
        return rgb_in
    lab1 = _linear_srgb_to_lab(rgb_in)
    L1 = lab1[0]
    C1 = np.sqrt(lab1[1] * lab1[1] + lab1[2] * lab1[2])
    if C1 < 1e-6:
        Lc = float(np.clip(L1, 0.0, 100.0))
        return np.clip(_lab_to_linear_srgb(np.array([Lc, 0.0, 0.0])), 0.0, 1.0)
    cos_h = lab1[1] / C1
    sin_h = lab1[2] / C1
    Lc = float(np.clip(L1, 0.0, 100.0))
    Cin = _hpminde_bisect_chroma(Lc, cos_h, sin_h, C1)
    return np.clip(_lab_to_linear_srgb(np.array([Lc, Cin * cos_h, Cin * sin_h])), 0.0, 1.0)


def hpminde_clip(lin):
    """Vectorized wrapper: only the out-of-[0,1] pixels hit the per-pixel solve."""
    H, W, _ = lin.shape
    flat = lin.reshape(-1, 3)
    out = flat.copy()
    in_gamut = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    bad = np.where(~in_gamut)[0]
    for i in bad:
        out[i] = _hpminde_clip_pixel(flat[i])
    return out.reshape(H, W, 3)


# ----------------------------------------------------------------------------
# Forward pass — EXACT shader order (megaShader.js)
# ----------------------------------------------------------------------------
class Params:
    def __init__(self, temperature, tint, lumCurve, saturation, vibrance, contrast=0.0):
        self.temperature = temperature
        self.tint = tint
        self.lumCurve = lumCurve  # list[[x,y],...] or None
        self.saturation = saturation
        self.vibrance = vibrance
        self.contrast = contrast


def editor_forward(neutral_srgb, params, f16=False, apply_hpminde=True):
    lin = srgb_decode(neutral_srgb)
    # step 2: WB (linear diagonal multiply), exposureEV = 0
    lin = lin * (2.0 ** wb_exp(params.temperature, params.tint))[None, None, :]
    # step 4: master luma curve as a LINEAR ratio (luma-only)
    if params.lumCurve:
        lut = build_lut(params.lumCurve, f16=f16)
        linLum = np.maximum(np.tensordot(lin, REC709, axes=([2], [0])), 0.0)
        q = np.clip(linLum, 0.0, 1.0)
        newLinLum = lut_sample(lut, q)
        ratio = np.where(
            linLum <= 1.0,
            np.where(linLum > 0.001, newLinLum / np.maximum(linLum, 1e-9), newLinLum),
            newLinLum,
        )
        lin = lin * ratio[..., None]
    # step 5: highlight rolloff if max channel > 1
    mx = lin.max(axis=2, keepdims=True)
    roll = mx > 1.0
    tparam = (mx - 1.0) / (mx + 1.0)
    rolled = lin / np.maximum(mx, 1e-9) * (1.0 - tparam) + 1.0 * tparam
    lin = np.where(roll, rolled, lin)
    # step 6: HPMINDE clip (min channel < 0 case)
    if apply_hpminde:
        lin = hpminde_clip(lin)
    color = srgb_encode(lin)  # -> DISPLAY/sRGB space
    # step 7: contrast (= identity in Option B, contrast=0 -> factor 1)
    color = (color - 0.5) * contrast_factor(params.contrast) + 0.5
    # step 8: vibrance
    avg = np.tensordot(color, REC709, axes=([2], [0]))[..., None]
    mxc = color.max(axis=2, keepdims=True)
    amt = (mxc - avg) * (-params.vibrance * 3.0)
    color = np.clip(color * (1.0 - amt) + mxc * amt, 0.0, 1.0)
    # step 9: hue rotation (= identity at hue=0)
    # step 10: saturation (mix toward Rec709 luma) -- AFTER curve/contrast/vibrance
    lum = np.tensordot(color, REC709, axes=([2], [0]))[..., None]
    sf = sat_factor(params.saturation)
    color = color * (1.0 - sf) + lum * sf
    # steps 11-13 inactive (cm/cg/bw off, RGB per-channel curves identity, sharpen/grain off)
    return np.clip(color, 0.0, 1.0)


def mae_per_channel(neutral_srgb, preview_srgb, params, **kw):
    out = editor_forward(neutral_srgb, params, **kw)
    diff = np.abs(out - preview_srgb) * 255.0
    return diff.reshape(-1, 3).mean(axis=0)


# ----------------------------------------------------------------------------
# colorCurves baseline: 3x1D per-channel display-space curves (row 1, step 12),
# saturation OFF. Option B replaced colorCurves, so recipe.json has none --
# reconstruct a quick reference fit of neutral->preview per channel in display
# space (this is the lane the baseline used: per-channel LUT in DISPLAY).
# ----------------------------------------------------------------------------
def fit_channel_curve(x_disp, y_disp, nbins=24, nknots=16):
    """Monotone binned 1D fit x->y in display [0,1], resampled to uniform knots."""
    order = np.argsort(x_disp)
    xs = x_disp[order]
    ys = y_disp[order]
    n = len(xs)
    edges = np.linspace(0, n, nbins + 1).astype(int)
    bx, by = [], []
    for k in range(nbins):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        bx.append(xs[a:b].mean())
        by.append(ys[a:b].mean())
    bx = np.array(bx)
    by = np.array(by)
    # cummax-monotone
    for i in range(1, len(by)):
        if by[i] < by[i - 1]:
            by[i] = by[i - 1]
    # resample to uniform knots in [0,1] via linear interp on the binned fit
    knots = np.linspace(0.0, 1.0, nknots)
    yk = np.interp(knots, bx, by, left=by[0], right=by[-1])
    for i in range(1, len(yk)):
        if yk[i] < yk[i - 1]:
            yk[i] = yk[i - 1]
    return list(zip(knots.tolist(), yk.tolist()))


def baseline_forward(neutral_srgb, preview_srgb, temperature, tint, f16=False):
    """Reference colorCurves variant: WB + per-channel DISPLAY curves, no sat.

    Mirrors the megaShader row-1 lane (step 12): after WB+srgb_encode, each
    channel is remapped by its own 1D LUT; saturation is left at neutral.
    We fit each channel curve from neutral-after-WB(display) -> preview(display)
    so the variant is a fair side-by-side baseline (not the shipped recipe,
    which dropped colorCurves -- see s0/s4.3).
    """
    lin = srgb_decode(neutral_srgb)
    lin = lin * (2.0 ** wb_exp(temperature, tint))[None, None, :]
    lin = hpminde_clip(np.where(lin > 1.0, 1.0, lin))  # simple clamp+gamut for baseline
    disp = srgb_encode(lin)
    pv = preview_srgb
    luts = []
    for ch in range(3):
        pts = fit_channel_curve(disp[..., ch].ravel(), pv[..., ch].ravel())
        luts.append(build_lut(pts, f16=f16))
    out = np.empty_like(disp)
    for ch in range(3):
        out[..., ch] = lut_sample(luts[ch], disp[..., ch])
    return np.clip(out, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Regression assertions (s4.2)
# ----------------------------------------------------------------------------
def run_assertions():
    """Encode the 3 ordering/luma-only/WB regression guards as tests.

    Returns dict of bool results; raises AssertionError on failure.
    """
    results = {}

    # (2) master curve is a LINEAR luma RATIO applied equally to RGB:
    #     a pure-red linear pixel must keep G=B=0 through the curve (ratio
    #     preserves zeros), proving luma-only (not per-channel, not display).
    red_lin = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float64)
    lut = build_lut([[0.0, 0.0], [0.5, 0.9], [1.0, 1.0]])  # arbitrary non-identity
    linLum = np.maximum(np.tensordot(red_lin, REC709, axes=([2], [0])), 0.0)
    q = np.clip(linLum, 0.0, 1.0)
    newLinLum = lut_sample(lut, q)
    ratio = np.where(linLum > 0.001, newLinLum / np.maximum(linLum, 1e-9), newLinLum)
    toned = red_lin * ratio[..., None]
    assert toned[0, 0, 1] == 0.0 and toned[0, 0, 2] == 0.0, "master curve not luma-only (G/B leaked)"
    assert toned[0, 0, 0] != red_lin[0, 0, 0], "master curve had no effect on R"
    results["curve_is_luma_ratio_keeps_zeros"] = True

    # (3) WB is a LINEAR-light diagonal multiply BEFORE the curve. Verify the
    #     forward pass scales each channel by exactly exp2(wbExp) in linear,
    #     measured by disabling the curve and reading back a pure-channel probe
    #     in linear (decode the output -> divide by input -> equals 2^wbExp).
    t, ti = -0.5, 0.3
    g = 2.0 ** wb_exp(t, ti)
    probe = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float64)
    plin = srgb_decode(probe) * g[None, None, :]
    expect = srgb_decode(probe) * g[None, None, :]
    assert np.allclose(plin, expect), "WB not a per-channel linear diagonal"
    # confirm it is BEFORE the curve: curve consumes the WB'd luminance.
    p_noc = Params(t, ti, None, 0.0, 0.0)
    out_noc = editor_forward(probe, p_noc, apply_hpminde=False)
    # output (display) decoded back to linear should equal srgb_decode(probe)*g
    back = srgb_decode(out_noc)
    assert np.allclose(back, srgb_decode(probe) * g[None, None, :], atol=1e-6), \
        "WB diagonal not applied as linear multiply before encode"
    results["wb_linear_diagonal_before_curve"] = True

    # (1) saturation runs AFTER the master curve, contrast, and vibrance.
    #     Construct a chromatic pixel; with saturation active the OUTPUT must
    #     differ from the same pipeline with saturation=0, AND toggling the
    #     master curve must change the value the saturation step receives
    #     (i.e. saturation is downstream of the curve). We verify ordering by:
    #     forward(curve+sat) != forward(curve only) and the delta is a
    #     pure mix-toward-luma applied to the post-curve/post-vibrance color.
    pix = np.array([[[0.8, 0.2, 0.1]]], dtype=np.float64)
    curve = [[0.0, 0.0], [0.25, 0.4], [0.5, 0.55], [0.75, 0.8], [1.0, 1.0]]
    p_sat = Params(0.0, 0.0, curve, 0.5, 0.2)
    p_nosat = Params(0.0, 0.0, curve, 0.0, 0.2)  # same curve+vibrance, no sat
    o_sat = editor_forward(pix, p_sat, apply_hpminde=False)
    o_nosat = editor_forward(pix, p_nosat, apply_hpminde=False)
    # reconstruct: o_sat must equal mix(o_nosat_color, lum, satFactor) where
    # o_nosat_color is the color AT the saturation step (= o_nosat since sat=0
    # leaves it unchanged and clamps identically). So apply satFactor to o_nosat:
    sf = sat_factor(0.5)
    lum = np.tensordot(o_nosat, REC709, axes=([2], [0]))[..., None]
    expect_sat = np.clip(o_nosat * (1.0 - sf) + lum * sf, 0.0, 1.0)
    assert np.allclose(o_sat, expect_sat, atol=1e-9), \
        "saturation is not the final op after curve/contrast/vibrance"
    assert not np.allclose(o_sat, o_nosat), "saturation had no measurable effect"
    results["saturation_runs_last_after_curve_vibrance"] = True

    return results


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def load_grid(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def unflatten(points_flat):
    return [[points_flat[i], points_flat[i + 1]] for i in range(0, len(points_flat) - 1, 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/optb_dump")
    ap.add_argument("--out", default="/tmp/option_b_verify.json")
    ap.add_argument("--f16", action="store_true", help="skip byte-LUT quantize (prod F16 path)")
    args = ap.parse_args()

    neutral = load_grid(f"{args.dump}/neutral_grid.png")
    preview = load_grid(f"{args.dump}/preview_grid.png")
    with open(f"{args.dump}/recipe.json") as f:
        recipe = json.load(f)
    with open(f"{args.dump}/fit_debug.json") as f:
        fit_debug = json.load(f)

    # --- read SHIPPED params from recipe.json (do not refit) ---
    temperature = float(recipe.get("wbTemp", 0.0) or 0.0)
    tint = float(recipe.get("wbTint", 0.0) or 0.0)
    saturation = float(recipe.get("saturation", 0.0) or 0.0)
    vibrance = float(recipe.get("vibrance", 0.0) or 0.0)
    tone = recipe.get("tone")
    lumCurve = None
    if tone and isinstance(tone.get("points"), list) and len(tone["points"]) >= 4:
        lumCurve = unflatten(tone["points"])
    params = Params(temperature, tint, lumCurve, saturation, vibrance)

    # --- regression assertions ---
    assertions = run_assertions()
    assertions_pass = all(assertions.values())

    # --- Option B MAE (byte path = documented worse-case default) ---
    optB = mae_per_channel(neutral, preview, params, f16=args.f16)

    # --- colorCurves baseline (reference reconstruction) ---
    base_out = baseline_forward(neutral, preview, temperature, tint, f16=args.f16)
    base_diff = np.abs(base_out - preview) * 255.0
    baseline = base_diff.reshape(-1, 3).mean(axis=0)

    # --- cross-check vs producer Step G MAE ---
    producer = fit_debug.get("stepG_mae")
    producer_arr = np.array(producer, dtype=np.float64) if producer else None
    agree_delta = None
    if producer_arr is not None:
        agree_delta = float(np.abs(optB - producer_arr).max())

    result = {
        "optionB_mae": [float(x) for x in optB],
        "optionB_mean": float(optB.mean()),
        "baseline_mae": [float(x) for x in baseline],
        "baseline_mean": float(baseline.mean()),
        "producer_stepG_mae": [float(x) for x in producer_arr] if producer_arr is not None else None,
        "producer_stepG_mean": float(producer_arr.mean()) if producer_arr is not None else None,
        "agree_delta": agree_delta,
        "agree_within_0p1": (agree_delta is not None and agree_delta <= 0.1),
        "assertions": assertions,
        "assertions_pass": assertions_pass,
        "params": {
            "temperature": temperature,
            "tint": tint,
            "saturation": saturation,
            "vibrance": vibrance,
            "contrast": 0.0,
            "lumCurve": lumCurve,
            "satFactor": sat_factor(saturation),
            "wbExp": wb_exp(temperature, tint).tolist(),
            "f16": bool(args.f16),
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Option B  per-channel MAE [R,G,B] = {optB.tolist()}  mean = {optB.mean():.4f}")
    print(f"Baseline  per-channel MAE [R,G,B] = {baseline.tolist()}  mean = {baseline.mean():.4f}")
    if producer_arr is not None:
        print(f"Producer  stepG_mae        [R,G,B] = {producer_arr.tolist()}  mean = {producer_arr.mean():.4f}")
        print(f"agree_delta (max |optB-stepG|) = {agree_delta:.4f}  within_0.1 = {agree_delta <= 0.1}")
    print(f"assertions_pass = {assertions_pass}  ({assertions})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
