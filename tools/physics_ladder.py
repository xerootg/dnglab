#!/usr/bin/env python3
"""PHYSICS ABLATION LADDER — decompose the Nikon in-body look into model rungs.

REUSES the exact ports from verify_option_b.py (srgb_decode/encode, Paint.NET
natural-cubic SplineInterpolator + byte-LUT quantize, satFactor, contrastFactor,
wbExp, hpminde_clip, REC709). Every rung is measured on the SAME metric: encode
the model output to sRGB display and compare to preview_grid, per-channel + MEAN
display MAE on the 0-255 scale.

Pipeline contract (megaShader.js op order, from /tmp/option_b_design.md s1):
  LINEAR space (after srgb_decode):
    - WB / 3x3 matrix act in linear, BEFORE the tone curve.
    - tone is a luma RATIO in linear (editor lumPoints lane, row 0).
    - highlight rolloff (max>1), HPMINDE clip (any<0).
  DISPLAY space (after srgb_encode):
    - saturation / per-hue residual act in display.

RUNGS:
  M0 identity                 : out = neutral (raw gap)
  M1 WB free linear diagonal  : fit per-channel gain G (LSQ nlin->plin), encode
  M2 WB(stops)+luma tone curve: Option B WB (t,ti) + 16-knot linear luma ratio
  M3 = M2 + scalar saturation : == OPTION B (shipped recipe). ANCHOR ~7.05 @5580
  M4 3x3 linear matrix only   : LSQ nlin->plin in linear, encode
  M5 3x3 matrix + luma tone   : M (LSQ) then residual 16-knot luma ratio.
                                also chroma-decoupled variant; report best.
  M6 = M5 + per-hue residual  : 12 hue bins, per-bin sat+hue twist in HSV
  M7 colorCurves              : 3 independent 1-D display per-channel curves
                                (WB + clamp/hpminde -> encode -> 24-bin monotone
                                16-knot LUT, no saturation). ANCHOR ~2.79 @5580
  M8 global upper bound       : global degree-3 poly RGB->RGB (LSQ), encode

Pure numpy + PIL.
"""

import json
import sys
import traceback

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Exact ports from verify_option_b.py
# ----------------------------------------------------------------------------
REC709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def srgb_decode(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb_encode(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(
        c <= 0.0031308,
        12.92 * c,
        1.055 * np.power(np.clip(c, 0.0, None), 1.0 / 2.4) - 0.055,
    )


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
    idx = np.clip(np.round(np.clip(x, 0.0, 1.0) * 255.0), 0, 255).astype(int)
    return lut[idx]


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


# HPMINDE (LAB constant-L chroma bisect) — verbatim from verify_option_b.py
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
    H, W, _ = lin.shape
    flat = lin.reshape(-1, 3)
    out = flat.copy()
    in_gamut = np.all((flat >= 0.0) & (flat <= 1.0), axis=1)
    bad = np.where(~in_gamut)[0]
    for i in bad:
        out[i] = _hpminde_clip_pixel(flat[i])
    return out.reshape(H, W, 3)


# ----------------------------------------------------------------------------
# Shared linear-domain post-processing: highlight rolloff -> hpminde -> encode
# (same as verify_option_b.editor_forward steps 5,6,encode)
# ----------------------------------------------------------------------------
def lin_rolloff_hpminde_encode(lin, apply_hpminde=True):
    mx = lin.max(axis=2, keepdims=True)
    roll = mx > 1.0
    tparam = (mx - 1.0) / (mx + 1.0)
    rolled = lin / np.maximum(mx, 1e-9) * (1.0 - tparam) + 1.0 * tparam
    lin = np.where(roll, rolled, lin)
    if apply_hpminde:
        lin = hpminde_clip(lin)
    return srgb_encode(lin)


def apply_saturation(color, s):
    lum = np.tensordot(color, REC709, axes=([2], [0]))[..., None]
    sf = sat_factor(s)
    return np.clip(color * (1.0 - sf) + lum * sf, 0.0, 1.0)


def apply_vibrance(color, vibrance):
    if vibrance == 0.0:
        return color
    avg = np.tensordot(color, REC709, axes=([2], [0]))[..., None]
    mxc = color.max(axis=2, keepdims=True)
    amt = (mxc - avg) * (-vibrance * 3.0)
    return np.clip(color * (1.0 - amt) + mxc * amt, 0.0, 1.0)


def disp_mae(out_disp, preview_srgb):
    diff = np.abs(out_disp - preview_srgb) * 255.0
    return diff.reshape(-1, 3).mean(axis=0)


# ----------------------------------------------------------------------------
# Tone-curve luma ratio application (megaShader.js:434-443) using a prebuilt LUT.
# Operates on a LINEAR image; x and y are linear luminance in [0,1].
# ----------------------------------------------------------------------------
def apply_tone_ratio(lin, lut):
    linLum = np.maximum(np.tensordot(lin, REC709, axes=([2], [0])), 0.0)
    q = np.clip(linLum, 0.0, 1.0)
    newLinLum = lut_sample(lut, q)
    ratio = np.where(
        linLum <= 1.0,
        np.where(linLum > 0.001, newLinLum / np.maximum(linLum, 1e-9), newLinLum),
        newLinLum,
    )
    return lin * ratio[..., None]


# ----------------------------------------------------------------------------
# Tone-curve FIT: fit a 16-knot UNIFORM-x linear-luma ratio curve mapping
# x_lin (input linear luma) -> y_lin (target linear luma), per s2 Step D.
# 24 quantile bins -> per-bin mean -> cummax-monotone -> resample to 16 uniform
# knots in [0,1]. Returns points list [[x,y],...] for build_lut.
# ----------------------------------------------------------------------------
def fit_luma_curve(x_lin, y_lin, nbins=24, nknots=16):
    x = np.clip(x_lin.ravel(), 0.0, 1.0)
    y = np.clip(y_lin.ravel(), 0.0, 1.0)
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
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
    for i in range(1, len(by)):
        if by[i] < by[i - 1]:
            by[i] = by[i - 1]
    knots = np.linspace(0.0, 1.0, nknots)
    yk = np.interp(knots, bx, by, left=by[0], right=by[-1])
    for i in range(1, len(yk)):
        if yk[i] < yk[i - 1]:
            yk[i] = yk[i - 1]
    return list(zip(knots.tolist(), yk.tolist()))


# ----------------------------------------------------------------------------
# colorCurves baseline 1-D channel fit (verify_option_b.fit_channel_curve port)
# ----------------------------------------------------------------------------
def fit_channel_curve(x_disp, y_disp, nbins=24, nknots=16):
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
    for i in range(1, len(by)):
        if by[i] < by[i - 1]:
            by[i] = by[i - 1]
    knots = np.linspace(0.0, 1.0, nknots)
    yk = np.interp(knots, bx, by, left=by[0], right=by[-1])
    for i in range(1, len(yk)):
        if yk[i] < yk[i - 1]:
            yk[i] = yk[i - 1]
    return list(zip(knots.tolist(), yk.tolist()))


# ----------------------------------------------------------------------------
# WB-from-stops solve (s2 Step B): given a diagonal linear gain G, factor into
# (t, ti) shader-uniform WB by the wbExp decomposition. Exposure E is folded
# into the tone curve, so it is NOT applied to the WB (returns only t, ti).
# ----------------------------------------------------------------------------
def solve_wb_stops(G):
    Lr, Lg, Lb = np.log2(np.maximum(G, 1e-9))
    t = (Lr - Lb) / 0.90
    ti = (2.0 * Lg - Lr - Lb) / 0.96
    t = float(np.clip(t, -1.0, 1.0))
    ti = float(np.clip(ti, -1.0, 1.0))
    return t, ti


# ----------------------------------------------------------------------------
# RUNGS
# ----------------------------------------------------------------------------
def rung_M0(nlin, plin, preview, neutral):
    """identity: out = neutral."""
    out = neutral.copy()
    return disp_mae(out, preview), {}


def rung_M1(nlin, plin, preview):
    """WB free linear diagonal: per-channel gain G by LSQ nlin[c]->plin[c]."""
    N = nlin.reshape(-1, 3)
    P = plin.reshape(-1, 3)
    G = np.empty(3)
    for c in range(3):
        x = N[:, c]
        y = P[:, c]
        denom = float(np.dot(x, x))
        G[c] = float(np.dot(x, y) / denom) if denom > 1e-12 else 1.0
    out_lin = nlin * G[None, None, :]
    out = lin_rolloff_hpminde_encode(out_lin)
    return disp_mae(out, preview), {"G": G.tolist()}


def rung_M2_M3(neutral, preview, recipe):
    """M2 = Option B WB(stops) + 16-knot linear luma curve, no saturation.
    M3 = M2 + shipped scalar saturation (+ vibrance) == OPTION B.

    Uses the SHIPPED recipe params (WB t/ti, tone curve, saturation, vibrance)
    so M3 reproduces the validated 7.05 anchor exactly. M2 is the same forward
    pass with saturation=0 and vibrance=0.
    """
    t = float(recipe.get("wbTemp", 0.0) or 0.0)
    ti = float(recipe.get("wbTint", 0.0) or 0.0)
    sat = float(recipe.get("saturation", 0.0) or 0.0)
    vib = float(recipe.get("vibrance", 0.0) or 0.0)
    tone = recipe.get("tone")
    pts = None
    if tone and isinstance(tone.get("points"), list) and len(tone["points"]) >= 4:
        fl = tone["points"]
        pts = [[fl[i], fl[i + 1]] for i in range(0, len(fl) - 1, 2)]

    lin0 = srgb_decode(neutral)
    lin = lin0 * (2.0 ** wb_exp(t, ti))[None, None, :]
    if pts:
        lut = build_lut(pts)
        lin = apply_tone_ratio(lin, lut)
    color_pre_sat = lin_rolloff_hpminde_encode(lin)  # display, pre-saturation
    # contrast = identity (0). vibrance then saturation.

    # M2: no saturation, no vibrance
    out_M2 = color_pre_sat.copy()
    mae_M2 = disp_mae(out_M2, preview)

    # M3 = Option B: vibrance then saturation (shader order steps 8 then 10)
    color_vib = apply_vibrance(color_pre_sat, vib)
    out_M3 = apply_saturation(color_vib, sat)
    mae_M3 = disp_mae(out_M3, preview)

    return mae_M2, mae_M3, {"t": t, "ti": ti, "saturation": sat, "vibrance": vib}


def fit_matrix_lsq(nlin, plin):
    """Closed-form LSQ 3x3 M mapping nlin -> plin in linear (P ~= N @ M.T)."""
    N = nlin.reshape(-1, 3)
    P = plin.reshape(-1, 3)
    # Solve for A in N @ A = P  (A is M.T, shape 3x3). M = A.T.
    A, *_ = np.linalg.lstsq(N, P, rcond=None)
    M = A.T
    return M


def rung_M4(nlin, plin, preview):
    """3x3 linear color matrix ONLY: fit M (LSQ nlin->plin), encode(clip(N@M.T))."""
    M = fit_matrix_lsq(nlin, plin)
    out_lin = nlin @ M.T
    out = lin_rolloff_hpminde_encode(out_lin)
    return disp_mae(out, preview), M


def m5_coupled_output(nlin, plin, preview):
    """Variant A (coupled): fit M to full nlin->plin; then fit a residual luma
    ratio curve on luma(N@M.T) -> luma(plin); apply tone ratio to (N@M.T)."""
    M = fit_matrix_lsq(nlin, plin)
    matE = nlin @ M.T
    x_lin = np.maximum(np.tensordot(matE, REC709, axes=([2], [0])), 0.0)
    y_lin = np.maximum(np.tensordot(plin, REC709, axes=([2], [0])), 0.0)
    lut = build_lut(fit_luma_curve(x_lin, y_lin))
    toned = apply_tone_ratio(matE, lut)
    out = lin_rolloff_hpminde_encode(toned)
    return out, M


def m5_decoupled_output(nlin, plin, preview):
    """Variant B (chroma-decoupled): fit M to luma-normalized chromaticity
    (nlin/luma -> plin/luma), tone curve to luma directly; reconstruct
    out = chromaticity(M) * tone(luma)."""
    eps = 1e-6
    nlum = np.maximum(np.tensordot(nlin, REC709, axes=([2], [0])), eps)
    plum = np.maximum(np.tensordot(plin, REC709, axes=([2], [0])), eps)
    nchrom = nlin / nlum[..., None]
    pchrom = plin / plum[..., None]
    M = fit_matrix_lsq(nchrom, pchrom)
    chrom_out = nchrom @ M.T
    lut = build_lut(fit_luma_curve(nlum, plum))
    out_lum = lut_sample(lut, np.clip(nlum, 0.0, 1.0))
    chrom_lum = np.maximum(np.tensordot(chrom_out, REC709, axes=([2], [0])), eps)
    chrom_norm = chrom_out / chrom_lum[..., None]
    recon = chrom_norm * out_lum[..., None]
    out = lin_rolloff_hpminde_encode(recon)
    return out, M


def m5_matrix_only_output(nlin, plin, preview):
    """No-tone fallback: the tone curve is identity-able, so M5 capacity strictly
    includes the bare matrix (M4-style). Including this guarantees M5 <= M4."""
    M = fit_matrix_lsq(nlin, plin)
    out = lin_rolloff_hpminde_encode(nlin @ M.T)
    return out, M


def rung_M5(nlin, plin, preview):
    """3x3 matrix + luma tone curve. Reports the BEST of three valid variants
    (coupled residual tone / chroma-decoupled / matrix-only). All three lie
    inside the model's capacity (a luma curve can be identity), so reporting the
    minimum is the faithful upper-capacity estimate and guarantees M5 <= M4."""
    out_a, M_a = m5_coupled_output(nlin, plin, preview)
    out_b, M_b = m5_decoupled_output(nlin, plin, preview)
    out_c, M_c = m5_matrix_only_output(nlin, plin, preview)
    mae_a, mae_b, mae_c = disp_mae(out_a, preview), disp_mae(out_b, preview), disp_mae(out_c, preview)
    means = {"coupled": float(mae_a.mean()),
             "chroma_decoupled": float(mae_b.mean()),
             "matrix_only_tone_identity": float(mae_c.mean())}
    cands = [("coupled", mae_a, M_a, out_a),
             ("chroma_decoupled", mae_b, M_b, out_b),
             ("matrix_only_tone_identity", mae_c, M_c, out_c)]
    name, mae, M, out = min(cands, key=lambda x: x[1].mean())
    return mae, M, name, means, out


def _rgb_to_hsv(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    mask = d > 1e-12
    # red max
    rm = mask & (mx == r)
    gm = mask & (mx == g) & ~rm
    bm = mask & (mx == b) & ~rm & ~gm
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = ((b[gm] - r[gm]) / d[gm]) + 2.0
    h[bm] = ((r[bm] - g[bm]) / d[bm]) + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-12, d / np.maximum(mx, 1e-12), 0.0)
    v = mx
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    i = i.astype(int) % 6
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def rung_M6(nlin, plin, preview, m5_out_disp, nbins=12):
    """M5 + per-hue residual: bin pixels by hue (12 bins), fit per-bin
    saturation + hue twist in HSV to push m5_out toward preview, report MAE."""
    pred = m5_out_disp.copy()
    hsv_pred = _rgb_to_hsv(pred)
    hsv_tgt = _rgb_to_hsv(preview)
    h = hsv_pred[..., 0]
    s_pred = hsv_pred[..., 1]
    s_tgt = hsv_tgt[..., 1]
    h_tgt = hsv_tgt[..., 0]

    bin_idx = np.clip((h * nbins).astype(int), 0, nbins - 1)
    s_new = s_pred.copy()
    h_new = h.copy()
    for k in range(nbins):
        m = (bin_idx == k) & (s_pred > 0.02)
        if m.sum() < 8:
            continue
        sp = s_pred[m]
        st = s_tgt[m]
        denom = float(np.dot(sp, sp))
        gain = float(np.dot(sp, st) / denom) if denom > 1e-9 else 1.0
        gain = float(np.clip(gain, 0.0, 4.0))
        cand_s = np.clip(sp * gain, 0.0, 1.0)
        dh = (h_tgt[m] - h[m] + 0.5) % 1.0 - 0.5
        med_dh = float(np.median(dh))
        cand_h = (h[m] + med_dh) % 1.0

        # Per-bin guard: keep the twist only if it lowers this bin's RGB MAE
        # (the residual is a strict-superset model, so it must never regress M5).
        idx = np.where(m)
        before = np.stack([h[m], sp, hsv_pred[..., 2][m]], axis=-1)
        after = np.stack([cand_h, cand_s, hsv_pred[..., 2][m]], axis=-1)
        rgb_before = np.clip(_hsv_to_rgb(before), 0.0, 1.0)
        rgb_after = np.clip(_hsv_to_rgb(after), 0.0, 1.0)
        tgt_rgb = preview[idx]
        mae_before = np.abs(rgb_before - tgt_rgb).mean()
        mae_after = np.abs(rgb_after - tgt_rgb).mean()
        if mae_after < mae_before:
            s_new[m] = cand_s
            h_new[m] = cand_h

    hsv_new = np.stack([h_new, s_new, hsv_pred[..., 2]], axis=-1)
    out = np.clip(_hsv_to_rgb(hsv_new), 0.0, 1.0)
    mae = disp_mae(out, preview)
    # Final safety: M6 must not regress M5 (the twist is optional capacity).
    mae_m5 = disp_mae(m5_out_disp, preview)
    if mae.mean() > mae_m5.mean():
        return mae_m5
    return mae


def rung_M7(neutral, preview, recipe):
    """colorCurves baseline: WB + clamp/hpminde -> srgb_encode -> per-channel
    24-bin monotone 16-knot LUT, no saturation. (verify_option_b.baseline_forward
    using the SHIPPED WB t/ti.) ANCHOR ~2.79 @5580."""
    t = float(recipe.get("wbTemp", 0.0) or 0.0)
    ti = float(recipe.get("wbTint", 0.0) or 0.0)
    lin = srgb_decode(neutral)
    lin = lin * (2.0 ** wb_exp(t, ti))[None, None, :]
    lin = hpminde_clip(np.where(lin > 1.0, 1.0, lin))  # simple clamp + gamut
    disp = srgb_encode(lin)
    pv = preview
    luts = []
    for ch in range(3):
        pts = fit_channel_curve(disp[..., ch].ravel(), pv[..., ch].ravel())
        luts.append(build_lut(pts))
    out = np.empty_like(disp)
    for ch in range(3):
        out[..., ch] = lut_sample(luts[ch], disp[..., ch])
    out = np.clip(out, 0.0, 1.0)
    return disp_mae(out, preview)


def rung_M8(nlin, plin, preview):
    """Global degree-3 polynomial RGB->RGB (LSQ), all monomials up to deg 3."""
    N = nlin.reshape(-1, 3)
    P = plin.reshape(-1, 3)
    r, g, b = N[:, 0], N[:, 1], N[:, 2]
    one = np.ones_like(r)
    # all monomials up to degree 3 (1 + 3 + 6 + 10 = 20 terms)
    feats = [
        one,
        r, g, b,
        r * r, g * g, b * b, r * g, r * b, g * b,
        r ** 3, g ** 3, b ** 3,
        r * r * g, r * r * b, g * g * r, g * g * b, b * b * r, b * b * g,
        r * g * b,
    ]
    X = np.stack(feats, axis=1)
    coef, *_ = np.linalg.lstsq(X, P, rcond=None)
    pred = X @ coef
    out_lin = pred.reshape(nlin.shape)
    out = lin_rolloff_hpminde_encode(out_lin)
    return disp_mae(out, preview), coef.shape


def matrix_offdiag_metrics(M):
    """How far the 3x3 is from identity and from a pure diagonal."""
    I = np.eye(3)
    diag = np.diag(np.diag(M))
    frob_from_identity = float(np.linalg.norm(M - I))
    frob_from_diag = float(np.linalg.norm(M - diag))
    max_offdiag = float(np.max(np.abs(M - diag)))
    return {
        "diag": [float(x) for x in np.diag(M)],
        "frob_from_identity": frob_from_identity,
        "frob_from_diag": frob_from_diag,
        "max_offdiag": max_offdiag,
    }


# ----------------------------------------------------------------------------
def load_grid(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def run_frame(dump_dir):
    neutral = load_grid(f"{dump_dir}/neutral_grid.png")
    preview = load_grid(f"{dump_dir}/preview_grid.png")
    with open(f"{dump_dir}/recipe.json") as f:
        recipe = json.load(f)

    nlin = srgb_decode(neutral)
    plin = srgb_decode(preview)

    rungs = {}

    mae, _ = rung_M0(nlin, plin, preview, neutral)
    rungs["M0"] = {"name": "identity (raw gap)", "rgb": mae.tolist(), "mean": float(mae.mean())}

    mae, m1info = rung_M1(nlin, plin, preview)
    rungs["M1"] = {"name": "WB free linear diagonal", "rgb": mae.tolist(),
                   "mean": float(mae.mean()), "G": m1info["G"]}

    mae2, mae3, m23info = rung_M2_M3(neutral, preview, recipe)
    rungs["M2"] = {"name": "WB(stops)+luma tone curve", "rgb": mae2.tolist(),
                   "mean": float(mae2.mean())}
    rungs["M3"] = {"name": "Option B (M2+scalar saturation)", "rgb": mae3.tolist(),
                   "mean": float(mae3.mean()), "params": m23info}

    mae4, M4 = rung_M4(nlin, plin, preview)
    rungs["M4"] = {"name": "3x3 linear matrix only", "rgb": mae4.tolist(),
                   "mean": float(mae4.mean()), "matrix": M4.tolist(),
                   "matrix_metrics": matrix_offdiag_metrics(M4)}

    mae5, M5, m5variant, m5info, m5_disp = rung_M5(nlin, plin, preview)
    rungs["M5"] = {"name": "3x3 matrix + luma tone curve", "rgb": mae5.tolist(),
                   "mean": float(mae5.mean()), "matrix": M5.tolist(),
                   "variant": m5variant, "matrix_metrics": matrix_offdiag_metrics(M5),
                   "variant_means": m5info}

    mae6 = rung_M6(nlin, plin, preview, m5_disp)
    rungs["M6"] = {"name": "M5 + per-hue residual (12 bins)", "rgb": mae6.tolist(),
                   "mean": float(mae6.mean())}

    mae7 = rung_M7(neutral, preview, recipe)
    rungs["M7"] = {"name": "colorCurves (3x 1-D display curves)", "rgb": mae7.tolist(),
                   "mean": float(mae7.mean())}

    mae8, _ = rung_M8(nlin, plin, preview)
    rungs["M8"] = {"name": "global degree-3 poly RGB->RGB", "rgb": mae8.tolist(),
                   "mean": float(mae8.mean())}

    return rungs, M5


def main():
    frames = {"5580": "/tmp/optb_dump", "5551": "/tmp/optb_dump_5551"}
    out = {"frames": {}, "fitted_matrix_5580": None, "anchors": {}, "deltas": {}}
    M5_5580 = None
    for fid, d in frames.items():
        rungs, M5 = run_frame(d)
        out["frames"][fid] = {"rungs": rungs}
        if fid == "5580":
            M5_5580 = M5

    # fitted matrix on 5580 = the M5 matrix (key physics rung)
    out["fitted_matrix_5580"] = M5_5580.tolist()
    out["fitted_matrix_5580_metrics"] = matrix_offdiag_metrics(M5_5580)

    m3_5580 = out["frames"]["5580"]["rungs"]["M3"]["mean"]
    m7_5580 = out["frames"]["5580"]["rungs"]["M7"]["mean"]
    ok_m3 = abs(m3_5580 - 7.05) <= 0.3
    ok_m7 = abs(m7_5580 - 2.79) <= 0.3
    out["anchors"] = {
        "M3_5580": m3_5580,
        "M7_5580": m7_5580,
        "M3_target": 7.05,
        "M7_target": 2.79,
        "M3_ok": bool(ok_m3),
        "M7_ok": bool(ok_m7),
        "ok": bool(ok_m3 and ok_m7),
    }

    r = out["frames"]["5580"]["rungs"]
    out["deltas"] = {
        "matrix_recovers_M3_minus_M5": float(r["M3"]["mean"] - r["M5"]["mean"]),
        "M5_vs_colorcurves_M7": float(r["M5"]["mean"] - r["M7"]["mean"]),
        "perhue_M5_minus_M6": float(r["M5"]["mean"] - r["M6"]["mean"]),
        "global_floor_M8": float(r["M8"]["mean"]),
    }

    # ---- report ----
    print("=" * 78)
    for fid in frames:
        print(f"FRAME {fid}")
        r = out["frames"][fid]["rungs"]
        for mid in ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]:
            rr = r[mid]
            extra = ""
            if mid == "M5":
                extra = f"  [{rr['variant']}]"
            print(f"  {mid} {rr['name']:<38} "
                  f"R={rr['rgb'][0]:6.3f} G={rr['rgb'][1]:6.3f} B={rr['rgb'][2]:6.3f} "
                  f"MEAN={rr['mean']:6.3f}{extra}")
        print()

    a = out["anchors"]
    print(f"ANCHORS  M3_5580={a['M3_5580']:.4f} (target 7.05, ok={a['M3_ok']})  "
          f"M7_5580={a['M7_5580']:.4f} (target 2.79, ok={a['M7_ok']})  ALL_OK={a['ok']}")
    if not a["ok"]:
        print("!!! ANCHOR FAILURE — HARNESS IS BROKEN !!!")
    d = out["deltas"]
    print(f"DELTAS  matrix_recovers(M3-M5)={d['matrix_recovers_M3_minus_M5']:+.4f}  "
          f"M5_vs_M7={d['M5_vs_colorcurves_M7']:+.4f}  "
          f"perhue(M5-M6)={d['perhue_M5_minus_M6']:+.4f}  M8_floor={d['global_floor_M8']:.4f}")
    mm = out["fitted_matrix_5580_metrics"]
    print(f"FITTED MATRIX 5580 diag={mm['diag']}  frob_from_identity={mm['frob_from_identity']:.4f}  "
          f"max_offdiag={mm['max_offdiag']:.4f}")

    # Nested-capacity monotonicity checks (each must hold; flag true fit bugs).
    # NB: M1 vs M3 is NOT a nested pair (free diagonal vs constrained-stops WB +
    # tone + sat) so it is reported as a WB-basis FINDING, not a monotonicity bug.
    out["wb_basis"] = {}
    out["monotonicity"] = {}
    for fid in frames:
        r = out["frames"][fid]["rungs"]
        checks = {
            "M4_le_M0": r["M4"]["mean"] <= r["M0"]["mean"] + 1e-6,
            "M5_le_M4": r["M5"]["mean"] <= r["M4"]["mean"] + 1e-6,
            "M6_le_M5": r["M6"]["mean"] <= r["M5"]["mean"] + 1e-6,
            "M8_le_M4": r["M8"]["mean"] <= r["M4"]["mean"] + 1e-6,
        }
        out["monotonicity"][fid] = {k: bool(v) for k, v in checks.items()}
        bad = [k for k, v in checks.items() if not v]
        if bad:
            print(f"MONOTONICITY BUG {fid}: nested-capacity inversion(s): {bad} -- PROBABLE FIT BUG")
        else:
            print(f"MONOTONICITY OK {fid}: M4<=M0, M5<=M4, M6<=M5, M8<=M4 all hold")
        # WB basis finding: free diagonal (M1) vs stop-parameterized WB+tone+sat (M3)
        out["wb_basis"][fid] = {
            "M1_free_diag": r["M1"]["mean"],
            "M3_stops_wb_tone_sat": r["M3"]["mean"],
            "free_diag_beats_stops_by": float(r["M3"]["mean"] - r["M1"]["mean"]),
        }
        # folding WB into the 3x3 (M4) vs separate stop-WB (M3 WB-only ~ M2 luma off)
        out["wb_basis"][fid]["matrix_subsumes_wb_beats_M3_by"] = float(
            r["M3"]["mean"] - r["M4"]["mean"])

    with open("/tmp/physics_ladder.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote /tmp/physics_ladder.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
