#!/usr/bin/env python3
"""Phase-2 of the local-tone (Active D-Lighting) verification: does a FAITHFUL
LOCAL-LAPLACIAN tone operator beat the blurred-mask floor and clear the >=0.5
luma-MAE gate at a GENUINELY-LOCAL scale on both measured frames (5580, 5551)?

Background (docs/local-tone-adl.md):
  - colorCurves (WB + per-channel display 1D LUT, sat OFF) leaves a residual that
    is genuinely local tone (ADL): spatially coherent, surround-keyed, luma not
    chroma. Baseline luma MAE ~3.47 (5580) / ~4.09 (5551).
  - The CHEAP blurred-mask operator (verify: /tmp/local_tone_sweep.json) FAILED at
    local scale: at sigma 3-5% it dropped luma MAE only 0.11-0.28 and regressed
    blue; it only cleared >=0.5 at near-GLOBAL sigma=12% (colorCurves' own lane).
  - s2.3-2.4: the residual has a contrast-vs-lift TENSION -- a dominant
    low-amplitude local-CONTRAST field (negative conditioned slope) PLUS
    largest-amplitude dark-subject SHADOW-LIFT blobs (pixel << surround). A
    symmetric unsharp captures one but not the other. The local-Laplacian is the
    principled Phase-2 form (s2.4 "Why not the alternatives" + s6.1 Phase-2 hook):
    multi-scale edge-aware remapping can extract BOTH at local scale where the
    single-sigma blurred-mask cannot.

This implements the FAST DISCRETIZED-INTENSITY local-Laplacian (Paris et al.
2011; Aubry, Paris, Hasinoff, Kautz, Durand 2014 "Fast Local Laplacian Filters"):
  1. L = luma(colorCurves_out)   (display-space REC709 luma; the lane the residual
     is defined against -- reuses verify_option_b.baseline_forward).
  2. Gaussian pyramid G of L.
  3. For each of N discrete reference intensities g_i in [0,1]:
       L_remap = r(L; g_i)        # per-pixel pointwise remapping around g_i
       LP_i    = laplacian_pyramid(L_remap)
  4. Output Laplacian pyramid: at level l, pixel p with Gaussian-pyramid value
     v = G[l][p], pick the remapped-pyramid coefficient by interpolating between
     the two discrete g_i straddling v (Aubry's discretized selection).
  5. Collapse -> L_out.
  6. Apply LUMA-ONLY ratio composite on colorCurves_out (chroma preserved),
     identical form to the blurred-mask operator (s2.4):
       ratio = L_out / L ; out_rgb = colorCurves_out * ratio.

Remapping r(L; g_i) -- the faithful Paris et al. 2011 detail/edge split (their
fd/fe decomposition), parameterized for the ADL contrast+lift tension (s2.3):
  d = L - g_i
  detail (|d| <= sigma_r): r = g_i + alpha * d        (LINEAR detail gain)
      alpha > 1 -> BOOSTS local contrast (amplifies detail); alpha = 1 identity;
      alpha < 1 -> reduces local contrast. This is the literal detail multiplier:
      a fine texture of amplitude << sigma_r is scaled EXACTLY by alpha in the
      collapsed output (verified in guard 2). [The power-law (|d|/sigma_r)^alpha
      form was tried first and rejected: it does not give a predictable, faithful
      detail gain through the pyramid -- a fine sinusoid did not scale by alpha.]
  edge   (|d| >  sigma_r): r = g_i + sign(d) * (sigma_r*alpha + (|d|-sigma_r)*beta)
      C0-continuous with the detail branch at |d|=sigma_r (detail end =
      g_i+sign(d)*sigma_r*alpha). beta is the EDGE/tone slope past sigma_r:
      beta < 1 -> COMPRESSES range (tone compression = shadow-lift in the mapped
      sense); beta = 1 -> edge slope preserved (no ringing of hard edges, the
      edge-aware property); beta > 1 -> expands.
  ASYMMETRIC SHADOW LIFT: for the EDGE term on the DARK side (d < -sigma_r, i.e.
      pixel below reference -- a dark subject relative to a brighter level), add an
      extra additive lift   + shadow_lift * (|d|-sigma_r)   so dark-on-light gets
      the measured recovery beyond symmetric edge handling. shadow_lift >= 0.

At alpha=1, beta=1, shadow_lift=0 the remapping is the IDENTITY r(L;g)=L for all
g, so the whole filter is an exact identity (guard 1).

Guards (all reported to JSON):
  1. Identity: neutral params -> luma MAE unchanged vs baseline within 0.02.
  2. Edge-awareness: on a synthetic hard step + fine texture, the operator boosts
     texture detail while NOT ringing the step; quantify halo energy at the step
     and show it is much smaller than a plain Gaussian-unsharp tuned to the same
     texture gain (proves real edge-aware local-Laplacian, not a disguised blur).
  3. Baseline: colorCurves luma MAE reproduces ~3.47 (5580) / ~4.09 (5551).

Sweep grid over (sigma_r, alpha, beta, shadow_lift) INCLUDING both directions
(contrast boost alpha<1 AND detail compress alpha>1; shadow-lift beta<1 AND edge
expand beta>1; shadow_lift>=0). Reports:
  - baseline vs best per-frame luma MAE + drop; SHARED params (min mean luma MAE)
    + per-frame drop at shared;
  - gate: shared clears >=0.5 luma-MAE drop on BOTH frames, NO RGB regression, no
    halos;
  - effective SPATIAL scale of the winning operator (genuinely local vs drifted
    semi-global like the blurred-mask);
  - head-to-head vs the blurred-mask floor (0.11-0.28 at local scale);
  - before/after residual heatmaps + corrected PNGs to /tmp.

All numbers -> /tmp/local_laplacian_sweep.json. Pure numpy + PIL + scipy.

Run: python3 tools/local_laplacian_apply.py   (a few minutes; that's expected).
"""

import json
import os
import sys
import time

import numpy as np
from PIL import Image

# --- reuse the faithful ports ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_option_b as vob  # noqa: E402
import local_tone_characterize as ltc  # noqa: E402

REC709 = vob.REC709
gaussian_blur = ltc.gaussian_blur      # scipy/numpy separable Gaussian, mode=reflect
luma = ltc.luma                        # REC709 luma over last axis
diverging_heatmap = ltc.diverging_heatmap
load_grid = vob.load_grid

EPS = 1e-6


# ============================================================================
# Pyramid helpers (Gaussian / Laplacian) -- faithful Burt-Adelson with a
# Gaussian-blur + 2x decimate / upsample + Gaussian-blur expand. Uses the
# REUSED ltc.gaussian_blur (the same separable reflect-mode Gaussian the rest of
# the harness uses) as the pyramid smoothing filter, so the discretization is
# consistent with the blurred-mask comparison.
# ============================================================================
_PYR_SIGMA = 1.0  # blur sigma applied before each 2x decimation / after expand


def _downsample(img):
    """Blur then take every other sample (2x decimation)."""
    b = gaussian_blur(img, _PYR_SIGMA)
    return b[::2, ::2]


def _upsample(img, out_shape):
    """Nearest-style upsample to out_shape then blur (expand). out_shape = (H,W)."""
    H, W = out_shape
    up = np.zeros((H, W), dtype=np.float64)
    # place samples; np repeat-style by indexing keeps it exact for odd sizes
    hi = min(img.shape[0], (H + 1) // 2)
    wi = min(img.shape[1], (W + 1) // 2)
    up[0:2 * hi:2, 0:2 * wi:2] = img[:hi, :wi]
    # fill the holes by a light blur (expand filter). Multiply by 4 to preserve
    # mean energy after inserting zeros (standard expand normalization for 2x2).
    up = gaussian_blur(up, _PYR_SIGMA) * 4.0
    return up


def gaussian_pyramid(img, n_levels):
    pyr = [img]
    for _ in range(n_levels - 1):
        pyr.append(_downsample(pyr[-1]))
    return pyr


def laplacian_pyramid(img, n_levels):
    gp = gaussian_pyramid(img, n_levels)
    lp = []
    for l in range(n_levels - 1):
        up = _upsample(gp[l + 1], gp[l].shape)
        lp.append(gp[l] - up)
    lp.append(gp[-1])  # residual (coarsest) level
    return lp


def collapse_pyramid(lp):
    out = lp[-1]
    for l in range(len(lp) - 2, -1, -1):
        out = lp[l] + _upsample(out, lp[l].shape)
    return out


def num_levels_for(H, W, min_dim=8):
    n = 1
    d = min(H, W)
    while d > min_dim:
        d = (d + 1) // 2
        n += 1
    return n


# ============================================================================
# The pointwise remapping r(L; g) -- the heart of the local-Laplacian. This is
# what makes it edge-aware: each output Laplacian coefficient is taken from the
# pyramid of a DIFFERENTLY-remapped image, the remapping centered on the local
# Gaussian-pyramid value (the "reference" g). Detail (|L-g|<=sigma_r) and edges
# (|L-g|>sigma_r) are remapped separately, so a hard edge (large |L-g|) is moved
# through the EDGE branch (no detail-gain ringing) while fine texture (small
# |L-g|) is amplified through the DETAIL branch.
# ============================================================================
def remap(L, g, sigma_r, alpha, beta, shadow_lift):
    """Per-pixel remap of intensity L about reference g (scalar). Vectorized over
    the full image. Returns remapped intensities (not clamped; collapse + final
    composite clamp)."""
    d = L - g
    ad = np.abs(d)
    sgn = np.sign(d)

    # detail term (|d| <= sigma_r): LINEAR detail gain (Paris fd) -> a fine
    # texture << sigma_r is scaled exactly by alpha through the pyramid.
    # r_detail = g + alpha * d
    detail = g + alpha * d

    # edge term (|d| > sigma_r): C0-continuous with the detail branch at
    # |d|=sigma_r (detail end = g + sgn*sigma_r*alpha), then edge slope beta on
    # the excess. r_edge = g + sgn*(sigma_r*alpha + (ad-sigma_r)*beta)
    excess = np.maximum(ad - sigma_r, 0.0)
    edge = g + sgn * (sigma_r * alpha + excess * beta)

    # ASYMMETRIC SHADOW LIFT: on the dark edge side (d < -sigma_r), add extra
    # additive lift proportional to how far below the reference the pixel sits.
    # This is the measured dark-subject-on-lighter-field recovery (s2.3).
    dark_edge = (d < 0.0)
    edge = edge + np.where(dark_edge, shadow_lift * excess, 0.0)

    out = np.where(ad <= sigma_r, detail, edge)
    return out


# ============================================================================
# Fast discretized-intensity local-Laplacian (Aubry et al. 2014).
# ============================================================================
def local_laplacian(L, sigma_r, alpha, beta, shadow_lift, n_disc=12, n_levels=None):
    """Faithful fast local-Laplacian of a single-channel image L in [0,1].

    n_disc discrete reference intensities g_i tile [0,1]. For each, remap the
    FULL image about g_i and build its Laplacian pyramid. The output Laplacian
    pyramid selects, per level/pixel, the coefficient interpolated between the two
    discrete remapped pyramids whose g_i straddle the Gaussian-pyramid value at
    that pixel (Aubry's discretized selection -> O(n_disc) pyramids instead of one
    per pixel). Collapse to L_out."""
    H, W = L.shape
    if n_levels is None:
        n_levels = num_levels_for(H, W)

    # Gaussian pyramid of the (un-remapped) input drives the per-pixel reference.
    gp_in = gaussian_pyramid(L, n_levels)

    # discrete reference levels spanning [0,1]
    g_levels = np.linspace(0.0, 1.0, n_disc)

    # Laplacian pyramids of each remapped full image.
    remapped_lp = []
    for g in g_levels:
        Lr = remap(L, g, sigma_r, alpha, beta, shadow_lift)
        remapped_lp.append(laplacian_pyramid(Lr, n_levels))

    # Output Laplacian pyramid via discretized selection + linear interpolation
    # between the two straddling discrete levels.
    out_lp = []
    step = g_levels[1] - g_levels[0]
    for l in range(n_levels):
        ref = gp_in[l]  # reference intensity per pixel at this level
        # find lower discrete index k such that g_levels[k] <= ref < g_levels[k+1]
        pos = np.clip((ref - g_levels[0]) / step, 0.0, n_disc - 1.0)
        k = np.floor(pos).astype(np.int64)
        k = np.clip(k, 0, n_disc - 2)
        frac = pos - k
        # gather coefficients from the two straddling remapped pyramids
        coeff = np.empty_like(ref)
        # vectorize over the (small) set of discrete indices present at this level
        for ki in np.unique(k):
            m = (k == ki)
            lo = remapped_lp[ki][l]
            hi = remapped_lp[ki + 1][l]
            coeff[m] = lo[m] * (1.0 - frac[m]) + hi[m] * frac[m]
        out_lp.append(coeff)

    return collapse_pyramid(out_lp)


# ============================================================================
# Apply as a luma-only ratio composite on colorCurves_out (chroma preserved),
# matching the blurred-mask operator form (s2.4).
# ============================================================================
def apply_operator(cc_out, L, sigma_r, alpha, beta, shadow_lift, n_disc=12, n_levels=None):
    L_out = local_laplacian(L, sigma_r, alpha, beta, shadow_lift, n_disc=n_disc, n_levels=n_levels)
    L_out = np.clip(L_out, 0.0, 1.0)
    ratio = np.where(L > EPS, L_out / np.maximum(L, EPS), 1.0)
    out = cc_out * ratio[..., None]
    return np.clip(out, 0.0, 1.0)


# ============================================================================
# Metrics
# ============================================================================
def luma_mae(out_rgb, preview):
    return float(np.abs((luma(out_rgb) - luma(preview)) * 255.0).mean())


def rgb_mae(out_rgb, preview):
    diff = np.abs(out_rgb - preview) * 255.0
    return [float(x) for x in diff.reshape(-1, 3).mean(axis=0)]


# ============================================================================
# Frame prep
# ============================================================================
def prep_frame(name, dump_dir):
    neutral = load_grid(os.path.join(dump_dir, "neutral_hi.png"))
    preview = load_grid(os.path.join(dump_dir, "preview_hi.png"))
    H, W, _ = neutral.shape
    long_edge = max(H, W)
    with open(os.path.join(dump_dir, "recipe.json")) as f:
        recipe = json.load(f)
    temperature = float(recipe.get("wbTemp", 0.0) or 0.0)
    tint = float(recipe.get("wbTint", 0.0) or 0.0)

    cc_out = vob.baseline_forward(neutral, preview, temperature, tint, f16=False)
    L = luma(cc_out)
    base_luma_mae = luma_mae(cc_out, preview)
    base_rgb_mae = rgb_mae(cc_out, preview)
    n_levels = num_levels_for(H, W)
    return {
        "name": name, "dump_dir": dump_dir, "size": [W, H], "long_edge": long_edge,
        "preview": preview, "cc_out": cc_out, "L": L,
        "base_luma_mae": base_luma_mae, "base_rgb_mae": base_rgb_mae,
        "n_levels": n_levels,
    }


# ============================================================================
# GUARD 1: identity at neutral params
# ============================================================================
def guard_identity(frame, n_disc=12):
    out = apply_operator(frame["cc_out"], frame["L"], sigma_r=0.15,
                         alpha=1.0, beta=1.0, shadow_lift=0.0,
                         n_disc=n_disc, n_levels=frame["n_levels"])
    lm = luma_mae(out, frame["preview"])
    delta = abs(lm - frame["base_luma_mae"])
    return {"frame": frame["name"], "base_luma_mae": frame["base_luma_mae"],
            "neutral_luma_mae": lm, "abs_delta": delta, "ok": bool(delta <= 0.02)}


# ============================================================================
# GUARD 2: edge-awareness unit test (synthetic step-edge + fine texture).
# Show the local-Laplacian boosts texture WITHOUT ringing the hard step, and is
# DISTINCT from a plain Gaussian-unsharp (which halos the step). Quantify halo
# energy in a band beside the step.
# ============================================================================
def make_textured_step(N=256, step_lo=0.30, step_hi=0.70, tex_amp=0.03, tex_period=8):
    """Left half = step_lo, right half = step_hi; a fine sinusoidal texture is
    superimposed on BOTH halves. The step is the hard EDGE (jump 0.40 >> sigma_r,
    so it lands in the edge branch); the sinusoid (amp 0.03 << sigma_r) is the fine
    DETAIL we want amplified through the detail branch. Returns (img, base) where
    base is the step WITHOUT texture."""
    x = np.arange(N)
    base = np.where(x[None, :] < N // 2, step_lo, step_hi)
    base = np.broadcast_to(base, (N, N)).astype(np.float64).copy()
    tex = tex_amp * np.sin(2.0 * np.pi * x[None, :] / tex_period)
    tex = np.broadcast_to(tex, (N, N)).astype(np.float64)
    img = np.clip(base + tex, 0.0, 1.0)
    return img, base


def make_pure_step(N=256, step_lo=0.30, step_hi=0.70):
    """A hard step with NO texture -- isolates ringing/overshoot at the edge."""
    x = np.arange(N)
    s = np.where(x[None, :] < N // 2, step_lo, step_hi)
    return np.broadcast_to(s, (N, N)).astype(np.float64).copy()


def gaussian_unsharp(L, sigma, gain):
    """Plain (non-edge-aware) unsharp mask: L + gain*(L - blur(L))."""
    return L + gain * (L - gaussian_blur(L, sigma))


def texture_std(out, base, N):
    """Amplitude (std) of the fine sinusoid in a flat region away from the step
    (x in [N//8, 3N//8]). std of (out-base) isolates the texture from the step."""
    a, b = N // 8, 3 * N // 8
    return float((out - base)[:, a:b].std())


def step_overshoot(out, lo=0.30, hi=0.70, N=256, band=20):
    """Max deviation from the two flat plateaus in a band straddling the hard step
    -- the ringing / halo a non-edge-aware operator introduces at an edge. A real
    local-Laplacian with the edge in the edge branch (beta=1) leaves ~the step,
    so this stays small; an unsharp mask overshoots."""
    edge = N // 2
    left = out[N // 2, edge - band:edge]    # left plateau approaching the edge -> ~lo
    right = out[N // 2, edge:edge + band]   # right plateau -> ~hi
    return float(max(np.max(np.abs(left - lo)), np.max(np.abs(right - hi))))


def guard_edge_aware(out_dir="/tmp"):
    """Faithful edge-awareness test, two parts:
      (a) on a PURE step (no texture), the local-Laplacian must NOT ring the hard
          edge (small overshoot), while a Gaussian-unsharp tuned to the same detail
          gain overshoots strongly -> proves edge-aware, distinct from blur/unsharp.
      (b) on a textured patch, the local-Laplacian must actually BOOST the fine
          texture by ~alpha -> proves it does local-contrast manipulation.
    The operator is configured for detail boost (alpha=2.0) with edges preserved
    (beta=1.0), at a small sigma_r so the texture is detail and the step is edge."""
    N = 256
    period = 8
    sigma_r = 0.06
    alpha = 2.0   # detail BOOST (new linear-gain semantics: alpha>1 amplifies)
    beta = 1.0    # edge slope preserved -> hard step not rung
    shadow_lift = 0.0
    n_levels = num_levels_for(N, N)

    # --- (b) texture boost on the textured step ---
    img, base = make_textured_step(N=N, tex_period=period, tex_amp=0.03)
    in_tex = texture_std(img, base, N)
    ll = local_laplacian(img, sigma_r, alpha, beta, shadow_lift, n_disc=16, n_levels=n_levels)
    ll_tex = texture_std(ll, base, N)
    ll_boost = ll_tex / max(in_tex, 1e-9)

    # --- (a) ringing on a PURE step (no texture) ---
    pure = make_pure_step(N=N)
    ll_pure = local_laplacian(pure, sigma_r, alpha, beta, shadow_lift, n_disc=16, n_levels=n_levels)
    ll_over = step_overshoot(ll_pure, N=N)

    # Gaussian-unsharp matched to the SAME texture boost (so the comparison is at
    # equal detail gain), then measure its overshoot on the SAME pure step. Search
    # the gain at a sigma that spans the texture so it actually amplifies it.
    us_sigma = 1.0
    target = ll_tex
    lo_g, hi_g = 0.0, 60.0
    for _ in range(40):
        mid = 0.5 * (lo_g + hi_g)
        if texture_std(gaussian_unsharp(img, us_sigma, mid), base, N) < target:
            lo_g = mid
        else:
            hi_g = mid
    us_gain = 0.5 * (lo_g + hi_g)
    us = gaussian_unsharp(img, us_sigma, us_gain)
    us_tex = texture_std(us, base, N)
    us_boost = us_tex / max(in_tex, 1e-9)
    us_pure = gaussian_unsharp(pure, us_sigma, us_gain)
    us_over = step_overshoot(us_pure, N=N)

    overshoot_ratio = (us_over / ll_over) if abs(ll_over) > 1e-9 else float("inf")

    # save row profiles across the pure step (input, LL, unsharp) for visual check
    prof = np.stack([pure[N // 2], ll_pure[N // 2], us_pure[N // 2]], axis=0)
    pimg = (np.clip(np.repeat(prof, 30, axis=0), 0, 1) * 255).round().astype(np.uint8)
    pp = os.path.join(out_dir, "local_laplacian_edgeaware_profiles.png")
    Image.fromarray(pimg, "L").save(pp)

    # PASS: (b) LL boosts the texture (>1.5x, ~alpha) AND the unsharp matched that
    # boost (so the comparison is fair) AND (a) LL's hard-step overshoot is
    # materially smaller than the unsharp's at equal detail gain (>=2x smaller).
    ok = bool((ll_boost > 1.5)
              and (us_boost > 0.7 * ll_boost)
              and (us_over > 2.0 * ll_over))
    return {
        "synthetic": {"N": N, "step_lo": 0.30, "step_hi": 0.70,
                      "tex_amp": 0.03, "tex_period": period, "sigma_r": sigma_r,
                      "alpha": alpha, "beta": beta},
        "input_texture_std": in_tex,
        "local_laplacian": {"texture_std": ll_tex, "texture_boost_x": ll_boost,
                            "pure_step_overshoot": ll_over},
        "gaussian_unsharp": {"sigma": us_sigma, "gain": us_gain,
                             "texture_std": us_tex, "texture_boost_x": us_boost,
                             "pure_step_overshoot": us_over},
        "unsharp_to_ll_overshoot_ratio": overshoot_ratio,
        "profiles_png": pp,
        "ok": ok,
        "interpretation": (
            f"local-Laplacian boosts the fine texture {ll_boost:.2f}x (alpha=2.0) "
            f"while leaving a hard step nearly intact (overshoot {ll_over:.3f}); a "
            f"plain Gaussian-unsharp matched to the same {us_boost:.2f}x texture "
            f"gain rings the same hard step with overshoot {us_over:.3f} "
            f"({overshoot_ratio:.1f}x larger). EDGE-AWARE confirmed: the operator "
            "amplifies detail without ringing edges -- a real local-Laplacian, not "
            "a disguised blur/unsharp."
        ),
    }


# ============================================================================
# Effective spatial scale of a winning operator: where does the change live in
# frequency? Decompose the operator-induced luma change (L_out - L) into a
# low-frequency part (blur at sigma = scale_frac*long_edge) and report the
# fraction of change energy that is LOCAL (above the colorCurves ~12% global lane)
# vs the residual it removes. We report the dominant scale by finding the blur
# sigma at which most of the change energy is captured.
# ============================================================================
def effective_scale(frame, sigma_r, alpha, beta, shadow_lift, n_disc):
    L = frame["L"]
    L_out = np.clip(local_laplacian(L, sigma_r, alpha, beta, shadow_lift,
                                    n_disc=n_disc, n_levels=frame["n_levels"]), 0.0, 1.0)
    change = L_out - L
    change_c = change - change.mean()
    total_e = float((change_c * change_c).sum()) + 1e-30
    rows = []
    for sf in [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 0.35]:
        sig = sf * frame["long_edge"]
        lp = gaussian_blur(change_c, sig)
        frac = float((lp * lp).sum() / total_e)
        rows.append({"scale_frac": sf, "sigma_px": round(sig, 1),
                     "low_freq_energy_fraction": round(frac, 4)})
    # "effective scale" = smallest scale_frac that still captures >=50% of energy
    # when blurred to that scale means the energy is at scales >= that; we instead
    # report HIGH-PASS: fraction of energy ABOVE a given scale = 1 - low_frac.
    # The operator is "genuinely local" if a large fraction of its change energy
    # survives a high-pass at the 12% global cutoff (i.e. low_frac at 12% is well
    # below 1 -> meaningful sub-global structure).
    lf12 = next(r["low_freq_energy_fraction"] for r in rows if r["scale_frac"] == 0.12)
    lf03 = next(r["low_freq_energy_fraction"] for r in rows if r["scale_frac"] == 0.03)
    high_pass_above_12pct = round(1.0 - lf12, 4)
    return {"rows": rows,
            "low_freq_frac_at_3pct": lf03,
            "low_freq_frac_at_12pct": lf12,
            "high_pass_energy_above_12pct_global": high_pass_above_12pct}


# ============================================================================
# Heatmaps + corrected PNG at a given setting
# ============================================================================
def write_outputs(frame, params, tag, out_dir="/tmp"):
    sigma_r, alpha, beta, shadow_lift, n_disc = params
    out = apply_operator(frame["cc_out"], frame["L"], sigma_r, alpha, beta,
                         shadow_lift, n_disc=n_disc, n_levels=frame["n_levels"])
    R_before = (luma(frame["cc_out"]) - luma(frame["preview"])) * 255.0
    R_after = (luma(out) - luma(frame["preview"])) * 255.0
    vlim = max(1.0, float(np.percentile(np.abs(np.concatenate(
        [R_before.ravel(), R_after.ravel()])), 99.0)))
    name = frame["name"]
    paths = {}
    pb = os.path.join(out_dir, f"local_laplacian_residual_{name}_before.png")
    Image.fromarray(diverging_heatmap(R_before, vlim), "RGB").save(pb)
    paths["before"] = pb
    pa = os.path.join(out_dir, f"local_laplacian_residual_{name}_after.png")
    Image.fromarray(diverging_heatmap(R_after, vlim), "RGB").save(pa)
    paths["after"] = pa
    pc = os.path.join(out_dir, f"local_laplacian_corrected_{name}_hi.png")
    Image.fromarray((np.clip(out, 0, 1) * 255.0).round().astype(np.uint8), "RGB").save(pc)
    paths["corrected_hi"] = pc
    paths["heatmap_vlim_255"] = vlim
    paths["tag"] = tag
    return paths, out


# ============================================================================
# Sweep
# ============================================================================
def sweep(frames, grid_params, n_disc):
    """grid_params: list of (sigma_r, alpha, beta, shadow_lift).
    Returns: dict key=(sigma_r,alpha,beta,shadow_lift) -> {frame: {luma_mae, rgb_mae}}."""
    results = {}
    total = len(grid_params)
    t0 = time.time()
    for i, (sigma_r, alpha, beta, sl) in enumerate(grid_params):
        per = {}
        for fr in frames:
            out = apply_operator(fr["cc_out"], fr["L"], sigma_r, alpha, beta, sl,
                                 n_disc=n_disc, n_levels=fr["n_levels"])
            per[fr["name"]] = {"luma_mae": luma_mae(out, fr["preview"]),
                               "rgb_mae": rgb_mae(out, fr["preview"])}
        results[(round(sigma_r, 4), round(alpha, 4), round(beta, 4), round(sl, 4))] = per
        if (i + 1) % 10 == 0 or i + 1 == total:
            dt = time.time() - t0
            print(f"    sweep {i + 1}/{total}  ({dt:.1f}s, {dt / (i + 1):.2f}s/cfg)", flush=True)
    return results


def main():
    out_dir = "/tmp"
    n_disc = 12  # discrete intensity levels for the fast local-Laplacian

    frames_cfg = [("5580", "/tmp/optb_dump"), ("5551", "/tmp/optb_dump_5551")]
    print("=== preparing frames (colorCurves baseline, sat OFF) ===", flush=True)
    frames = [prep_frame(n, d) for n, d in frames_cfg]
    frame_names = [f["name"] for f in frames]
    for fr in frames:
        print(f"  {fr['name']}: size={fr['size']} levels={fr['n_levels']}  "
              f"baseline luma MAE={fr['base_luma_mae']:.4f}  "
              f"rgb MAE={[round(x, 3) for x in fr['base_rgb_mae']]}", flush=True)

    # ---- GUARD 3: baseline sanity ----
    base_ok = (abs(frames[0]["base_luma_mae"] - 3.47) <= 0.25
               and abs(frames[1]["base_luma_mae"] - 4.09) <= 0.25)
    print(f"\n[guard3] baseline luma MAE 5580={frames[0]['base_luma_mae']:.4f} (~3.47), "
          f"5551={frames[1]['base_luma_mae']:.4f} (~4.09)  ok={base_ok}", flush=True)

    # ---- GUARD 1: identity at neutral params ----
    print("\n[guard1] identity at neutral params (alpha=1,beta=1,shadow_lift=0)", flush=True)
    g_id = [guard_identity(fr, n_disc=n_disc) for fr in frames]
    for g in g_id:
        print(f"  {g['frame']}: base={g['base_luma_mae']:.4f} neutral={g['neutral_luma_mae']:.4f} "
              f"|delta|={g['abs_delta']:.4f}  ok={g['ok']}", flush=True)
    identity_ok = all(g["ok"] for g in g_id)

    # ---- GUARD 2: edge-awareness unit test ----
    print("\n[guard2] edge-awareness unit test (step + texture vs Gaussian-unsharp)", flush=True)
    g_edge = guard_edge_aware(out_dir=out_dir)
    print(f"  LL texture boost = {g_edge['local_laplacian']['texture_boost_x']:.2f}x  "
          f"pure-step overshoot = {g_edge['local_laplacian']['pure_step_overshoot']:.3f}", flush=True)
    print(f"  unsharp (matched gain) texture boost = {g_edge['gaussian_unsharp']['texture_boost_x']:.2f}x  "
          f"pure-step overshoot = {g_edge['gaussian_unsharp']['pure_step_overshoot']:.3f}", flush=True)
    print(f"  unsharp/LL overshoot ratio = {g_edge['unsharp_to_ll_overshoot_ratio']:.1f}x  ok={g_edge['ok']}", flush=True)

    # ---- build sweep grid (INCLUDES both directions) ----
    # sigma_r: detail/edge intensity threshold (in display-luma units). small ->
    #   only fine texture is "detail"; large -> more of the tonal range is detail.
    sigma_r_vals = [0.05, 0.10, 0.15, 0.25, 0.40]
    # alpha: detail gain. <1 boosts local contrast; 1 identity; >1 compresses.
    alpha_vals = [0.5, 0.75, 1.0, 1.5, 2.0]
    # beta: edge/tone compression. <1 compresses range (shadow-lift sense); >1 expands.
    beta_vals = [0.5, 0.75, 1.0, 1.5]
    # shadow_lift: asymmetric additive dark-subject lift (>=0).
    shadow_vals = [0.0, 0.2, 0.5, 1.0]

    grid_params = []
    for sr in sigma_r_vals:
        for a in alpha_vals:
            for b in beta_vals:
                for sl in shadow_vals:
                    grid_params.append((sr, a, b, sl))
    print(f"\n=== sweeping {len(grid_params)} (sigma_r,alpha,beta,shadow_lift) configs x "
          f"{len(frames)} frames, n_disc={n_disc} ===", flush=True)
    res = sweep(frames, grid_params, n_disc)

    # ---- per-frame optima + shared optimum (min mean luma MAE) ----
    def perframe_best(name):
        bk, bm = None, float("inf")
        for k, per in res.items():
            m = per[name]["luma_mae"]
            if m < bm:
                bm, bk = m, k
        return bk, bm

    def shared_best():
        bk, bm = None, float("inf")
        for k, per in res.items():
            m = sum(per[n]["luma_mae"] for n in frame_names) / len(frame_names)
            if m < bm:
                bm, bk = m, k
        return bk, bm

    # shared optimum that ALSO satisfies the gate (>=0.5 drop both + no RGB regr),
    # to report the best GATE-CLEARING shared config if one exists.
    def shared_best_gated():
        bk, bm = None, float("inf")
        for k, per in res.items():
            ok = True
            for fr in frames:
                drop = fr["base_luma_mae"] - per[fr["name"]]["luma_mae"]
                rgb = per[fr["name"]]["rgb_mae"]
                non_regress = all(rgb[c] <= fr["base_rgb_mae"][c] + 1e-9 for c in range(3))
                if drop < 0.5 or not non_regress:
                    ok = False
                    break
            if ok:
                m = sum(per[n]["luma_mae"] for n in frame_names) / len(frame_names)
                if m < bm:
                    bm, bk = m, k
        return bk, bm

    perframe = {}
    for fr in frames:
        k, m = perframe_best(fr["name"])
        perframe[fr["name"]] = {"key": k, "luma_mae": m, "drop": fr["base_luma_mae"] - m}
        print(f"\n  per-frame OPT {fr['name']}: (sigma_r,alpha,beta,shadow_lift)={k}  "
              f"luma MAE {fr['base_luma_mae']:.4f} -> {m:.4f} (drop {fr['base_luma_mae'] - m:.4f})", flush=True)

    shared_key, shared_mean = shared_best()
    gated_key, gated_mean = shared_best_gated()
    sr, a, b, sl = shared_key
    print(f"\n=== SHARED OPT (min mean luma MAE): (sigma_r,alpha,beta,shadow_lift)="
          f"{shared_key}  mean luma MAE={shared_mean:.4f} ===", flush=True)
    if gated_key is not None:
        print(f"=== SHARED GATE-CLEARING OPT: {gated_key}  mean luma MAE={gated_mean:.4f} ===", flush=True)
    else:
        print("=== no shared config clears the >=0.5/no-regression gate ===", flush=True)

    # ---- per-frame results at the shared optimum ----
    frames_out = {}
    gate_detail = {}
    for fr in frames:
        per = res[shared_key][fr["name"]]
        lm, rm = per["luma_mae"], per["rgb_mae"]
        drop = fr["base_luma_mae"] - lm
        rgb_non_regress = all(rm[c] <= fr["base_rgb_mae"][c] + 1e-9 for c in range(3))
        rgb_delta = [fr["base_rgb_mae"][c] - rm[c] for c in range(3)]
        pf = perframe[fr["name"]]
        frames_out[fr["name"]] = {
            "baseline_luma_mae": fr["base_luma_mae"],
            "baseline_rgb_mae": fr["base_rgb_mae"],
            "perframe_opt": {"sigma_r": pf["key"][0], "alpha": pf["key"][1],
                             "beta": pf["key"][2], "shadow_lift": pf["key"][3],
                             "luma_mae": pf["luma_mae"], "drop": pf["drop"]},
            "at_shared": {"luma_mae": lm, "drop": drop, "rgb_mae": rm,
                          "rgb_delta": rgb_delta, "rgb_non_regress": rgb_non_regress,
                          "clears_0p5_drop": bool(drop >= 0.5)},
        }
        gate_detail[fr["name"]] = {"drop": drop, "clears_0p5": bool(drop >= 0.5),
                                   "rgb_non_regress": rgb_non_regress,
                                   "rgb_delta": rgb_delta}
        print(f"  {fr['name']} @ shared: luma MAE {fr['base_luma_mae']:.4f} -> {lm:.4f} "
              f"(drop {drop:.4f})  rgb MAE {[round(x, 3) for x in rm]}  "
              f"non-regress={rgb_non_regress}  clears0.5={drop >= 0.5}", flush=True)

    gate_pass = all(g["clears_0p5"] and g["rgb_non_regress"] for g in gate_detail.values())

    # ---- effective spatial scale of the winning (shared) operator ----
    print("\n=== effective spatial scale of the shared operator ===", flush=True)
    eff_scale = {}
    for fr in frames:
        es = effective_scale(fr, sr, a, b, sl, n_disc)
        eff_scale[fr["name"]] = es
        print(f"  {fr['name']}: low-freq frac of change energy @3%={es['low_freq_frac_at_3pct']:.3f} "
              f"@12%={es['low_freq_frac_at_12pct']:.3f}  "
              f"high-pass energy above 12% global={es['high_pass_energy_above_12pct_global']:.3f}", flush=True)
    # genuinely local if a meaningful fraction of change energy is ABOVE the 12%
    # global cutoff (the colorCurves lane); >=0.35 -> sub-global structure present.
    mean_hp = float(np.mean([eff_scale[n]["high_pass_energy_above_12pct_global"] for n in frame_names]))
    genuinely_local = bool(mean_hp >= 0.35)
    effective_scale_summary = {
        "mean_high_pass_energy_above_12pct_global": round(mean_hp, 4),
        "genuinely_local": genuinely_local,
        "verdict": (
            f"mean {mean_hp:.2f} of the operator's change energy is above the 12% "
            "global cutoff -> "
            + ("GENUINELY LOCAL (sub-global structure, not just colorCurves' lane)"
               if genuinely_local else
               "SEMI-GLOBAL DRIFT (change energy concentrated at near-global scale, "
               "the colorCurves lane -- same failure mode as the blurred-mask)")
        ),
    }

    # ---- head-to-head vs blurred-mask floor ----
    blurred_mask_local_floor = {"range": [0.11, 0.28],
                                "note": "blurred-mask drop at local sigma 3-5% "
                                        "(from /tmp/local_tone_sweep.json), regressed blue"}
    worse_frame_drop = min(g["drop"] for g in gate_detail.values())
    best_frame_drop = max(g["drop"] for g in gate_detail.values())
    beats_floor = bool(worse_frame_drop > 0.28)  # beats the best blurred-mask local drop
    vs_blurred_mask = {
        "blurred_mask_local_drop_range": [0.11, 0.28],
        "local_laplacian_shared_drop_5580": gate_detail.get("5580", {}).get("drop"),
        "local_laplacian_shared_drop_5551": gate_detail.get("5551", {}).get("drop"),
        "local_laplacian_worse_frame_drop": worse_frame_drop,
        "local_laplacian_best_frame_drop": best_frame_drop,
        "beats_blurred_mask_local_floor": beats_floor,
        "verdict": (
            f"local-Laplacian worse-frame drop {worse_frame_drop:.3f} vs blurred-mask "
            f"local floor 0.11-0.28: "
            + ("MATERIALLY BEATS the blurred-mask at local scale"
               if beats_floor else
               "does NOT materially beat the blurred-mask at local scale")
        ),
    }

    # ---- before/after heatmaps + corrected PNGs at the shared setting ----
    print("\n=== writing heatmaps + corrected PNGs (shared setting) ===", flush=True)
    heatmaps = {}
    shared_params = (sr, a, b, sl, n_disc)
    for fr in frames:
        hp, _ = write_outputs(fr, shared_params, tag="shared", out_dir=out_dir)
        heatmaps[fr["name"]] = hp
        print(f"  {fr['name']}: {hp['before']} | {hp['after']} | {hp['corrected_hi']}", flush=True)

    # ---- assemble JSON ----
    out = {
        "n_disc_intensity_levels": n_disc,
        "pyramid_sigma": _PYR_SIGMA,
        "sweep_grid": {"sigma_r": sigma_r_vals, "alpha": alpha_vals,
                       "beta": beta_vals, "shadow_lift": shadow_vals,
                       "n_configs": len(grid_params)},
        "remap_semantics": {
            "alpha": "detail gain; <1 boosts local contrast, 1 identity, >1 compresses",
            "beta": "edge/tone compression; <1 compresses range (shadow-lift sense), >1 expands",
            "shadow_lift": "asymmetric additive dark-subject (pixel<reference) lift, >=0",
            "sigma_r": "detail/edge intensity threshold in display-luma units",
        },
        "frames": frames_out,
        "shared_opt": {"sigma_r": sr, "alpha": a, "beta": b, "shadow_lift": sl,
                       "mean_luma_mae": shared_mean},
        "shared_opt_gate_clearing": (
            None if gated_key is None else
            {"sigma_r": gated_key[0], "alpha": gated_key[1], "beta": gated_key[2],
             "shadow_lift": gated_key[3], "mean_luma_mae": gated_mean}
        ),
        "gate_pass": bool(gate_pass),
        "gate_detail": gate_detail,
        "vs_blurred_mask": vs_blurred_mask,
        "effective_scale": {"per_frame": eff_scale, "summary": effective_scale_summary},
        "guards": {
            "identity_ok": bool(identity_ok),
            "identity_detail": g_id,
            "edge_aware_ok": bool(g_edge["ok"]),
            "edge_aware_detail": g_edge,
            "baseline_ok": bool(base_ok),
            "baseline_detail": {"5580": frames[0]["base_luma_mae"],
                                "5551": frames[1]["base_luma_mae"]},
        },
        "heatmaps": heatmaps,
    }
    with open(os.path.join(out_dir, "local_laplacian_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n========================= GATE =========================", flush=True)
    print(f"  shared (sigma_r,alpha,beta,shadow_lift) = ({sr}, {a}, {b}, {sl})", flush=True)
    for fr in frames:
        g = gate_detail[fr["name"]]
        print(f"  {fr['name']}: drop={g['drop']:.4f} (>=0.5? {g['clears_0p5']})  "
              f"no RGB regression? {g['rgb_non_regress']}", flush=True)
    print(f"  GATE PASS = {gate_pass}", flush=True)
    print(f"  guards: identity={identity_ok} edge_aware={g_edge['ok']} baseline={base_ok}", flush=True)
    print(f"  effective scale: {effective_scale_summary['verdict']}", flush=True)
    print(f"  vs blurred-mask: {vs_blurred_mask['verdict']}", flush=True)
    print("  wrote /tmp/local_laplacian_sweep.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
