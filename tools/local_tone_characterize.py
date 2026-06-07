#!/usr/bin/env python3
"""Characterize the SPATIAL signature of the residual left by the GLOBAL
colorCurves model (WB + per-channel display 1D LUT, no saturation) against the
camera JPEG/preview, to confirm it is LOCAL-TONE (Active D-Lighting) and not
noise / per-hue.

Reuses the faithful ports in verify_option_b.py:
  - srgb_decode / srgb_encode          (IEC 61966-2-1)
  - REC709 luma weights
  - wb_exp                             (WB linear diagonal)
  - hpminde_clip / build_lut / lut_sample
  - fit_channel_curve                  (24-bin monotone, 16-knot per-channel fit)
  - baseline_forward                   (the colorCurves lane: WB + 3x1D display LUT, no sat)

For each frame (5580 = /tmp/optb_dump, 5551 = /tmp/optb_dump_5551):
  1. Fit colorCurves per-channel on the hi-res neutral->preview and render
     colorCurves_out = baseline_forward(neutral_hi).
  2. Residual R = colorCurves_out - preview_hi (display [0,1] and 0-255).
     Report residual MAE (sanity ~2.8-3.4).
  3. SPATIAL: fraction of residual ENERGY that is LOW-FREQUENCY, comparing R to
     a large-radius Gaussian-blurred R (sigma ~ 3% of image width).
       lowFreqFraction = ||blur(R)||^2 / ||R||^2  (per channel + mean)
     High => spatially-coherent local tone; low => high-freq quant/noise.
  4. ADL SIGNATURE: PARTIAL correlation of residual-luma with the LOCAL-AVERAGE
     luma of the neutral (large Gaussian), CONTROLLING for the per-pixel neutral
     luma. A nonzero partial corr = shadow-lift keyed to local surround (ADL
     hallmark) beyond any global tone.
  5. Write residual heatmap PNGs per frame (diverging colormap) to /tmp.
  6. Write all stats to /tmp/local_tone_stats.json.

Pure numpy + PIL; scipy.ndimage.gaussian_filter used if available, else a
separable-Gaussian convolution implemented in numpy (verified equivalent).
"""

import json
import os
import sys

import numpy as np
from PIL import Image

# --- reuse the faithful ports from verify_option_b.py ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_option_b as vob  # noqa: E402

REC709 = vob.REC709

# scipy gaussian if available, else numpy separable fallback
try:
    from scipy.ndimage import gaussian_filter as _scipy_gaussian

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def _gaussian_kernel1d(sigma, truncate=4.0):
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _conv1d_reflect(arr, k, axis):
    """1D convolution along `axis` with 'reflect' edge handling (mirror, no edge
    repeat) to match scipy's default mode='reflect'."""
    radius = (len(k) - 1) // 2
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (radius, radius)
    a = np.pad(arr, pad, mode="reflect")
    # build output via sliding sum (vectorized over taps)
    out = np.zeros_like(arr, dtype=np.float64)
    n = arr.shape[axis]
    for t, w in enumerate(k):
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice(t, t + n)
        out += w * a[tuple(sl)]
    return out


def gaussian_blur(arr, sigma):
    """Separable Gaussian blur of a 2-D (H,W) or 3-D (H,W,C) array.

    Uses scipy.ndimage.gaussian_filter (mode='reflect') if available; otherwise
    a numpy separable convolution with reflect padding (equivalent)."""
    arr = np.asarray(arr, dtype=np.float64)
    if _HAVE_SCIPY:
        if arr.ndim == 2:
            return _scipy_gaussian(arr, sigma=sigma, mode="reflect")
        out = np.empty_like(arr)
        for c in range(arr.shape[2]):
            out[..., c] = _scipy_gaussian(arr[..., c], sigma=sigma, mode="reflect")
        return out
    # numpy fallback
    k = _gaussian_kernel1d(sigma)
    if arr.ndim == 2:
        tmp = _conv1d_reflect(arr, k, axis=0)
        return _conv1d_reflect(tmp, k, axis=1)
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        tmp = _conv1d_reflect(arr[..., c], k, axis=0)
        out[..., c] = _conv1d_reflect(tmp, k, axis=1)
    return out


def luma(rgb):
    return np.tensordot(rgb, REC709, axes=([rgb.ndim - 1], [0]))


def diverging_heatmap(R_signed, vlim):
    """Map a signed single-channel residual R_signed in [-vlim, vlim] to an RGB
    diverging colormap (blue=neg, white=0, red=pos). Returns uint8 (H,W,3)."""
    x = np.clip(R_signed / vlim, -1.0, 1.0)  # [-1,1]
    # blue->white->red
    pos = np.clip(x, 0.0, 1.0)  # 0..1 for positive
    neg = np.clip(-x, 0.0, 1.0)  # 0..1 for negative
    r = 1.0 - neg  # red stays high unless negative
    g = 1.0 - pos - neg
    b = 1.0 - pos
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def partial_corr(a, b, c):
    """Partial correlation of a and b controlling for c.
    corr(resid(a~c), resid(b~c)) via the standard formula on pairwise corrs."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    c = c.ravel().astype(np.float64)
    rab = np.corrcoef(a, b)[0, 1]
    rac = np.corrcoef(a, c)[0, 1]
    rbc = np.corrcoef(b, c)[0, 1]
    denom = np.sqrt(max(1e-12, (1.0 - rac * rac) * (1.0 - rbc * rbc)))
    return float((rab - rac * rbc) / denom), {
        "r_ab": float(rab),
        "r_ac": float(rac),
        "r_bc": float(rbc),
    }


def analyze_frame(name, dump_dir, out_dir="/tmp"):
    neutral = vob.load_grid(os.path.join(dump_dir, "neutral_hi.png"))
    preview = vob.load_grid(os.path.join(dump_dir, "preview_hi.png"))
    H, W, _ = neutral.shape

    with open(os.path.join(dump_dir, "recipe.json")) as f:
        recipe = json.load(f)
    temperature = float(recipe.get("wbTemp", 0.0) or 0.0)
    tint = float(recipe.get("wbTint", 0.0) or 0.0)

    # (1) fit colorCurves (WB + per-channel display 1D LUT, no saturation) on
    #     the hi-res data and render. baseline_forward fits each channel curve
    #     from neutral-after-WB(display) -> preview(display).
    cc_out = vob.baseline_forward(neutral, preview, temperature, tint, f16=False)

    # (2) residual in display [0,1] and 0-255
    R = cc_out - preview  # signed, display [0,1]
    R255 = R * 255.0
    abs255 = np.abs(R255)
    mae_per_ch = abs255.reshape(-1, 3).mean(axis=0)
    mae_mean = float(abs255.mean())
    R_luma = luma(R)  # signed residual luma, display [0,1]

    # (3) low-frequency energy fraction. sigma ~ 3% of image width.
    sigma = 0.03 * W
    # work on mean-removed residual so the constant offset (a trivial "global"
    # shift) doesn't dominate -- we want SPATIAL coherence, not a DC bias.
    Rc = R - R.reshape(-1, 3).mean(axis=0)[None, None, :]
    R_blur = gaussian_blur(Rc, sigma)
    energy_total = (Rc * Rc).reshape(-1, 3).sum(axis=0)  # per channel
    energy_low = (R_blur * R_blur).reshape(-1, 3).sum(axis=0)
    low_frac_per_ch = energy_low / np.maximum(energy_total, 1e-30)
    # mean over channels weighted by total energy (so it reflects overall signal)
    low_frac_mean = float(energy_low.sum() / max(1e-30, energy_total.sum()))

    # also a luma-only version (most relevant for ADL shadow-lift)
    Rl_c = R_luma - R_luma.mean()
    Rl_blur = gaussian_blur(Rl_c, sigma)
    low_frac_luma = float((Rl_blur * Rl_blur).sum() / max(1e-30, (Rl_c * Rl_c).sum()))

    # (4) ADL signature: partial corr of residual-luma with LOCAL-AVERAGE luma
    #     of neutral (large Gaussian), controlling for per-pixel neutral luma.
    n_luma = luma(neutral)  # per-pixel neutral luma (display [0,1])
    n_local = gaussian_blur(n_luma, sigma)  # local-average (surround) luma
    pc, pc_detail = partial_corr(R_luma, n_local, n_luma)
    # also the simple (non-partial) corrs of residual-luma vs per-pixel luma,
    # for context (the GLOBAL tone the colorCurves model should have absorbed).
    r_resid_perpixel = float(np.corrcoef(R_luma.ravel(), n_luma.ravel())[0, 1])
    r_resid_local = float(np.corrcoef(R_luma.ravel(), n_local.ravel())[0, 1])

    # (5) heatmaps: per-channel + luma, diverging colormap, robust vlim (p99)
    heat_paths = {}
    for ch, chn in enumerate("RGB"):
        vlim = float(np.percentile(np.abs(R255[..., ch]), 99.0))
        vlim = max(vlim, 1.0)
        img = diverging_heatmap(R255[..., ch], vlim)
        p = os.path.join(out_dir, f"local_tone_residual_{name}_{chn}.png")
        Image.fromarray(img, "RGB").save(p)
        heat_paths[chn] = p
    vlim_l = max(1.0, float(np.percentile(np.abs(R_luma * 255.0), 99.0)))
    img_l = diverging_heatmap(R_luma * 255.0, vlim_l)
    p_l = os.path.join(out_dir, f"local_tone_residual_{name}_luma.png")
    Image.fromarray(img_l, "RGB").save(p_l)
    heat_paths["luma"] = p_l
    # low-freq (blurred) luma residual heatmap -- shows the coherent local-tone field
    vlim_lf = max(1.0, float(np.percentile(np.abs(Rl_blur * 255.0), 99.0)))
    img_lf = diverging_heatmap(Rl_blur * 255.0, vlim_lf)
    p_lf = os.path.join(out_dir, f"local_tone_residual_{name}_luma_lowfreq.png")
    Image.fromarray(img_lf, "RGB").save(p_lf)
    heat_paths["luma_lowfreq"] = p_lf

    return {
        "frame": name,
        "dump_dir": dump_dir,
        "size": [W, H],
        "wb": {"temperature": temperature, "tint": tint},
        "gaussian_sigma_px": float(sigma),
        "sigma_pct_width": 3.0,
        "scipy_used": _HAVE_SCIPY,
        "residual_mae_per_channel_RGB": [float(x) for x in mae_per_ch],
        "residual_mae_mean": mae_mean,
        "residual_luma_mae_255": float(np.abs(R_luma * 255.0).mean()),
        "residual_signed_mean_255_RGB": [float(x) for x in R255.reshape(-1, 3).mean(axis=0)],
        "low_freq_fraction_per_channel_RGB": [float(x) for x in low_frac_per_ch],
        "low_freq_fraction_mean": low_frac_mean,
        "low_freq_fraction_luma": low_frac_luma,
        "adl_partial_corr_residualLuma_localLuma_given_perpixelLuma": pc,
        "adl_partial_corr_detail": pc_detail,
        "corr_residualLuma_vs_perpixelLuma": r_resid_perpixel,
        "corr_residualLuma_vs_localLuma": r_resid_local,
        "heatmaps": heat_paths,
    }


def verdict(frames):
    lf = np.mean([f["low_freq_fraction_luma"] for f in frames])
    pc = np.mean([abs(f["adl_partial_corr_residualLuma_localLuma_given_perpixelLuma"]) for f in frames])
    is_local = lf >= 0.5  # majority of residual energy is spatially coherent
    parts = []
    parts.append(
        f"mean low-freq luma fraction = {lf:.3f} "
        f"({'LOCAL-TONE: residual energy is spatially coherent' if is_local else 'NON-LOCAL: residual is high-freq (noise/quant)'})"
    )
    parts.append(
        f"mean |ADL partial corr| (residual-luma vs local surround | per-pixel luma) = {pc:.3f} "
        f"({'shadow/tone lift keyed to local surround beyond global tone -- ADL hallmark present' if pc >= 0.1 else 'weak local-surround dependence'})"
    )
    return is_local, "; ".join(parts)


def main():
    frames_cfg = [
        ("5580", "/tmp/optb_dump"),
        ("5551", "/tmp/optb_dump_5551"),
    ]
    results = []
    for name, d in frames_cfg:
        r = analyze_frame(name, d)
        results.append(r)
        print(f"=== frame {name} ({d}) ===")
        print(f"  residual MAE RGB = {r['residual_mae_per_channel_RGB']}  mean = {r['residual_mae_mean']:.4f}")
        print(f"  residual luma MAE (255) = {r['residual_luma_mae_255']:.4f}")
        print(f"  low-freq fraction  RGB = {r['low_freq_fraction_per_channel_RGB']}")
        print(f"  low-freq fraction mean = {r['low_freq_fraction_mean']:.4f}  luma = {r['low_freq_fraction_luma']:.4f}")
        print(f"  ADL partial corr (residLuma vs localLuma | perpixelLuma) = {r['adl_partial_corr_residualLuma_localLuma_given_perpixelLuma']:.4f}")
        print(f"    detail r_ab(resid,local)={r['adl_partial_corr_detail']['r_ab']:.4f} "
              f"r_ac(resid,perpix)={r['adl_partial_corr_detail']['r_ac']:.4f} "
              f"r_bc(local,perpix)={r['adl_partial_corr_detail']['r_bc']:.4f}")
        print(f"  heatmaps: {list(r['heatmaps'].values())}")

    is_local, summary = verdict(results)
    out = {
        "scipy_used": _HAVE_SCIPY,
        "frames": results,
        "verdict_is_local_tone": bool(is_local),
        "verdict_summary": summary,
    }
    with open("/tmp/local_tone_stats.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"VERDICT is_local_tone = {is_local}")
    print(f"  {summary}")
    print("wrote /tmp/local_tone_stats.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
