#!/usr/bin/env python3
"""Fabrication-resistant validation of the SHIPPED Clarity GPU operator.

Clarity is the shippable, edge-aware, halo-limited, midtone-weighted, luma-only
LOCAL-CONTRAST operator added to the Lightbox WebGL editor (megaShader.js). It is
the practical embodiment of the validated multi-scale local-Laplacian finding
(docs/local-tone-adl.md, docs/look-physics-decomposition.md): edge-awareness via
a large-radius SURROUND + a SOFT DETAIL LIMIT, WITHOUT the full pyramid.

This script ports the EXACT shipped fragment-shader math (same surround blur,
same soft-limit, same midtone weight, same luma-only ratio composite) and checks:

  (1) IDENTITY at clarity == 0  -> output == input, bit-for-bit (the op is
      skipped on the GPU; here gain 0 -> ratio 1.0 everywhere).

  (2) EDGE-AWARENESS on the synthetic step + fine-texture image from
      local_laplacian_apply.make_textured_step(): Clarity boosts the fine texture
      while keeping the hard step's overshoot LOW.  Compared head-to-head against
      a naive Gaussian-unsharp matched to the SAME texture gain, on the same pure
      step -- the unsharp's step overshoot should be markedly LARGER (overshoot
      ratio >> 1), demonstrating the edge-awareness comes from the soft-limit, not
      from a disguised blur.

  (3) NO CLIPPING BLOWUP on /tmp/optb_dump/neutral_hi.png at a moderate clarity
      (+50): report min/max per channel before/after and confirm Clarity does not
      manufacture out-of-range channels or collapse the image; write a corrected
      PNG for visual inspection.

Honest about approximation: this is NOT the full local-Laplacian. It is a
single-surround edge-aware unsharp with a soft detail limiter + midtone window.
The oracle (local_laplacian_apply.py) remains the reference for halo behaviour;
guard (2) here measures Clarity's halo against the SAME naive-unsharp baseline the
oracle uses, so a markedly lower overshoot ratio is the falsifiable evidence that
Clarity's soft-limit really does suppress edge ringing.

All numbers -> /tmp/clarity_validate.json. Pure numpy + PIL (+ scipy via the
reused harness). Run: python3 tools/clarity_validate.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image

# Reuse the SAME synthetic generators + Gaussian + metrics the local-Laplacian
# oracle uses, so the comparison is apples-to-apples.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_laplacian_apply as lla  # noqa: E402
import local_tone_characterize as ltc  # noqa: E402

REC709 = np.array([0.2126, 0.7152, 0.0722])
gaussian_blur = ltc.gaussian_blur

# ── Shipped operator constants (must match megaShader.js) ───────────────────
CLARITY_SCALE = 1.2      # slider/100 -> gain, max gain at |slider|=100
CLARITY_THRESH = 0.08    # soft-limit knee in display-luma units
EPS = 1e-3               # clL > 0.001 ? ratio : 1.0


def display_luma(rgb):
    """REC709 luma of clamped display-space rgb (matches the shader's clL)."""
    return np.dot(np.clip(rgb, 0.0, 1.0), REC709)


def clarity_operator(rgb, clarity_slider, surround_sigma_px, thresh=CLARITY_THRESH):
    """EXACT port of the shipped megaShader Clarity stage.

    rgb: HxWx3 display-space image in [0,1].
    clarity_slider: [-100,100]; 0 -> exact identity (ratio 1.0 everywhere).
    surround_sigma_px: Gaussian sigma (px) for the large-radius surround L_blur.
    Returns (out_rgb, info) where info carries L, L_blur, detail, ratio for guards.

    Shader lines (megaShader.js):
      if (clarityGain != 0.0) {
        clL    = dot(clamp(color.rgb,0,1), REC709);
        clBlur = <surround blur of display luma>;
        clDetail = clL - clBlur;
        clT    = max(clarityThresh, 1e-4);
        clD    = clDetail * clT / (clT + abs(clDetail));
        clW    = clamp(4*clL*(1-clL), 0, 1);
        clOut  = clamp(clL + clarityGain*clW*clD, 0, 1);
        clRatio= clL > 0.001 ? clOut/clL : 1.0;
        color.rgb = clamp(color.rgb * clRatio, 0, 1);
      }
    """
    gain = (clarity_slider / 100.0) * CLARITY_SCALE
    L = display_luma(rgb)
    if gain == 0.0:
        # GPU short-circuits the whole op -> exact identity.
        ratio = np.ones_like(L)
        out = np.clip(rgb, 0.0, 1.0)
        return out, {"L": L, "L_blur": L.copy(), "detail": np.zeros_like(L),
                     "ratio": ratio, "gain": 0.0}
    # Large-radius surround = Gaussian blur of the display luma. (On the GPU this
    # is the quarter-res clarity prepass; the math is the same low-frequency
    # surround.)
    L_blur = gaussian_blur(L, surround_sigma_px)
    detail = L - L_blur
    clT = max(thresh, 1e-4)
    d = detail * clT / (clT + np.abs(detail))            # soft limit
    w = np.clip(4.0 * L * (1.0 - L), 0.0, 1.0)           # midtone weight
    L_out = np.clip(L + gain * w * d, 0.0, 1.0)
    ratio = np.where(L > EPS, L_out / np.maximum(L, EPS), 1.0)
    out = np.clip(rgb * ratio[..., None], 0.0, 1.0)
    return out, {"L": L, "L_blur": L_blur, "detail": detail, "ratio": ratio, "gain": gain}


# ── Guard (2) helpers (reuse oracle generators + metrics) ───────────────────
def texture_std(out_L, base_L, N):
    a, b = N // 8, 3 * N // 8
    return float((out_L - base_L)[:, a:b].std())


def step_overshoot(out_L, lo=0.30, hi=0.70, N=256, band=20):
    edge = N // 2
    left = out_L[N // 2, edge - band:edge]
    right = out_L[N // 2, edge:edge + band]
    return float(max(np.max(np.abs(left - lo)), np.max(np.abs(right - hi))))


def gaussian_unsharp_L(L, sigma, gain):
    return L + gain * (L - gaussian_blur(L, sigma))


def gray_rgb(L):
    """Promote a single-channel luma field to a neutral-gray RGB image so the
    clarity operator (which works on rgb) runs on it. For a neutral image,
    display_luma(gray_rgb(L)) == L and the ratio composite reduces to L_out, so
    the operator acts directly on L -- exactly the luma path we want to probe."""
    return np.repeat(L[..., None], 3, axis=2)


def guard_identity(rgb):
    out, info = clarity_operator(rgb, clarity_slider=0.0, surround_sigma_px=10.0)
    max_abs = float(np.max(np.abs(out - np.clip(rgb, 0.0, 1.0))))
    # Also confirm a nonzero slider actually changes something (so "identity" is
    # meaningful and not a dead operator).
    out_on, _ = clarity_operator(rgb, clarity_slider=50.0, surround_sigma_px=10.0)
    changed = float(np.max(np.abs(out_on - np.clip(rgb, 0.0, 1.0))))
    return {
        "max_abs_diff_at_clarity_0": max_abs,
        "identity_ok": bool(max_abs == 0.0),
        "max_abs_diff_at_clarity_50": changed,
        "operator_is_live": bool(changed > 1e-4),
    }


def guard_edge_aware(out_dir="/tmp"):
    """Clarity vs naive Gaussian-unsharp at matched texture gain, on the SAME
    synthetic step+texture the local-Laplacian oracle uses."""
    N = 256
    period = 8
    tex_amp = 0.03
    # Surround sigma in px for a 256px synthetic at the shipped ~3% long-edge
    # radius -> sigma ~ 0.03*256/2 ~ 4 px (matches the GPU surround scale).
    surround_sigma = 4.0
    clarity_slider = 100.0   # max boost to make the texture gain clearly measurable

    img_L, base_L = lla.make_textured_step(N=N, tex_period=period, tex_amp=tex_amp)
    pure_L = lla.make_pure_step(N=N)
    in_tex = texture_std(img_L, base_L, N)

    # --- Clarity on the textured step + on the pure step ---
    cl_tex_out, _ = clarity_operator(gray_rgb(img_L), clarity_slider, surround_sigma)
    cl_tex_L = display_luma(cl_tex_out)
    cl_boost = texture_std(cl_tex_L, base_L, N) / max(in_tex, 1e-9)

    cl_pure_out, _ = clarity_operator(gray_rgb(pure_L), clarity_slider, surround_sigma)
    cl_pure_L = display_luma(cl_pure_out)
    cl_over = step_overshoot(cl_pure_L, N=N)

    # --- Gaussian-unsharp matched to the SAME texture boost ---
    target = texture_std(cl_tex_L, base_L, N)
    us_sigma = surround_sigma
    lo_g, hi_g = 0.0, 200.0
    for _ in range(48):
        mid = 0.5 * (lo_g + hi_g)
        if texture_std(gaussian_unsharp_L(img_L, us_sigma, mid), base_L, N) < target:
            lo_g = mid
        else:
            hi_g = mid
    us_gain = 0.5 * (lo_g + hi_g)
    us_tex_L = gaussian_unsharp_L(img_L, us_sigma, us_gain)
    us_boost = texture_std(us_tex_L, base_L, N) / max(in_tex, 1e-9)
    us_pure_L = gaussian_unsharp_L(pure_L, us_sigma, us_gain)
    us_over = step_overshoot(us_pure_L, N=N)

    overshoot_ratio = (us_over / cl_over) if abs(cl_over) > 1e-9 else float("inf")

    # row profiles across the pure step for visual inspection
    prof = np.stack([pure_L[N // 2], cl_pure_L[N // 2],
                     np.clip(us_pure_L[N // 2], 0, 1)], axis=0)
    pimg = (np.clip(np.repeat(prof, 30, axis=0), 0, 1) * 255).round().astype(np.uint8)
    pp = os.path.join(out_dir, "clarity_edgeaware_profiles.png")
    Image.fromarray(pimg, "L").save(pp)

    # PASS: Clarity boosts texture (>1.2x), the unsharp matched that boost (fair),
    # and Clarity's hard-step overshoot is markedly smaller (>=1.5x lower) -- the
    # soft-limit's edge-awareness.
    ok = bool((cl_boost > 1.2)
              and (us_boost > 0.7 * cl_boost)
              and (us_over > 1.5 * cl_over))
    return {
        "synthetic": {"N": N, "tex_amp": tex_amp, "tex_period": period,
                      "surround_sigma_px": surround_sigma,
                      "clarity_slider": clarity_slider, "thresh": CLARITY_THRESH},
        "input_texture_std": in_tex,
        "clarity": {"texture_boost_x": cl_boost, "pure_step_overshoot": cl_over},
        "gaussian_unsharp": {"sigma": us_sigma, "gain": us_gain,
                             "texture_boost_x": us_boost,
                             "pure_step_overshoot": us_over},
        "unsharp_to_clarity_overshoot_ratio": overshoot_ratio,
        "profiles_png": pp,
        "edge_aware_ok": ok,
        "interpretation": (
            f"Clarity boosts the fine texture {cl_boost:.2f}x while leaving a hard "
            f"step's overshoot at {cl_over:.4f}; a Gaussian-unsharp matched to the "
            f"same {us_boost:.2f}x texture gain overshoots the same step at "
            f"{us_over:.4f} ({overshoot_ratio:.1f}x larger). The soft detail limit "
            "is what holds the edge down -- edge-aware, not a disguised unsharp. "
            "(Approximation note: single-surround, not the full local-Laplacian "
            "pyramid; oracle = local_laplacian_apply.py.)"
        ),
    }


def guard_no_clipping(dump_dir="/tmp/optb_dump", out_dir="/tmp", clarity_slider=50.0):
    path = os.path.join(dump_dir, "neutral_hi.png")
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    H, W, _ = img.shape
    long_edge = max(H, W)
    surround_sigma = 0.03 * long_edge / 2.0   # ~3% long-edge radius -> sigma

    out, info = clarity_operator(img, clarity_slider, surround_sigma)

    in_min = [float(img[..., c].min()) for c in range(3)]
    in_max = [float(img[..., c].max()) for c in range(3)]
    out_min = [float(out[..., c].min()) for c in range(3)]
    out_max = [float(out[..., c].max()) for c in range(3)]
    # Fraction of pixels pushed to hard 0 or 1 by the operator that weren't
    # already there in the input -> "manufactured clipping".
    new_hi = float(np.mean((out >= 1.0) & (img < 1.0)))
    new_lo = float(np.mean((out <= 0.0) & (img > 0.0)))
    ratio = info["ratio"]
    # local-contrast sanity: did mid-frequency luma variance go UP (more local
    # contrast) without the global mean drifting much?
    L_in, L_out = info["L"], display_luma(out)
    band_in = L_in - gaussian_blur(L_in, surround_sigma)
    band_out = L_out - gaussian_blur(L_out, surround_sigma)
    local_contrast_gain = float(band_out.std() / max(band_in.std(), 1e-9))
    mean_drift = float(abs(L_out.mean() - L_in.mean()))

    cp = os.path.join(out_dir, "clarity_corrected_neutral_hi.png")
    Image.fromarray((out * 255.0).round().astype(np.uint8), "RGB").save(cp)

    no_blowup = bool(new_hi < 0.02 and new_lo < 0.02
                     and all(out_max[c] <= 1.0 + 1e-9 for c in range(3))
                     and all(out_min[c] >= -1e-9 for c in range(3)))
    sane_local_contrast = bool(local_contrast_gain > 1.05 and mean_drift < 0.03)
    return {
        "image": path, "size": [W, H], "clarity_slider": clarity_slider,
        "surround_sigma_px": surround_sigma,
        "in_min": in_min, "in_max": in_max,
        "out_min": out_min, "out_max": out_max,
        "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
        "new_clipped_high_frac": new_hi, "new_clipped_low_frac": new_lo,
        "local_contrast_gain_x": local_contrast_gain,
        "global_mean_luma_drift": mean_drift,
        "no_clipping_blowup": no_blowup,
        "sane_local_contrast": sane_local_contrast,
        "corrected_png": cp,
    }


def main():
    out_dir = "/tmp"

    # An RGB test image for the identity guard: reuse the neutral dump if present,
    # else a synthetic gradient.
    dump = os.path.join("/tmp/optb_dump", "neutral_hi.png")
    if os.path.exists(dump):
        rgb = np.asarray(Image.open(dump).convert("RGB"), dtype=np.float64) / 255.0
    else:
        g = np.linspace(0, 1, 256)
        rgb = np.stack(np.meshgrid(g, g), -1)
        rgb = np.dstack([rgb[..., 0], rgb[..., 1], (rgb[..., 0] + rgb[..., 1]) * 0.5])

    print("=== guard 1: identity at clarity == 0 ===", flush=True)
    g1 = guard_identity(rgb)
    print(f"  max|out-in| @0 = {g1['max_abs_diff_at_clarity_0']:.3e}  "
          f"identity_ok={g1['identity_ok']}  live@50={g1['operator_is_live']}", flush=True)

    print("=== guard 2: edge-awareness vs Gaussian-unsharp ===", flush=True)
    g2 = guard_edge_aware(out_dir=out_dir)
    print(f"  clarity texture boost = {g2['clarity']['texture_boost_x']:.2f}x  "
          f"step overshoot = {g2['clarity']['pure_step_overshoot']:.4f}", flush=True)
    print(f"  unsharp (matched)     = {g2['gaussian_unsharp']['texture_boost_x']:.2f}x  "
          f"step overshoot = {g2['gaussian_unsharp']['pure_step_overshoot']:.4f}", flush=True)
    print(f"  unsharp/clarity overshoot ratio = "
          f"{g2['unsharp_to_clarity_overshoot_ratio']:.1f}x  ok={g2['edge_aware_ok']}", flush=True)

    print("=== guard 3: no-clipping on neutral_hi.png @ +50 ===", flush=True)
    if os.path.exists(dump):
        g3 = guard_no_clipping(out_dir=out_dir)
        print(f"  out min/max = {[round(x,4) for x in g3['out_min']]} / "
              f"{[round(x,4) for x in g3['out_max']]}", flush=True)
        print(f"  new clip hi/lo frac = {g3['new_clipped_high_frac']:.4f} / "
              f"{g3['new_clipped_low_frac']:.4f}  "
              f"local-contrast gain = {g3['local_contrast_gain_x']:.3f}x  "
              f"mean drift = {g3['global_mean_luma_drift']:.4f}", flush=True)
        print(f"  no_clipping_blowup={g3['no_clipping_blowup']}  "
              f"sane_local_contrast={g3['sane_local_contrast']}", flush=True)
    else:
        g3 = {"skipped": True, "reason": f"{dump} not found"}
        print(f"  SKIPPED ({dump} not found)", flush=True)

    result = {
        "operator": {
            "form": "edge-aware halo-limited midtone-weighted luma-only local contrast",
            "shader": "frontend/src/lib/glfx-es6/filters/adjust/megaShader.js",
            "clarity_scale": CLARITY_SCALE, "thresh": CLARITY_THRESH,
            "approximation_note": (
                "Single large-radius surround + soft detail limit + midtone window; "
                "NOT the full local-Laplacian pyramid. Oracle for halo behaviour: "
                "tools/local_laplacian_apply.py."
            ),
        },
        "guard1_identity": g1,
        "guard2_edge_aware": g2,
        "guard3_no_clipping": g3,
        "all_pass": bool(
            g1["identity_ok"] and g1["operator_is_live"]
            and g2["edge_aware_ok"]
            and (g3.get("skipped") or (g3["no_clipping_blowup"] and g3["sane_local_contrast"]))
        ),
    }
    with open(os.path.join(out_dir, "clarity_validate.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\n========================= SUMMARY =========================", flush=True)
    print(f"  identity_ok        = {g1['identity_ok']}", flush=True)
    print(f"  edge_aware_ok      = {g2['edge_aware_ok']}  "
          f"(overshoot ratio {g2['unsharp_to_clarity_overshoot_ratio']:.1f}x)", flush=True)
    if not g3.get("skipped"):
        print(f"  no_clipping_blowup = {g3['no_clipping_blowup']}  "
              f"sane_local_contrast = {g3['sane_local_contrast']}", flush=True)
    print(f"  ALL PASS = {result['all_pass']}", flush=True)
    print("  wrote /tmp/clarity_validate.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
