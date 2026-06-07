#!/usr/bin/env python3
"""Measure the IMPLIED LOCAL-TONE TRANSFER of the colorCurves residual.

Sibling to local_tone_characterize.py. Where that script proves the residual is
spatially-coherent local tone (low-freq fraction + ADL partial-correlation), THIS
script quantifies the OPERATOR ITSELF: the curve "how much does the camera lift /
darken luma as a function of the LOCAL surround brightness", which is what we need
to know to pick and seed an operator.

Pipeline per frame (reuses faithful ports via local_tone_characterize / verify_option_b):
  colorCurves_out = baseline_forward(neutral_hi, preview_hi, wbTemp, wbTint)
      = WB(linear diagonal) + per-channel DISPLAY 1D LUT (fit neutral->preview), no sat.
  R_luma  = luma(colorCurves_out) - luma(preview_hi)         [display, 0..1]
  L_local = large-Gaussian-blur(luma(colorCurves_out)), sigma = sigma_frac * W (~3%)
  L_pix   = luma(colorCurves_out)                            [per-pixel control]

Sign: R_luma = colorCurves - preview. R_luma < 0 => preview brighter than the
GLOBAL colorCurves fit => the camera LIFTED that region. camera_lift = -R_luma.

Outputs:
  (1) mean R_luma binned by L_local  -> the implied LOCAL-tone transfer curve.
  (2) mean R_luma binned by L_pix    -> GLOBAL control; should be ~flat near 0
      because colorCurves already absorbed the global luma transfer. Contrast (1)/(2)
      isolates the LOCAL effect.
  (3) highlight behavior: mean R_luma in the top L_local bins (compression/protection).
  (4) cross-frame: overlay 5580 vs 5551 L_local->R_luma; rms/max diff, Pearson r,
      rms-as-fraction-of-span -> fixed scene-independent operator vs scene-adaptive.

Writes /tmp/local_tone_transfer.json (+ optional overlay PNG).

Usage:
  python3 local_tone_transfer.py
    [--dumpA /tmp/optb_dump --labelA 5580]
    [--dumpB /tmp/optb_dump_5551 --labelB 5551]
    [--nbins 16] [--sigma-frac 0.03]
    [--out /tmp/local_tone_transfer.json] [--plot /tmp/local_tone_transfer.png]
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_tone_characterize as ltc  # noqa: E402  (gaussian_blur, luma, etc.)
import verify_option_b as vob  # noqa: E402


def bin_by(control, resid, nbins, edges=None):
    """Mean residual per equal-WIDTH bin of `control` over [0,1].

    Equal-width (not equal-count) so 5580 and 5551 share identical L_local bin
    edges -> the cross-frame overlay is apples-to-apples. Empty bins -> NaN.
    """
    if edges is None:
        edges = np.linspace(0.0, 1.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(control.ravel(), edges) - 1, 0, nbins - 1)
    rflat = resid.ravel()
    means = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=np.int64)
    for b in range(nbins):
        m = idx == b
        c = int(m.sum())
        counts[b] = c
        if c > 0:
            means[b] = float(rflat[m].mean())
    return centers, means, counts, edges


def _wt_mean(vals, cnts, sl):
    v = vals[sl]
    w = cnts[sl].astype(np.float64)
    m = np.isfinite(v) & (w > 0)
    if not m.any():
        return float("nan")
    return float(np.average(v[m], weights=w[m]))


def render_frame(dump, sigma_frac):
    """Render colorCurves, return the per-pixel residual + control signals."""
    neutral = vob.load_grid(os.path.join(dump, "neutral_hi.png"))
    preview = vob.load_grid(os.path.join(dump, "preview_hi.png"))
    with open(os.path.join(dump, "recipe.json")) as f:
        recipe = json.load(f)
    temperature = float(recipe.get("wbTemp", 0.0) or 0.0)
    tint = float(recipe.get("wbTint", 0.0) or 0.0)
    H, W, _ = neutral.shape
    sigma = sigma_frac * W
    # colorCurves lane: WB + per-channel DISPLAY 1D LUT (fit neutral->preview), no sat.
    cc = vob.baseline_forward(neutral, preview, temperature, tint, f16=False)
    cc_luma = ltc.luma(cc)
    resid = cc_luma - ltc.luma(preview)  # R_luma = colorCurves - preview
    L_local = ltc.gaussian_blur(cc_luma, sigma)
    return {
        "wbTemp": temperature, "wbTint": tint, "W": W, "H": H, "sigma": sigma,
        "resid": resid, "L_local": L_local, "L_pix": cc_luma,
    }


def conditioned_local_curve(frame, nbins, local_edges, pix_lo, pix_hi):
    """Isolate the LOCAL effect from the GLOBAL one: bin R_luma by L_local but
    ONLY over pixels whose per-pixel luma L_pix is inside [pix_lo, pix_hi].

    Holding per-pixel luma roughly fixed removes the global tone confound (L_pix
    and L_local are strongly correlated), so any remaining slope vs L_local is the
    pure LOCAL-surround operator -- the thing a global 1D curve cannot reproduce.
    """
    resid, L_local, L_pix = frame["resid"], frame["L_local"], frame["L_pix"]
    band = (L_pix >= pix_lo) & (L_pix < pix_hi)
    cen, mean_loc, cnt_loc, _ = bin_by(L_local[band], resid[band], nbins, edges=local_edges)
    return cen, mean_loc, cnt_loc, int(band.sum())


def characterize(frame, nbins, local_edges=None):
    W, H, sigma = frame["W"], frame["H"], frame["sigma"]
    resid, L_local, L_pix = frame["resid"], frame["L_local"], frame["L_pix"]

    cen, mean_loc, cnt_loc, edges = bin_by(L_local, resid, nbins, edges=local_edges)
    _, mean_pix, cnt_pix, _ = bin_by(L_pix, resid, nbins, edges=edges)

    # Quartiles relative to the POPULATED L_local support (these scenes are dark:
    # L_local rarely reaches the bright end, so fixed [0,1] quartiles would leave
    # "highlight" empty). Use the populated bin range to define shadow/mid/highlight.
    pop = np.where(cnt_loc > 0)[0]
    if pop.size:
        plo, phi = int(pop.min()), int(pop.max())
        span = phi - plo + 1
        q = max(1, span // 4)
        shadow_wt = _wt_mean(mean_loc, cnt_loc, slice(plo, plo + q))
        highlight_wt = _wt_mean(mean_loc, cnt_loc, slice(phi - q + 1, phi + 1))
        mid_wt = _wt_mean(mean_loc, cnt_loc, slice(plo + q, phi - q + 1))
        populated_local_range = [float(cen[plo]), float(cen[phi])]
    else:
        shadow_wt = highlight_wt = mid_wt = float("nan")
        populated_local_range = [None, None]

    # "Flatness" of the GLOBAL control: weighted std of the per-pixel-binned curve.
    pv = mean_pix
    pw = cnt_pix.astype(np.float64)
    pm = np.isfinite(pv) & (pw > 0)
    if pm.any():
        pmean = np.average(pv[pm], weights=pw[pm])
        pstd = float(np.sqrt(np.average((pv[pm] - pmean) ** 2, weights=pw[pm])))
    else:
        pstd = float("nan")
    lv = mean_loc
    lw = cnt_loc.astype(np.float64)
    lm = np.isfinite(lv) & (lw > 0)
    if lm.any():
        lmean = np.average(lv[lm], weights=lw[lm])
        lstd = float(np.sqrt(np.average((lv[lm] - lmean) ** 2, weights=lw[lm])))
    else:
        lstd = float("nan")

    return {
        "wbTemp": frame["wbTemp"],
        "wbTint": frame["wbTint"],
        "image_size": [int(W), int(H)],
        "sigma_px": float(sigma),
        "resid_global_mean": float(resid.mean()),
        "resid_global_std": float(resid.std()),
        "local_bin_centers": cen.tolist(),
        "local_bin_resid_mean_x255": [None if not np.isfinite(v) else float(v * 255.0) for v in mean_loc],
        "local_bin_resid_mean": [None if not np.isfinite(v) else float(v) for v in mean_loc],
        "local_bin_camera_lift_x255": [None if not np.isfinite(v) else float(-v * 255.0) for v in mean_loc],
        "local_bin_count": cnt_loc.tolist(),
        "pix_bin_centers": cen.tolist(),
        "pix_bin_resid_mean_x255": [None if not np.isfinite(v) else float(v * 255.0) for v in mean_pix],
        "pix_bin_resid_mean": [None if not np.isfinite(v) else float(v) for v in mean_pix],
        "pix_bin_count": cnt_pix.tolist(),
        "populated_local_range": populated_local_range,
        "shadow_quartile_resid_mean_x255": shadow_wt * 255.0 if np.isfinite(shadow_wt) else None,
        "midtone_resid_mean_x255": mid_wt * 255.0 if np.isfinite(mid_wt) else None,
        "highlight_quartile_resid_mean_x255": highlight_wt * 255.0 if np.isfinite(highlight_wt) else None,
        "local_curve_weighted_std_x255": lstd * 255.0 if np.isfinite(lstd) else None,
        "global_control_weighted_std_x255": pstd * 255.0 if np.isfinite(pstd) else None,
        "local_over_global_amplitude_ratio": (lstd / pstd) if (np.isfinite(lstd) and np.isfinite(pstd) and pstd > 0) else None,
        "_arrays": {
            "cen": cen,
            "mean_loc": mean_loc,
            "cnt_loc": cnt_loc,
            "mean_pix": mean_pix,
            "cnt_pix": cnt_pix,
        },
    }


def cross_frame(A, B):
    a = np.asarray(A["_arrays"]["mean_loc"], dtype=np.float64)
    b = np.asarray(B["_arrays"]["mean_loc"], dtype=np.float64)
    ca = np.asarray(A["_arrays"]["cnt_loc"], dtype=np.float64)
    cb = np.asarray(B["_arrays"]["cnt_loc"], dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b) & (ca > 0) & (cb > 0)
    diffs = a - b
    da, db = a[valid], b[valid]
    if valid.any():
        rms = float(np.sqrt(np.mean((da - db) ** 2)))
        mx = float(np.max(np.abs(da - db)))
        span_a = float(da.max() - da.min())
        span_b = float(db.max() - db.min())
    else:
        rms = mx = span_a = span_b = float("nan")
    if valid.sum() >= 2 and da.std() > 0 and db.std() > 0:
        r = float(np.corrcoef(da, db)[0, 1])
    else:
        r = float("nan")
    span_mean = np.nanmean([span_a, span_b])
    rms_frac = float(rms / span_mean) if (np.isfinite(span_mean) and span_mean > 0) else float("nan")
    return {
        "shared_bins": int(valid.sum()),
        "rms_diff_x255": rms * 255.0 if np.isfinite(rms) else None,
        "max_diff_x255": mx * 255.0 if np.isfinite(mx) else None,
        "pearson_r": r,
        "curve_span_A_x255": span_a * 255.0 if np.isfinite(span_a) else None,
        "curve_span_B_x255": span_b * 255.0 if np.isfinite(span_b) else None,
        "rms_as_frac_of_curve_span": rms_frac,
        "per_bin_diff_x255": [None if not np.isfinite(d) else float(d * 255.0) for d in diffs],
        "verdict": _verdict(rms, rms_frac, r),
    }


def _verdict(rms, rms_frac, r):
    if not np.isfinite(rms):
        return "insufficient shared support"
    rms255 = rms * 255.0
    if rms255 < 6.0 and (np.isfinite(rms_frac) and rms_frac < 0.35):
        return (f"CONSISTENT (rms={rms255:.2f}/255, {rms_frac*100:.0f}% of curve span, r={r:.2f}): "
                f"fixed, scene-independent operator -> seedable from ADL strength")
    if rms255 < 12.0:
        return (f"PARTIALLY consistent (rms={rms255:.2f}/255, {rms_frac*100:.0f}% of span, r={r:.2f}): "
                f"shared SHAPE, scene-dependent MAGNITUDE -> seed shape, scale by ADL")
    return f"DIVERGENT (rms={rms255:.2f}/255, r={r:.2f}): scene-adaptive operator"


def make_plot(A, B, labelA, labelB, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, key, title in [
        (axes[0], "mean_loc", "binned by LOCAL-AVG luma L_local\n(implied LOCAL-tone transfer)"),
        (axes[1], "mean_pix", "binned by per-pixel luma L_pix\n(GLOBAL control -- should be ~flat)"),
    ]:
        for D, lab, col in [(A, labelA, "C0"), (B, labelB, "C1")]:
            cen = D["_arrays"]["cen"]
            y = np.asarray(D["_arrays"][key], dtype=np.float64) * 255.0
            ax.plot(cen, y, "-o", color=col, label=lab, ms=4)
        ax.axhline(0.0, color="k", lw=0.7, ls="--")
        ax.set_xlabel("control luma (display, 0..1)")
        ax.set_title(title, fontsize=9)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean R_luma = colorCurves - preview  (/255)\n(<0 = camera lifted)")
    fig.suptitle("colorCurves residual: implied LOCAL-tone transfer vs GLOBAL control", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def _strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumpA", default="/tmp/optb_dump")
    ap.add_argument("--labelA", default="5580")
    ap.add_argument("--dumpB", default="/tmp/optb_dump_5551")
    ap.add_argument("--labelB", default="5551")
    ap.add_argument("--nbins", type=int, default=16)
    ap.add_argument("--sigma-frac", type=float, default=0.03)
    ap.add_argument("--out", default="/tmp/local_tone_transfer.json")
    ap.add_argument("--plot", default="/tmp/local_tone_transfer.png")
    args = ap.parse_args()

    fA = render_frame(args.dumpA, args.sigma_frac)
    fB = render_frame(args.dumpB, args.sigma_frac)

    # Shared L_local bin edges from the UNION distribution via equal-COUNT
    # (percentile) edges, so all nbins are populated in both frames AND identical
    # across frames -> the cross-frame overlay compares the same surround bins.
    # (These scenes are dark; fixed [0,1] edges would leave most bins empty.)
    union_local = np.concatenate([fA["L_local"].ravel(), fB["L_local"].ravel()])
    qs = np.linspace(0.0, 1.0, args.nbins + 1)
    shared_edges = np.quantile(union_local, qs)
    shared_edges[0] = 0.0
    shared_edges[-1] = 1.0
    shared_edges = np.maximum.accumulate(shared_edges)  # strictly nondecreasing

    A = characterize(fA, args.nbins, local_edges=shared_edges)
    B = characterize(fB, args.nbins, local_edges=shared_edges)
    cons = cross_frame(A, B)

    # Conditioned (per-pixel-luma-held-fixed) LOCAL curve to isolate the LOCAL
    # surround effect from the GLOBAL one. Use a mid per-pixel band where both
    # frames have pixels and L_local still varies. Coarser bins (enough samples).
    union_pix = np.concatenate([fA["L_pix"].ravel(), fB["L_pix"].ravel()])
    pix_lo = float(np.quantile(union_pix, 0.40))
    pix_hi = float(np.quantile(union_pix, 0.75))
    cnbins = max(4, args.nbins // 2)
    # shared L_local edges WITHIN the band (percentiles of band-restricted union)
    bandA = (fA["L_pix"] >= pix_lo) & (fA["L_pix"] < pix_hi)
    bandB = (fB["L_pix"] >= pix_lo) & (fB["L_pix"] < pix_hi)
    band_union_local = np.concatenate([fA["L_local"][bandA].ravel(), fB["L_local"][bandB].ravel()])
    cqs = np.linspace(0.0, 1.0, cnbins + 1)
    cedges = np.quantile(band_union_local, cqs)
    cedges[0] = 0.0
    cedges[-1] = 1.0
    cedges = np.maximum.accumulate(cedges)
    condA = conditioned_local_curve(fA, cnbins, cedges, pix_lo, pix_hi)
    condB = conditioned_local_curve(fB, cnbins, cedges, pix_lo, pix_hi)
    cond = {
        "pix_band": [pix_lo, pix_hi],
        "pix_band_percentiles": [0.40, 0.75],
        "nbins": cnbins,
        "bin_centers": condA[0].tolist(),
        args.labelA: {
            "n_pixels": condA[3],
            "resid_mean_x255": [None if not np.isfinite(v) else float(v * 255.0) for v in condA[1]],
            "count": condA[2].tolist(),
        },
        args.labelB: {
            "n_pixels": condB[3],
            "resid_mean_x255": [None if not np.isfinite(v) else float(v * 255.0) for v in condB[1]],
            "count": condB[2].tolist(),
        },
    }
    # slope of conditioned curve (camera_lift = -R vs L_local): negative R slope
    # with increasing L_local => brighter surround pushed even brighter (local
    # contrast); positive R slope => brighter surround darkened (compression).
    def cond_slope(cc):
        cen, mean, cnt, _ = cc
        m = np.isfinite(mean) & (cnt > 0)
        if m.sum() < 2:
            return None
        return float(np.polyfit(cen[m], mean[m], 1)[0] * 255.0)  # /255 per unit L_local
    cond["resid_slope_per_unit_Llocal_x255"] = {
        args.labelA: cond_slope(condA),
        args.labelB: cond_slope(condB),
    }
    plotted = make_plot(A, B, args.labelA, args.labelB, args.plot) if args.plot else False

    result = {
        "config": {
            "nbins": args.nbins,
            "sigma_frac": args.sigma_frac,
            "control_space": "display-sRGB Rec709 luma in [0,1]",
            "residual_def": "R_luma = luma(colorCurves_out) - luma(preview_hi)",
            "colorCurves_lane": "WB(linear diagonal) + per-channel DISPLAY 1D LUT fit neutral->preview, saturation OFF",
            "sign_note": "R_luma<0 => preview brighter than global colorCurves fit (camera lifted); camera_lift = -R_luma",
            "binning": "equal-width bins over [0,1] so both frames share identical L_local edges",
        },
        args.labelA: _strip(A),
        args.labelB: _strip(B),
        "cross_frame_consistency": cons,
        "conditioned_local_curve": cond,
        "plot": args.plot if plotted else None,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    def fmt(D, key):
        cen = D["_arrays"]["cen"]
        y = np.asarray(D["_arrays"][key], dtype=np.float64)
        return "  ".join(
            f"{c:.3f}:{(v*255):+.2f}" if np.isfinite(v) else f"{c:.3f}:--"
            for c, v in zip(cen, y)
        )

    def f2(x):
        return f"{x:+.2f}" if x is not None else "--"

    print("=== IMPLIED LOCAL-TONE TRANSFER  (R_luma = colorCurves - preview, /255;  <0 = camera lifted) ===")
    for D, lab in [(A, args.labelA), (B, args.labelB)]:
        pr = D["populated_local_range"]
        prs = f"[{pr[0]:.3f},{pr[1]:.3f}]" if pr[0] is not None else "[--]"
        print(f"\n[{lab}]  sigma={D['sigma_px']:.1f}px  resid_mean={D['resid_global_mean']*255:+.2f}/255  "
              f"populated L_local range={prs}")
        print(f"  L_local -> R_luma (LOCAL transfer): {fmt(D, 'mean_loc')}")
        print(f"  L_pix   -> R_luma (GLOBAL control): {fmt(D, 'mean_pix')}")
        print(f"  shadow={f2(D['shadow_quartile_resid_mean_x255'])}  "
              f"mid={f2(D['midtone_resid_mean_x255'])}  "
              f"highlight={f2(D['highlight_quartile_resid_mean_x255'])}  (/255, over populated support)")
        print(f"  local curve std={f2(D['local_curve_weighted_std_x255'])}  "
              f"global control std={f2(D['global_control_weighted_std_x255'])}  "
              f"local/global amplitude={D['local_over_global_amplitude_ratio']}")
    print("\n=== CONDITIONED LOCAL CURVE (per-pixel luma held in mid band; isolates LOCAL surround) ===")
    print(f"  per-pixel band L_pix in [{cond['pix_band'][0]:.3f},{cond['pix_band'][1]:.3f}] (40-75 pct)")
    for lab in (args.labelA, args.labelB):
        c = cond[lab]
        cen = cond["bin_centers"]
        s = "  ".join(
            f"{cc:.3f}:{v:+.2f}" if v is not None else f"{cc:.3f}:--"
            for cc, v in zip(cen, c["resid_mean_x255"])
        )
        print(f"  [{lab}] L_local -> R_luma | fixed L_pix (n={c['n_pixels']}): {s}")
    print(f"  conditioned R slope per unit L_local (/255): "
          f"{args.labelA}={cond['resid_slope_per_unit_Llocal_x255'][args.labelA]}  "
          f"{args.labelB}={cond['resid_slope_per_unit_Llocal_x255'][args.labelB]}")

    print("\n=== CROSS-FRAME CONSISTENCY (L_local->R_luma overlay) ===")
    print(f"  shared_bins={cons['shared_bins']}  rms={cons['rms_diff_x255']:.2f}/255  "
          f"max={cons['max_diff_x255']:.2f}/255  r={cons['pearson_r']:.3f}  "
          f"rms/span={cons['rms_as_frac_of_curve_span']:.2f}")
    print(f"  VERDICT: {cons['verdict']}")
    print(f"\nwrote {args.out}" + (f" and {args.plot}" if plotted else " (no matplotlib, no plot)"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
