#!/usr/bin/env python3
"""Validate the proposed LOCAL-TONE (Active D-Lighting emulation) operator with a
parameter sweep on the two measured frames (5580, 5551). This is the go/no-go
gate from docs/local-tone-adl.md s6.

The question: does a SINGLE SHARED (k_pos, k_lift, sigma) drop luma MAE by >=0.5
on BOTH frames (operator works / parametric), or do the per-frame optima diverge
(overfit / needs more data)?

Reuses the FAITHFUL ports already written:
  - verify_option_b.py            : srgb_decode/encode, REC709, fit_channel_curve,
                                    build_lut, lut_sample, baseline_forward
                                    (WB + per-channel display 1D LUT, saturation OFF)
  - local_tone_characterize.py    : gaussian_blur (scipy or numpy fallback), luma,
                                    diverging_heatmap, load_grid

Operator (docs s2.4, applied in DISPLAY space, luma-only, chroma preserved):
  cc_out = baseline_forward(neutral_hi)        # colorCurves lane, sat OFF
  L      = luma(cc_out)                         # display-space REC709 luma
  L_blur = Gaussian(L, sigma)
  detail = L - L_blur                          # signed local contrast
  g      = (detail >= 0) ? k_pos : k_lift      # asymmetric gain
  L_out  = clamp(L + g*detail, 0, 1)
  ratio  = (L > eps) ? L_out/L : 1
  out_rgb = cc_out * ratio                      # luminance ratio composite

Sweep INCLUDES NEGATIVE gains -- the design's verbal sign is ambiguous (the
measured CONDITIONED slope of camera-lift vs surround is positive, which may map
to a NEGATIVE unsharp gain = local-contrast reduction / surround-follow). Let the
data decide the sign.

Outputs:
  - per-frame optimal (k_pos,k_lift,sigma) and its luma-MAE drop
  - SHARED (k_pos,k_lift,sigma) minimizing MEAN luma MAE; per-frame drop at shared
  - gate: shared clears >=0.5 luma-MAE drop on BOTH frames with NO per-channel regr
  - shared-vs-perframe optimum distance (parametric-confirmed vs divergent)
  - SIGN of the winning operator (contrast-increase vs surround-follow/lift)
  - before/after residual heatmaps + corrected_hi PNGs to /tmp
  - all numbers to /tmp/local_tone_sweep.json
"""

import json
import os
import sys

import numpy as np
from PIL import Image

# --- reuse the faithful ports ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_option_b as vob  # noqa: E402
import local_tone_characterize as ltc  # noqa: E402

REC709 = vob.REC709
gaussian_blur = ltc.gaussian_blur
luma = ltc.luma
diverging_heatmap = ltc.diverging_heatmap
load_grid = vob.load_grid

EPS = 1e-6


def apply_operator(cc_out, L, L_blur, k_pos, k_lift):
    """Asymmetric blurred-luma-mask operator (docs s2.4), luma-only ratio
    composite. cc_out/L/L_blur are precomputed (per-sigma) so the inner sweep is
    cheap. Returns corrected display-space RGB in [0,1]."""
    detail = L - L_blur
    g = np.where(detail >= 0.0, k_pos, k_lift)
    L_out = np.clip(L + g * detail, 0.0, 1.0)
    ratio = np.where(L > EPS, L_out / np.maximum(L, EPS), 1.0)
    out = cc_out * ratio[..., None]
    return np.clip(out, 0.0, 1.0)


def luma_mae(out_rgb, preview):
    """Residual luma MAE vs preview, /255 scale (matches characterize)."""
    return float(np.abs((luma(out_rgb) - luma(preview)) * 255.0).mean())


def rgb_mae(out_rgb, preview):
    diff = np.abs(out_rgb - preview) * 255.0
    return [float(x) for x in diff.reshape(-1, 3).mean(axis=0)]


def prep_frame(name, dump_dir):
    """Render the colorCurves baseline and precompute per-sigma blurred luma."""
    neutral = load_grid(os.path.join(dump_dir, "neutral_hi.png"))
    preview = load_grid(os.path.join(dump_dir, "preview_hi.png"))
    H, W, _ = neutral.shape
    long_edge = max(H, W)

    with open(os.path.join(dump_dir, "recipe.json")) as f:
        recipe = json.load(f)
    temperature = float(recipe.get("wbTemp", 0.0) or 0.0)
    tint = float(recipe.get("wbTint", 0.0) or 0.0)

    # (1) colorCurves baseline forward (WB + per-channel display 1D LUT, sat OFF)
    cc_out = vob.baseline_forward(neutral, preview, temperature, tint, f16=False)
    L = luma(cc_out)
    base_luma_mae = luma_mae(cc_out, preview)
    base_rgb_mae = rgb_mae(cc_out, preview)

    return {
        "name": name,
        "dump_dir": dump_dir,
        "size": [W, H],
        "long_edge": long_edge,
        "preview": preview,
        "cc_out": cc_out,
        "L": L,
        "base_luma_mae": base_luma_mae,
        "base_rgb_mae": base_rgb_mae,
        "blur_cache": {},  # sigma_px -> L_blur
    }


def get_blur(frame, sigma_px):
    cache = frame["blur_cache"]
    key = round(sigma_px, 4)
    if key not in cache:
        cache[key] = gaussian_blur(frame["L"], sigma_px)
    return cache[key]


def sweep(frames, k_values, sigma_fracs):
    """Evaluate luma MAE for every (k_pos, k_lift, sigma) on every frame.

    Returns:
      grid: dict keyed by (k_pos, k_lift, sigma_frac) -> {frame_name: luma_mae}
      The sigma is expressed as a fraction of the long edge so it is comparable
      across frames of identical size (both 1024-long here, but kept general).
    """
    grid = {}
    for sf in sigma_fracs:
        # precompute L_blur per frame for this sigma
        blurs = {}
        for fr in frames:
            sigma_px = sf * fr["long_edge"]
            blurs[fr["name"]] = (fr, get_blur(fr, sigma_px))
        for k_pos in k_values:
            for k_lift in k_values:
                key = (round(k_pos, 4), round(k_lift, 4), round(sf, 4))
                per = {}
                for fr in frames:
                    _, L_blur = blurs[fr["name"]]
                    out = apply_operator(fr["cc_out"], fr["L"], L_blur, k_pos, k_lift)
                    per[fr["name"]] = luma_mae(out, fr["preview"])
                grid[key] = per
    return grid


def perframe_optimum(grid, frame_name):
    best_key, best_mae = None, float("inf")
    for key, per in grid.items():
        m = per[frame_name]
        if m < best_mae:
            best_mae, best_key = m, key
    return best_key, best_mae


def shared_optimum(grid, frame_names):
    best_key, best_mean = None, float("inf")
    for key, per in grid.items():
        m = sum(per[n] for n in frame_names) / len(frame_names)
        if m < best_mean:
            best_mean, best_key = m, key
    return best_key, best_mean


def sigma_boundary_scan(frames, k_values, sigma_fracs_ext):
    """The s2.4 operator is meant to be LOCAL (doc-measured surround sigma = 3% of
    the long edge). If the in-grid shared optimum lands on the largest sigma, the
    gate may only be reachable by enlarging the surround until the 'blur' is no
    longer local (it degenerates toward a whole-image average = a near-GLOBAL tone
    nudge, which is the lane colorCurves already owns). This scan reports, per
    sigma, the best SHARED setting (maximizing the WORSE frame's luma-MAE drop --
    the gate-binding metric) plus its full per-channel non-regression, so the
    verdict can state WHERE (and whether-still-local) the gate first clears."""
    rows = []
    for sf in sigma_fracs_ext:
        blurs = {fr["name"]: get_blur(fr, sf * fr["long_edge"]) for fr in frames}
        best = None
        for k_pos in k_values:
            for k_lift in k_values:
                per = {}
                for fr in frames:
                    out = apply_operator(fr["cc_out"], fr["L"], blurs[fr["name"]], k_pos, k_lift)
                    lm = luma_mae(out, fr["preview"])
                    rm = rgb_mae(out, fr["preview"])
                    per[fr["name"]] = {
                        "drop": fr["base_luma_mae"] - lm,
                        "non_regress": all(rm[c] <= fr["base_rgb_mae"][c] + 1e-9 for c in range(3)),
                    }
                worse_drop = min(per[n]["drop"] for n in per)
                if best is None or worse_drop > best["worse_drop"]:
                    best = {"k_pos": round(k_pos, 4), "k_lift": round(k_lift, 4),
                            "worse_drop": worse_drop, "per": per}
        full_gate = (best["worse_drop"] >= 0.5) and all(best["per"][n]["non_regress"] for n in best["per"])
        rows.append({
            "sigma_frac": round(sf, 4),
            "sigma_px_at_1024": round(sf * 1024.0, 1),
            "shared_k_pos": best["k_pos"], "shared_k_lift": best["k_lift"],
            "drop_5580": round(best["per"].get("5580", {}).get("drop", float("nan")), 4),
            "drop_5551": round(best["per"].get("5551", {}).get("drop", float("nan")), 4),
            "worse_frame_drop": round(best["worse_drop"], 4),
            "both_clear_0p5": best["worse_drop"] >= 0.5,
            "full_gate_no_regression": bool(full_gate),
        })
    first_clear = next((r for r in rows if r["full_gate_no_regression"]), None)
    return {"rows": rows, "first_full_gate": first_clear,
            "doc_measured_local_sigma_frac": 0.03}


def sign_finding(k_pos, k_lift):
    """Interpret the SIGN of the winning operator physically.

    detail = L - L_blur (pixel minus surround).
    g > 0 : out = L + g*detail amplifies the pixel's deviation from surround =>
            local-CONTRAST INCREASE (unsharp mask). Brighter-than-surround gets
            brighter, darker-than-surround gets darker.
    g < 0 : out moves the pixel TOWARD its surround => local-contrast REDUCTION /
            SURROUND-FOLLOW. A pixel darker than a bright surround is LIFTED toward
            the surround; a pixel brighter than a dark surround is pulled DOWN.
    g_pos acts on detail>=0 (pixel >= surround); g_lift on detail<0 (pixel <
    surround, i.e. dark subject on lighter field)."""
    def lab(g, region):
        if abs(g) < 1e-9:
            return f"{region}: g={g:+.2f} (identity / no effect)"
        if g > 0:
            return f"{region}: g={g:+.2f} -> CONTRAST-INCREASE (unsharp; amplify deviation from surround)"
        return f"{region}: g={g:+.2f} -> SURROUND-FOLLOW / contrast-reduction (pull pixel toward surround)"

    pos = lab(k_pos, "k_pos (pixel>=surround, highlights-vs-field)")
    lift = lab(k_lift, "k_lift (pixel<surround, dark-subject-on-lighter-field)")

    # Overall character: the dark-subject lift term (k_lift) is the one the doc
    # cares about (the coherent blobs). For a pixel BELOW its surround:
    #   k_lift < 0 -> LIFTED toward surround (shadow recovery / surround-follow).
    #   k_lift > 0 -> pushed DOWN, away from surround (deepens shadows; contrast).
    if k_lift < -1e-9:
        physical = ("WINNING SIGN: NEGATIVE lift gain = SURROUND-FOLLOW / shadow-lift. "
                    "Dark subjects sitting on lighter fields are LIFTED toward their "
                    "surround -- this is the ADL shadow-recovery direction, NOT a "
                    "positive unsharp. The measured positive conditioned slope of "
                    "camera-lift vs surround corresponds to this negative unsharp gain.")
    elif k_lift > 1e-9:
        physical = ("WINNING SIGN: POSITIVE lift gain = CONTRAST-INCREASE (unsharp). "
                    "Dark-on-light subjects are pushed DARKER (deviation from surround "
                    "amplified). This is local-contrast enhancement, the literal "
                    "unsharp-mask reading of the doc's s2.4.")
    else:
        physical = "WINNING SIGN: k_lift ~ 0, no asymmetric lift component."

    return {"k_pos_interpretation": pos, "k_lift_interpretation": lift, "physical": physical}


def write_heatmaps(frame, shared_key, out_dir="/tmp"):
    """Write before/after residual luma heatmaps + corrected_hi PNG at the SHARED
    setting, so halos / over-lift are visually inspectable. Shared vlim so before
    and after are directly comparable."""
    name = frame["name"]
    k_pos, k_lift, sf = shared_key
    sigma_px = sf * frame["long_edge"]
    L_blur = get_blur(frame, sigma_px)
    out = apply_operator(frame["cc_out"], frame["L"], L_blur, k_pos, k_lift)

    R_before = (luma(frame["cc_out"]) - luma(frame["preview"])) * 255.0
    R_after = (luma(out) - luma(frame["preview"])) * 255.0
    vlim = max(1.0, float(np.percentile(np.abs(np.concatenate(
        [R_before.ravel(), R_after.ravel()])), 99.0)))

    paths = {}
    pb = os.path.join(out_dir, f"local_tone_residual_{name}_before.png")
    Image.fromarray(diverging_heatmap(R_before, vlim), "RGB").save(pb)
    paths["before"] = pb
    pa = os.path.join(out_dir, f"local_tone_residual_{name}_after.png")
    Image.fromarray(diverging_heatmap(R_after, vlim), "RGB").save(pa)
    paths["after"] = pa
    pc = os.path.join(out_dir, f"local_tone_corrected_{name}_hi.png")
    Image.fromarray((np.clip(out, 0, 1) * 255.0).round().astype(np.uint8), "RGB").save(pc)
    paths["corrected_hi"] = pc
    paths["heatmap_vlim_255"] = vlim
    return paths


def main():
    frames_cfg = [
        ("5580", "/tmp/optb_dump"),
        ("5551", "/tmp/optb_dump_5551"),
    ]
    frames = [prep_frame(n, d) for n, d in frames_cfg]
    frame_names = [f["name"] for f in frames]

    print("=== baseline (colorCurves lane, saturation OFF) ===")
    for fr in frames:
        print(f"  {fr['name']}: baseline luma MAE = {fr['base_luma_mae']:.4f}  "
              f"rgb MAE = {[round(x,3) for x in fr['base_rgb_mae']]}")

    # sweep grid: k in [-1.0, 1.0] step 0.1 (INCLUDES NEGATIVE); sigma in
    # {0.02, 0.03, 0.05} * long_edge
    k_values = [round(-1.0 + 0.1 * i, 4) for i in range(21)]  # -1.0 .. 1.0
    sigma_fracs = [0.02, 0.03, 0.05]
    print(f"\n=== sweeping k_pos,k_lift in [{k_values[0]},{k_values[-1]}] "
          f"({len(k_values)} steps) x sigma {sigma_fracs} "
          f"= {len(k_values)**2 * len(sigma_fracs)} settings/frame ===")

    grid = sweep(frames, k_values, sigma_fracs)

    # per-frame optima
    perframe = {}
    for fr in frames:
        key, mae = perframe_optimum(grid, fr["name"])
        perframe[fr["name"]] = {"key": key, "luma_mae": mae,
                                "drop": fr["base_luma_mae"] - mae}
        print(f"\n  per-frame OPTIMUM {fr['name']}: k_pos={key[0]} k_lift={key[1]} "
              f"sigma_frac={key[2]}  luma MAE {fr['base_luma_mae']:.4f} -> {mae:.4f}  "
              f"(drop {fr['base_luma_mae']-mae:.4f})")

    # shared optimum (min mean luma MAE)
    shared_key, shared_mean = shared_optimum(grid, frame_names)
    k_pos, k_lift, sf = shared_key
    print(f"\n=== SHARED OPTIMUM (min mean luma MAE): k_pos={k_pos} k_lift={k_lift} "
          f"sigma_frac={sf}  mean luma MAE = {shared_mean:.4f} ===")

    # per-frame results AT the shared setting (luma + rgb)
    frames_out = {}
    gate_per_frame = {}
    for fr in frames:
        sigma_px = sf * fr["long_edge"]
        L_blur = get_blur(fr, sigma_px)
        out = apply_operator(fr["cc_out"], fr["L"], L_blur, k_pos, k_lift)
        lm = luma_mae(out, fr["preview"])
        rm = rgb_mae(out, fr["preview"])
        drop = fr["base_luma_mae"] - lm
        # per-channel regression check: each of R/G/B MAE must be non-worse
        rgb_non_regress = all(rm[c] <= fr["base_rgb_mae"][c] + 1e-9 for c in range(3))
        rgb_delta = [fr["base_rgb_mae"][c] - rm[c] for c in range(3)]  # >0 = improved
        clears_drop = drop >= 0.5
        gate_per_frame[fr["name"]] = {"drop": drop, "clears_0p5": clears_drop,
                                      "rgb_non_regress": rgb_non_regress}
        pf = perframe[fr["name"]]
        frames_out[fr["name"]] = {
            "baseline_luma_mae": fr["base_luma_mae"],
            "baseline_rgb_mae": fr["base_rgb_mae"],
            "perframe_opt": {
                "k_pos": pf["key"][0], "k_lift": pf["key"][1], "sigma": pf["key"][2],
                "luma_mae": pf["luma_mae"], "drop": pf["drop"],
            },
            "at_shared": {
                "luma_mae": lm, "drop": drop, "rgb_mae": rm,
                "rgb_delta_vs_baseline": rgb_delta,
                "rgb_non_regress": rgb_non_regress,
                "clears_0p5_drop": clears_drop,
            },
        }
        print(f"  {fr['name']} @ shared: luma MAE {fr['base_luma_mae']:.4f} -> {lm:.4f} "
              f"(drop {drop:.4f})  rgb MAE {[round(x,3) for x in rm]}  "
              f"non-regress={rgb_non_regress}  clears0.5={clears_drop}")

    # gate: shared clears >=0.5 luma drop on BOTH frames with NO per-channel regr
    gate_pass = all(g["clears_0p5"] and g["rgb_non_regress"] for g in gate_per_frame.values())

    # distance shared-vs-perframe optimum (parametric-confirmed vs divergent)
    perframe_vs_shared = {}
    for fr in frames:
        pk = perframe[fr["name"]]["key"]
        dk_pos = abs(pk[0] - k_pos)
        dk_lift = abs(pk[1] - k_lift)
        dsigma = abs(pk[2] - sf)
        # how much luma MAE is given up by using shared instead of per-frame
        mae_penalty = frames_out[fr["name"]]["at_shared"]["luma_mae"] - perframe[fr["name"]]["luma_mae"]
        perframe_vs_shared[fr["name"]] = {
            "d_k_pos": round(dk_pos, 4), "d_k_lift": round(dk_lift, 4),
            "d_sigma_frac": round(dsigma, 4),
            "luma_mae_penalty_vs_perframe": mae_penalty,
        }
        print(f"  {fr['name']} shared-vs-perframe: dk_pos={dk_pos:.2f} dk_lift={dk_lift:.2f} "
              f"dsigma={dsigma:.3f}  mae penalty={mae_penalty:.4f}")

    # parametric verdict: optima close in k-space and shared barely worse than perframe
    max_dk = max(max(v["d_k_pos"], v["d_k_lift"]) for v in perframe_vs_shared.values())
    max_dsigma = max(v["d_sigma_frac"] for v in perframe_vs_shared.values())
    max_penalty = max(v["luma_mae_penalty_vs_perframe"] for v in perframe_vs_shared.values())
    parametric_confirmed = (max_dk <= 0.3) and (max_dsigma <= 0.02) and (max_penalty <= 0.15)
    parametric_verdict = (
        f"max |dk|={max_dk:.2f}, max dsigma={max_dsigma:.3f}, max luma-MAE penalty="
        f"{max_penalty:.4f}: "
        + ("PARAMETRIC-CONFIRMED (shared optimum tracks both per-frame optima)"
           if parametric_confirmed else
           "DIVERGENT (per-frame optima pull apart -> overfit / needs more data)")
    )

    sign = sign_finding(k_pos, k_lift)

    # sigma-boundary diagnostic: the in-grid shared optimum sits on the largest
    # sigma, so probe larger surrounds to find WHERE the gate first clears and
    # whether that sigma is still a LOCAL operator (vs degenerate near-global).
    sigma_fracs_ext = [0.03, 0.05, 0.08, 0.12, 0.20, 0.35, 0.50]
    print(f"\n=== sigma-boundary scan {sigma_fracs_ext} (gate-binding = worse frame's drop) ===")
    sigma_scan = sigma_boundary_scan(frames, k_values, sigma_fracs_ext)
    for r in sigma_scan["rows"]:
        print(f"  sigma_frac={r['sigma_frac']:.2f} (~{r['sigma_px_at_1024']:.0f}px) "
              f"shared(kp={r['shared_k_pos']},kl={r['shared_k_lift']})  "
              f"drop 5580={r['drop_5580']:.3f} 5551={r['drop_5551']:.3f}  "
              f"worse={r['worse_frame_drop']:.3f}  full_gate={r['full_gate_no_regression']}")
    fc = sigma_scan["first_full_gate"]
    if fc:
        print(f"  -> gate first FULLY clears at sigma_frac={fc['sigma_frac']} "
              f"(~{fc['sigma_px_at_1024']:.0f}px = {fc['sigma_frac']/0.03:.1f}x the "
              f"doc-measured 3% local-surround sigma)")
    else:
        print("  -> gate NEVER fully clears across the extended sigma range")

    # heatmaps + corrected PNGs at the shared setting
    print("\n=== writing before/after heatmaps + corrected_hi PNGs (shared setting) ===")
    heatmaps = {}
    for fr in frames:
        hp = write_heatmaps(fr, shared_key)
        heatmaps[fr["name"]] = hp
        print(f"  {fr['name']}: {hp['before']} | {hp['after']} | {hp['corrected_hi']}")

    out = {
        "k_grid": {"min": k_values[0], "max": k_values[-1], "step": 0.1,
                   "n": len(k_values), "includes_negative": True},
        "sigma_fracs_of_long_edge": sigma_fracs,
        "frames": frames_out,
        "shared_opt": {"k_pos": k_pos, "k_lift": k_lift, "sigma": sf,
                       "mean_luma_mae": shared_mean},
        "gate_pass": bool(gate_pass),
        "gate_detail": gate_per_frame,
        "perframe_vs_shared_distance": perframe_vs_shared,
        "parametric_confirmed": bool(parametric_confirmed),
        "parametric_verdict": parametric_verdict,
        "sign_finding": sign,
        "sigma_boundary_scan": sigma_scan,
        "heatmaps": heatmaps,
    }
    with open("/tmp/local_tone_sweep.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=========================== GATE ===========================")
    print(f"  shared (k_pos,k_lift,sigma) = ({k_pos}, {k_lift}, {sf})")
    for fr in frames:
        g = gate_per_frame[fr["name"]]
        print(f"  {fr['name']}: drop={g['drop']:.4f} (>=0.5? {g['clears_0p5']})  "
              f"no per-channel regression? {g['rgb_non_regress']}")
    print(f"  GATE PASS = {gate_pass}")
    print(f"  {parametric_verdict}")
    print(f"  SIGN: {sign['physical']}")
    print("  wrote /tmp/local_tone_sweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
