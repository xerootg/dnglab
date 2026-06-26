// SPDX-License-Identifier: LGPL-2.1
// Copyright 2025 dnglab contributors

//! Adobe LCP lens correction profile database.
//!
//! This module loads lens correction profiles extracted from Adobe Camera Raw
//! LCP files and provides lookup + interpolation for generating DNG opcodes
//! (WarpRectilinear, FixVignetteRadial) at conversion time.
//!
//! The profiles store Adobe PerspectiveModel v2 coefficients. Conversion to
//! DNG WarpRectilinear coordinate space happens at lookup time based on the
//! actual DNG image dimensions.

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::dng::opcodes;

static LENS_PROFILES_TOML: &str = include_str!(concat!(env!("OUT_DIR"), "/lens_profiles.toml"));

lazy_static! {
  static ref PROFILES_DB: Vec<LensProfile> = parse_profiles().unwrap_or_default();
}

#[derive(Debug, Clone, Deserialize)]
pub struct CalibrationPoint {
  pub focal: f64,
  pub aperture: f64,
  pub flx: f64,
  pub cx: f64,
  pub cy: f64,
  /// Radial distortion coefficients (Adobe PerspectiveModel v2 space)
  pub k1: f64,
  pub k2: f64,
  pub k3: f64,
  /// Vignetting coefficients (optional)
  pub v1: Option<f64>,
  pub v2: Option<f64>,
  pub v3: Option<f64>,
  /// Chromatic aberration radial ScaleFactor per channel (relative to green).
  /// Emitted as a per-plane WarpRectilinear opcode when present.
  #[serde(default)]
  pub ca_red_scale: Option<f64>,
  #[serde(default)]
  pub ca_blue_scale: Option<f64>,
  /// ResidualMeanError from LCP: lower = tighter polynomial fit.
  /// Carried through for provenance; not currently used at lookup time.
  #[serde(default)]
  pub residual_error: Option<f64>,
  /// Subject distance the calibration was captured at (metres).
  #[serde(default)]
  pub focus_distance: Option<f64>,
  /// Optional per-vignette-model centre / focal-length-x overrides. Adobe LCP
  /// files can ship a vignette model whose centre and FocalLengthX differ from
  /// the distortion model's; when present these are used for the FixVignette
  /// radial normalisation instead of `cx`/`cy`/`flx`. Absent in every profile
  /// shipped today, so this is a latent superset over the current data.
  #[serde(default)]
  pub vig_flx: Option<f64>,
  #[serde(default)]
  pub vig_cx: Option<f64>,
  #[serde(default)]
  pub vig_cy: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LensProfile {
  pub lens_name: String,
  pub lens_id: Option<i64>,
  pub crop_factor: f64,
  pub image_width: f64,
  pub image_height: f64,
  pub calibration: Vec<CalibrationPoint>,
}

/// Result of looking up a lens profile for specific shooting parameters.
/// Contains DNG-ready opcode coefficients.
pub struct LensOpcodes {
  pub opcode_list1: Vec<u8>,
  pub opcode_list3: Vec<u8>,
}

/// DNG WarpRectilinear distortion coefficients (single plane), already
/// converted out of Adobe PerspectiveModel v2 space.
#[derive(Debug, Clone, Serialize)]
pub struct DngDistortion {
  /// `[1.0, k1/ratio², k2/ratio⁴, k3/ratio⁶]`.
  pub k: [f64; 4],
  /// Tangential terms (always `[0.0, 0.0]` for Adobe radial models).
  pub kt: [f64; 2],
  pub cx: f64,
  pub cy: f64,
}

/// DNG transverse chromatic-aberration radial scale factors (green = 1.0).
#[derive(Debug, Clone, Serialize)]
pub struct DngTca {
  pub kr: f64,
  pub kb: f64,
}

/// DNG FixVignetteRadial coefficients. The trailing two entries are always
/// `0.0` (Adobe vignette models only carry three radial terms).
#[derive(Debug, Clone, Serialize)]
pub struct DngVignette {
  pub k: [f64; 5],
  pub cx: f64,
  pub cy: f64,
}

/// The full set of DNG-space lens-correction coefficients for one shooting
/// configuration. This is the single canonical Adobe→DNG conversion: the DNG
/// opcode-byte path ([`generate_opcodes`]) and the numeric path consumed by the
/// backend's `/calibration` endpoint (via the `dnglab_py` binding) both derive
/// from it.
///
/// Serialises to exactly the dict shape the frontend's uniform compiler reads:
/// `{distortion:{k,kt,cx,cy}, tca?:{kr,kb}, vignetting?:{k,cx,cy}}`.
#[derive(Debug, Clone, Serialize)]
pub struct DngLensCalibration {
  pub distortion: DngDistortion,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tca: Option<DngTca>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub vignetting: Option<DngVignette>,
}

/// DNG normalisation ratio: `(w·flx) / MaxDistancePointToRect(centre, rect)`.
///
/// `MaxDistancePointToRect` is the farthest corner distance from the optical
/// centre, which equals the half-diagonal only when the centre sits at
/// `(0.5, 0.5)`. Guards a zero max-distance the same way the reference Python
/// implementation did (degenerate centres/dimensions → ratio 1.0).
fn dng_ratio(cx: f64, cy: f64, flx: f64, w: f64, h: f64) -> f64 {
  let cx_px = cx * w;
  let cy_px = cy * h;
  let dcx = cx_px.max(w - cx_px);
  let dcy = cy_px.max(h - cy_px);
  let max_dist = (dcx * dcx + dcy * dcy).sqrt();
  if max_dist <= 0.0 {
    1.0
  } else {
    (w * flx) / max_dist
  }
}

/// Compute DNG-space lens-correction coefficients for the given calibration
/// points and shooting parameters. This is the canonical Adobe PerspectiveModel
/// v2 → DNG WarpRectilinear / FixVignetteRadial conversion.
///
/// `focal` / `aperture` are optional to mirror the EXIF-less case: a `None`
/// focal uses the first calibration point verbatim (no interpolation); a `None`
/// aperture takes the first point at the chosen focal length.
///
/// Returns `None` when there are no calibration points or the output
/// dimensions are zero (matching the backend `get_calibration` guards).
///
/// Note the chromatic-aberration semantics: a `tca` entry is emitted whenever
/// *either* `ca_red_scale` or `ca_blue_scale` is present — even at an identity
/// scale of 1.0 — to preserve the backend `/calibration` API contract. The DNG
/// opcode-byte path keeps its own `>1e-6` identity threshold in
/// [`generate_opcodes`]; that threshold is intentionally *not* applied here.
pub fn calibration_coeffs(
  points: &[CalibrationPoint],
  focal: Option<f64>,
  aperture: Option<f64>,
  width: u32,
  height: u32,
) -> Option<DngLensCalibration> {
  if points.is_empty() || width == 0 || height == 0 {
    return None;
  }
  let w = width as f64;
  let h = height as f64;

  // Distortion. Adobe normalises r by (w·flx); DNG by MaxDistancePointToRect.
  // At the same physical pixel r_dng = r_adobe·ratio, so k_dng = k_adobe/ratioᴺ.
  let dist = interpolate_point(points, focal, aperture);
  let ratio = dng_ratio(dist.cx, dist.cy, dist.flx, w, h);
  let distortion = DngDistortion {
    k: [
      1.0,
      dist.k1 / (ratio * ratio),
      dist.k2 / ratio.powi(4),
      dist.k3 / ratio.powi(6),
    ],
    kt: [0.0, 0.0],
    cx: dist.cx,
    cy: dist.cy,
  };

  // Chromatic aberration: linear per-channel radial scale relative to green.
  // Emitted whenever either scale is present (no identity threshold here).
  let tca = if dist.ca_red_scale.is_some() || dist.ca_blue_scale.is_some() {
    Some(DngTca {
      kr: dist.ca_red_scale.unwrap_or(1.0),
      kb: dist.ca_blue_scale.unwrap_or(1.0),
    })
  } else {
    None
  };

  // Vignetting: uses its own centre/flx overrides when the profile provides
  // them, otherwise the distortion centre/flx.
  let vignetting = match find_vignette_point(points, focal, aperture) {
    Some(vig) if vig.v1.is_some() => {
      let vig_cx = vig.vig_cx.unwrap_or(vig.cx);
      let vig_cy = vig.vig_cy.unwrap_or(vig.cy);
      let vig_flx = vig.vig_flx.unwrap_or(vig.flx);
      let vr = dng_ratio(vig_cx, vig_cy, vig_flx, w, h);
      Some(DngVignette {
        k: [
          vig.v1.unwrap_or(0.0) / vr.powi(2),
          vig.v2.unwrap_or(0.0) / vr.powi(4),
          vig.v3.unwrap_or(0.0) / vr.powi(6),
          0.0,
          0.0,
        ],
        cx: vig_cx,
        cy: vig_cy,
      })
    }
    _ => None,
  };

  Some(DngLensCalibration { distortion, tca, vignetting })
}

/// Look up a lens profile by name and generate DNG opcodes for the given
/// focal length, aperture, and output image dimensions.
pub fn lookup_lens_opcodes(
  lens_name: &str,
  focal_mm: f64,
  aperture: f64,
  dng_width: u32,
  dng_height: u32,
) -> Option<LensOpcodes> {
  let name_lower = lens_name.to_lowercase();
  let profile = PROFILES_DB.iter().find(|p| p.lens_name.to_lowercase() == name_lower)?;
  generate_opcodes(profile, focal_mm, aperture, dng_width, dng_height)
}

/// Look up a lens profile by Adobe LensID and generate DNG opcodes.
pub fn lookup_lens_opcodes_by_id(
  lens_id: i64,
  focal_mm: f64,
  aperture: f64,
  dng_width: u32,
  dng_height: u32,
) -> Option<LensOpcodes> {
  let profile = PROFILES_DB.iter().find(|p| p.lens_id == Some(lens_id))?;
  generate_opcodes(profile, focal_mm, aperture, dng_width, dng_height)
}

fn generate_opcodes(
  profile: &LensProfile,
  focal_mm: f64,
  aperture: f64,
  dng_width: u32,
  dng_height: u32,
) -> Option<LensOpcodes> {
  // Single canonical Adobe→DNG conversion (interpolation + coordinate-space
  // change). The byte encoding below derives entirely from these coefficients.
  let cal = calibration_coeffs(
    &profile.calibration,
    Some(focal_mm),
    Some(aperture),
    dng_width,
    dng_height,
  )?;

  log::debug!(
    "LCP WarpRectilinear for '{}' @ {:.0}mm f/{:.1}: k1={:.6} k2={:.6} k3={:.6}",
    profile.lens_name, focal_mm, aperture, cal.distortion.k[1], cal.distortion.k[2], cal.distortion.k[3]
  );

  let kr = [cal.distortion.k];
  let kt = [cal.distortion.kt];
  let warp_opcode = opcodes::encode_warp_rectilinear(&kr, &kt, cal.distortion.cx, cal.distortion.cy, opcodes::FLAG_OPTIONAL);
  let mut list3_opcodes: Vec<Vec<u8>> = vec![warp_opcode];

  // Chromatic aberration: Adobe stores a radial ScaleFactor per channel
  // relative to green. DNG readers can express this as a per-plane
  // WarpRectilinear (3 planes: R, G, B) applied after the distortion opcode.
  // Only emit when the scales actually differ from identity — tiny amounts
  // of CA correction aren't worth the extra sampling pass. (The coefficient
  // struct carries `tca` whenever present; the identity threshold lives only
  // here, on the byte path.)
  if let Some(tca) = &cal.tca {
    let sr = tca.kr;
    let sb = tca.kb;
    if (sr - 1.0).abs() > 1e-6 || (sb - 1.0).abs() > 1e-6 {
      let ca_kr = [
        [sr, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [sb, 0.0, 0.0, 0.0],
      ];
      let ca_kt = [[0.0_f64, 0.0_f64]; 3];
      let ca_opcode = opcodes::encode_warp_rectilinear(&ca_kr, &ca_kt, cal.distortion.cx, cal.distortion.cy, opcodes::FLAG_OPTIONAL);
      list3_opcodes.push(ca_opcode);
      log::debug!(
        "LCP CA for '{}' @ {:.0}mm: r_scale={:.6} b_scale={:.6}",
        profile.lens_name, focal_mm, sr, sb
      );
    }
  }

  let opcode_list3 = opcodes::encode_opcode_list(&list3_opcodes);

  // Vignetting correction
  let opcode_list1 = if let Some(vig) = &cal.vignetting {
    log::debug!(
      "LCP FixVignetteRadial for '{}' @ {:.0}mm f/{:.1}: k0={:.6} k1={:.6} k2={:.6}",
      profile.lens_name, focal_mm, aperture, vig.k[0], vig.k[1], vig.k[2]
    );
    let vig_opcode = opcodes::encode_fix_vignette_radial(vig.k[0], vig.k[1], vig.k[2], vig.k[3], vig.k[4], vig.cx, vig.cy, opcodes::FLAG_OPTIONAL);
    opcodes::encode_opcode_list(&[vig_opcode])
  } else {
    Vec::new()
  };

  Some(LensOpcodes { opcode_list1, opcode_list3 })
}

/// Find the best calibration point by interpolating between the two closest
/// focal lengths, then picking the closest aperture.
///
/// `focal == None` returns the first calibration point verbatim (the EXIF-less
/// case); `aperture == None` takes the first point at the chosen focal length.
fn interpolate_point(points: &[CalibrationPoint], focal: Option<f64>, aperture: Option<f64>) -> CalibrationPoint {
  let focal = match focal {
    None => return points[0].clone(),
    Some(f) => f,
  };

  // Get unique focal lengths
  let mut focals: Vec<f64> = points.iter().map(|p| p.focal).collect();
  focals.sort_by(|a, b| a.partial_cmp(b).unwrap());
  focals.dedup();

  if focals.len() == 1 {
    // Single focal length — just find closest aperture
    return find_closest_aperture(points, focal, aperture);
  }

  // Find the two bracketing focal lengths
  let (f_lo, f_hi) = bracket(&focals, focal);

  if (f_lo - f_hi).abs() < 0.01 {
    return find_closest_aperture(points, f_lo, aperture);
  }

  // Get the best point at each focal length
  let p_lo = find_closest_aperture(points, f_lo, aperture);
  let p_hi = find_closest_aperture(points, f_hi, aperture);

  // Linear interpolation factor
  let t = (focal - f_lo) / (f_hi - f_lo);

  CalibrationPoint {
    focal,
    aperture: aperture.unwrap_or(p_lo.aperture),
    flx: lerp(p_lo.flx, p_hi.flx, t),
    cx: lerp(p_lo.cx, p_hi.cx, t),
    cy: lerp(p_lo.cy, p_hi.cy, t),
    k1: lerp(p_lo.k1, p_hi.k1, t),
    k2: lerp(p_lo.k2, p_hi.k2, t),
    k3: lerp(p_lo.k3, p_hi.k3, t),
    v1: lerp_opt(p_lo.v1, p_hi.v1, t),
    v2: lerp_opt(p_lo.v2, p_hi.v2, t),
    v3: lerp_opt(p_lo.v3, p_hi.v3, t),
    ca_red_scale: lerp_opt(p_lo.ca_red_scale, p_hi.ca_red_scale, t),
    ca_blue_scale: lerp_opt(p_lo.ca_blue_scale, p_hi.ca_blue_scale, t),
    residual_error: None,
    focus_distance: None,
    vig_flx: lerp_opt(p_lo.vig_flx, p_hi.vig_flx, t),
    vig_cx: lerp_opt(p_lo.vig_cx, p_hi.vig_cx, t),
    vig_cy: lerp_opt(p_lo.vig_cy, p_hi.vig_cy, t),
  }
}

/// Find the calibration point with the closest aperture at the closest focal
/// length. `aperture == None` takes the first point at that focal length.
fn find_closest_aperture(points: &[CalibrationPoint], focal: f64, aperture: Option<f64>) -> CalibrationPoint {
  // Filter to points at this focal length (or closest)
  let at_focal: Vec<&CalibrationPoint> = points.iter().filter(|p| (p.focal - focal).abs() < 0.5).collect();
  if at_focal.is_empty() {
    // Fallback: find closest focal
    return points
      .iter()
      .min_by(|a, b| {
        let da = (a.focal - focal).abs();
        let db = (b.focal - focal).abs();
        da.partial_cmp(&db).unwrap()
      })
      .cloned()
      .unwrap();
  }
  match aperture {
    None => at_focal[0].clone(),
    // Find closest aperture
    Some(ap) => at_focal
      .iter()
      .min_by(|a, b| {
        let da = (a.aperture - ap).abs();
        let db = (b.aperture - ap).abs();
        da.partial_cmp(&db).unwrap()
      })
      .cloned()
      .cloned()
      .unwrap(),
  }
}

/// Find the best vignetting calibration point (must have v1 present).
///
/// `focal == None` returns the first vignetting point verbatim.
fn find_vignette_point(points: &[CalibrationPoint], focal: Option<f64>, aperture: Option<f64>) -> Option<CalibrationPoint> {
  let vig: Vec<CalibrationPoint> = points.iter().filter(|p| p.v1.is_some()).cloned().collect();
  if vig.is_empty() {
    return None;
  }

  let focal = match focal {
    None => return Some(vig[0].clone()),
    Some(f) => f,
  };

  // Get unique focal lengths with vignetting
  let mut focals: Vec<f64> = vig.iter().map(|p| p.focal).collect();
  focals.sort_by(|a, b| a.partial_cmp(b).unwrap());
  focals.dedup();

  let (f_lo, f_hi) = bracket(&focals, focal);

  if (f_lo - f_hi).abs() < 0.01 {
    return Some(find_closest_aperture(&vig, f_lo, aperture));
  }

  // Interpolate between focal lengths
  let p_lo = find_closest_aperture(&vig, f_lo, aperture);
  let p_hi = find_closest_aperture(&vig, f_hi, aperture);

  let t = (focal - f_lo) / (f_hi - f_lo);
  Some(CalibrationPoint {
    focal,
    aperture: aperture.unwrap_or(p_lo.aperture),
    flx: lerp(p_lo.flx, p_hi.flx, t),
    cx: lerp(p_lo.cx, p_hi.cx, t),
    cy: lerp(p_lo.cy, p_hi.cy, t),
    k1: lerp(p_lo.k1, p_hi.k1, t),
    k2: lerp(p_lo.k2, p_hi.k2, t),
    k3: lerp(p_lo.k3, p_hi.k3, t),
    v1: lerp_opt(p_lo.v1, p_hi.v1, t),
    v2: lerp_opt(p_lo.v2, p_hi.v2, t),
    v3: lerp_opt(p_lo.v3, p_hi.v3, t),
    ca_red_scale: lerp_opt(p_lo.ca_red_scale, p_hi.ca_red_scale, t),
    ca_blue_scale: lerp_opt(p_lo.ca_blue_scale, p_hi.ca_blue_scale, t),
    residual_error: None,
    focus_distance: None,
    vig_flx: lerp_opt(p_lo.vig_flx, p_hi.vig_flx, t),
    vig_cx: lerp_opt(p_lo.vig_cx, p_hi.vig_cx, t),
    vig_cy: lerp_opt(p_lo.vig_cy, p_hi.vig_cy, t),
  })
}

/// Find the two values in a sorted list that bracket `target`.
fn bracket(sorted: &[f64], target: f64) -> (f64, f64) {
  if sorted.is_empty() {
    return (0.0, 0.0);
  }
  if target <= sorted[0] {
    return (sorted[0], sorted[0]);
  }
  if target >= sorted[sorted.len() - 1] {
    return (sorted[sorted.len() - 1], sorted[sorted.len() - 1]);
  }
  for w in sorted.windows(2) {
    if target >= w[0] && target <= w[1] {
      return (w[0], w[1]);
    }
  }
  (sorted[0], sorted[sorted.len() - 1])
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
  a + (b - a) * t
}

fn lerp_opt(a: Option<f64>, b: Option<f64>, t: f64) -> Option<f64> {
  match (a, b) {
    (Some(a), Some(b)) => Some(lerp(a, b, t)),
    (Some(a), None) => Some(a),
    (None, Some(b)) => Some(b),
    (None, None) => None,
  }
}

fn parse_profiles() -> Option<Vec<LensProfile>> {
  let toml: toml::Value = LENS_PROFILES_TOML.parse().ok()?;
  let profiles = toml.get("profiles")?.as_array()?;
  let mut result = Vec::with_capacity(profiles.len());
  for p in profiles {
    if let Ok(profile) = toml::from_str::<LensProfile>(&p.to_string()) {
      result.push(profile);
    } else {
      // Try manual parsing for robustness
      if let Some(profile) = parse_profile_manual(p) {
        result.push(profile);
      }
    }
  }
  log::info!("Loaded {} lens correction profiles", result.len());
  Some(result)
}

fn parse_profile_manual(v: &toml::Value) -> Option<LensProfile> {
  let lens_name = v.get("lens_name")?.as_str()?.to_string();
  let lens_id = v.get("lens_id").and_then(|v| v.as_integer());
  let crop_factor = v.get("crop_factor").and_then(|v| v.as_float()).unwrap_or(1.0);
  let image_width = v.get("image_width").and_then(|v| v.as_float()).unwrap_or(0.0);
  let image_height = v.get("image_height").and_then(|v| v.as_float()).unwrap_or(0.0);

  let cals = v.get("calibration")?.as_array()?;
  let mut calibration = Vec::with_capacity(cals.len());
  for c in cals {
    calibration.push(CalibrationPoint {
      focal: c.get("focal").and_then(|v| v.as_float()).unwrap_or(0.0),
      aperture: c.get("aperture").and_then(|v| v.as_float()).unwrap_or(0.0),
      flx: c.get("flx").and_then(|v| v.as_float()).unwrap_or(1.0),
      cx: c.get("cx").and_then(|v| v.as_float()).unwrap_or(0.5),
      cy: c.get("cy").and_then(|v| v.as_float()).unwrap_or(0.5),
      k1: c.get("k1").and_then(|v| v.as_float()).unwrap_or(0.0),
      k2: c.get("k2").and_then(|v| v.as_float()).unwrap_or(0.0),
      k3: c.get("k3").and_then(|v| v.as_float()).unwrap_or(0.0),
      v1: c.get("v1").and_then(|v| v.as_float()),
      v2: c.get("v2").and_then(|v| v.as_float()),
      v3: c.get("v3").and_then(|v| v.as_float()),
      ca_red_scale: c.get("ca_red_scale").and_then(|v| v.as_float()),
      ca_blue_scale: c.get("ca_blue_scale").and_then(|v| v.as_float()),
      residual_error: c.get("residual_error").and_then(|v| v.as_float()),
      focus_distance: c.get("focus_distance").and_then(|v| v.as_float()),
      vig_flx: c.get("vig_flx").and_then(|v| v.as_float()),
      vig_cx: c.get("vig_cx").and_then(|v| v.as_float()),
      vig_cy: c.get("vig_cy").and_then(|v| v.as_float()),
    });
  }

  Some(LensProfile {
    lens_name,
    lens_id,
    crop_factor,
    image_width,
    image_height,
    calibration,
  })
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Build a distortion-only calibration point (all optional fields cleared).
  fn pt(focal: f64, aperture: f64, flx: f64, cx: f64, cy: f64, k1: f64, k2: f64, k3: f64) -> CalibrationPoint {
    CalibrationPoint {
      focal,
      aperture,
      flx,
      cx,
      cy,
      k1,
      k2,
      k3,
      v1: None,
      v2: None,
      v3: None,
      ca_red_scale: None,
      ca_blue_scale: None,
      residual_error: None,
      focus_distance: None,
      vig_flx: None,
      vig_cx: None,
      vig_cy: None,
    }
  }

  /// Parity tolerance vs the reference Python implementation. The two sides
  /// compute the same closed form on bit-identical inputs; only the `x**n`
  /// (C `pow`) vs `x.powi(n)` rounding differs, far below this bound.
  fn approx(got: f64, expected: f64) {
    assert!(
      (got - expected).abs() < 1e-9,
      "expected {expected}, got {got} (diff {})",
      (got - expected).abs()
    );
  }

  // Golden values below were produced by the original Python lcp_db math
  // (backend/utils/lcp_db.py) on the same inputs, captured before deleting it.

  #[test]
  fn sample1_distortion_only() {
    let pts = vec![pt(50.0, 2.8, 1.0, 0.5, 0.5, 0.1, -0.05, 0.02)];
    let cal = calibration_coeffs(&pts, Some(50.0), Some(2.8), 6000, 4000).unwrap();
    approx(cal.distortion.k[0], 1.0);
    approx(cal.distortion.k[1], 0.03611111111111112);
    approx(cal.distortion.k[2], -0.0065200617283950645);
    approx(cal.distortion.k[3], 0.0009417866941015095);
    assert_eq!(cal.distortion.kt, [0.0, 0.0]);
    approx(cal.distortion.cx, 0.5);
    approx(cal.distortion.cy, 0.5);
    assert!(cal.tca.is_none());
    assert!(cal.vignetting.is_none());
  }

  #[test]
  fn sample2_distortion_tca_vignette() {
    let mut p = pt(24.0, 4.0, 1.2, 0.5, 0.5, -0.2, 0.1, -0.03);
    p.v1 = Some(-1.5);
    p.v2 = Some(0.8);
    p.v3 = Some(-0.2);
    p.ca_red_scale = Some(1.0002);
    p.ca_blue_scale = Some(0.9995);
    let pts = vec![p];
    let cal = calibration_coeffs(&pts, Some(24.0), Some(4.0), 4000, 3000).unwrap();
    approx(cal.distortion.k[1], -0.054253472222222224);
    approx(cal.distortion.k[2], 0.007358598120418597);
    approx(cal.distortion.k[3], -0.0005988442480809405);
    let tca = cal.tca.unwrap();
    approx(tca.kr, 1.0002);
    approx(tca.kb, 0.9995);
    let vig = cal.vignetting.unwrap();
    approx(vig.k[0], -0.4069010416666667);
    approx(vig.k[1], 0.058868784963348776);
    approx(vig.k[2], -0.00399229498720627);
    approx(vig.k[3], 0.0);
    approx(vig.k[4], 0.0);
    approx(vig.cx, 0.5);
    approx(vig.cy, 0.5);
  }

  #[test]
  fn sample3_canon_real_point() {
    // Canon EF 100-400mm f/4.5-5.6L IS II USM, focal=100 aperture=5.6, 5640x3752.
    let mut p = pt(100.0, 5.6, 2.987735, 0.5, 0.5, -0.291151, -1.1918, 1.114907);
    p.v1 = Some(-2.182246);
    p.v2 = Some(-101.904841);
    p.v3 = Some(1062.806708);
    let pts = vec![p];
    let cal = calibration_coeffs(&pts, Some(100.0), Some(5.6), 5640, 3752).unwrap();
    approx(cal.distortion.k[1], -0.011762688252819233);
    approx(cal.distortion.k[2], -0.0019452704203300967);
    approx(cal.distortion.k[3], 7.351966932995161e-05);
    assert!(cal.tca.is_none());
    let vig = cal.vignetting.unwrap();
    approx(vig.k[0], -0.088164146401564);
    approx(vig.k[1], -0.16633031791050654);
    approx(vig.k[2], 0.07008404982102942);
  }

  #[test]
  fn none_focal_uses_first_point() {
    // focal=None / aperture=None must use the first point verbatim.
    let mut p = pt(24.0, 4.0, 1.2, 0.5, 0.5, -0.2, 0.1, -0.03);
    p.v1 = Some(-1.5);
    p.v2 = Some(0.8);
    p.v3 = Some(-0.2);
    p.ca_red_scale = Some(1.0002);
    p.ca_blue_scale = Some(0.9995);
    let pts = vec![p];
    let cal = calibration_coeffs(&pts, None, None, 6000, 4000).unwrap();
    approx(cal.distortion.k[1], -0.05015432098765433);
    approx(cal.distortion.k[2], 0.006288639784331658);
    approx(cal.distortion.k[3], -0.00047310368747865484);
    let vig = cal.vignetting.unwrap();
    approx(vig.k[0], -0.37615740740740744);
  }

  #[test]
  fn guard_cases_return_none() {
    let pts = vec![pt(50.0, 2.8, 1.0, 0.5, 0.5, 0.1, -0.05, 0.02)];
    assert!(calibration_coeffs(&pts, Some(50.0), Some(2.8), 0, 4000).is_none());
    assert!(calibration_coeffs(&pts, Some(50.0), Some(2.8), 6000, 0).is_none());
    assert!(calibration_coeffs(&[], Some(50.0), Some(2.8), 6000, 4000).is_none());
  }

  #[test]
  fn tca_emitted_at_identity_scale() {
    // Python emits tca whenever a CA scale is present, even at exactly 1.0.
    let mut p = pt(50.0, 2.8, 1.0, 0.5, 0.5, 0.1, -0.05, 0.02);
    p.ca_red_scale = Some(1.0);
    let pts = vec![p];
    let cal = calibration_coeffs(&pts, Some(50.0), Some(2.8), 6000, 4000).unwrap();
    let tca = cal.tca.expect("tca present even at identity red scale");
    approx(tca.kr, 1.0);
    approx(tca.kb, 1.0);
  }

  #[test]
  fn serialized_dict_shape() {
    // The JSON the binding hands to Python must use exactly these keys, omitting
    // absent optional sub-dicts.
    let pts = vec![pt(50.0, 2.8, 1.0, 0.5, 0.5, 0.1, -0.05, 0.02)];
    let cal = calibration_coeffs(&pts, Some(50.0), Some(2.8), 6000, 4000).unwrap();
    let json = serde_json::to_value(&cal).unwrap();
    assert!(json.get("distortion").is_some());
    assert!(json["distortion"].get("k").is_some());
    assert!(json["distortion"].get("kt").is_some());
    assert!(json["distortion"].get("cx").is_some());
    assert!(json["distortion"].get("cy").is_some());
    assert!(json.get("tca").is_none(), "tca key omitted when absent");
    assert!(json.get("vignetting").is_none(), "vignetting key omitted when absent");
  }
}
