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
use serde::Deserialize;

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
  pub opcode_list2: Vec<u8>,
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
  if profile.calibration.is_empty() {
    return None;
  }

  // Find the best calibration point for distortion (closest focal length,
  // then closest aperture among those)
  let dist_point = interpolate_point(&profile.calibration, focal_mm, aperture);

  // Convert Adobe coefficients to DNG WarpRectilinear coordinate space.
  //
  // Adobe normalizes r by (width * FocalLengthX).
  // DNG WarpRectilinear normalizes r by max(width, height).
  //
  // For the polynomial 1 + k1*r² + k2*r⁴ + k3*r⁶:
  //   r_dng = r_adobe * (width * flx) / max(width, height)
  //
  // For landscape images (width >= height): ratio = flx
  // For portrait images: ratio = width * flx / height
  let max_dim = dng_width.max(dng_height) as f64;
  let ratio = (dng_width as f64 * dist_point.flx) / max_dim;

  let kr1 = dist_point.k1 * ratio * ratio;
  let kr2 = dist_point.k2 * ratio.powi(4);
  let kr3 = dist_point.k3 * ratio.powi(6);

  log::debug!(
    "LCP WarpRectilinear for '{}' @ {:.0}mm f/{:.1}: k1={:.6} k2={:.6} k3={:.6} (ratio={:.4})",
    profile.lens_name, focal_mm, aperture, kr1, kr2, kr3, ratio
  );

  let kr = [[1.0_f64, kr1, kr2, kr3]];
  let kt = [[0.0_f64, 0.0_f64]];
  let warp_opcode = opcodes::encode_warp_rectilinear(&kr, &kt, dist_point.cx, dist_point.cy, opcodes::FLAG_OPTIONAL);
  let opcode_list2 = opcodes::encode_opcode_list(&[warp_opcode]);

  // Vignetting correction
  let opcode_list1 = if let Some(vig_point) = find_vignette_point(&profile.calibration, focal_mm, aperture) {
    // Vignette model uses the same coordinate conversion.
    // FixVignetteRadial normalizes r to 1.0 at the image corner (half-diagonal).
    // Adobe's vignette model also uses FocalLengthX normalization.
    //
    // Adobe vignette radius: r_adobe = dist_from_center / (width * flx)
    // DNG vignette radius:   r_dng = dist_from_center / half_diag
    //   where half_diag = sqrt(w² + h²) / 2
    //
    // r_dng = r_adobe * (width * flx) / half_diag
    // ratio_vig = (width * flx) / half_diag
    let w = dng_width as f64;
    let h = dng_height as f64;
    let half_diag = (w * w + h * h).sqrt() / 2.0;
    let vig_flx = vig_point.flx;
    let vig_ratio = (w * vig_flx) / half_diag;

    let vk0 = vig_point.v1.unwrap_or(0.0) * vig_ratio.powi(2);
    let vk1 = vig_point.v2.unwrap_or(0.0) * vig_ratio.powi(4);
    let vk2 = vig_point.v3.unwrap_or(0.0) * vig_ratio.powi(6);

    log::debug!(
      "LCP FixVignetteRadial for '{}' @ {:.0}mm f/{:.1}: k0={:.6} k1={:.6} k2={:.6}",
      profile.lens_name, focal_mm, aperture, vk0, vk1, vk2
    );

    let vig_opcode = opcodes::encode_fix_vignette_radial(vk0, vk1, vk2, 0.0, 0.0, vig_point.cx, vig_point.cy, opcodes::FLAG_OPTIONAL);
    opcodes::encode_opcode_list(&[vig_opcode])
  } else {
    Vec::new()
  };

  Some(LensOpcodes { opcode_list1, opcode_list2 })
}

/// Find the best calibration point by interpolating between the two closest
/// focal lengths, then picking the closest aperture.
fn interpolate_point(points: &[CalibrationPoint], focal: f64, aperture: f64) -> CalibrationPoint {
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
    aperture,
    flx: lerp(p_lo.flx, p_hi.flx, t),
    cx: lerp(p_lo.cx, p_hi.cx, t),
    cy: lerp(p_lo.cy, p_hi.cy, t),
    k1: lerp(p_lo.k1, p_hi.k1, t),
    k2: lerp(p_lo.k2, p_hi.k2, t),
    k3: lerp(p_lo.k3, p_hi.k3, t),
    v1: lerp_opt(p_lo.v1, p_hi.v1, t),
    v2: lerp_opt(p_lo.v2, p_hi.v2, t),
    v3: lerp_opt(p_lo.v3, p_hi.v3, t),
  }
}

/// Find the calibration point with the closest aperture at the closest focal length.
fn find_closest_aperture(points: &[CalibrationPoint], focal: f64, aperture: f64) -> CalibrationPoint {
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
  // Find closest aperture
  at_focal
    .iter()
    .min_by(|a, b| {
      let da = (a.aperture - aperture).abs();
      let db = (b.aperture - aperture).abs();
      da.partial_cmp(&db).unwrap()
    })
    .cloned()
    .cloned()
    .unwrap()
}

/// Find the best vignetting calibration point (must have v1 present).
fn find_vignette_point(points: &[CalibrationPoint], focal: f64, aperture: f64) -> Option<CalibrationPoint> {
  let vig_points: Vec<&CalibrationPoint> = points.iter().filter(|p| p.v1.is_some()).collect();
  if vig_points.is_empty() {
    return None;
  }

  // Get unique focal lengths with vignetting
  let mut focals: Vec<f64> = vig_points.iter().map(|p| p.focal).collect();
  focals.sort_by(|a, b| a.partial_cmp(b).unwrap());
  focals.dedup();

  let (f_lo, f_hi) = bracket(&focals, focal);

  if (f_lo - f_hi).abs() < 0.01 {
    let at_focal: Vec<&&CalibrationPoint> = vig_points.iter().filter(|p| (p.focal - f_lo).abs() < 0.5).collect();
    return at_focal
      .iter()
      .min_by(|a, b| {
        let da = (a.aperture - aperture).abs();
        let db = (b.aperture - aperture).abs();
        da.partial_cmp(&db).unwrap()
      })
      .map(|p| (**p).clone());
  }

  // Interpolate between focal lengths
  let p_lo = vig_points
    .iter()
    .filter(|p| (p.focal - f_lo).abs() < 0.5)
    .min_by(|a, b| (a.aperture - aperture).abs().partial_cmp(&(b.aperture - aperture).abs()).unwrap())?;
  let p_hi = vig_points
    .iter()
    .filter(|p| (p.focal - f_hi).abs() < 0.5)
    .min_by(|a, b| (a.aperture - aperture).abs().partial_cmp(&(b.aperture - aperture).abs()).unwrap())?;

  let t = (focal - f_lo) / (f_hi - f_lo);
  Some(CalibrationPoint {
    focal,
    aperture,
    flx: lerp(p_lo.flx, p_hi.flx, t),
    cx: lerp(p_lo.cx, p_hi.cx, t),
    cy: lerp(p_lo.cy, p_hi.cy, t),
    k1: lerp(p_lo.k1, p_hi.k1, t),
    k2: lerp(p_lo.k2, p_hi.k2, t),
    k3: lerp(p_lo.k3, p_hi.k3, t),
    v1: lerp_opt(p_lo.v1, p_hi.v1, t),
    v2: lerp_opt(p_lo.v2, p_hi.v2, t),
    v3: lerp_opt(p_lo.v3, p_hi.v3, t),
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
