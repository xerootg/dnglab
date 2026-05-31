//! Vendor-neutral camera "recipe" — the in-body look (Nikon Picture Control,
//! Olympus Picture Mode, …) normalized into one schema so common infrastructure
//! (the DNG) can carry it and editors can seed slider defaults from it.
//!
//! See `docs/camera-recipes.md` for the full design. Key rules:
//!
//! * Every field is optional (absent = "camera didn't specify / leave neutral").
//! * Scalar adjustments are normalized to neutral-centered conventions
//!   (`0.0` = neutral, range roughly `[-1, 1]`) so the editor mapping is uniform
//!   across vendors. The raw vendor values are preserved in [`Recipe::extras`].
//! * This is a *display-referred, editable* description (slider lane), NOT a
//!   baked profile. Producers must not also bake it into `ProfileToneCurve`
//!   (the "one-lane rule").
//!
//! The struct serializes to the compact JSON embedded as `lb:recipe` in the
//! DNG XMP packet.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Current `Recipe` schema version. Bump when the shape changes in a way that
/// needs migration on the consumer side. Mirrors RawTherapee's `ppVersion`
/// approach (a mandatory version field that gates load-time migration).
pub const RECIPE_SCHEMA_VERSION: u32 = 1;

/// How to interpret a [`ToneCurve::points`] control-point list. Values mirror
/// RawTherapee's `DiagonalCurveType` so the encoding is battle-tested.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CurveType {
  /// Identity / no curve.
  Empty,
  /// Smooth spline through the control points (Catmull-Rom-ish).
  Spline,
  /// Parametric (shadow/light/etc. weighted) — reserved; rarely emitted.
  Parametric,
  /// Straight linear segments between control points.
  Linear,
}

impl Default for CurveType {
  fn default() -> Self {
    CurveType::Empty
  }
}

/// A tone curve as control points in `[0,1] × [0,1]`, scene-linear-input →
/// display-output. Empty `points` with `Empty` type = identity.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToneCurve {
  #[serde(rename = "type")]
  pub curve_type: CurveType,
  /// `[x0, y0, x1, y1, …]` flattened pairs in `[0,1]`.
  pub points: Vec<f32>,
}

impl ToneCurve {
  pub fn is_empty(&self) -> bool {
    self.curve_type == CurveType::Empty || self.points.len() < 4
  }
}

/// Black/white input and output levels plus a gamma — the "levels" primitive
/// (Nikon Flexible Color, generic levels tools). Neutral = `0/1/0/1`, gamma 1.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Levels {
  /// Input black point, `[0,1]`. Neutral 0.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub black_in: Option<f32>,
  /// Input white point, `[0,1]`. Neutral 1.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub white_in: Option<f32>,
  /// Output black point, `[0,1]`. Neutral 0.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub black_out: Option<f32>,
  /// Output white point, `[0,1]`. Neutral 1.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub white_out: Option<f32>,
  /// Gamma exponent, `[0.05, 6.0]`. Neutral 1.0.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub gamma: Option<f32>,
}

/// Split-toning / monochrome toning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Toning {
  pub enabled: bool,
  /// Toning hue in degrees `[0,360)` (or `[-180,180]`).
  #[serde(skip_serializing_if = "Option::is_none")]
  pub hue: Option<f32>,
  /// Toning saturation/strength, `[0,1]`.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub saturation: Option<f32>,
}

/// The normalized, vendor-neutral recipe. All look-bearing fields optional.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Recipe {
  /// Schema version (always written; see [`RECIPE_SCHEMA_VERSION`]).
  pub version: u32,

  /// Provenance, `"<vendor>:<base/mode name>"`, e.g. `"nikon:FlexibleColor"`.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub source: Option<String>,
  /// Vendor version + recipe name (free text), e.g. `"0310/KG200"`.
  #[serde(rename = "sourceParams", skip_serializing_if = "Option::is_none")]
  pub source_params: Option<String>,

  // --- tone (the dominant term) ---
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tone: Option<ToneCurve>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub levels: Option<Levels>,

  /// Per-channel R/G/B curves in **display space** (sRGB-display input →
  /// sRGB-display output, control points in `[0,1]×[0,1]`). Unlike [`tone`]
  /// (scene-linear, the picture-control look), these are *measured* at convert
  /// time as the transform from a neutral develop to the camera's own embedded
  /// JPEG preview — they capture the in-body look (tone + colour + WB + Active
  /// D-Lighting's global component) in one portable primitive. The editor seeds
  /// its per-channel RGB curves (`colorRed/colorGreen/colorBlue`) from these,
  /// which it applies in display space at the end of its pipeline — so the
  /// pristine-open render matches the embedded JPEG. `[R, G, B]` order; any
  /// channel may be `Empty` (identity). See `docs/camera-recipes.md`.
  #[serde(rename = "colorCurves", skip_serializing_if = "Option::is_none")]
  pub color_curves: Option<[ToneCurve; 3]>,

  // --- scalar adjustments (neutral 0; roughly [-1,1] unless noted) ---
  /// Exposure/brightness seed in EV stops (neutral 0). Carries the in-body
  /// auto-brightening that is NOT part of the tone curve — e.g. Nikon Active
  /// D-Lighting's shadow lift, mapped to an EV bump per level. The editor
  /// seeds its exposure slider from this. See `docs/camera-recipes.md`.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub exposure: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub contrast: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub brightness: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub highlights: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub shadows: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub saturation: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub vibrance: Option<f32>,
  /// Hue rotation in degrees `[-180,180]`. Neutral 0.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub hue: Option<f32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub clarity: Option<f32>,
  /// Sharpening amount `[0,1]`. Neutral 0.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub sharpening: Option<f32>,

  // --- toning / monochrome ---
  #[serde(skip_serializing_if = "Option::is_none")]
  pub toning: Option<Toning>,
  /// True if the base look is monochrome.
  #[serde(skip_serializing_if = "Option::is_none")]
  pub monochrome: Option<bool>,

  /// Raw, non-portable vendor fields kept verbatim for fidelity/debugging
  /// (e.g. Nikon viewpoint list, filter id, original slider integers). String
  /// values keep this serde-simple and schema-stable.
  #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
  pub extras: BTreeMap<String, String>,
}

impl Recipe {
  /// A fresh recipe stamped with the current schema version and a `source`.
  pub fn new(source: impl Into<String>) -> Self {
    Recipe {
      version: RECIPE_SCHEMA_VERSION,
      source: Some(source.into()),
      ..Default::default()
    }
  }

  /// Serialize to the compact JSON embedded as `lb:recipe` in the DNG XMP.
  pub fn to_json(&self) -> Result<String, serde_json::Error> {
    serde_json::to_string(self)
  }

  /// Parse a recipe back from an XMP packet that carries an `lb:recipe`
  /// element (as written by the DNG converter). Returns `None` when the packet
  /// has no `lb:recipe` or it fails to parse. Inverse of the producer in
  /// `dng/convert.rs`. The element body is XML-entity-unescaped before JSON
  /// decoding (mirrors the producer's `&amp;`/`&lt;`/`&gt;` escaping).
  pub fn from_xmp(xmp: &[u8]) -> Option<Recipe> {
    let s = std::str::from_utf8(xmp).ok()?;
    Self::from_xmp_str(s)
  }

  /// String form of [`Recipe::from_xmp`].
  pub fn from_xmp_str(s: &str) -> Option<Recipe> {
    // Element name may carry the `lb:` prefix (as we emit) or, if a future
    // serializer changes the prefix, fall back to a suffix match on `:recipe>`.
    let open_tag = "<lb:recipe>";
    let close_tag = "</lb:recipe>";
    let (start, body_from) = if let Some(i) = s.find(open_tag) {
      (i, i + open_tag.len())
    } else {
      // Generic: find any `…:recipe>` opening tag.
      let i = s.find(":recipe>")?;
      let body = i + ":recipe>".len();
      // Walk back to the '<' that begins this element to keep the close search aligned.
      let lt = s[..i].rfind('<')?;
      (lt, body)
    };
    let _ = start;
    let end = s[body_from..].find(close_tag).map(|o| body_from + o)?;
    let body = &s[body_from..end];
    let json = body.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&");
    serde_json::from_str(&json).ok()
  }

  /// True if the recipe carries no look-bearing data (only version/source).
  /// Producers should skip emitting an empty recipe.
  pub fn is_meaningful(&self) -> bool {
    self.tone.as_ref().map(|t| !t.is_empty()).unwrap_or(false)
      || self.color_curves.as_ref().map(|c| c.iter().any(|t| !t.is_empty())).unwrap_or(false)
      || self.levels.is_some()
      || self.exposure.is_some()
      || self.contrast.is_some()
      || self.brightness.is_some()
      || self.highlights.is_some()
      || self.shadows.is_some()
      || self.saturation.is_some()
      || self.vibrance.is_some()
      || self.hue.is_some()
      || self.clarity.is_some()
      || self.sharpening.is_some()
      || self.toning.is_some()
      || self.monochrome.is_some()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn empty_recipe_not_meaningful() {
    let r = Recipe::new("nikon:Standard");
    assert!(!r.is_meaningful());
  }

  #[test]
  fn scalar_makes_meaningful() {
    let mut r = Recipe::new("nikon:Vivid");
    r.saturation = Some(0.3);
    assert!(r.is_meaningful());
  }

  #[test]
  fn xmp_round_trip() {
    let mut r = Recipe::new("nikon:FLEXIBLE COLOR");
    r.source_params = Some("0310/KG200".to_string());
    r.tone = Some(ToneCurve { curve_type: CurveType::Spline, points: vec![0.0, 0.0, 0.5, 0.55, 1.0, 1.0] });
    r.saturation = Some(0.2);
    r.extras.insert("nikon.pictureControl.base".to_string(), "FLEXIBLE COLOR".to_string());
    let json = r.to_json().unwrap();
    // Wrap exactly as the DNG converter does (with XML escaping).
    let esc = json.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
    let xmp = format!(
      "<?xpacket?><x:xmpmeta><rdf:RDF><rdf:Description xmlns:lb=\"https://lightbox.photo/ns/recipe/1.0/\"><lb:recipe>{}</lb:recipe></rdf:Description></rdf:RDF></x:xmpmeta>",
      esc
    );
    let back = Recipe::from_xmp(xmp.as_bytes()).expect("parse");
    assert_eq!(back, r);
  }

  #[test]
  fn xmp_absent_returns_none() {
    let xmp = b"<x:xmpmeta><rdf:RDF><rdf:Description crs:Contrast=\"0\"/></rdf:RDF></x:xmpmeta>";
    assert!(Recipe::from_xmp(xmp).is_none());
  }

  #[test]
  fn tone_identity_not_meaningful() {
    let mut r = Recipe::new("x");
    r.tone = Some(ToneCurve { curve_type: CurveType::Empty, points: vec![] });
    assert!(!r.is_meaningful());
    r.tone = Some(ToneCurve { curve_type: CurveType::Spline, points: vec![0.0, 0.0, 1.0, 1.0] });
    assert!(r.is_meaningful());
  }
}
