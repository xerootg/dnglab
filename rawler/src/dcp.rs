// SPDX-License-Identifier: LGPL-2.1
// Copyright 2024 RAW-Manager contributors

//! DNG Camera Profile (DCP) loader.
//!
//! Reads a `.dcp` file (which is a standard TIFF) and exposes the
//! color-profile tags so they can be copied into an output DNG.
//!
//! Tags extracted:
//!   - UniqueCameraModel     (50708) — used for matching only
//!   - CalibrationIlluminant1/2 (50778/50779)
//!   - ColorMatrix1/2        (50721/50722)
//!   - ForwardMatrix1/2      (50964/50965)
//!   - ProfileHueSatMapDims  (50937)
//!   - ProfileHueSatMapData1 (50938)
//!   - ProfileHueSatMapData2 (50939)
//!   - ProfileToneCurve      (50940)
//!   - ProfileName           (50936)
//!   - ProfileEmbedPolicy    (50941)
//!   - ProfileCopyright      (50942)

use std::path::{Path, PathBuf};

use crate::{
  formats::tiff::{GenericTiffReader, IFD, Value, reader::TiffReader},
  tags::DngTag,
};

/// Tags from a DCP file that we want to copy into the output DNG.
///
/// All fields are `Option` because individual profiles may omit them.
#[derive(Debug, Clone)]
pub struct DcpProfile {
  /// The raw IFD loaded from the DCP file.  We keep the whole IFD so the
  /// caller can use `ifd.value_iter()` with a tag filter to copy exactly
  /// what it needs.
  pub ifd: IFD,
}

impl DcpProfile {
  /// Read a DCP file from `path` and return the parsed profile.
  pub fn load(path: &Path) -> crate::Result<Self> {
    let bytes = std::fs::read(path)?;
    let tiff = GenericTiffReader::new_with_buffer(bytes, 0, 0, None)?;
    let ifd = tiff.root_ifd().clone();
    Ok(Self { ifd })
  }

  /// Returns `true` if this profile has at least a ForwardMatrix1.
  pub fn has_forward_matrix(&self) -> bool {
    self.ifd.get_entry(DngTag::ForwardMatrix1).is_some()
  }

  /// Returns the `ProfileName` string from the DCP, if present.
  pub fn profile_name(&self) -> Option<String> {
    self
      .ifd
      .get_entry(DngTag::ProfileName)
      .and_then(|e| e.value.as_string().cloned())
  }

  /// Returns the `BaselineExposureOffset` from the DCP as an f64 EV value.
  ///
  /// Adobe bakes this offset into the output DNG's `BaselineExposure` tag
  /// rather than writing a separate `BaselineExposureOffset` tag.
  pub fn baseline_exposure_offset(&self) -> Option<f64> {
    self.ifd.get_entry(DngTag::BaselineExposureOffset).and_then(|e| match &e.value {
      Value::SRational(v) if !v.is_empty() && v[0].d != 0 => Some(v[0].n as f64 / v[0].d as f64),
      Value::Rational(v) if !v.is_empty() && v[0].d != 0 => Some(v[0].n as f64 / v[0].d as f64),
      _ => None,
    })
  }
}

/// Tags from a DCP that we propagate into the output DNG root IFD.
///
/// UniqueCameraModel is deliberately excluded — the rawler conversion
/// already writes the correct model string for the camera being converted.
const DCP_COPY_TAGS: &[u16] = &[
  DngTag::CalibrationIlluminant1 as u16,
  DngTag::CalibrationIlluminant2 as u16,
  DngTag::ColorMatrix1 as u16,
  DngTag::ColorMatrix2 as u16,
  DngTag::ForwardMatrix1 as u16,
  DngTag::ForwardMatrix2 as u16,
  DngTag::ProfileHueSatMapDims as u16,
  DngTag::ProfileHueSatMapData1 as u16,
  DngTag::ProfileHueSatMapData2 as u16,
  DngTag::ProfileHueSatMapData3 as u16,
  DngTag::ProfileHueSatMapEncoding as u16,
  DngTag::ProfileLookTableDims as u16,
  DngTag::ProfileLookTableData as u16,
  DngTag::ProfileLookTableEncoding as u16,
  DngTag::ProfileToneCurve as u16,
  DngTag::ProfileName as u16,
  DngTag::ProfileEmbedPolicy as u16,
  DngTag::ProfileCopyright as u16,
];

impl DcpProfile {
  /// Returns an iterator over the subset of DCP tags that should be
  /// copied into the output DNG root IFD.
  pub fn copy_tags_iter(&self) -> impl Iterator<Item = (&u16, &Value)> {
    self.ifd.value_iter().filter(|(tag, _)| DCP_COPY_TAGS.contains(tag))
  }
}

/// Resolve a DCP file path given a profiles directory and a camera's
/// `UniqueCameraModel` string (e.g. `"Nikon D600"`).
///
/// `picture_style` is an optional lowercase hint from the camera's
/// Picture-Control / Picture-Style metadata (e.g. `"standard"`, `"vivid"`,
/// `"monochrome"`).  When provided it is tried first so that the selected
/// profile matches the look the photographer chose in-camera.
///
/// Lookup order:
/// 1. `<dir>/<model>.dcp` — case-sensitive exact match (fast path).
/// 2. Case-insensitive flat scan for `<model>.dcp` in `<dir>/` (backward compat).
/// 3. A subdirectory whose name case-insensitively matches `<model>` (Adobe layout:
///    `<dir>/<Model>/<Model> Camera <Mode>.dcp`).  Within that subdirectory the
///    "best" profile is chosen:
///      a. (if `picture_style` is set) A file whose name contains the hint
///         without a "v2" suffix — e.g. `picture_style = "vivid"` selects
///         `"Nikon Z f Camera Vivid.dcp"`.
///      b. (if `picture_style` is set) Same but allowing "v2" variants.
///      c. A file containing "standard" (case-insensitive), excluding "hdr"
///         and "v2" — the generic camera Standard calibration.
///      d. Same but including "v2" variants.
///      e. Any file that does NOT contain "monochrome", excluding "hdr"/"v2".
///      f. Same but including "v2".
///      g. Any non-monochrome file.
///      h. The lexicographically first `.dcp` file (last resort).
///
/// Monochrome picture-style DCPs are only chosen by the hint path (step a/b)
/// or as an absolute last resort (step h).
pub fn find_dcp(dcp_dir: &Path, unique_camera_model: &str, picture_style: Option<&str>) -> Option<PathBuf> {
  // 1. Exact flat match
  let candidate = dcp_dir.join(format!("{}.dcp", unique_camera_model));
  if candidate.exists() {
    return Some(candidate);
  }

  // 2. Case-insensitive flat scan
  let target = unique_camera_model.to_lowercase();
  let target_dcp = format!("{}.dcp", target);
  if let Ok(entries) = std::fs::read_dir(dcp_dir) {
    for entry in entries.flatten() {
      let name = entry.file_name();
      let name_lower = name.to_string_lossy().to_lowercase();
      if name_lower == target_dcp {
        return Some(entry.path());
      }
    }
  }

  // 3. Nested subdirectory (Adobe layout: `<dir>/<Model>/`)
  if let Ok(entries) = std::fs::read_dir(dcp_dir) {
    for entry in entries.flatten() {
      if !entry.file_type().map_or(false, |ft| ft.is_dir()) {
        continue;
      }
      let dir_name = entry.file_name();
      if dir_name.to_string_lossy().to_lowercase() != target {
        continue;
      }

      // Collect .dcp files from the matching subdirectory
      let mut candidates: Vec<PathBuf> = std::fs::read_dir(entry.path())
        .into_iter()
        .flatten()
        .flatten()
        .filter(|f| {
          f.file_type().map_or(false, |ft| ft.is_file())
            && f.file_name().to_string_lossy().to_lowercase().ends_with(".dcp")
        })
        .map(|f| f.path())
        .collect();

      if candidates.is_empty() {
        break;
      }
      candidates.sort();

      // Helper: file name in lowercase
      let fname = |p: &PathBuf| p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
      let is_v2  = |p: &PathBuf| fname(p).contains(" v2.");
      let is_hdr = |p: &PathBuf| fname(p).contains(" hdr");

      // a/b. If the caller knows which picture style was active, try that first.
      if let Some(style) = picture_style {
        if let Some(p) = candidates.iter().find(|p| fname(p).contains(style) && !is_v2(p)) {
          return Some(p.clone());
        }
        if let Some(p) = candidates.iter().find(|p| fname(p).contains(style)) {
          return Some(p.clone());
        }
      }

      // c. "standard", not hdr, not v2   ("Camera Standard.dcp")
      if let Some(p) = candidates.iter().find(|p| fname(p).contains("standard") && !is_hdr(p) && !is_v2(p)) {
        return Some(p.clone());
      }
      // d. "standard", not hdr, any variant
      if let Some(p) = candidates.iter().find(|p| fname(p).contains("standard") && !is_hdr(p)) {
        return Some(p.clone());
      }
      // e. "standard" (includes HDR variants — still better than Vivid/Landscape)
      if let Some(p) = candidates.iter().find(|p| fname(p).contains("standard")) {
        return Some(p.clone());
      }
      // f. not monochrome, not hdr, not v2
      if let Some(p) = candidates.iter().find(|p| !fname(p).contains("monochrome") && !is_hdr(p) && !is_v2(p)) {
        return Some(p.clone());
      }
      // g. not monochrome, not hdr
      if let Some(p) = candidates.iter().find(|p| !fname(p).contains("monochrome") && !is_hdr(p)) {
        return Some(p.clone());
      }
      // h. not monochrome
      if let Some(p) = candidates.iter().find(|p| !fname(p).contains("monochrome")) {
        return Some(p.clone());
      }
      // i. anything
      return candidates.into_iter().next();
    }
  }

  None
}

/// Automatically discover a DCP profile for the given camera model.
///
/// This searches for profiles in the following order:
/// 1. The `DNGLAB_DCP_DIR` environment variable (if set).
/// 2. Standard Adobe CameraRaw system directories.
///
/// Returns the loaded profile if found.
pub fn auto_find_dcp(unique_camera_model: &str, picture_style: Option<&str>) -> Option<PathBuf> {
  // 1. Environment variable override
  if let Ok(env_dir) = std::env::var("DNGLAB_DCP_DIR") {
    let dir = PathBuf::from(&env_dir);
    if dir.is_dir() {
      if let Some(path) = find_dcp(&dir, unique_camera_model, picture_style) {
        return Some(path);
      }
    }
  }

  // 2. Standard Adobe CameraRaw paths
  for dir in system_dcp_dirs() {
    if dir.is_dir() {
      if let Some(path) = find_dcp(&dir, unique_camera_model, picture_style) {
        return Some(path);
      }
    }
  }

  None
}

/// Returns the platform-specific standard directories where Adobe stores
/// DCP camera profiles.
fn system_dcp_dirs() -> Vec<PathBuf> {
  let mut dirs = Vec::new();

  #[cfg(target_os = "linux")]
  {
    // Adobe DNG Converter installed via Wine or native
    if let Ok(home) = std::env::var("HOME") {
      // darktable camera profiles (common community path)
      dirs.push(PathBuf::from(format!("{}/.local/share/darktable/color/out", home)));
      // Wine-installed Adobe DNG Converter
      dirs.push(PathBuf::from(format!(
        "{}/.wine/drive_c/ProgramData/Adobe/CameraRaw/CameraProfiles",
        home
      )));
      dirs.push(PathBuf::from(format!(
        "{}/.wine/drive_c/Program Files/Adobe/Adobe DNG Converter/CameraProfiles",
        home
      )));
    }
  }

  #[cfg(target_os = "macos")]
  {
    dirs.push(PathBuf::from(
      "/Library/Application Support/Adobe/CameraRaw/CameraProfiles",
    ));
    if let Ok(home) = std::env::var("HOME") {
      dirs.push(PathBuf::from(format!(
        "{}/Library/Application Support/Adobe/CameraRaw/CameraProfiles",
        home
      )));
    }
  }

  #[cfg(target_os = "windows")]
  {
    if let Ok(appdata) = std::env::var("APPDATA") {
      dirs.push(PathBuf::from(format!(
        "{}/Adobe/CameraRaw/CameraProfiles",
        appdata
      )));
    }
    if let Ok(programdata) = std::env::var("ProgramData") {
      dirs.push(PathBuf::from(format!(
        "{}/Adobe/CameraRaw/CameraProfiles",
        programdata
      )));
    }
    dirs.push(PathBuf::from(
      "C:/Program Files/Adobe/Adobe DNG Converter/CameraProfiles",
    ));
  }

  dirs
}
