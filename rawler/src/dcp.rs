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
/// `UniqueCameraModel` string (e.g. `"NIKON Z F"`).
///
/// The function tries `<dir>/<model>.dcp` (case-sensitive) first,
/// then falls back to a case-insensitive scan of the directory.
pub fn find_dcp(dcp_dir: &Path, unique_camera_model: &str) -> Option<PathBuf> {
  // Fast path: exact match
  let candidate = dcp_dir.join(format!("{}.dcp", unique_camera_model));
  if candidate.exists() {
    return Some(candidate);
  }

  // Slow path: case-insensitive scan
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
  None
}
