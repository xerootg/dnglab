// SPDX-License-Identifier: LGPL-2.1
// Copyright 2024 - rawler authors

//! Nikon Z-series encrypted ShotInfo (makernote tag 0x0091) parser.
//!
//! ShotInfo is an encrypted binary blob appended to the makernote of recent
//! Nikon mirrorless cameras.  The first four bytes ("0808" for the Nikon Zf)
//! are not encrypted; everything from byte 4 onwards uses the same XOR-based
//! stream cipher that is also applied to `LensData`.
//!
//! Layout for version "0808" (Nikon Zf, shares table with Z7 II):
//!
//! ```text
//! Offset  Size  Field
//! 0x0000  4     ShotInfoVersion   (plain ASCII, not encrypted)
//! 0x0004  8     FirmwareVersion   (encrypted ASCII, e.g. "03.00.g0")
//! 0x000E  8     FirmwareVersion2  (encrypted)
//! 0x0018  8     FirmwareVersion3  (encrypted)
//! 0x0020  4     <padding / reserved>
//! 0x0024  4     NumberOffsets     (u32LE, count of entries in offset table)
//! 0x0028  N×4   OffsetTable       (N = NumberOffsets, absolute offsets into
//!                                   this blob for sub-sections)
//! …
//! 0x0088  4     OrientationOffset (absolute offset → OrientationInfo,
//!                                   Zf only; 0x0098 for Z7 II / Z6 II)
//! …
//! ```
//!
//! `OrientationInfo` (12 bytes at OrientationOffset):
//! ```text
//! +0  fixed32u  RollAngle   (degrees, clockwise; subtract 360 if > 180)
//! +4  fixed32u  PitchAngle  (degrees, upward tilt)
//! +8  fixed32u  YawAngle    (portrait yaw)
//! ```
//! `fixed32u` = u32 / 65536.0

use crate::Result;
use crate::bits::LEu32;
use crate::decoders::nef::NikonMakernote;
use crate::formats::tiff::IFD;

/// Camera orientation angles decoded from `OrientationInfo`.
#[derive(Debug, Clone, Default)]
pub struct OrientationInfo {
  /// Camera roll in degrees (positive = clockwise tilt).
  pub roll_angle: f64,
  /// Camera pitch in degrees (positive = upward tilt).
  pub pitch_angle: f64,
  /// Camera yaw when shooting in portrait orientation (degrees).
  pub yaw_angle: f64,
}

/// Per-shot header fields from the ShotInfo blob.
#[derive(Debug, Clone, Default)]
pub struct ShotInfoHeader {
  /// Four-byte ASCII version string (e.g. `"0808"` for the Nikon Zf).
  pub version: String,
  /// Firmware build string stored at offset 0x04.
  pub firmware_version: String,
  /// Secondary firmware string at offset 0x0E.
  pub firmware_version2: String,
  /// Tertiary firmware string at offset 0x18.
  pub firmware_version3: String,
  /// Number of entries in the variable-length offset directory.
  pub num_offsets: u32,
}

/// Decoded ShotInfo for Nikon Z-series cameras (version "0803"/"0808").
///
/// Only the fields that are available unconditionally for the Zf (version
/// "0808") are represented here.  Additional sub-sections (interval info,
/// portrait impression, menu settings) can be added when needed.
#[derive(Debug, Clone, Default)]
pub struct NefShotInfoZ7II {
  pub header: ShotInfoHeader,
  /// Camera orientation data (roll/pitch/yaw) from `OrientationInfo`.
  pub orientation: OrientationInfo,
}

/// Parse a `fixed32u` value as a signed angle in degrees.
///
/// A `fixed32u` is an unsigned 32-bit fixed-point number with 16 fractional
/// bits (i.e. `value / 65536.0`).  Exiftool treats values > 180 as negative
/// by subtracting 360, matching the physical sign convention.
#[inline]
fn fixed32u_to_angle(raw: u32) -> f64 {
  let f = raw as f64 / 65536.0;
  if f > 180.0 {
    f - 360.0
  } else {
    f
  }
}

/// Read a null-terminated ASCII string from `buf[offset..]`, capped at `len`
/// bytes.  Non-ASCII bytes after the null terminator are ignored.
fn read_ascii(buf: &[u8], offset: usize, len: usize) -> String {
  let end = (offset + len).min(buf.len());
  let slice = &buf[offset..end];
  let nul = slice.iter().position(|&b| b == 0).unwrap_or(slice.len());
  String::from_utf8_lossy(&slice[..nul]).into_owned()
}

/// Attempt to decrypt and deserialise the Nikon ShotInfo tag (0x0091).
///
/// Returns `None` if the tag is absent, if the version is not recognised, or
/// if the buffer is too short to hold the expected sub-sections.
/// Returns an error only when cryptographic key material is missing and the
/// decryption helper fails.
pub(super) fn parse_shot_info(makernote: &IFD) -> Result<Option<NefShotInfoZ7II>> {
  let entry = match makernote.get_entry(NikonMakernote::ShotInfo) {
    Some(e) => e,
    None => return Ok(None),
  };

  let mut buf = entry.get_data().to_vec();
  if buf.len() < 0xa4 {
    return Ok(None);
  }

  // The first four bytes are the plain-text version identifier; encryption
  // begins at byte 4.
  let version = read_ascii(&buf, 0, 4);
  match version.as_str() {
    "0808" | "0803" => {} // Nikon Zf (0808) and Z7 II / Z6 II (0803)
    _ => return Ok(None),
  }

  // Decrypt in-place from byte 4 onwards.
  super::decrypt::nef_decrypt(&mut buf, 4, makernote)?;

  // --- Header ---
  let firmware_version = read_ascii(&buf, 0x04, 8);
  let firmware_version2 = read_ascii(&buf, 0x0e, 8);
  let firmware_version3 = read_ascii(&buf, 0x18, 8);
  let num_offsets = LEu32(&buf, 0x24);

  // Sanity-check: the offset table must fit inside the buffer.
  let table_end = 0x28usize + num_offsets as usize * 4;
  if table_end > buf.len() {
    return Ok(None);
  }

  // --- OrientationInfo ---
  // For version "0808" (Zf) the orientation pointer lives at byte 0x88.
  // For version "0803" (Z7 II / Z6 II) it lives at byte 0x98.
  let ori_ptr_offset: usize = if version == "0808" { 0x88 } else { 0x98 };

  // Ensure the pointer field itself is within the already-decrypted table.
  if ori_ptr_offset + 4 > table_end {
    return Ok(None);
  }

  let ori_offset = LEu32(&buf, ori_ptr_offset) as usize;
  if ori_offset == 0 || ori_offset + 12 > buf.len() {
    return Ok(None);
  }

  let roll = fixed32u_to_angle(LEu32(&buf, ori_offset));
  let pitch = fixed32u_to_angle(LEu32(&buf, ori_offset + 4));
  let yaw = fixed32u_to_angle(LEu32(&buf, ori_offset + 8));

  Ok(Some(NefShotInfoZ7II {
    header: ShotInfoHeader {
      version,
      firmware_version,
      firmware_version2,
      firmware_version3,
      num_offsets,
    },
    orientation: OrientationInfo {
      roll_angle: roll,
      pitch_angle: pitch,
      yaw_angle: yaw,
    },
  }))
}
