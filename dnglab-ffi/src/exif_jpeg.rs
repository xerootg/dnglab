// SPDX-License-Identifier: LGPL-2.1
//! EXIF authoring for JPEG output.
//!
//! Builds a standards compliant EXIF TIFF structure from rawler's
//! [`RawMetadata`] (root IFD, Exif sub-IFD, GPS sub-IFD, lens tags) and
//! splices it into a JPEG byte stream as an APP1 segment, replacing any EXIF
//! segment that is already present (camera embedded previews sometimes carry
//! their own, usually stale, EXIF block).

use std::io::Cursor;

use rawler::decoders::RawMetadata;
use rawler::formats::tiff::writer::{DirectoryWriter, TiffWriter};
use rawler::tags::{ExifTag, TiffCommonTag};

use crate::error::{Error, Result};

/// Maximum payload of a single APP1 segment: 65535 minus the 2 length bytes.
const APP1_MAX_PAYLOAD: usize = 65533;
const EXIF_HEADER: &[u8; 6] = b"Exif\x00\x00";

/// Build the TIFF structure of an EXIF block (without the `Exif\0\0` header).
///
/// * `orientation` overrides the orientation stored in the RAW file. Pass `1`
///   when the pixels have been rotated upright, or the RAW orientation when the
///   pixels are still in sensor orientation (embedded JPEG passthrough).
/// * MakerNotes are always stripped: they are opaque vendor blobs with
///   file-relative offsets that are meaningless once relocated, and they are
///   routinely larger than the 64 KiB an APP1 segment can hold.
pub fn build_exif_tiff(md: &RawMetadata, orientation: u16, software: &str) -> Result<Vec<u8>> {
  let mut md = md.clone();
  md.exif.orientation = Some(orientation);
  md.exif.makernotes = None;

  let mut buf = Cursor::new(Vec::new());
  let mut tiff = TiffWriter::new(&mut buf)?;
  let mut root_ifd = DirectoryWriter::new();
  let mut exif_ifd = DirectoryWriter::new();

  if !md.make.is_empty() {
    root_ifd.add_tag(TiffCommonTag::Make, md.make.as_str());
  }
  if !md.model.is_empty() {
    root_ifd.add_tag(TiffCommonTag::Model, md.model.as_str());
  }
  if !software.is_empty() {
    root_ifd.add_tag(TiffCommonTag::Software, software);
  }
  // EXIF version 2.30
  exif_ifd.add_tag_undefined(ExifTag::ExifVersion, b"0230".to_vec());

  // Root IFD fields, Exif sub-IFD fields, GPS sub-IFD (incl. GPSInfo pointer).
  md.write_exif_tags(&mut tiff, &mut root_ifd, &mut exif_ifd)?;

  if exif_ifd.entry_count() > 0 {
    let exif_offset = exif_ifd.build(&mut tiff)?;
    root_ifd.add_tag(TiffCommonTag::ExifIFDPointer, exif_offset);
  }
  tiff.build(root_ifd)?;
  Ok(buf.into_inner())
}

/// Build the complete APP1 payload (`Exif\0\0` + TIFF) for [`splice_exif_app1`]
/// or `jpeg_encoder::Encoder::add_app_segment`.
pub fn build_exif_app1_payload(md: &RawMetadata, orientation: u16, software: &str) -> Result<Vec<u8>> {
  let tiff = build_exif_tiff(md, orientation, software)?;
  let mut payload = Vec::with_capacity(EXIF_HEADER.len() + tiff.len());
  payload.extend_from_slice(EXIF_HEADER);
  payload.extend_from_slice(&tiff);
  if payload.len() > APP1_MAX_PAYLOAD {
    return Err(Error::new(format!(
      "EXIF block too large for a single APP1 segment ({} bytes > {})",
      payload.len(),
      APP1_MAX_PAYLOAD
    )));
  }
  Ok(payload)
}

/// Return the JPEG with `app1_payload` (`Exif\0\0` + TIFF) inserted as an APP1
/// segment. Existing EXIF APP1 segments are removed. A leading JFIF APP0 segment
/// keeps its position (the JFIF spec requires it to be first); the EXIF block is
/// inserted directly after it, otherwise directly after SOI.
pub fn splice_exif_app1(jpeg: &[u8], app1_payload: &[u8]) -> Result<Vec<u8>> {
  if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
    return Err(Error::new("Input is not a valid JPEG (missing SOI marker)"));
  }
  if app1_payload.len() > APP1_MAX_PAYLOAD {
    return Err(Error::new("EXIF block too large for a single APP1 segment"));
  }

  // Walk the header segments (APPn / COM) until the first non-header marker.
  // Collect everything except EXIF APP1 segments, remembering where a leading
  // APP0 ends so the new EXIF block can go right behind it.
  let mut out = Vec::with_capacity(jpeg.len() + app1_payload.len() + 4);
  out.extend_from_slice(&jpeg[..2]);
  let mut pos = 2usize;
  let mut insert_at = out.len();
  let mut first_segment = true;

  while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF {
    let marker = jpeg[pos + 1];
    // Only APPn (E0..EF) and COM (FE) segments are header segments we can skip over.
    if !((0xE0..=0xEF).contains(&marker) || marker == 0xFE) {
      break;
    }
    let seg_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
    if seg_len < 2 || pos + 2 + seg_len > jpeg.len() {
      return Err(Error::new("Corrupt JPEG header segment"));
    }
    let seg_end = pos + 2 + seg_len;
    let is_exif = marker == 0xE1 && seg_len >= 2 + EXIF_HEADER.len() && &jpeg[pos + 4..pos + 4 + EXIF_HEADER.len()] == EXIF_HEADER;
    if !is_exif {
      out.extend_from_slice(&jpeg[pos..seg_end]);
      if first_segment && marker == 0xE0 {
        // JFIF APP0 stays first.
        insert_at = out.len();
      }
    }
    first_segment = false;
    pos = seg_end;
  }

  let mut segment = Vec::with_capacity(4 + app1_payload.len());
  segment.extend_from_slice(&[0xFF, 0xE1]);
  segment.extend_from_slice(&((app1_payload.len() + 2) as u16).to_be_bytes());
  segment.extend_from_slice(app1_payload);
  out.splice(insert_at..insert_at, segment);

  out.extend_from_slice(&jpeg[pos..]);
  Ok(out)
}

/// Parse the dimensions (width, height) from the SOF marker of a JPEG stream.
/// Returns `(0, 0)` when no frame header is found.
pub fn jpeg_dimensions(data: &[u8]) -> (u32, u32) {
  let mut i = 0usize;
  while i + 1 < data.len() {
    if data[i] != 0xFF {
      i += 1;
      continue;
    }
    let marker = data[i + 1];
    i += 2;
    if marker == 0xFF || marker == 0x00 || marker == 0xD8 || (0xD0..=0xD7).contains(&marker) || marker == 0xD9 {
      continue;
    }
    if i + 2 > data.len() {
      break;
    }
    let len = ((data[i] as usize) << 8) | data[i + 1] as usize;
    let is_sof = matches!(marker, 0xC0..=0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF);
    if is_sof {
      if i + 7 <= data.len() {
        let height = ((data[i + 3] as u32) << 8) | data[i + 4] as u32;
        let width = ((data[i + 5] as u32) << 8) | data[i + 6] as u32;
        return (width, height);
      }
      break;
    }
    if marker == 0xDA {
      // Start of scan without SOF: give up.
      break;
    }
    if len < 2 || i + len > data.len() {
      break;
    }
    i += len;
  }
  (0, 0)
}

/// Read the EXIF orientation (1..8) from a JPEG's APP1 segment, if any.
/// Camera embedded previews that already carry EXIF usually store `1` even
/// when the sensor data is rotated, so this is informational only.
pub fn jpeg_exif_orientation(data: &[u8]) -> Option<u16> {
  let mut pos = 2usize;
  if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
    return None;
  }
  while pos + 4 <= data.len() && data[pos] == 0xFF {
    let marker = data[pos + 1];
    if !((0xE0..=0xEF).contains(&marker) || marker == 0xFE) {
      break;
    }
    let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
    if seg_len < 2 || pos + 2 + seg_len > data.len() {
      return None;
    }
    if marker == 0xE1 && seg_len >= 2 + EXIF_HEADER.len() && &data[pos + 4..pos + 4 + EXIF_HEADER.len()] == EXIF_HEADER {
      let tiff = &data[pos + 4 + EXIF_HEADER.len()..pos + 2 + seg_len];
      return orientation_from_tiff(tiff);
    }
    pos += 2 + seg_len;
  }
  None
}

fn orientation_from_tiff(tiff: &[u8]) -> Option<u16> {
  if tiff.len() < 8 {
    return None;
  }
  let le = match &tiff[0..2] {
    b"II" => true,
    b"MM" => false,
    _ => return None,
  };
  let rd16 = |o: usize| -> Option<u16> {
    let b = tiff.get(o..o + 2)?;
    Some(if le {
      u16::from_le_bytes([b[0], b[1]])
    } else {
      u16::from_be_bytes([b[0], b[1]])
    })
  };
  let rd32 = |o: usize| -> Option<u32> {
    let b = tiff.get(o..o + 4)?;
    Some(if le {
      u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    } else {
      u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    })
  };
  let ifd0 = rd32(4)? as usize;
  let count = rd16(ifd0)? as usize;
  for i in 0..count {
    let e = ifd0 + 2 + i * 12;
    if rd16(e)? == 0x0112 {
      return rd16(e + 8);
    }
  }
  None
}

#[cfg(test)]
mod tests {
  use super::*;

  // Smallest JPEG-ish byte stream with SOI, APP0, SOF0 and EOI.
  fn tiny_jpeg(with_exif: bool) -> Vec<u8> {
    let mut v = vec![0xFF, 0xD8];
    // APP0 JFIF
    v.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x10]);
    v.extend_from_slice(b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00");
    if with_exif {
      let payload = b"Exif\0\0II*\0\x08\0\0\0\0\0";
      v.extend_from_slice(&[0xFF, 0xE1]);
      v.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
      v.extend_from_slice(payload);
    }
    // SOF0: len 17, precision 8, height 0x0010, width 0x0020, 3 comps
    v.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08, 0x00, 0x10, 0x00, 0x20, 0x03]);
    v.extend_from_slice(&[1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    v.extend_from_slice(&[0xFF, 0xD9]);
    v
  }

  #[test]
  fn dimensions_from_sof() {
    assert_eq!(jpeg_dimensions(&tiny_jpeg(false)), (32, 16));
    assert_eq!(jpeg_dimensions(&tiny_jpeg(true)), (32, 16));
    assert_eq!(jpeg_dimensions(&[0xFF, 0xD8, 0xFF, 0xD9]), (0, 0));
  }

  #[test]
  fn splice_inserts_after_app0_and_replaces_existing_exif() {
    let payload = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
    for with_exif in [false, true] {
      let out = splice_exif_app1(&tiny_jpeg(with_exif), &payload).unwrap();
      // SOI, APP0 (18 bytes), then our APP1
      assert_eq!(&out[0..2], &[0xFF, 0xD8]);
      assert_eq!(&out[2..4], &[0xFF, 0xE0]);
      assert_eq!(&out[20..22], &[0xFF, 0xE1]);
      assert_eq!(&out[24..30], b"Exif\0\0");
      // Exactly one EXIF segment and dimensions still parse.
      let count = out.windows(6).filter(|w| *w == b"Exif\0\0").count();
      assert_eq!(count, 1);
      assert_eq!(jpeg_dimensions(&out), (32, 16));
      assert_eq!(jpeg_exif_orientation(&out), Some(6));
    }
  }

  #[test]
  fn splice_rejects_non_jpeg() {
    assert!(splice_exif_app1(b"not a jpeg", b"Exif\0\0").is_err());
  }
}
