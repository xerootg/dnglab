use std::io::Cursor;

use crate::common::check_md5_equal;
use crate::common::rawdb_file;
use rawler::dng::convert::ConvertParams;
use rawler::dng::convert::convert_raw_file;
use rawler::formats::jfif::Jfif;
use rawler::rawsource::RawSource;
use rawler::{analyze::raw_pixels_digest, decoders::RawDecodeParams};

#[test]
fn dnglab_354_dng_mismatch_tile_dim_vs_ljpeg_sof_dim() -> std::result::Result<(), Box<dyn std::error::Error>> {
  let path = rawdb_file("issues/dnglab_354/dnglab_354.dng");
  let digest = raw_pixels_digest(path, &RawDecodeParams::default())?;
  check_md5_equal(digest, "e5fcd3fd81a3f8e2d9709b92f3b8f546");
  Ok(())
}

#[test]
fn dnglab_366_monochrome_dng_support() -> std::result::Result<(), Box<dyn std::error::Error>> {
  let path = rawdb_file("issues/dnglab_366/dnglab_366.dng");
  let digest = raw_pixels_digest(&path, &RawDecodeParams::default())?;
  check_md5_equal(digest, "f3549fafda97fca90b9993c1278bcd90");
  let mut dng = Cursor::new(Vec::new());
  convert_raw_file(&path, &mut dng, &ConvertParams::default())?;
  Ok(())
}

#[test]
fn dnglab_376_canon_crx_craw_qstep_shl_bug() -> std::result::Result<(), Box<dyn std::error::Error>> {
  {
    let path = rawdb_file("issues/dnglab_376/Canon_EOS_R6M2_CRAW_ISO_25600.CR3");
    let digest = raw_pixels_digest(&path, &RawDecodeParams::default())?;
    check_md5_equal(digest, "66c9fcb6541c90bdfb06d876be5984ec");
    let mut dng = Cursor::new(Vec::new());
    convert_raw_file(&path, &mut dng, &ConvertParams::default())?;
  }
  {
    let path = rawdb_file("issues/dnglab_376/_MGC9382.CR3");
    let digest = raw_pixels_digest(&path, &RawDecodeParams::default())?;
    check_md5_equal(digest, "aef96546a58e5265fb2f7b9e7498cbd0");
    let mut dng = Cursor::new(Vec::new());
    convert_raw_file(&path, &mut dng, &ConvertParams::default())?;
  }
  Ok(())
}

#[test]
fn dnglab_386_catch_jpeg_exif_tiff_ifd_error() -> std::result::Result<(), Box<dyn std::error::Error>> {
  let path = rawdb_file("issues/dnglab_386/jpeg_ifd_error.jpg");
  let rawfile = RawSource::new(&path)?;
  let jfif = Jfif::new(&rawfile)?;
  assert!(jfif.exif_ifd().is_none());
  Ok(())
}

#[test]
fn dnglab_477_jpeg_quantization_table_with_zero_value() -> std::result::Result<(), Box<dyn std::error::Error>> {
  let image = image::open(rawdb_file("issues/dnglab_477/dnglab_477.jpg"))?;
  let _ = image.to_rgb8();
  Ok(())
}

#[test]
fn dnglab_619_silverfast_scan_missing_illuminant() -> std::result::Result<(), Box<dyn std::error::Error>> {
  let path = rawdb_file("issues/dnglab_619/silverfast_scan.dng");
  let mut dng = Cursor::new(Vec::new());
  convert_raw_file(&path, &mut dng, &ConvertParams::default())?;
  Ok(())
}

/// Regression test: repeated convert_raw_file calls in the same process must
/// produce identical DNG output.  A buffer-reuse bug caused BlackLevel tag data
/// to be overwritten with zeros after the first few conversions, leading to
/// wrong colour rendering (pink/magenta cast).
#[test]
fn repeated_conversion_preserves_blacklevel() -> std::result::Result<(), Box<dyn std::error::Error>> {
  let path = std::path::PathBuf::from("/home/xero/Downloads/7215093.ORF");
  if !path.exists() {
    eprintln!("Skipping test: {} not found", path.display());
    return Ok(());
  }
  // Match the Python binding's params: no thumbnail, no preview
  let params = ConvertParams {
    thumbnail: false,
    preview: false,
    embedded: false,
    ..ConvertParams::default()
  };

  // Convert 6 times sequentially — same pattern as the Python binding
  let expected_bl = [256u16, 254, 254, 255];
  for i in 0..6 {
    let mut dng = Cursor::new(Vec::new());
    convert_raw_file(&path, &mut dng, &params)?;
    let data = dng.into_inner();
    let bl = find_blacklevel_values(&data);
    assert_eq!(
      bl.as_deref(), Some(expected_bl.as_slice()),
      "Conversion {i}: BlackLevel is {bl:?}, expected {expected_bl:?}"
    );
  }

  // Also test with thread spawning (simulates py.detach GIL release)
  for i in 0..6 {
    let p = path.clone();
    let par = params.clone();
    let data = std::thread::spawn(move || {
      let mut dng = Cursor::new(Vec::new());
      convert_raw_file(&p, &mut dng, &par).unwrap();
      dng.into_inner()
    }).join().unwrap();
    let bl = find_blacklevel_values(&data);
    assert_eq!(
      bl.as_deref(), Some(expected_bl.as_slice()),
      "Threaded conversion {i}: BlackLevel is {bl:?}, expected {expected_bl:?}"
    );
  }
  Ok(())
}

/// Extract BlackLevel (tag 50714) u16 values from a DNG byte buffer.
fn find_blacklevel_values(data: &[u8]) -> Option<Vec<u16>> {
  if data.len() < 8 { return None; }
  let is_le = data[0] == b'I';
  let r16 = |o: usize| -> u16 {
    if is_le { u16::from_le_bytes([data[o], data[o+1]]) }
    else { u16::from_be_bytes([data[o], data[o+1]]) }
  };
  let r32 = |o: usize| -> u32 {
    if is_le { u32::from_le_bytes([data[o], data[o+1], data[o+2], data[o+3]]) }
    else { u32::from_be_bytes([data[o], data[o+1], data[o+2], data[o+3]]) }
  };

  // Walk IFD chain looking for tag 50714
  let mut ifd_off = r32(4) as usize;
  while ifd_off > 0 && ifd_off + 2 < data.len() {
    let num = r16(ifd_off) as usize;
    for i in 0..num {
      let eoff = ifd_off + 2 + i * 12;
      if eoff + 12 > data.len() { break; }
      let tag = r16(eoff);
      if tag == 50714 {
        let dtype = r16(eoff + 2);
        let count = r32(eoff + 4) as usize;
        if dtype != 3 { return None; } // expect SHORT
        let total = count * 2;
        let voff = if total <= 4 { eoff + 8 } else { r32(eoff + 8) as usize };
        if voff + total > data.len() { return None; }
        return Some((0..count).map(|j| r16(voff + j * 2)).collect());
      }
    }
    // Next IFD
    let next_off_pos = ifd_off + 2 + num * 12;
    if next_off_pos + 4 > data.len() { break; }
    ifd_off = r32(next_off_pos) as usize;
  }
  None
}
