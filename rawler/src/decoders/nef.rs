use std::rc::Rc;

use image::DynamicImage;
use log::debug;
use log::warn;
use serde::Deserialize;
use serde::Serialize;

use crate::RawImage;
use crate::RawLoader;
use crate::RawlerError;
use crate::Result;
use crate::alloc_image_plain;
use crate::analyze::FormatDump;
use crate::bits::BEu16;
use crate::bits::BEu32;
use crate::bits::Endian;
use crate::bits::LEu16;
use crate::bits::LEu32;
use crate::bits::LookupTable;
use crate::bits::clampbits;
use crate::buffer::PaddedBuf;
use crate::decoders::dynamic_image_from_ifd;
use crate::decoders::dynamic_image_from_jpeg_interchange_format;
use crate::decoders::nef::lensdata::NefLensData;
use crate::decompressors::decompress_lines_fn;
use crate::decompressors::ljpeg::huffman::HuffTable;
use crate::decompressors::packed::*;
use crate::exif::Exif;
use crate::dng::opcodes;
use crate::formats::tiff::entry::Entry;
use crate::formats::tiff::GenericTiffReader;
use crate::formats::tiff::IFD;
use crate::formats::tiff::Value;
use crate::formats::tiff::ifd::OffsetMode;
use crate::formats::tiff::reader::TiffReader;
use crate::imgop::Dim2;
use crate::imgop::Point;
use crate::imgop::Rect;
use crate::lens::LensDescription;
use crate::lens::LensResolver;
use crate::pixarray::PixU16;
use crate::pumps::BitPump;
use crate::pumps::BitPumpMSB;
use crate::pumps::ByteStream;
use crate::rawimage::CFAConfig;
use crate::rawimage::RawPhotometricInterpretation;
use crate::rawimage::WhiteLevel;
use crate::rawsource::RawSource;
use crate::tags::DngTag;
use crate::tags::ExifTag;
use crate::tags::TiffCommonTag;

use super::BlackLevel;
use super::Camera;
use super::Decoder;
use super::FormatHint;
use super::RawDecodeParams;
use super::RawMetadata;
use super::WellKnownIFD;

mod decrypt;
pub mod lensdata;
mod noisecal;
pub mod shotinfo;

const NIKON_F_MOUNT: &str = "F-mount";
const NIKON_Z_MOUNT: &str = "Z-mount";

// NEF Huffman tables in order. First two are the normal huffman definitions.
// Third one are weird shifts that are used in the lossy split encodings only
// Values are extracted from dcraw with the shifts unmangled out.
const NIKON_TREE: [[[u8; 16]; 3]; 6] = [
  [
    // 12-bit lossy
    [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0],
    [5, 4, 3, 6, 2, 7, 1, 0, 8, 9, 11, 10, 12, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    // 12-bit lossy after split
    [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0],
    [6, 5, 5, 5, 5, 5, 4, 3, 2, 1, 0, 11, 12, 12, 0, 0],
    [3, 5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    // 12-bit lossless
    [0, 0, 1, 4, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, 11, 12, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    // 14-bit lossy
    [0, 0, 1, 4, 3, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0],
    [5, 6, 4, 7, 8, 3, 9, 2, 1, 0, 10, 11, 12, 13, 14, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    // 14-bit lossy after split
    [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0],
    [8, 7, 7, 7, 7, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14, 0],
    [0, 5, 4, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  [
    // 14-bit lossless
    [0, 0, 1, 4, 2, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0],
    [7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
];

// We use this for the D50 and D2X whacky WB "encryption"
const WB_SERIALMAP: [u8; 256] = [
  0xc1, 0xbf, 0x6d, 0x0d, 0x59, 0xc5, 0x13, 0x9d, 0x83, 0x61, 0x6b, 0x4f, 0xc7, 0x7f, 0x3d, 0x3d, 0x53, 0x59, 0xe3, 0xc7, 0xe9, 0x2f, 0x95, 0xa7, 0x95, 0x1f,
  0xdf, 0x7f, 0x2b, 0x29, 0xc7, 0x0d, 0xdf, 0x07, 0xef, 0x71, 0x89, 0x3d, 0x13, 0x3d, 0x3b, 0x13, 0xfb, 0x0d, 0x89, 0xc1, 0x65, 0x1f, 0xb3, 0x0d, 0x6b, 0x29,
  0xe3, 0xfb, 0xef, 0xa3, 0x6b, 0x47, 0x7f, 0x95, 0x35, 0xa7, 0x47, 0x4f, 0xc7, 0xf1, 0x59, 0x95, 0x35, 0x11, 0x29, 0x61, 0xf1, 0x3d, 0xb3, 0x2b, 0x0d, 0x43,
  0x89, 0xc1, 0x9d, 0x9d, 0x89, 0x65, 0xf1, 0xe9, 0xdf, 0xbf, 0x3d, 0x7f, 0x53, 0x97, 0xe5, 0xe9, 0x95, 0x17, 0x1d, 0x3d, 0x8b, 0xfb, 0xc7, 0xe3, 0x67, 0xa7,
  0x07, 0xf1, 0x71, 0xa7, 0x53, 0xb5, 0x29, 0x89, 0xe5, 0x2b, 0xa7, 0x17, 0x29, 0xe9, 0x4f, 0xc5, 0x65, 0x6d, 0x6b, 0xef, 0x0d, 0x89, 0x49, 0x2f, 0xb3, 0x43,
  0x53, 0x65, 0x1d, 0x49, 0xa3, 0x13, 0x89, 0x59, 0xef, 0x6b, 0xef, 0x65, 0x1d, 0x0b, 0x59, 0x13, 0xe3, 0x4f, 0x9d, 0xb3, 0x29, 0x43, 0x2b, 0x07, 0x1d, 0x95,
  0x59, 0x59, 0x47, 0xfb, 0xe5, 0xe9, 0x61, 0x47, 0x2f, 0x35, 0x7f, 0x17, 0x7f, 0xef, 0x7f, 0x95, 0x95, 0x71, 0xd3, 0xa3, 0x0b, 0x71, 0xa3, 0xad, 0x0b, 0x3b,
  0xb5, 0xfb, 0xa3, 0xbf, 0x4f, 0x83, 0x1d, 0xad, 0xe9, 0x2f, 0x71, 0x65, 0xa3, 0xe5, 0x07, 0x35, 0x3d, 0x0d, 0xb5, 0xe9, 0xe5, 0x47, 0x3b, 0x9d, 0xef, 0x35,
  0xa3, 0xbf, 0xb3, 0xdf, 0x53, 0xd3, 0x97, 0x53, 0x49, 0x71, 0x07, 0x35, 0x61, 0x71, 0x2f, 0x43, 0x2f, 0x11, 0xdf, 0x17, 0x97, 0xfb, 0x95, 0x3b, 0x7f, 0x6b,
  0xd3, 0x25, 0xbf, 0xad, 0xc7, 0xc5, 0xc5, 0xb5, 0x8b, 0xef, 0x2f, 0xd3, 0x07, 0x6b, 0x25, 0x49, 0x95, 0x25, 0x49, 0x6d, 0x71, 0xc7,
];

const WB_KEYMAP: [u8; 256] = [
  0xa7, 0xbc, 0xc9, 0xad, 0x91, 0xdf, 0x85, 0xe5, 0xd4, 0x78, 0xd5, 0x17, 0x46, 0x7c, 0x29, 0x4c, 0x4d, 0x03, 0xe9, 0x25, 0x68, 0x11, 0x86, 0xb3, 0xbd, 0xf7,
  0x6f, 0x61, 0x22, 0xa2, 0x26, 0x34, 0x2a, 0xbe, 0x1e, 0x46, 0x14, 0x68, 0x9d, 0x44, 0x18, 0xc2, 0x40, 0xf4, 0x7e, 0x5f, 0x1b, 0xad, 0x0b, 0x94, 0xb6, 0x67,
  0xb4, 0x0b, 0xe1, 0xea, 0x95, 0x9c, 0x66, 0xdc, 0xe7, 0x5d, 0x6c, 0x05, 0xda, 0xd5, 0xdf, 0x7a, 0xef, 0xf6, 0xdb, 0x1f, 0x82, 0x4c, 0xc0, 0x68, 0x47, 0xa1,
  0xbd, 0xee, 0x39, 0x50, 0x56, 0x4a, 0xdd, 0xdf, 0xa5, 0xf8, 0xc6, 0xda, 0xca, 0x90, 0xca, 0x01, 0x42, 0x9d, 0x8b, 0x0c, 0x73, 0x43, 0x75, 0x05, 0x94, 0xde,
  0x24, 0xb3, 0x80, 0x34, 0xe5, 0x2c, 0xdc, 0x9b, 0x3f, 0xca, 0x33, 0x45, 0xd0, 0xdb, 0x5f, 0xf5, 0x52, 0xc3, 0x21, 0xda, 0xe2, 0x22, 0x72, 0x6b, 0x3e, 0xd0,
  0x5b, 0xa8, 0x87, 0x8c, 0x06, 0x5d, 0x0f, 0xdd, 0x09, 0x19, 0x93, 0xd0, 0xb9, 0xfc, 0x8b, 0x0f, 0x84, 0x60, 0x33, 0x1c, 0x9b, 0x45, 0xf1, 0xf0, 0xa3, 0x94,
  0x3a, 0x12, 0x77, 0x33, 0x4d, 0x44, 0x78, 0x28, 0x3c, 0x9e, 0xfd, 0x65, 0x57, 0x16, 0x94, 0x6b, 0xfb, 0x59, 0xd0, 0xc8, 0x22, 0x36, 0xdb, 0xd2, 0x63, 0x98,
  0x43, 0xa1, 0x04, 0x87, 0x86, 0xf7, 0xa6, 0x26, 0xbb, 0xd6, 0x59, 0x4d, 0xbf, 0x6a, 0x2e, 0xaa, 0x2b, 0xef, 0xe6, 0x78, 0xb6, 0x4e, 0xe0, 0x2f, 0xdc, 0x7c,
  0xbe, 0x57, 0x19, 0x32, 0x7e, 0x2a, 0xd0, 0xb8, 0xba, 0x29, 0x00, 0x3c, 0x52, 0x7d, 0xa8, 0x49, 0x3b, 0x2d, 0xeb, 0x25, 0x49, 0xfa, 0xa3, 0xaa, 0x39, 0xa7,
  0xc5, 0xa7, 0x50, 0x11, 0x36, 0xfb, 0xc6, 0x67, 0x4a, 0xf5, 0xa5, 0x12, 0x65, 0x7e, 0xb0, 0xdf, 0xaf, 0x4e, 0xb3, 0x61, 0x7f, 0x2f,
];

/// NEF format encapsulation for analyzer
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NefFormat {
  tiff: GenericTiffReader,
}

#[derive(Debug, Clone)]
pub struct NefDecoder<'a> {
  #[allow(unused)]
  rawloader: &'a RawLoader,
  tiff: GenericTiffReader,
  makernote: IFD,
  camera: Camera,
  /// Pre-computed OpcodeList1 blob (vignette correction, applied before demosaicing).
  /// Empty when no NikonNEFInfo lens correction data is found.
  opcode_list1: Vec<u8>,
  /// Pre-computed OpcodeList3 blob (distortion correction, applied after demosaicing).
  opcode_list3: Vec<u8>,
}

impl<'a> NefDecoder<'a> {
  pub fn new(file: &RawSource, tiff: GenericTiffReader, rawloader: &'a RawLoader) -> Result<NefDecoder<'a>> {
    let raw = tiff
      .find_first_ifd_with_tag(TiffCommonTag::CFAPattern)
      .or_else(|| tiff.find_ifd_with_new_subfile_type(0))
      .ok_or_else(|| RawlerError::DecoderFailed(format!("Failed to find a suitable IFD in NEF decoder")))?;
    let bps = fetch_tiff_tag!(raw, TiffCommonTag::BitsPerSample).force_usize(0);

    // Make sure we always use a 12/14 bit mode to get correct white/blackpoints
    let mode = format!("{}bit", bps);
    let camera = rawloader.check_supported_with_mode(tiff.root_ifd(), &mode)?;

    let makernote = if let Some(exif) = tiff.find_first_ifd_with_tag(ExifTag::MakerNotes) {
      exif.parse_makernote(&mut file.reader(), OffsetMode::Absolute, &[])?
    } else {
      warn!("NEF makernote not found");
      None
    }
    .ok_or("File has not makernotes")?;

    //makernote.dump::<ExifTag>(0).iter().for_each(|line| eprintln!("DUMP: {}", line));

    // Parse NikonNEFInfo tag 0xc7d5 from the raw IFD for lens correction opcodes.
    let (opcode_list1, opcode_list3) = {
      let raw2 = tiff
        .find_first_ifd_with_tag(TiffCommonTag::CFAPattern)
        .or_else(|| tiff.find_ifd_with_new_subfile_type(0));
      raw2
        .and_then(|r| r.get_entry(0xc7d5_u16))
        .and_then(|entry| {
          if let Value::Undefined(data) = &entry.value {
            parse_nikon_nef_opcodes(data)
          } else {
            None
          }
        })
        .unwrap_or_default()
    };

    Ok(NefDecoder {
      tiff,
      rawloader,
      makernote,
      camera,
      opcode_list1,
      opcode_list3,
    })
  }
}

impl<'a> Decoder for NefDecoder<'a> {
  fn raw_image(&self, file: &RawSource, _params: &RawDecodeParams, dummy: bool) -> Result<RawImage> {
    let raw = self
      .tiff
      .find_first_ifd_with_tag(TiffCommonTag::CFAPattern)
      .or_else(|| self.tiff.find_ifd_with_new_subfile_type(0))
      .ok_or_else(|| RawlerError::DecoderFailed(format!("Failed to find a suitable IFD in NEF decoder")))?;
    let mut width = fetch_tiff_tag!(raw, TiffCommonTag::ImageWidth).force_usize(0);
    let height = fetch_tiff_tag!(raw, TiffCommonTag::ImageLength).force_usize(0);
    let bps = fetch_tiff_tag!(raw, TiffCommonTag::BitsPerSample).force_usize(0);
    let mut cpp = fetch_tiff_tag!(raw, TiffCommonTag::BitsPerSample).count(); // Linear files don't have SamplesPerPixel
    let compression = fetch_tiff_tag!(raw, TiffCommonTag::Compression).force_usize(0);

    let nef_compression = if let Some(z_makernote) = self.makernote.get_entry(NikonMakernote::Makernotes0x51) {
      // For new Z models, a new tag 0x51 for makernotes appears. This contains
      // The new-old NEFCompression tag. The old tag is unavailable in this models.
      Some(NefCompression::try_from(crate::bits::LEu16(z_makernote.get_data(), 10)).map_err(RawlerError::from)?)
    } else {
      self
        .makernote
        .get_entry(NikonMakernote::NefCompression)
        .map(|entry| entry.force_u16(0))
        .map(NefCompression::try_from)
        .transpose()
        .map_err(RawlerError::from)?
    };
    debug!("TIFF compression flag: {}, NEF compression mode: {:?}", compression, nef_compression);

    if matches!(nef_compression, Some(NefCompression::HighEfficency)) || matches!(nef_compression, Some(NefCompression::HighEfficencyStar)) {
      return Err(RawlerError::DecoderFailed(format!("NEF compression {:?} is not supported", nef_compression)));
    }

    let offset = fetch_tiff_tag!(raw, TiffCommonTag::StripOffsets).force_usize(0);
    let size = fetch_tiff_tag!(raw, TiffCommonTag::StripByteCounts).force_usize(0);
    let rows_per_strip = fetch_tiff_tag!(raw, TiffCommonTag::RowsPerStrip).get_usize(0).ok().flatten().unwrap_or(height);

    // That's little bit hacky here. Some files like D500 using multiple strips.
    // Because the strips has no holes between and are perfectly aligned, we can process the whole
    // chunk at once, instead of iterating over every strip.
    // It would be safer to process each strip offset, but it is not need for any known model so far.
    let src = if rows_per_strip == height {
      file.subview_padded(offset as u64, size as u64)?
    } else {
      let full_size: u32 = match fetch_tiff_tag!(raw, TiffCommonTag::StripByteCounts) {
        Value::Long(data) => data.iter().copied().sum(),
        _ => {
          return Err("StripByteCounts is not of type LONG".into());
        }
      };
      file.subview_padded(offset as u64, full_size as u64)?
    };

    let coeffs = normalize_wb(self.get_wb()?);
    debug!("WB coeff: {:?}", coeffs);

    assert_eq!(self.tiff.little_endian(), self.makernote.endian == Endian::Little);

    let mut linearization_table: Option<Vec<u16>> = None;
    let image = if self.camera.model == "NIKON D100" {
      width = 3040;
      decompress_12be_wcontrol(&src, width, height, dummy)?
    } else if self.camera.find_hint("coolpixsplit") {
      decompress_12be_interlaced_unaligned(&src, width, height, dummy)?
    } else if self.camera.find_hint("msb32") {
      decompress_12be_msb32(&src, width, height, dummy)?
    } else if self.camera.find_hint("unpacked") {
      // P7800 and others is LE, but data is BE, so we use hints here
      if (self.tiff.little_endian() || self.camera.find_hint("little_endian")) && !self.camera.find_hint("big_endian") {
        decompress_16le(&src, width, height, dummy)?
      } else {
        decompress_16be(&src, width, height, dummy)?
      }
    } else if let Some(padding) = self.is_uncompressed(raw)? {
      debug!("NEF uncompressed row padding: {}, little-endian: {}", padding, self.tiff.little_endian());
      match bps {
        16 => {
          // Used by Coolscan scanners
          if self.tiff.little_endian() {
            decompress_16le(&src, width * cpp, height, dummy)?
          } else {
            decompress_16be(&src, width * cpp, height, dummy)?
          }
        }
        14 => {
          if (self.tiff.little_endian() || self.camera.find_hint("little_endian")) && !self.camera.find_hint("big_endian") {
            // Models like D6 uses packed instead of unpacked 14le encoding. And D6 uses
            // row padding.
            if matches!(nef_compression, Some(NefCompression::Packed14Bits)) {
              decompress_14le_padded(&src, width, height, (width * bps / u8::BITS as usize) + padding, dummy)?
            } else {
              decompress_14le_unpacked(&src, width, height, dummy)?
            }
          } else {
            decompress_14be_unpacked(&src, width, height, dummy)?
          }
        }
        12 => {
          if (self.tiff.little_endian() || self.camera.find_hint("little_endian")) && !self.camera.find_hint("big_endian") {
            decompress_12le_padded(&src, width, height, (width * bps / u8::BITS as usize) + padding, dummy)?
          } else {
            decompress_12be(&src, width, height, dummy)?
          }
        }
        x => return Err(RawlerError::unsupported(&self.camera, format!("Don't know uncompressed bps {}", x))),
      }
    } else if size == width * height * 3 {
      cpp = 3;
      Self::decode_snef_compressed(&src, coeffs, width, height, dummy)?
    } else if compression == 34713 {
      let (pixels, table) = self.decode_compressed(&src, width, height, bps, dummy)?;
      linearization_table = table;
      pixels
    } else {
      return Err(RawlerError::unsupported(&self.camera, format!("NEF: Don't know compression {}", compression)));
    };

    assert_eq!(image.width, width * cpp);
    let blacklevel = self.get_blacklevel(bps)?;
    let whitelevel = None;
    let photometric = match cpp {
      1 => RawPhotometricInterpretation::Cfa(CFAConfig::new_from_camera(&self.camera)),
      3 => RawPhotometricInterpretation::LinearRaw,
      _ => todo!(),
    };
    let mut img = RawImage::new(self.camera.clone(), image, cpp, coeffs, photometric, blacklevel, whitelevel, dummy);
    img.linearization_table = linearization_table;

    if let Some(crop) = self.get_crop()? {
      debug!("RAW Crops: {:?}", crop);
      img.crop_area = Some(crop);
    }

    if cpp == 3 {
      // Reset levels to defaults (0)
      img.blacklevel = BlackLevel::zero(1, 1, cpp);
      img.whitelevel = WhiteLevel::new(vec![65535; cpp]);
    }

    // Compute per-ISO noise profile if no static one is defined in the camera TOML
    if img.camera.noise_profile.is_none() {
      let iso = self
        .tiff
        .root_ifd()
        .get_sub_ifd(ExifTag::ExifOffset)
        .and_then(|exif_ifd| exif_ifd.get_entry(ExifTag::ISOSpeedRatings))
        .map(|e| e.force_u32(0));
      if let Some(iso) = iso {
        if let Some(np) = noisecal::noise_profile_for_iso(&self.camera.model, iso) {
          debug!("Computed NoiseProfile for ISO {}: {:?}", iso, np);
          img.camera.noise_profile = Some(np);
        }
      }
    }

    Ok(img)
  }

  fn format_dump(&self) -> FormatDump {
    FormatDump::Nef(NefFormat { tiff: self.tiff.clone() })
  }

  fn raw_metadata(&self, _file: &RawSource, _params: &RawDecodeParams) -> Result<RawMetadata> {
    let exif = Exif::new(self.tiff.root_ifd())?;
    if let Ok(Some(shot_info)) = self.shot_info() {
      let ori = &shot_info.orientation;
      log::debug!(
        "NEF ShotInfo: firmware={}, roll={:.1}, pitch={:.1}, yaw={:.1}",
        shot_info.header.firmware_version,
        ori.roll_angle,
        ori.pitch_angle,
        ori.yaw_angle,
      );
    }
    Ok(match self.get_lens_description() {
      Ok(lens_data) => RawMetadata::new_with_lens(&self.camera, exif, lens_data.cloned()),
      Err(err) => {
        log::warn!("Failed to read lens information: {:?}", err);
        RawMetadata::new(&self.camera, exif)
      }
    })
  }

  fn preview_image(&self, file: &RawSource, params: &RawDecodeParams) -> Result<Option<DynamicImage>> {
    if params.image_index != 0 {
      return Ok(None);
    }
    // High resolution preview image is stored in JPEGInterchangeFormat tag.
    // Search for all IFDs and use the best match.
    let mut ifds = self.tiff.find_ifds_with_filter(|ifd| {
      if ifd.get_new_sub_file_type() == Some(1) {
        ifd.get_entry(ExifTag::JPEGInterchangeFormatLength).is_some()
      } else {
        false
      }
    });

    ifds.sort_by(|a, b| {
      a.get_entry(ExifTag::JPEGInterchangeFormatLength)
        .map(|x| x.force_u32(0))
        .cmp(&b.get_entry(ExifTag::JPEGInterchangeFormatLength).map(|x| x.force_u32(0)))
    });

    // Take the IFD with the largest JPEG stream size
    if let Some(jpeg_ifd) = ifds.last() {
      return Ok(Some(dynamic_image_from_jpeg_interchange_format(jpeg_ifd, file)?));
    } else {
      // No matching IFDs found, use root IFD (possibly bad resolution)
      Ok(Some(dynamic_image_from_ifd(self.tiff.root_ifd(), file)?))
    }
  }

  fn preview_jpeg(&self, file: &RawSource, params: &RawDecodeParams) -> Result<Option<(Vec<u8>, u32, u32)>> {
    if params.image_index != 0 {
      return Ok(None);
    }
    let mut ifds = self.tiff.find_ifds_with_filter(|ifd| {
      if ifd.get_new_sub_file_type() == Some(1) {
        ifd.get_entry(ExifTag::JPEGInterchangeFormatLength).is_some()
      } else {
        false
      }
    });
    ifds.sort_by(|a, b| {
      a.get_entry(ExifTag::JPEGInterchangeFormatLength)
        .map(|x| x.force_u32(0))
        .cmp(&b.get_entry(ExifTag::JPEGInterchangeFormatLength).map(|x| x.force_u32(0)))
    });
    if let Some(jpeg_ifd) = ifds.last() {
      let offset = fetch_tiff_tag!(jpeg_ifd, ExifTag::JPEGInterchangeFormat).force_usize(0) as u64;
      let size = fetch_tiff_tag!(jpeg_ifd, ExifTag::JPEGInterchangeFormatLength).force_usize(0) as u64;
      let buf = file.subview(offset, size)?;
      let mut width = jpeg_ifd.get_entry(ExifTag::ImageWidth).map(|e| e.force_u32(0)).unwrap_or(0);
      let mut height = jpeg_ifd.get_entry(ExifTag::ImageHeight).map(|e| e.force_u32(0)).unwrap_or(0);
      // If IFD doesn't have dimensions, parse from JPEG SOF marker
      if (width == 0 || height == 0) && buf.len() > 2 {
        let (w, h) = jpeg_dimensions(buf);
        width = w;
        height = h;
      }
      Ok(Some((buf.to_vec(), width, height)))
    } else {
      Ok(None)
    }
  }

  fn format_hint(&self) -> FormatHint {
    FormatHint::NEF
  }

  fn xpacket(&self, _file: &RawSource, _params: &RawDecodeParams) -> crate::Result<Option<Vec<u8>>> {
    Ok(self.tiff.get_entry(TiffCommonTag::Xmp).map(|e| e.get_data().to_vec()))
  }

  fn ifd(&self, wk_ifd: WellKnownIFD) -> crate::Result<Option<Rc<IFD>>> {
    match wk_ifd {
      WellKnownIFD::VirtualDngRootTags => {
        let mut ifd = IFD::default();
        // MakerNoteSafety = 1 tells the DNG converter that the Nikon MakerNote
        // (including encrypted ShotInfo, NefMeta, ContrastCurve, LensData, etc.)
        // has absolute offsets and is safe to preserve verbatim in the output DNG.
        ifd.entries.insert(
          DngTag::MakerNoteSafety.into(),
          Entry { tag: DngTag::MakerNoteSafety.into(), value: Value::Short(vec![1]), embedded: None },
        );

        // Extract NefMeta1 tone curve control points as DNG ProfileToneCurve.
        // NefMeta1 layout: byte 8 = num_points, bytes 10+ = (input, output) u8 pairs.
        // The last point is a sentinel (output=0) and is skipped.
        if let Some(meta1) = self.makernote.get_entry(TiffCommonTag::NefMeta1) {
          let data = meta1.get_data();
          if data.len() >= 10 {
            let num_points = data[8] as usize;
            if data.len() >= 10 + num_points * 2 {
              let mut curve_points: Vec<f32> = Vec::new();
              // Add implicit (0.0, 0.0) start point
              curve_points.push(0.0);
              curve_points.push(0.0);
              for i in 0..num_points {
                let input = data[10 + i * 2] as f32 / 255.0;
                let output = data[10 + i * 2 + 1] as f32 / 255.0;
                // Skip sentinel points (output == 0 with non-zero input)
                if output == 0.0 && input > 0.0 {
                  continue;
                }
                curve_points.push(input);
                curve_points.push(output);
              }
              // Add implicit (1.0, 1.0) end point if not already present
              if curve_points.len() >= 2 {
                let last_in = curve_points[curve_points.len() - 2];
                let last_out = curve_points[curve_points.len() - 1];
                if last_in < 1.0 || last_out < 1.0 {
                  curve_points.push(1.0);
                  curve_points.push(1.0);
                }
              }
              if curve_points.len() >= 4 {
                ifd.entries.insert(
                  DngTag::ProfileToneCurve.into(),
                  Entry {
                    tag: DngTag::ProfileToneCurve.into(),
                    value: Value::Float(curve_points),
                    embedded: None,
                  },
                );
              }
            }
          }
        }

        // Extract ICC profile from source file as AsShotICCProfile.
        if let Some(icc_entry) = self.tiff.get_entry(ExifTag::IccProfile) {
          let icc_data = icc_entry.get_data().to_vec();
          if !icc_data.is_empty() {
            ifd.entries.insert(
              DngTag::AsShotICCProfile.into(),
              Entry {
                tag: DngTag::AsShotICCProfile.into(),
                value: Value::Undefined(icc_data),
                embedded: None,
              },
            );
          }
        }

        Ok(Some(Rc::new(ifd)))
      }
      WellKnownIFD::VirtualDngRawTags => {
        let (opc1, opc3) = if !self.opcode_list1.is_empty() || !self.opcode_list3.is_empty() {
          // Use embedded Nikon correction data (tag 0xc7d5)
          (self.opcode_list1.clone(), self.opcode_list3.clone())
        } else {
          // Fall back to Adobe LCP lens correction profile
          self.lcp_fallback_opcodes()
        };

        if opc1.is_empty() && opc3.is_empty() {
          return Ok(None);
        }
        let mut ifd = IFD::default();
        if !opc1.is_empty() {
          ifd.entries.insert(
            DngTag::OpcodeList1.into(),
            Entry { tag: DngTag::OpcodeList1.into(), value: Value::Undefined(opc1), embedded: None },
          );
        }
        if !opc3.is_empty() {
          ifd.entries.insert(
            DngTag::OpcodeList3.into(),
            Entry { tag: DngTag::OpcodeList3.into(), value: Value::Undefined(opc3), embedded: None },
          );
        }
        Ok(Some(Rc::new(ifd)))
      }
      _ => Ok(None),
    }
  }
}

impl<'a> NefDecoder<'a> {
  /// For older formats, we use the camera definitions and this here
  /// is useless. But if we found here the levels in makernotes, we
  /// use these instead. For 12 bit images, the blacklevels are still relative to
  /// 14 bit image data. So we need to reduce them by 2 bits.
  fn get_blacklevel(&self, bps: usize) -> Result<Option<BlackLevel>> {
    if let Some(levels) = self.makernote.get_entry(NikonMakernote::BlackLevel) {
      let mut black = [levels.force_u16(0), levels.force_u16(1), levels.force_u16(2), levels.force_u16(3)];
      if bps == 12 {
        black.iter_mut().for_each(|v| *v >>= 14 - 12);
      }
      Ok(Some(BlackLevel::new(&black, self.camera.cfa.width, self.camera.cfa.height, 1)))
    } else {
      Ok(None)
    }
  }

  fn get_crop(&self) -> Result<Option<Rect>> {
    if let Some(crop) = self.makernote.get_entry(NikonMakernote::CropArea) {
      let values = [crop.force_u16(0), crop.force_u16(1), crop.force_u16(2), crop.force_u16(3)];
      let rect = Rect::new(
        Point::new(values[0] as usize, values[1] as usize),
        Dim2::new(values[2] as usize, values[3] as usize),
      );
      Ok(Some(rect))
    } else {
      Ok(None)
    }
  }

  /// Get lens description by analyzing TIFF tags and makernotes
  /// Decrypt and parse the Nikon ShotInfo tag (0x0091).
  ///
  /// Returns `None` if the tag is absent or the version is not supported.
  /// Currently handles version "0808" (Nikon Zf) and "0803" (Z7 II / Z6 II).
  pub fn shot_info(&self) -> Result<Option<shotinfo::NefShotInfoZ7II>> {
    shotinfo::parse_shot_info(&self.makernote)
  }

  fn get_lens_description(&self) -> Result<Option<&'static LensDescription>> {
    if let Some(lensdata) = lensdata::from_makernote(&self.makernote)? {
      if let Some(lenstype) = self.makernote.get_entry(NikonMakernote::LensType) {
        match lensdata {
          NefLensData::FMount(oldv) => {
            let composite_id = oldv.composite_id(lenstype.force_u8(0));
            log::debug!("NEF lens composite ID: {}", composite_id);
            let resolver = LensResolver::new()
              .with_nikon_id(Some(composite_id))
              .with_camera(&self.camera)
              .with_mounts(&[NIKON_F_MOUNT.into()]);
            return Ok(resolver.resolve());
          }
          NefLensData::ZMount(newv) => {
            let resolver = LensResolver::new()
              .with_lens_id((newv.lens_id as u32, 0))
              .with_camera(&self.camera)
              .with_mounts(&[NIKON_Z_MOUNT.into()]);
            return Ok(resolver.resolve());
          }
        }
      }
    }
    Ok(None)
  }

  /// Fall back to Adobe LCP lens correction profiles when embedded Nikon
  /// correction data (tag 0xc7d5) is absent.
  fn lcp_fallback_opcodes(&self) -> (Vec<u8>, Vec<u8>) {
    let lens = match self.get_lens_description() {
      Ok(Some(l)) => l,
      _ => return Default::default(),
    };

    // Get focal length and aperture from EXIF
    let exif_ifd = self.tiff.root_ifd().get_sub_ifd(ExifTag::ExifOffset);
    let focal_mm = exif_ifd
      .as_ref()
      .and_then(|ifd| ifd.get_entry(ExifTag::FocalLength))
      .and_then(|e| match &e.value {
        Value::Rational(r) => r.first().map(|r| r.as_f32() as f64),
        _ => None,
      })
      .unwrap_or(0.0);

    let aperture = exif_ifd
      .as_ref()
      .and_then(|ifd| ifd.get_entry(ExifTag::FNumber))
      .and_then(|e| match &e.value {
        Value::Rational(r) => r.first().map(|r| r.as_f32() as f64),
        _ => None,
      })
      .unwrap_or(0.0);

    if focal_mm <= 0.0 {
      return Default::default();
    }

    // Get image dimensions from the raw IFD
    let raw_ifd = self
      .tiff
      .find_first_ifd_with_tag(TiffCommonTag::CFAPattern)
      .or_else(|| self.tiff.find_ifd_with_new_subfile_type(0));
    let (width, height) = match raw_ifd {
      Some(ifd) => {
        let w = ifd.get_entry(TiffCommonTag::ImageWidth).map(|e| e.force_u32(0)).unwrap_or(0);
        let h = ifd.get_entry(TiffCommonTag::ImageLength).map(|e| e.force_u32(0)).unwrap_or(0);
        (w, h)
      }
      None => return Default::default(),
    };

    if width == 0 || height == 0 {
      return Default::default();
    }

    log::debug!(
      "LCP fallback: lens='{}', focal={:.1}mm, f/{:.1}, {}x{}",
      lens.lens_name, focal_mm, aperture, width, height
    );

    match crate::lens_profiles::lookup_lens_opcodes(&lens.lens_name, focal_mm, aperture, width, height) {
      Some(opcodes) => {
        log::info!("Using Adobe LCP correction for '{}'", lens.lens_name);
        (opcodes.opcode_list1, opcodes.opcode_list3)
      }
      None => Default::default(),
    }
  }

  fn get_wb(&self) -> Result<[f32; 4]> {
    if self.camera.find_hint("nowb") {
      Ok([f32::NAN, f32::NAN, f32::NAN, f32::NAN])
    } else if let Some(levels) = self.makernote.get_entry(TiffCommonTag::NefWB0) {
      Ok([levels.force_f32(0), 1.0, 1.0, levels.force_f32(1)])
    } else if let Some(levels) = self.makernote.get_entry(TiffCommonTag::NrwWB) {
      let data = levels.get_data();
      if data[0..3] == b"NRW"[..] {
        let offset = if data[4..8] == b"0100"[..] { 1556 } else { 56 };
        Ok([
          (LEu32(data, offset) << 2) as f32,
          (LEu32(data, offset + 4) + LEu32(data, offset + 8)) as f32,
          (LEu32(data, offset + 4) + LEu32(data, offset + 8)) as f32,
          (LEu32(data, offset + 12) << 2) as f32,
        ])
      } else {
        Ok([BEu16(data, 1248) as f32, 256.0, 256.0, BEu16(data, 1250) as f32])
      }
    } else if let Some(levels) = self.makernote.get_entry(TiffCommonTag::NefWB1) {
      let mut version: u32 = 0;
      for i in 0..4 {
        version = (version << 4) + (levels.get_data()[i] - b'0') as u32;
      }
      let buf = levels.get_data();
      debug!("NEF Color balance version: 0x{:x}", version);

      match version {
        0x100 => Ok([
          BEu16(buf, 36 * 2) as f32,
          BEu16(buf, 38 * 2) as f32,
          BEu16(buf, 38 * 2) as f32,
          BEu16(buf, 37 * 2) as f32,
        ]),
        // Nikon D2H
        0x102 => Ok([
          BEu16(buf, 5 * 2) as f32,
          BEu16(buf, 6 * 2) as f32,
          BEu16(buf, 6 * 2) as f32,
          BEu16(buf, 8 * 2) as f32,
        ]),
        // Nikon D70
        0x103 => Ok([
          BEu16(buf, 10 * 2) as f32,
          BEu16(buf, 11 * 2) as f32,
          BEu16(buf, 11 * 2) as f32,
          BEu16(buf, 12 * 2) as f32,
        ]),
        0x803 => {
          let data = levels.get_data();
          let green = LEu16(data, 56) as f32;
          if green == 0.0 {
            Err(RawlerError::unsupported(&self.camera, "NEF: 0x0803 WB block has zero green reference".to_string()))
          } else {
            Ok([
              LEu16(data, 44) as f32 / green,
              1.0,
              1.0,
              LEu16(data, 46) as f32 / green,
            ])
          }
        }
        0x204 | 0x205 => {
          let serial = fetch_tiff_tag!(self.makernote, TiffCommonTag::NefSerial);
          let data = serial.get_data();
          let mut serialno = 0_usize;
          for i in 0..serial.count() as usize {
            if data[i] == 0 {
              break;
            }
            serialno = serialno * 10
              + if data[i] >= 48 && data[i] <= 57 {
                // "0" to "9"
                (data[i] - 48) as usize
              } else {
                (data[i] % 10) as usize
              };
          }

          // Get the "decryption" key
          let keydata = fetch_tiff_tag!(self.makernote, TiffCommonTag::NefKey).force_u32(0).to_le_bytes();
          let keyno = (keydata[0] ^ keydata[1] ^ keydata[2] ^ keydata[3]) as usize;

          let src = if version == 0x204 {
            &levels.get_data()[284..]
          } else {
            &levels.get_data()[4..]
          };

          let ci = WB_SERIALMAP[serialno & 0xff] as u32;
          let mut cj = WB_KEYMAP[keyno & 0xff] as u32;
          let mut ck = 0x60_u32;
          let mut buf = [0_u8; 280];
          for i in 0..280 {
            cj += ci * ck;
            ck += 1;
            buf[i] = src[i] ^ (cj as u8);
          }

          let off = if version == 0x204 { 6 } else { 14 };
          Ok([
            BEu16(&buf, off) as f32,
            BEu16(&buf, off + 2) as f32,
            BEu16(&buf, off + 4) as f32,
            BEu16(&buf, off + 6) as f32,
          ])
        }
        x => Err(RawlerError::unsupported(&self.camera, format!("NEF: Don't know about WB version 0x{:x}", x))),
      }
    } else {
      log::debug!("NEF: Don't know how to fetch WB, fallback to [1.0, 1.0, 1.0]");
      Ok([1.0, 1.0, 1.0, 1.0])
    }
  }

  fn create_hufftable(num: usize) -> std::result::Result<HuffTable, String> {
    let mut htable = HuffTable::empty();

    for i in 0..15 {
      htable.bits[i] = NIKON_TREE[num][0][i] as u32;
      htable.huffval[i] = NIKON_TREE[num][1][i] as u32;
      htable.shiftval[i] = NIKON_TREE[num][2][i] as u32;
    }

    htable.initialize()?;
    Ok(htable)
  }

  /// The compression flags in some raws are not reliable because of firmware bugs.
  /// We try to figure out the compression by some heuristics.
  /// The return value is None if the file is not uncompressed or Some(x)
  /// where x is the extra amount of bytes after each row.
  fn is_uncompressed(&self, raw: &IFD) -> Result<Option<usize>> {
    let width = fetch_tiff_tag!(raw, TiffCommonTag::ImageWidth).force_usize(0);
    let height = fetch_tiff_tag!(raw, TiffCommonTag::ImageLength).force_usize(0);
    let bps = fetch_tiff_tag!(raw, TiffCommonTag::BitsPerSample).force_usize(0);
    let compression = fetch_tiff_tag!(raw, TiffCommonTag::Compression).force_usize(0);
    let size = fetch_tiff_tag!(raw, TiffCommonTag::StripByteCounts).force_usize(0);

    fn div_round_up(a: usize, b: usize) -> usize {
      a.div_ceil(b) // (a + b - 1) / b
    }

    let req_pixels = width * height;
    let req_input_bits = bps * req_pixels;
    let req_input_bytes = div_round_up(req_input_bits, 8);

    Ok(if compression == 1 || size == width * height * bps / 8 {
      Some(0)
    } else if size >= req_input_bytes {
      // Some models (D6) using row padding, so the row width is slightly larger.
      // This should be no more than 16 extra bytes.
      let total_padding = size - req_input_bytes;
      let per_row_padding = total_padding / height;
      if total_padding % height != 0 {
        None
      } else if per_row_padding < 16 {
        Some(per_row_padding)
      } else {
        None
      }
    } else {
      None
    })
  }

  fn decode_compressed(&self, src: &PaddedBuf, width: usize, height: usize, bps: usize, dummy: bool) -> std::result::Result<(PixU16, Option<Vec<u16>>), String> {
    let meta = if let Some(meta) = self.makernote.get_entry(TiffCommonTag::NefMeta2) {
      debug!("Found NefMeta2");
      meta
    } else {
      debug!("Fallback NefMeta1");
      fetch_tiff_tag!(self.makernote, TiffCommonTag::NefMeta1)
    };
    Self::do_decode(src, meta.get_data(), self.makernote.endian, width, height, bps, dummy)
  }

  pub(crate) fn do_decode(
    src: &[u8],
    meta: &[u8],
    endian: Endian,
    width: usize,
    height: usize,
    bps: usize,
    dummy: bool,
  ) -> std::result::Result<(PixU16, Option<Vec<u16>>), String> {
    debug!("NEF decode with: endian: {:?}, width: {}, height: {}, bps: {}", endian, width, height, bps);
    if dummy {
      return Ok((PixU16::new_uninit(width, height), None));
    }
    let mut out = alloc_image_plain!(width, height, dummy);
    let mut stream = ByteStream::new(meta, endian);
    let v0 = stream.get_u8();
    let v1 = stream.get_u8();
    debug!("Nef version v0:{}, v1:{}", v0, v1);

    let mut huff_select = 0;
    if v0 == 73 || v1 == 88 {
      assert!(stream.remaining_bytes() >= 2110);
      stream.consume_bytes(2110);
    }
    if v0 == 70 {
      huff_select = 2;
    }
    if bps == 14 {
      huff_select += 3;
    }

    // Create the huffman table used to decode
    let mut htable = Self::create_hufftable(huff_select)?;

    // Setup the predictors
    let mut pred_up1: [i32; 2] = [stream.get_u16() as i32, stream.get_u16() as i32];
    let mut pred_up2: [i32; 2] = [stream.get_u16() as i32, stream.get_u16() as i32];

    // Get the linearization curve
    let mut points = [0_u16; 1 << 16];
    for i in 0..points.len() {
      points[i] = i as u16;
    }

    // Some models reports 14 bits, but the data is 12 bits.
    // So we reduce the bps to calculate the max value which
    // is needed in the next steps.
    let real_bps = if v0 == 68 && v1 == 64 {
      bps as u32 - 2 // Special for D780, Z7 and others
    } else {
      bps as u32
    };
    let mut max = 1 << real_bps;

    let csize = stream.get_u16() as usize;
    let mut split = 0_usize;
    let step = if csize > 1 { max / (csize - 1) } else { 0 };
    if v0 == 68 && (v1 == 32 || v1 == 64) && step > 0 {
      for i in 0..csize {
        points[i * step] = stream.get_u16();
      }
      for i in 0..max {
        let b_scale = i % step;
        let a_pos = i - b_scale;
        let b_pos = a_pos + step;
        //assert!(a_pos < max);
        //assert!(b_pos > 0);
        //assert!(b_pos < max);
        //assert!(a_pos < b_pos);
        let a_scale = step - b_scale;
        points[i] = ((a_scale * points[a_pos] as usize + b_scale * points[b_pos] as usize) / step) as u16;
      }
      split = endian.read_u16(meta, 562) as usize;
    } else if v0 != 70 && csize <= 0x4001 {
      for i in 0..csize {
        points[i] = stream.get_u16();
      }
      max = csize;
    }
    let curve = LookupTable::new(&points[0..max]);
    let is_identity = points[0..max].iter().enumerate().all(|(i, &v)| v == i as u16);
    let linearization_table = if is_identity { None } else { Some(points[0..max].to_vec()) };

    let mut pump = BitPumpMSB::new(src);
    let mut random = pump.peek_bits(24);

    for row in 0..height {
      if split > 0 && row == split {
        htable = Self::create_hufftable(huff_select + 1)?;
      }
      pred_up1[row & 1] += htable.huff_decode(&mut pump)?;
      pred_up2[row & 1] += htable.huff_decode(&mut pump)?;
      let mut pred_left1 = pred_up1[row & 1];
      let mut pred_left2 = pred_up2[row & 1];
      for col in (0..width).step_by(2) {
        if col > 0 {
          pred_left1 += htable.huff_decode(&mut pump)?;
          pred_left2 += htable.huff_decode(&mut pump)?;
        }
        if is_identity {
          out[row * width + col + 0] = curve.dither(clampbits(pred_left1, real_bps), &mut random);
          out[row * width + col + 1] = curve.dither(clampbits(pred_left2, real_bps), &mut random);
        } else {
          out[row * width + col + 0] = clampbits(pred_left1, real_bps);
          out[row * width + col + 1] = clampbits(pred_left2, real_bps);
        }
      }
    }

    Ok((out, linearization_table))
  }

  // Decodes 12 bit data in an YUY2-like pattern (2 Luma, 1 Chroma per 2 pixels).
  // We un-apply the whitebalance, so output matches lossless.
  pub(crate) fn decode_snef_compressed(src: &PaddedBuf, coeffs: [f32; 4], width: usize, height: usize, dummy: bool) -> std::result::Result<PixU16, String> {
    let inv_wb_r = (1024.0 / coeffs[0]) as i32;
    let inv_wb_b = (1024.0 / coeffs[2]) as i32;

    //println!("Got invwb {} {}", inv_wb_r, inv_wb_b);

    let snef_curve = {
      let g: f32 = 2.4;
      let f: f32 = 0.055;
      let min: f32 = 0.04045;
      let mul: f32 = 12.92;
      let curve = (0..4096)
        .map(|i| {
          let v = (i as f32) / 4095.0;
          let res = if v <= min { v / mul } else { ((v + f) / (1.0 + f)).powf(g) };
          clampbits((res * 65535.0 * 4.0) as i32, 16)
        })
        .collect::<Vec<u16>>();
      LookupTable::new(&curve)
    };

    decompress_lines_fn(
      width * 3,
      height,
      dummy,
      &(|out: &mut [u16], row| {
        let inb = &src[row * width * 3..];
        let mut random = BEu32(inb, 0);
        for (o, i) in out.chunks_exact_mut(6).zip(inb.chunks_exact(6)) {
          let g1: u16 = i[0] as u16;
          let g2: u16 = i[1] as u16;
          let g3: u16 = i[2] as u16;
          let g4: u16 = i[3] as u16;
          let g5: u16 = i[4] as u16;
          let g6: u16 = i[5] as u16;

          let y1 = (g1 | ((g2 & 0x0f) << 8)) as f32;
          let y2 = ((g2 >> 4) | (g3 << 4)) as f32;
          let cb = (g4 | ((g5 & 0x0f) << 8)) as f32 - 2048.0;
          let cr = ((g5 >> 4) | (g6 << 4)) as f32 - 2048.0;

          let r = snef_curve.dither(clampbits((y1 + 1.370705 * cr) as i32, 12), &mut random);
          let g = snef_curve.dither(clampbits((y1 - 0.337633 * cb - 0.698001 * cr) as i32, 12), &mut random);
          let b = snef_curve.dither(clampbits((y1 + 1.732446 * cb) as i32, 12), &mut random);
          // invert the white balance
          o[0] = clampbits((inv_wb_r * r as i32 + (1 << 9)) >> 10, 15);
          o[1] = g;
          o[2] = clampbits((inv_wb_b * b as i32 + (1 << 9)) >> 10, 15);

          let r = snef_curve.dither(clampbits((y2 + 1.370705 * cr) as i32, 12), &mut random);
          let g = snef_curve.dither(clampbits((y2 - 0.337633 * cb - 0.698001 * cr) as i32, 12), &mut random);
          let b = snef_curve.dither(clampbits((y2 + 1.732446 * cb) as i32, 12), &mut random);
          // invert the white balance
          o[3] = clampbits((inv_wb_r * r as i32 + (1 << 9)) >> 10, 15);
          o[4] = g;
          o[5] = clampbits((inv_wb_b * b as i32 + (1 << 9)) >> 10, 15);
        }
        Ok(())
      }),
    )
  }
}

/// Parse the NikonNEFInfo blob (tag 0xc7d5 in the raw SubIFD of Z-series NEF files)
/// and build OpcodeList1 (vignette) and OpcodeList3 (distortion) blobs.
///
/// NikonNEFInfo layout:
///   bytes  0- 5:  "Nikon\0"
///   bytes  6- 7:  version (e.g. 0x01 0x03)
///   bytes  8- 9:  0x00 0x00
///   bytes 10-11:  endian marker "II" (LE) or "MM" (BE)
///   bytes 12-13:  TIFF magic (42 LE)
///   bytes 14-17:  IFD offset from "II" marker (always 8 → IFD at byte 18)
///   bytes 18+  :  standard TIFF IFD
///     entry 0x0005 → DistortionInfo (UNDEFINED, 84 bytes)
///     entry 0x0006 → VignetteInfo   (UNDEFINED, 116 bytes)
///
/// DistortionInfo layout (LE):
///   0x00-0x03: version string "0100"
///   0x04:      DistortionCorrection flag (1 = on optional, 3 = on required)
///   0x10:      u32 number of coefficients (typically 4; last is often 0)
///   0x14 + 8×i: i32 numerator, i32 denominator  (rational64s, denom = 1048576)
///
/// VignetteInfo layout (LE):
///   0x00-0x03: version string "0100"
///   0x10:      u32 polynomial degree (always 8 → 4 coefficients for r²,r⁴,r⁶,r⁸)
///   0x24 + 16×j: i32 numerator, i32 denominator  (rational64s, denom = 1048576)
///     (j=0 → k0, j=1 → k1, j=2 → k2, optional j=3 → k3 usually 0)
///
/// Coefficient interpretation (see DNG 1.3 spec, opcodes 1 and 3):
///   FixVignetteRadial: pixel *= 1 + k0·r² + k1·r⁴ + k2·r⁶ (positive = brighten)
///   WarpRectilinear:   x' = cx + m·(kr0 + kr1·r² + kr2·r⁴ + kr3·r⁶)·dx
///     where kr0=1.0 means no overall scaling; Nikon d1,d2,d3 → kr1,kr2,kr3.
///
/// NOTE: Nikon's exact polynomial model is not publicly documented.  The
/// coefficient mapping here is a best-effort approximation based on reverse-
/// engineering by the ExifTool community (ref [28]) and matches ACR's
/// behaviour (which reads the DistortionCorrection flag but applies its own
/// built-in profile rather than these raw coefficients).
fn parse_nikon_nef_opcodes(data: &[u8]) -> Option<(Vec<u8>, Vec<u8>)> {
  // Verify the "Nikon\0" magic
  if data.len() < 20 || &data[0..6] != b"Nikon\0" {
    return None;
  }
  // Only handle little-endian sub-IFD for now
  if &data[10..12] != b"II" {
    debug!("parse_nikon_nef_opcodes: big-endian NikonNEFInfo not supported");
    return None;
  }
  // IFD offset is relative to the "II" marker at data[10]
  let ifd_offset = u32::from_le_bytes(data[14..18].try_into().ok()?) as usize;
  let ii_base = 10usize;
  let ifd_start = ii_base + ifd_offset; // = 18

  if ifd_start + 2 > data.len() {
    return None;
  }
  let num_entries = u16::from_le_bytes(data[ifd_start..ifd_start + 2].try_into().ok()?) as usize;

  let mut dist_blob: Option<&[u8]> = None;
  let mut vig_blob: Option<&[u8]> = None;

  for i in 0..num_entries {
    let e = ifd_start + 2 + i * 12;
    if e + 12 > data.len() {
      break;
    }
    let tag = u16::from_le_bytes(data[e..e + 2].try_into().ok()?);
    let typ = u16::from_le_bytes(data[e + 2..e + 4].try_into().ok()?);
    let count = u32::from_le_bytes(data[e + 4..e + 8].try_into().ok()?) as usize;
    let raw_offset = u32::from_le_bytes(data[e + 8..e + 12].try_into().ok()?) as usize;

    if typ == 7 && count > 4 {
      // UNDEFINED with external offset (relative to "II" marker at data[10])
      let blob_start = ii_base + raw_offset;
      let blob_end = blob_start + count;
      if blob_end <= data.len() {
        match tag {
          5 => dist_blob = Some(&data[blob_start..blob_end]),
          6 => vig_blob = Some(&data[blob_start..blob_end]),
          _ => {}
        }
      }
    }
  }

  let opcode_list3 = dist_blob.and_then(build_warp_rectilinear_opcode).unwrap_or_default();
  let opcode_list1 = vig_blob.and_then(build_fix_vignette_opcode).unwrap_or_default();

  if opcode_list1.is_empty() && opcode_list3.is_empty() {
    return None;
  }
  Some((opcode_list1, opcode_list3))
}

/// Read a `rational64s` (pair of i32 LE) from a blob at byte offset `off`.
/// Returns `None` if denominator is zero or data is too short.
fn read_rational64s(blob: &[u8], off: usize) -> Option<f64> {
  if off + 8 > blob.len() {
    return None;
  }
  let num = i32::from_le_bytes(blob[off..off + 4].try_into().ok()?) as f64;
  let den = i32::from_le_bytes(blob[off + 4..off + 8].try_into().ok()?) as f64;
  if den == 0.0 {
    None
  } else {
    Some(num / den)
  }
}

/// Build a WarpRectilinear OpcodeList blob from a DistortionInfo blob.
///
/// The correction flag at offset 0x04 must be 1 or 3 (on); if it is 0 or 2
/// (no lens / off) the function returns `None` so no opcode is written.
fn build_warp_rectilinear_opcode(blob: &[u8]) -> Option<Vec<u8>> {
  if blob.len() < 0x2C {
    return None;
  }
  // DistortionCorrection flag: 0=no lens, 1=on optional, 2=off, 3=on required
  let dc_flag = blob[0x04];
  if dc_flag == 0 || dc_flag == 2 {
    debug!("NEF DistortionInfo: correction is off (flag={}), skipping WarpRectilinear opcode", dc_flag);
    return None;
  }

  // Three radial correction coefficients at rational64s offsets 0x14, 0x1C, 0x24
  let d1 = read_rational64s(blob, 0x14).unwrap_or(0.0);
  let d2 = read_rational64s(blob, 0x1C).unwrap_or(0.0);
  let d3 = read_rational64s(blob, 0x24).unwrap_or(0.0);

  // DNG WarpRectilinear: x' = cx + m·(kr0 + kr1·r² + kr2·r⁴ + kr3·r⁶)·dx
  // kr0 = 1.0 (identity scale), Nikon coefficients map to kr1..kr3.
  let kr = [[1.0_f64, d1, d2, d3]];
  let kt = [[0.0_f64, 0.0_f64]];

  log::debug!("NEF WarpRectilinear: kr1={:.5} kr2={:.5} kr3={:.5}", d1, d2, d3);

  let opcode = opcodes::encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
  Some(opcodes::encode_opcode_list(&[opcode]))
}

/// Build a FixVignetteRadial OpcodeList blob from a VignetteInfo blob.
fn build_fix_vignette_opcode(blob: &[u8]) -> Option<Vec<u8>> {
  if blob.len() < 0x4C {
    return None;
  }

  // Three (or four) vignette coefficients.
  // Offsets 0x24, 0x34, 0x44 are standard rational64s slots (ExifTool ref [28]).
  // The optional 4th coefficient is at 0x4C with an alternative denominator of 1
  // (not 1048576 like the others): bytes 0x4C-0x4F = numerator, 0x50-0x53 = denominator.
  // ExifTool notes it "seems to always be 0".
  let k0 = read_rational64s(blob, 0x24).unwrap_or(0.0);
  let k1 = read_rational64s(blob, 0x34).unwrap_or(0.0);
  let k2 = read_rational64s(blob, 0x44).unwrap_or(0.0);
  // 4th coefficient: denominator is 1 (not 1048576), so typically = 0/1 = 0
  let k3 = if blob.len() >= 0x54 { read_rational64s(blob, 0x4C).unwrap_or(0.0) } else { 0.0 };

  log::debug!("NEF FixVignetteRadial: k0={:.5} k1={:.5} k2={:.5} k3={:.5}", k0, k1, k2, k3);

  let opcode = opcodes::encode_fix_vignette_radial(k0, k1, k2, k3, 0.0, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
  Some(opcodes::encode_opcode_list(&[opcode]))
}

fn normalize_wb(raw_wb: [f32; 4]) -> [f32; 4] {
  debug!("NEF raw wb: {:?}", raw_wb);
  // We never have more then RGB colors so far (no RGBE etc.)
  // So we combine G1 and G2 to get RGB wb.
  let div = raw_wb[1];
  let mut norm = raw_wb;
  norm.iter_mut().for_each(|v| {
    if v.is_normal() {
      *v /= div
    }
  });
  [norm[0], (norm[1] + norm[2]) / 2.0, norm[3], f32::NAN]
}

crate::tags::tiff_tag_enum!(NikonMakernote);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum NikonMakernote {
  MakernoteVersion = 0x0001,
  NefWB0 = 0x000C,
  PreviewIFD = 0x0011,
  NrwWB = 0x0014,
  NefSerial = 0x001d,
  ImageSizeRaw = 0x003e,
  CropArea = 0x0045,
  BlackLevel = 0x003d,
  Makernotes0x51 = 0x0051,
  LensType = 0x0083,
  NefMeta1 = 0x008c,
  NefMeta2 = 0x0096,
  ShotInfo = 0x0091,
  NefCompression = 0x0093,
  NefWB1 = 0x0097,
  LensData = 0x0098,
  NefKey = 0x00a7,
}

/// Known NEF compression formats
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum NefCompression {
  LossyType1 = 1,
  Uncompressed = 2,
  Lossless = 3,
  LossyType2 = 4,
  StripedPacked12Bits = 5,
  UncompressedReduced12Bits = 6,
  Unpacked12Bits = 7,
  Small = 8,
  Packed12Bits = 9,
  Packed14Bits = 10,
  HighEfficency = 13,
  HighEfficencyStar = 14,
}

impl TryFrom<u16> for NefCompression {
  type Error = String;

  fn try_from(v: u16) -> std::result::Result<Self, Self::Error> {
    Ok(match v {
      1 => Self::LossyType1,
      2 => Self::Uncompressed,
      3 => Self::Lossless,
      4 => Self::LossyType2,
      5 => Self::StripedPacked12Bits,
      6 => Self::UncompressedReduced12Bits,
      7 => Self::Unpacked12Bits,
      8 => Self::Small,
      9 => Self::Packed12Bits,
      10 => Self::Packed14Bits,
      13 => Self::HighEfficency,
      14 => Self::HighEfficencyStar,
      _ => return Err(format!("unknown nef compression: {}", v)),
    })
  }
}

// Re-export the shared helper for local use
use super::jpeg_dimensions;
