mod noisecal;

use std::cmp;
use std::collections::HashMap;
use std::io::Read;
use std::io::Seek;
use std::rc::Rc;

use crate::RawImage;
use crate::RawLoader;
use crate::RawlerError;
use crate::Result;
use crate::alloc_image;
use crate::analyze::FormatDump;
use crate::buffer::PaddedBuf;
use crate::decompressors::packed::*;
use crate::dng::opcodes;
use crate::exif::Exif;
use crate::formats::tiff::Entry;
use crate::formats::tiff::GenericTiffReader;
use crate::formats::tiff::IFD;
use crate::formats::tiff::Rational;
use crate::formats::tiff::Value;
use crate::formats::tiff::reader::TiffReader;
use crate::imgop::Dim2;
use crate::imgop::Point;
use crate::imgop::Rect;
use crate::imgop::xyz::Illuminant;
use crate::imgop::xyz::FlatColorMatrix;
use crate::lens::LensDescription;
use crate::lens::LensResolver;
use crate::pixarray::PixU16;
use crate::pumps::BitPump;
use crate::pumps::BitPumpMSB;
use crate::rawimage::CFAConfig;
use crate::rawimage::RawPhotometricInterpretation;
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

const MFT_MOUNT: &str = "MFT-mount";

#[derive(Debug, Clone)]
pub struct OrfDecoder<'a> {
  #[allow(unused)]
  rawloader: &'a RawLoader,
  tiff: GenericTiffReader,
  camera: Camera,
  makernote: IFD,
  /// Pre-computed OpcodeList3 blob (distortion correction, applied after demosaicing).
  opcode_list3: Vec<u8>,
}

pub fn parse_makernote<R: Read + Seek>(reader: &mut R, exif_ifd: &IFD) -> Result<Option<IFD>> {
  if let Some(exif) = exif_ifd.get_entry(ExifTag::MakerNotes) {
    let offset = exif.offset().unwrap() as u32;
    log::debug!("Makernote offset: {}", offset);
    match &exif.value {
      Value::Undefined(data) => {
        let mut off = 0;
        // Olympus starts the makernote with their own name, sometimes truncated
        if data[0..5] == b"OLYMP"[..] {
          off += 8;
          if data[0..7] == b"OLYMPUS"[..] {
            off += 4;
          }
        }
        // OM Digital Solutions put their name in front of the TIFF structure, too
        if data[0..9] == b"OM SYSTEM"[..] {
          off += 16;
          assert_eq!(data[12..14], b"II"[..]);
        }
        let endian = exif_ifd.endian;
        //assert!(data[off..off + 2] == b"II"[..] || data[off..off + 2] == b"MM"[..], "ORF: must contain endian marker in makernote IFD");
        //let endian = if data[off..off + 2] == b"II"[..] { Endian::Little } else { Endian::Big };
        //off += 4;

        let mut mainifd = IFD::new(reader, offset + off as u32, exif_ifd.base, exif_ifd.corr, endian, &[0x3000])?;

        // Parse the Olympus Equipment section if it exists
        if let Some(entry) = mainifd.get_entry_raw_with_len(OrfMakernotes::EquipmentIFD, reader, 4)? {
          // The entry is of type UNDEFINED and count = 1. This tag contains a single 32 bit
          // offset to the IFD.
          let ioff = entry.get_force_u32(0);
          log::debug!("Found EquipmentIFD at offset: {}", ioff);
          // The IFD start at offset+ioff, but all offsets inside the IFD a relative to the main makernote IFD offset.
          // So we use the main IFD as base offset, but start parsing IFD at ioff.
          let ifd = IFD::new(reader, ioff, offset, 0, endian, &[])?;
          mainifd.sub.insert(OrfMakernotes::EquipmentIFD.into(), vec![ifd]);
        }

        // Parse the Olympus CameraSettings section if it exists (contains preview image info)
        if let Some(entry) = mainifd.get_entry_raw_with_len(OrfMakernotes::CameraSettingsIFD, reader, 4)? {
          let ioff = entry.get_force_u32(0);
          log::debug!("Found CameraSettingsIFD at offset: {}", ioff);
          let ifd = IFD::new(reader, ioff, offset, 0, endian, &[])?;
          mainifd.sub.insert(OrfMakernotes::CameraSettingsIFD.into(), vec![ifd]);
        }

        // For Olympus or OM-System models
        if off == 12 || off == 16 {
          // Parse the Olympus ImgProc section if it exists
          let ioff = if let Some(entry) = mainifd.get_entry_raw_with_len(OrfMakernotes::ImageProcessingIFD, reader, 4)? {
            // The entry is of type UNDEFINED and count = 1. This tag contains a single 32 bit
            // offset to the IFD.
            entry.get_force_u32(0)
          } else {
            0
          };
          if ioff != 0 {
            log::debug!("Found ImageIFD at offset: {}", ioff);
            // The IFD start at offset+ioff, but all offsets inside the IFD a relative to the main makernote IFD offset.
            // So we use the main IFD as base offset, but start parsing IFD at ioff.
            let iprocifd = IFD::new(reader, ioff, offset, 0, endian, &[])?;
            mainifd.sub.insert(OrfMakernotes::ImageProcessingIFD.into(), vec![iprocifd]);
          } else {
            log::debug!("ORF ImageIFD not found");
          }
        }
        Ok(Some(mainifd))
      }
      _ => Err(RawlerError::DecoderFailed("EXIF makernote has unknown type".to_string())),
    }
  } else {
    Ok(None)
  }
}

impl<'a> OrfDecoder<'a> {
  pub fn new(file: &RawSource, tiff: GenericTiffReader, rawloader: &'a RawLoader) -> Result<OrfDecoder<'a>> {
    let camera = rawloader.check_supported(tiff.root_ifd())?;

    let makernote = if let Some(exif) = tiff.find_first_ifd_with_tag(ExifTag::MakerNotes) {
      parse_makernote(&mut file.reader(), exif)?
    } else {
      log::warn!("ORF makernote not found");
      None
    }
    .ok_or("File has not makernotes")?;

    //makernote.dump::<ExifTag>(0).iter().for_each(|line| eprintln!("DUMP: {}", line));

    let opcode_list3 = build_orf_warp_rectilinear(&makernote).unwrap_or_default();

    Ok(OrfDecoder {
      tiff,
      rawloader,
      camera,
      makernote,
      opcode_list3,
    })
  }
}

impl<'a> Decoder for OrfDecoder<'a> {
  fn raw_image(&self, file: &RawSource, _params: &RawDecodeParams, dummy: bool) -> Result<RawImage> {
    let raw = self
      .tiff
      .find_first_ifd_with_tag(TiffCommonTag::StripOffsets)
      .ok_or_else(|| RawlerError::DecoderFailed(format!("Failed to find a IFD with StripOffsets tag")))?;
    let width = fetch_tiff_tag!(raw, TiffCommonTag::ImageWidth).force_usize(0);
    let height = fetch_tiff_tag!(raw, TiffCommonTag::ImageLength).force_usize(0);
    let offset = fetch_tiff_tag!(raw, TiffCommonTag::StripOffsets).force_usize(0);
    let counts = fetch_tiff_tag!(raw, TiffCommonTag::StripByteCounts);
    let bps = match self.get_bits_per_pixel()? {
      Some(bps) if [12, 14].contains(&bps) => bps,
      Some(bps) => {
        log::warn!("Unsupported bps: {}", bps);
        bps
      }
      None => {
        log::debug!("No bps found, fallback to 12");
        12
      }
    } as usize;

    let mut size: usize = 0;
    for i in 0..counts.count() {
      size += counts.force_u32(i as usize) as usize;
    }

    let camera = if width >= self.camera.highres_width {
      self.rawloader.check_supported_with_mode(self.tiff.root_ifd(), "highres")?
    } else {
      self.camera.clone()
    };

    log::debug!(
      "ORF raw image size: {}, dim: {}x{}, total mp: {}, strip counts: {}",
      size,
      width,
      height,
      width * height,
      counts.count()
    );

    // These conditions are sorted in descending order.
    // All ORF files comes with no hints about the used compression.
    // But we need to differentiate between 12be-interlaced and
    // 12be-msb32 because they are in the same size range.
    let image = if size >= width * height * 2 {
      let src = file.subview(offset as u64, size as u64)?;
      if self.tiff.little_endian() {
        log::debug!("ORF: decompress_12le_unpacked_left_aligned");
        decompress_12le_unpacked_left_aligned(&src, width, height, dummy)?
      } else {
        log::debug!("ORF: decompress_12be_unpacked_left_aligned");
        decompress_12be_unpacked_left_aligned(&src, width, height, dummy)?
      }
    } else if size >= width * height / 10 * 16 {
      log::debug!("ORF: decompress_12le_wcontrol");
      let src = file.subview(offset as u64, size as u64)?;
      decompress_12le_wcontrol(&src, width, height, dummy)?
    } else if size >= width * height * 12 / 8 {
      if self.camera.find_hint("interlaced") {
        log::debug!("ORF: decompress_12be_interlaced");
        // If interlaced, there is a gap between the strips.
        // To prevent reassembly of strips, we calculate the gap
        // and increase the src buffer.
        let gap = {
          let half = (height + 1) >> 1;
          // Second field is 2048 byte aligned
          let second_field_offset = (((half * width * 3 / 2) >> 11) + 1) << 11;
          let second_field_offset_unaligned = half * width * 3 / 2;
          second_field_offset - second_field_offset_unaligned
        };
        let src = file.subview(offset as u64, (size + gap) as u64)?;
        decompress_12be_interlaced(&src, width, height, dummy)?
      } else {
        log::debug!("ORF: decompress_12be_msb32");
        let src = file.subview(offset as u64, size as u64)?;
        decompress_12be_msb32(&src, width, height, dummy)?
      }
    } else {
      log::debug!("ORF: fallback to decode_compressed");
      let src = file.subview_padded(offset as u64, size as u64)?;
      OrfDecoder::decode_compressed(&src, width, height, bps, dummy)
    };

    let cpp = 1;
    let blacklevel = self.get_blacklevel(bps)?;
    let whitelevel = None;
    let photometric = RawPhotometricInterpretation::Cfa(CFAConfig::new_from_camera(&self.camera));
    let mut img = RawImage::new(camera, image, cpp, normalize_wb(self.get_wb()?), photometric, blacklevel, whitelevel, dummy);
    if let Some(crop) = self.get_crop()? {
      img.crop_area = Some(crop);
    }
    if bps == 14 {
      // Blacklevel is already corrected, only required for whitelevel.
      // Encoded for 12 bps, whitelevel must be multiplied by 4.
      img.whitelevel.0.iter_mut().for_each(|level| *level = *level << 2);
    }

    // Only use MakerNote color matrices as a fallback when the TOML camera
    // definition provides none.  The MakerNote stores a camera-internal color
    // correction matrix (rows sum to 1.0) which is NOT a DNG ColorMatrix
    // (XYZ→camera).  The TOML values are the correct DNG-spec matrices.
    if img.color_matrix.is_empty() {
      if let Some(matrices) = self.get_imgproc_color_matrices().or_else(|| self.get_rawinfo_color_matrices()) {
        log::info!("ORF: Using color matrices from MakerNote as fallback ({} illuminants)", matrices.len());
        img.color_matrix = matrices;
      }
    }

    // Set linearization table from tone curve or DC7 gamma.
    if let Some(table) = self.get_tone_curve().or_else(|| self.get_dc7_gamma()) {
      log::info!("ORF: Using linearization table ({} entries)", table.len());
      img.linearization_table = Some(table);
    }

    // Set noise profile from MakerNote calibration data.
    if img.camera.noise_profile.is_none() {
      if let Some(np) = self.get_noise_calibration() {
        log::info!("ORF: Using noise profile from MakerNote calibration");
        img.camera.noise_profile = Some(np);
      }
    }

    // Fall back to per-ISO noise calibration lookup if still unset.
    if img.camera.noise_profile.is_none() {
      let iso = self
        .tiff
        .root_ifd()
        .get_sub_ifd(ExifTag::ExifOffset)
        .and_then(|exif_ifd| exif_ifd.get_entry(ExifTag::ISOSpeedRatings))
        .map(|e| e.force_u32(0));
      if let Some(iso) = iso {
        if let Some(np) = noisecal::noise_profile_for_iso(&self.camera.clean_model, iso) {
          log::debug!("ORF: Computed NoiseProfile for ISO {}: {:?}", iso, np);
          img.camera.noise_profile = Some(np);
        }
      }
    }

    // Apply WB fine-tuning from CameraSettings
    if let Some(wb_adjust) = self.get_wb_fine_tune() {
      log::debug!("ORF: Applying WB fine-tune: {:?}", wb_adjust);
      for i in 0..4 {
        if img.wb_coeffs[i].is_normal() && wb_adjust[i].is_normal() {
          img.wb_coeffs[i] *= wb_adjust[i];
        }
      }
    }

    Ok(img)
  }

  fn format_dump(&self) -> FormatDump {
    todo!()
  }

  fn raw_metadata(&self, _file: &RawSource, __params: &RawDecodeParams) -> Result<RawMetadata> {
    let mut exif = Exif::new(self.tiff.root_ifd())?;

    // Extract CameraSerialNumber and LensSerialNumber from MakerNotes EquipmentIFD
    if let Some(equip_ifd) = self.makernote.get_sub_ifd(OrfMakernotes::EquipmentIFD) {
      if exif.serial_number.is_none() {
        if let Some(entry) = equip_ifd.get_entry(OrfEquipmentTags::SerialNumber) {
          if let Value::Ascii(data) = &entry.value {
            if let Some(serial) = data.strings().first() {
              let serial = serial.trim().to_string();
              if !serial.is_empty() {
                exif.serial_number = Some(serial);
              }
            }
          }
        }
      }
      if exif.lens_serial_number.is_none() {
        if let Some(entry) = equip_ifd.get_entry(OrfEquipmentTags::LensSerialNumber) {
          if let Value::Ascii(data) = &entry.value {
            if let Some(serial) = data.strings().first() {
              let serial = serial.trim().to_string();
              if !serial.is_empty() {
                exif.lens_serial_number = Some(serial);
              }
            }
          }
        }
      }
    }

    let mdata = RawMetadata::new_with_lens(&self.camera, exif, self.get_lens_description()?.cloned());
    Ok(mdata)
  }

  fn format_hint(&self) -> FormatHint {
    FormatHint::ORF
  }

  fn preview_jpeg(&self, file: &RawSource, _params: &RawDecodeParams) -> Result<Option<(Vec<u8>, u32, u32)>> {
    if let Some(cs_ifd) = self.makernote.get_sub_ifd(OrfMakernotes::CameraSettingsIFD) {
      let valid = cs_ifd
        .get_entry(OrfCameraSettings::PreviewImageValid)
        .map(|e| e.force_u32(0))
        .unwrap_or(0);
      if valid == 0 {
        return Ok(None);
      }
      let rel_offset = cs_ifd
        .get_entry(OrfCameraSettings::PreviewImageStart)
        .map(|e| e.force_u32(0) as u64)
        .unwrap_or(0);
      let length = cs_ifd
        .get_entry(OrfCameraSettings::PreviewImageLength)
        .map(|e| e.force_u32(0) as u64)
        .unwrap_or(0);
      if rel_offset == 0 || length == 0 {
        return Ok(None);
      }
      // PreviewImageStart is relative to the MakerNotes base offset,
      // not an absolute file position.  Add the IFD base to get the
      // true file offset.
      let abs_offset = rel_offset + cs_ifd.base as u64;
      let buf = file.subview(abs_offset, length)?;
      let (width, height) = super::jpeg_dimensions(&buf);
      if width > 0 && height > 0 {
        return Ok(Some((buf.to_vec(), width, height)));
      }
    }
    Ok(None)
  }

  fn ifd(&self, wk_ifd: WellKnownIFD) -> crate::Result<Option<Rc<IFD>>> {
    match wk_ifd {
      WellKnownIFD::VirtualDngRootTags => {
        let mut ifd = IFD::default();
        // Olympus MakerNotes use offsets relative to the MakerNote start, not
        // the file start, so they are self-contained and safe to copy verbatim.
        ifd.entries.insert(
          DngTag::MakerNoteSafety.into(),
          Entry { tag: DngTag::MakerNoteSafety.into(), value: Value::Short(vec![1]), embedded: None },
        );
        Ok(Some(Rc::new(ifd)))
      }
      WellKnownIFD::VirtualDngRawTags => {
        let (mut opc1, opc3_blob) = if !self.opcode_list3.is_empty() {
          // Use embedded Olympus distortion correction data
          (Vec::new(), self.opcode_list3.clone())
        } else {
          // Fall back to Adobe LCP lens correction profile
          self.lcp_fallback_opcodes()
        };

        // Add lateral CA correction as an additional OpcodeList1 entry.
        // CA correction uses WarpRectilinear with 3 planes (R, G, B) and
        // should be applied before demosaicing (OpcodeList1).
        if let Some(ca_opcode) = self.get_lateral_ca_correction() {
          log::info!("ORF: Adding lateral CA correction opcode");
          if opc1.is_empty() {
            opc1 = opcodes::encode_opcode_list(&[ca_opcode]);
          } else {
            // Append to existing OpcodeList1 — need to re-parse and extend.
            // For simplicity, create a fresh list with just the CA opcode
            // since LCP opcodes already include CA correction.
            // Only add if we're not already using LCP fallback.
            if self.opcode_list3.is_empty() {
              // LCP fallback active — LCP already handles CA, skip
            } else {
              opc1 = opcodes::encode_opcode_list(&[ca_opcode]);
            }
          }
        }

        if opc1.is_empty() && opc3_blob.is_empty() {
          return Ok(None);
        }
        let mut ifd = IFD::default();
        if !opc1.is_empty() {
          ifd.entries.insert(
            DngTag::OpcodeList1.into(),
            Entry { tag: DngTag::OpcodeList1.into(), value: Value::Undefined(opc1), embedded: None },
          );
        }
        if !opc3_blob.is_empty() {
          ifd.entries.insert(
            DngTag::OpcodeList3.into(),
            Entry { tag: DngTag::OpcodeList3.into(), value: Value::Undefined(opc3_blob), embedded: None },
          );
        }
        Ok(Some(Rc::new(ifd)))
      }
      _ => Ok(None),
    }
  }
}

impl<'a> OrfDecoder<'a> {
  /* This is probably the slowest decoder of them all.
   * I cannot see any way to effectively speed up the prediction
   * phase, which is by far the slowest part of this algorithm.
   * Also there is no way to multithread this code, since prediction
   * is based on the output of all previous pixel (bar the first four)
   */

  pub fn decode_compressed(buf: &PaddedBuf, width: usize, height: usize, bps: usize, dummy: bool) -> PixU16 {
    let mut out = alloc_image!(width, height, dummy);

    /* Build a table to quickly look up "high" value */
    let mut bittable: [u8; 4096] = [0; 4096];
    for i in 0..4096 {
      let mut b = 12;
      for high in 0..12 {
        if ((i >> (11 - high)) & 1) != 0 {
          b = high;
          break;
        }
      }
      bittable[i] = b;
    }

    let mut left: [i32; 2] = [0; 2];
    let mut nw: [i32; 2] = [0; 2];
    let skip = if bps == 14 { 8 } else { 7 };
    let mut pump = BitPumpMSB::new(&buf[skip..]);

    for row in 0..height {
      let mut acarry: [[i32; 3]; 2] = [[0; 3]; 2];

      for c in 0..width / 2 {
        let col: usize = c * 2;
        for s in 0..2 {
          // Run twice for odd and even pixels
          let i = if acarry[s][2] < 3 { 2 } else { 0 };
          let mut nbits = 2 + i;
          while ((acarry[s][0] >> (nbits + i)) & 0xffff) > 0 {
            nbits += 1
          }
          nbits = cmp::min(nbits, 16);
          let b = pump.peek_ibits(15);

          let sign: i32 = -(b >> 14);
          let low: i32 = (b >> 12) & 3;
          let mut high: i32 = bittable[(b & 4095) as usize] as i32;

          // Skip bytes used above or read bits
          if high == 12 {
            pump.consume_bits(15);
            high = pump.get_ibits(16 - nbits) >> 1;
          } else {
            pump.consume_bits((high + 4) as u32);
          }

          acarry[s][0] = ((high << nbits) | pump.get_ibits(nbits)) as i32;
          let diff = (acarry[s][0] ^ sign) + acarry[s][1];
          acarry[s][1] = (diff * 3 + acarry[s][1]) >> 5;
          acarry[s][2] = if acarry[s][0] > 16 { 0 } else { acarry[s][2] + 1 };

          if row < 2 || col < 2 {
            // We're in a border, special care is needed
            let pred = if row < 2 && col < 2 {
              // We're in the top left corner
              0
            } else if row < 2 {
              // We're going along the top border
              left[s]
            } else {
              // col < 2, we're at the start of a line
              nw[s] = out[(row - 2) * width + (col + s)] as i32;
              nw[s]
            };
            left[s] = pred + ((diff << 2) | low);
            out[row * width + (col + s)] = left[s] as u16;
          } else {
            let up: i32 = out[(row - 2) * width + (col + s)] as i32;
            let left_minus_nw: i32 = left[s] - nw[s];
            let up_minus_nw: i32 = up - nw[s];
            // Check if sign is different, and one is not zero
            let pred = if left_minus_nw * up_minus_nw < 0 {
              if left_minus_nw.abs() > 32 || up_minus_nw.abs() > 32 {
                left[s] + up_minus_nw
              } else {
                (left[s] + up) >> 1
              }
            } else if left_minus_nw.abs() > up_minus_nw.abs() {
              left[s]
            } else {
              up
            };

            left[s] = pred + ((diff << 2) | low);
            nw[s] = up;
            out[row * width + (col + s)] = left[s] as u16;
          }
        }
      }
    }
    out
  }

  fn get_blacklevel(&self, bps: usize) -> Result<Option<BlackLevel>> {
    // Use the specific ImageProcessing sub-IFD instead of a generic recursive
    // search.  find_ifds_with_tag iterates HashMap-backed sub-IFDs in
    // non-deterministic order, so multiple sub-IFDs containing the same tag
    // number would return an unpredictable result.
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD);
    let ifd = match imgproc {
      Some(ifd) if ifd.get_entry(OrfImageProcessing::OrfBlackLevels).is_some() => ifd,
      _ => {
        log::info!("ORF: Couldn't find ImgProc IFD, unable to read blacklevel");
        return Ok(None);
      }
    };

    let blacks = fetch_tiff_tag!(ifd, OrfImageProcessing::OrfBlackLevels);
    let mut levels = [blacks.force_u16(0), blacks.force_u16(1), blacks.force_u16(2), blacks.force_u16(3)];
    if bps == 14 {
      // Blacklevel is encoded for 12 bits
      levels.iter_mut().for_each(|level| *level = *level << 2);
    }
    Ok(Some(BlackLevel::new(&levels, self.camera.cfa.width, self.camera.cfa.height, 1)))
  }

  fn get_bits_per_pixel(&self) -> Result<Option<u16>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD);
    let ifd = match imgproc {
      Some(ifd) if ifd.get_entry(OrfImageProcessing::ValidBits).is_some() => ifd,
      _ => return Ok(None),
    };
    Ok(Some(fetch_tiff_tag!(ifd, OrfImageProcessing::ValidBits).force_u16(0)))
  }

  fn get_crop(&self) -> Result<Option<Rect>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD);
    let ifd = match imgproc {
      Some(ifd) if ifd.get_entry(OrfImageProcessing::CropLeft).is_some() => ifd,
      _ => return Ok(None),
    };
    let crop_left = fetch_tiff_tag!(ifd, OrfImageProcessing::CropLeft).force_usize(0);
    let crop_top = fetch_tiff_tag!(ifd, OrfImageProcessing::CropTop).force_usize(0);
    let crop_width = fetch_tiff_tag!(ifd, OrfImageProcessing::CropWidth).force_usize(0);
    let crop_height = fetch_tiff_tag!(ifd, OrfImageProcessing::CropHeight).force_usize(0);
    Ok(Some(Rect::new(Point::new(crop_left, crop_top), Dim2::new(crop_width, crop_height))))
  }

  /// Get lens description by analyzing TIFF tags and makernotes
  fn get_lens_description(&self) -> Result<Option<&'static LensDescription>> {
    if let Some(ifd) = self.makernote.get_sub_ifd(OrfMakernotes::EquipmentIFD) {
      match ifd.get_entry(OrfEquipmentTags::LensType) {
        Some(Entry {
          value: Value::Byte(settings), ..
        }) => {
          log::debug!("Lens type tag: {:?}", settings);
          let make_id = settings[0];
          let model_id = settings[2];
          let submodel_id = settings[3];
          let composite_id = format!("{:02X} {:02X} {:02X}", make_id, model_id, submodel_id);
          log::debug!("ORF lens composite ID: {}", composite_id);
          let resolver = LensResolver::new()
            .with_olympus_id(Some(composite_id))
            .with_camera(&self.camera)
            .with_focal_len(self.get_focal_len()?)
            .with_mounts(&[MFT_MOUNT.into()]);
          return Ok(resolver.resolve());
        }
        _ => {
          log::warn!("Camera settings in makernote not found, no lens data available");
        }
      }
    }
    log::warn!("No lens data found");
    Ok(None)
  }

  fn get_focal_len(&self) -> Result<Option<Rational>> {
    if let Some(exif) = self.tiff.find_first_ifd_with_tag(ExifTag::MakerNotes) {
      if let Some(Entry {
        value: Value::Short(focal), ..
      }) = exif.get_entry(ExifTag::FocalLength)
      {
        return Ok(focal.get(1).map(|v| Rational::new(*v as u32, 1)));
      }
    }
    Ok(None)
  }

  /// Fall back to Adobe LCP lens correction profiles when embedded Olympus
  /// distortion correction data is absent or flagged invalid.
  fn lcp_fallback_opcodes(&self) -> (Vec<u8>, Vec<u8>) {
    let lens = match self.get_lens_description() {
      Ok(Some(l)) => l,
      _ => return Default::default(),
    };

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

    let raw_ifd = self
      .tiff
      .find_first_ifd_with_tag(TiffCommonTag::StripOffsets)
      .or_else(|| self.tiff.find_first_ifd_with_tag(TiffCommonTag::TileOffsets));
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
    let redmul = self.makernote.get_entry(OrfMakernotes::OlympusRedMul);
    let bluemul = self.makernote.get_entry(OrfMakernotes::OlympusBlueMul);
    match (redmul, bluemul) {
      (Some(redmul), Some(bluemul)) => Ok([redmul.force_u32(0) as f32, 256.0, 256.0, bluemul.force_u32(0) as f32]),
      _ => {
        // Access the ImageProcessing sub-IFD directly to avoid finding
        // the wrong IFD (CameraSettingsIFD also has tag 0x0600).
        let imgproc = self
          .makernote
          .get_sub_ifd(OrfMakernotes::ImageProcessingIFD)
          .ok_or_else(|| RawlerError::DecoderFailed("ORF: Couldn't find ImageProcessing IFD".to_string()))?;
        let wbs = imgproc
          .get_entry(OrfImageProcessing::WB_RBLevels)
          .ok_or_else(|| RawlerError::DecoderFailed("ORF: Couldn't find WB_RBLevels".to_string()))?;
        Ok([wbs.force_f32(0), 256.0, 256.0, wbs.force_f32(1)])
      }
    }
  }

  /// Read color matrices from the RawInfo sub-IFD (MakerNote → 0x3000).
  ///
  /// Two 3x3 camera-internal color correction matrices from RawInfo:
  /// - Tag 0x0200: first color matrix (signed ShortArray of 9, scaled by 256)
  /// - Tag 0x0240: second color matrix (signed ShortArray of 9, scaled by 256)
  ///
  /// These are camera-internal matrices (NOT DNG ColorMatrix).
  /// Only used as a fallback when no TOML camera definition is available.
  fn get_rawinfo_color_matrices(&self) -> Option<HashMap<Illuminant, FlatColorMatrix>> {
    let rawinfo = self.makernote.get_sub_ifd(OrfMakernotes::RawInfo)?;
    let mut matrices: HashMap<Illuminant, FlatColorMatrix> = HashMap::new();

    if let Some(entry) = rawinfo.get_entry(OrfRawInfo::ColorMatrix1) {
      if entry.count() >= 9 {
        let matrix: FlatColorMatrix = (0..9).map(|i| entry.force_i16(i) as f32 / 256.0).collect();
        if matrix.iter().any(|&v| v != 0.0) {
          log::debug!("ORF RawInfo ColorMatrix1 (0x0200): {:?}", matrix);
          matrices.insert(Illuminant::A, matrix);
        }
      }
    }

    if let Some(entry) = rawinfo.get_entry(OrfRawInfo::ColorMatrix2) {
      if entry.count() >= 9 {
        let matrix: FlatColorMatrix = (0..9).map(|i| entry.force_i16(i) as f32 / 256.0).collect();
        if matrix.iter().any(|&v| v != 0.0) {
          log::debug!("ORF RawInfo ColorMatrix2 (0x0240): {:?}", matrix);
          matrices.insert(Illuminant::D65, matrix);
        }
      }
    }

    if matrices.is_empty() { None } else { Some(matrices) }
  }

  /// Read color matrices from the ImageProcessing sub-IFD (MakerNote → 0x2040).
  ///
  /// Reads tags 0x0200-0x021a (27 per-illuminant 3x3 color matrices).
  /// We read the first four (0x0200-0x0203) which cover the most common illuminants.
  /// Each is a ShortArray of 9, with values scaled by 256.
  fn get_imgproc_color_matrices(&self) -> Option<HashMap<Illuminant, FlatColorMatrix>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD)?;
    let mut matrices: HashMap<Illuminant, FlatColorMatrix> = HashMap::new();

    // Map the first few color matrix tags to illuminants.
    // 0x0200 = A (Tungsten ~2856K), 0x0201 = D65 (Daylight ~6500K)
    let tag_illuminant_map = [
      (OrfImageProcessing::ColorMatrix0200, Illuminant::A),
      (OrfImageProcessing::ColorMatrix0201, Illuminant::D65),
      (OrfImageProcessing::ColorMatrix0202, Illuminant::D55),
      (OrfImageProcessing::ColorMatrix0203, Illuminant::D75),
    ];

    for (tag, illuminant) in &tag_illuminant_map {
      if let Some(entry) = imgproc.get_entry(*tag) {
        if entry.count() >= 9 {
          // Values are signed shorts scaled by 256.
          let matrix: FlatColorMatrix = (0..9).map(|i| entry.force_i16(i) as f32 / 256.0).collect();
          // Skip identity/zero matrices
          let is_identity_or_zero = matrix.iter().all(|&v| v == 0.0)
            || (matrix == vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
          if !is_identity_or_zero {
            log::debug!("ORF ImgProc ColorMatrix {:?} ({:?}): {:?}", tag, illuminant, matrix);
            matrices.insert(*illuminant, matrix);
          }
        }
      }
    }

    if matrices.is_empty() { None } else { Some(matrices) }
  }

  /// Read per-CFA linear correction pairs from the RawInfo sub-IFD.
  ///
  /// Reads tags 0x0100, 0x0120-0x0133 as ShortArray(2) pairs
  /// with default [0x100, 0x100]. These represent gain/offset corrections
  /// for each CFA channel. We combine them into a linearization table.
  #[allow(dead_code)]
  fn get_rawinfo_linear_correction(&self) -> Option<Vec<[u16; 2]>> {
    let rawinfo = self.makernote.get_sub_ifd(OrfMakernotes::RawInfo)?;
    let default = [0x100u16, 0x100u16];

    let tags = [
      OrfRawInfo::LinearCorrection0100,
      OrfRawInfo::LinearCorrection0120,
      OrfRawInfo::LinearCorrection0121,
      OrfRawInfo::LinearCorrection0122,
      OrfRawInfo::LinearCorrection0123,
      OrfRawInfo::LinearCorrection0130,
      OrfRawInfo::LinearCorrection0131,
      OrfRawInfo::LinearCorrection0132,
      OrfRawInfo::LinearCorrection0133,
    ];

    let pairs: Vec<[u16; 2]> = tags
      .iter()
      .map(|tag| {
        rawinfo
          .get_entry(*tag)
          .filter(|e| e.count() >= 2)
          .map(|e| [e.force_u16(0), e.force_u16(1)])
          .unwrap_or(default)
      })
      .collect();

    // If all pairs are the default, no correction is needed
    if pairs.iter().all(|p| *p == default) {
      return None;
    }

    log::debug!("ORF RawInfo linear correction pairs: {:?}", pairs);
    Some(pairs)
  }

  /// Read the tone/gamma curve from ImageProcessing tag 0x0400.
  ///
  /// Reads this as a ShortArray of 63 elements — a tone curve LUT.
  /// The curve maps 12-bit sensor values through a gamma/tone response.
  /// We convert this into a full linearization table for DNG.
  fn get_tone_curve(&self) -> Option<Vec<u16>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD)?;
    let entry = imgproc.get_entry(OrfImageProcessing::ToneCurve)?;
    if entry.count() < 33 {
      return None;
    }

    let count = entry.count() as usize;
    let curve: Vec<u16> = (0..count).map(|i| entry.force_u16(i)).collect();

    // Validate: skip if all zeros or all same value
    if curve.iter().all(|&v| v == 0) || curve.iter().all(|&v| v == curve[0]) {
      return None;
    }

    log::debug!("ORF tone curve ({} points): {:?}", count, &curve[..count.min(16)]);

    // The 63-point curve needs to be interpolated into a full
    // linearization table. Expand to 4096 entries (12-bit range).
    let num_points = curve.len();
    let max_output = 4095u16;
    let table_size = 4096usize;
    let mut table = vec![0u16; table_size];

    for i in 0..table_size {
      // Map table index to curve position
      let pos = (i as f64) * (num_points - 1) as f64 / (table_size - 1) as f64;
      let lo = pos.floor() as usize;
      let hi = (lo + 1).min(num_points - 1);
      let frac = pos - lo as f64;

      // Linear interpolation between curve points
      let val = curve[lo] as f64 * (1.0 - frac) + curve[hi] as f64 * frac;
      table[i] = (val.round() as u16).min(max_output);
    }

    Some(table)
  }

  /// Read lateral chromatic aberration correction data from ImageProcessing.
  ///
  /// Reads tags 0x0340-0x0343 as SShortArray(18) — four arrays
  /// of 18 signed shorts representing polynomial coefficients for lateral CA
  /// correction per color plane.
  ///
  /// These are converted to DNG WarpRectilinear opcodes with per-plane
  /// radial correction coefficients.
  fn get_lateral_ca_correction(&self) -> Option<Vec<u8>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD)?;

    let tags = [
      OrfImageProcessing::LateralCA_0,
      OrfImageProcessing::LateralCA_1,
      OrfImageProcessing::LateralCA_2,
      OrfImageProcessing::LateralCA_3,
    ];

    let mut ca_data: Vec<Vec<i16>> = Vec::new();
    for tag in &tags {
      if let Some(entry) = imgproc.get_entry(*tag) {
        if entry.count() >= 18 {
          let coeffs: Vec<i16> = (0..18).map(|i| entry.force_i16(i)).collect();
          ca_data.push(coeffs);
        }
      }
    }

    // Need at least 2 CA arrays (typically R and B plane corrections)
    if ca_data.len() < 2 {
      return None;
    }

    // Check if all coefficients are zero (no CA correction needed)
    if ca_data.iter().all(|arr| arr.iter().all(|&v| v == 0)) {
      return None;
    }

    log::debug!("ORF lateral CA: {} arrays of {} coefficients each", ca_data.len(), ca_data[0].len());

    // Convert Olympus CA polynomial coefficients to DNG WarpRectilinear.
    //
    // The Olympus CA arrays contain polynomial coefficients for radial
    // displacement of each color plane. We extract the first few coefficients
    // and scale them to DNG WarpRectilinear kr format.
    //
    // DNG WarpRectilinear with 3 planes: [R, G, B] each with [kr0, kr1, kr2, kr3]
    // kr0 = scale (1.0 = no change), kr1-kr3 = radial polynomial
    //
    // Olympus stores coefficients as signed shorts scaled by 16384 (2^14).
    let scale = 16384.0_f64;

    // Build per-plane kr coefficients: [scale, k1, k2, k3]
    // CA array 0 = R plane, CA array 1 = B plane (G is reference at 1.0)
    let r_kr = [
      1.0 + ca_data[0][0] as f64 / scale,
      ca_data[0][1] as f64 / scale,
      if ca_data[0].len() > 2 { ca_data[0][2] as f64 / scale } else { 0.0 },
      if ca_data[0].len() > 3 { ca_data[0][3] as f64 / scale } else { 0.0 },
    ];
    let g_kr = [1.0_f64, 0.0, 0.0, 0.0]; // Green is reference
    let b_kr = [
      1.0 + ca_data[1][0] as f64 / scale,
      ca_data[1][1] as f64 / scale,
      if ca_data[1].len() > 2 { ca_data[1][2] as f64 / scale } else { 0.0 },
      if ca_data[1].len() > 3 { ca_data[1][3] as f64 / scale } else { 0.0 },
    ];

    log::debug!("ORF CA kr R: {:?}, G: {:?}, B: {:?}", r_kr, g_kr, b_kr);

    let kr = [r_kr, g_kr, b_kr];
    let kt = [[0.0_f64, 0.0], [0.0, 0.0], [0.0, 0.0]];

    let opcode = opcodes::encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
    Some(opcode)
  }

  /// Read noise calibration data from ImageProcessing sub-IFD.
  ///
  /// Reads noise model from tags 0x1400-0x1409, conditional on
  /// tag 0x1406 (feature flag). When enabled, the noise model provides
  /// per-channel shot noise and read noise coefficients for DNG NoiseProfile.
  fn get_noise_calibration(&self) -> Option<Vec<f64>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD)?;

    // Check if noise calibration is enabled (tag 0x1406)
    let flag = imgproc.get_entry(OrfImageProcessing::NoiseCalFlag).map(|e| e.force_u16(0)).unwrap_or(0);
    if flag == 0 {
      return None;
    }

    // Read the noise model coefficients.
    // Tags 0x1400 (SShortArray) and 0x1401 (ShortArray) contain the primary noise model.
    // These typically encode shot noise (S) and read noise (O) per channel.
    let signed_entry = imgproc.get_entry(OrfImageProcessing::NoiseModel_Signed)?;
    let unsigned_entry = imgproc.get_entry(OrfImageProcessing::NoiseModel_Unsigned)?;

    // The noise model values are scaled. Convert to DNG NoiseProfile format:
    // [S_r, O_r, S_g, O_g, S_b, O_b] where S = shot noise scale, O = read noise offset.
    //
    // Olympus stores these as 16-bit values. The exact scaling depends on the
    // sensor bit depth and internal normalization. We scale to [0,1] normalized range.
    //
    // For a 12-bit sensor normalized to [0,1], the scaling factor is 1/(2^12)^2 = 1/16777216
    // for offset and 1/2^12 = 1/4096 for shot noise.
    let bit_scale = 4096.0_f64; // 2^12
    let bit_scale_sq = bit_scale * bit_scale;

    if signed_entry.count() >= 3 && unsigned_entry.count() >= 3 {
      // Interpret as per-channel [R, G, B] noise coefficients
      let noise_profile = vec![
        (signed_entry.force_i16(0) as f64).abs() / bit_scale,    // S_r
        (unsigned_entry.force_u16(0) as f64) / bit_scale_sq,     // O_r
        (signed_entry.force_i16(1) as f64).abs() / bit_scale,    // S_g
        (unsigned_entry.force_u16(1) as f64) / bit_scale_sq,     // O_g
        (signed_entry.force_i16(2) as f64).abs() / bit_scale,    // S_b
        (unsigned_entry.force_u16(2) as f64) / bit_scale_sq,     // O_b
      ];

      // Validate: all values should be positive and reasonable
      if noise_profile.iter().all(|&v| v > 0.0 && v < 1.0) {
        log::debug!("ORF noise calibration: {:?}", noise_profile);
        return Some(noise_profile);
      }
    }

    None
  }

  /// Read the large calibration arrays from RawInfo (0x0250-0x0252).
  ///
  /// These 99-element arrays contain noise/tone calibration
  /// data that can be used as a secondary noise profile source.
  #[allow(dead_code)]
  fn get_rawinfo_calibration_arrays(&self) -> Option<[Vec<u16>; 3]> {
    let rawinfo = self.makernote.get_sub_ifd(OrfMakernotes::RawInfo)?;

    let tags = [OrfRawInfo::CalibrationArray0, OrfRawInfo::CalibrationArray1, OrfRawInfo::CalibrationArray2];

    let mut arrays = Vec::new();
    for tag in &tags {
      if let Some(entry) = rawinfo.get_entry(*tag) {
        if entry.count() >= 99 {
          let arr: Vec<u16> = (0..99).map(|i| entry.force_u16(i)).collect();
          arrays.push(arr);
        } else {
          return None;
        }
      } else {
        return None;
      }
    }

    if arrays.len() == 3 {
      Some([arrays.remove(0), arrays.remove(0), arrays.remove(0)])
    } else {
      None
    }
  }

  /// Read WB fine-tuning data from CameraSettings sub-IFD (0x2020).
  ///
  /// Tags 0x0500-0x0507 contain white balance adjustments to refine
  /// the base WB coefficients.
  fn get_wb_fine_tune(&self) -> Option<[f32; 4]> {
    let cs_ifd = self.makernote.get_sub_ifd(OrfMakernotes::CameraSettingsIFD)?;

    // Tag 0x0500: White balance setting ID
    let wb_setting = cs_ifd.get_entry(OrfCameraSettings::WhiteBalance).map(|e| e.force_u16(0)).unwrap_or(0);

    // Tag 0x0501: White balance mode
    let _wb_mode = cs_ifd.get_entry(OrfCameraSettings::WBMode).map(|e| e.force_u16(0)).unwrap_or(0);

    // Tags 0x0505/0x0506 contain WB fine-tune adjustments as SShortArrays
    // These provide amber-blue and green-magenta offsets
    if let Some(entry0505) = cs_ifd.get_entry(OrfCameraSettings::WBParam0505) {
      if entry0505.count() >= 2 {
        let ab_adjust = entry0505.force_i16(0) as f32;
        let gm_adjust = entry0505.force_i16(1) as f32;

        // Only apply if there are actual adjustments
        if ab_adjust != 0.0 || gm_adjust != 0.0 {
          log::debug!("ORF WB fine-tune (setting={}): AB={}, GM={}", wb_setting, ab_adjust, gm_adjust);
          // Convert WB fine-tune to multiplier adjustments
          // AB (amber-blue) adjusts R/B ratio, GM (green-magenta) adjusts G
          let scale = 256.0; // Olympus WB fine-tune scale factor
          return Some([
            1.0 + ab_adjust / scale,  // R adjustment
            1.0 - gm_adjust / scale,  // G1 adjustment
            1.0 - gm_adjust / scale,  // G2 adjustment
            1.0 - ab_adjust / scale,  // B adjustment
          ]);
        }
      }
    }

    None
  }

  /// Read DC7 gamma handling data from ImageProcessing (0x0635/0x0636).
  ///
  /// Newer OM System cameras (DC7 series) store gamma unpacking data
  /// as LongArrays that are used by CDC7UnpackGamma for sensor linearization.
  fn get_dc7_gamma(&self) -> Option<Vec<u16>> {
    let imgproc = self.makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD)?;

    let entry0 = imgproc.get_entry(OrfImageProcessing::DC7Gamma0)?;

    // DC7 gamma tag is only useful when it contains a LongArray (u32 values).
    // Some cameras store a proprietary blob (type UNDEFINED, starts with "CMIO")
    // which cannot be parsed as a simple integer table.
    let values = match &entry0.value {
      Value::Long(v) => v,
      _ => {
        log::debug!("ORF DC7 gamma tag 0x0635: not a LongArray, skipping");
        return None;
      }
    };

    if values.len() < 2 {
      return None;
    }

    let count = values.len();
    log::debug!("ORF DC7 gamma tag 0x0635: {} entries", count);

    // Build a linearization table from the gamma data.
    // DC7 gamma values are stored as 32-bit unsigned, representing
    // a gamma decompression curve. Convert to 16-bit LUT.
    let max_val = values[0];
    if max_val == 0 {
      return None;
    }

    let mut table: Vec<u16> = Vec::with_capacity(count);
    for &val in values {
      // Scale to 16-bit range
      let scaled = ((val as u64 * 65535) / max_val.max(1) as u64).min(65535) as u16;
      table.push(scaled);
    }

    // Also read 0x0636 if available (second gamma component)
    if let Some(entry1) = imgproc.get_entry(OrfImageProcessing::DC7Gamma1) {
      log::debug!("ORF DC7 gamma tag 0x0636: {} entries", entry1.count());
      // Second gamma array refines the linearization; blend with first
      // For now, the primary table from 0x0635 is sufficient
    }

    if table.is_empty() || table.iter().all(|&v| v == table[0]) {
      return None;
    }

    Some(table)
  }
}

fn normalize_wb(raw_wb: [f32; 4]) -> [f32; 4] {
  log::debug!("ORF raw wb: {:?}", raw_wb);
  let div = raw_wb[1];
  let mut norm = raw_wb;
  norm.iter_mut().for_each(|v| {
    if v.is_normal() {
      *v /= div
    }
  });
  [norm[0], (norm[1] + norm[2]) / 2.0, norm[3], f32::NAN]
}

crate::tags::tiff_tag_enum!(OrfMakernotes);
crate::tags::tiff_tag_enum!(OrfImageProcessing);
crate::tags::tiff_tag_enum!(OrfEquipmentTags);
crate::tags::tiff_tag_enum!(OrfCameraSettings);
crate::tags::tiff_tag_enum!(OrfRawInfo);

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum OrfMakernotes {
  ImageProcessingIFD = 0x2040,
  CameraSettingsIFD = 0x2020,
  RawInfo = 0x3000,
  OlympusRedMul = 0x1017,
  OlympusBlueMul = 0x1018,
  EquipmentIFD = 0x2010,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum OrfCameraSettings {
  PreviewImageValid = 0x0100,
  PreviewImageStart = 0x0101,
  PreviewImageLength = 0x0102,
  /// Exposure mode.
  ExposureMode = 0x0200,
  /// White balance setting.
  WhiteBalance = 0x0500,
  /// White balance mode.
  WBMode = 0x0501,
  /// White balance bracket (SShortArray).
  WBBracket = 0x0502,
  /// White balance parameter (SShortArray).
  WBParam0503 = 0x0503,
  /// White balance parameter.
  WBParam0504 = 0x0504,
  /// White balance parameter (SShortArray).
  WBParam0505 = 0x0505,
  /// White balance parameter (SShortArray).
  WBParam0506 = 0x0506,
  /// White balance parameter.
  WBParam0507 = 0x0507,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum OrfImageProcessing {
  ImageProcessingVersion = 0x0000,
  WB_RBLevels = 0x0100,
  /// Color correction matrix (ShortArray of 9, 3x3, scaled by 256).
  /// Tags 0x0200-0x021a: 27 per-illuminant matrices.
  ColorMatrix0200 = 0x0200,
  ColorMatrix0201 = 0x0201,
  ColorMatrix0202 = 0x0202,
  ColorMatrix0203 = 0x0203,
  /// Lateral CA correction polynomial coefficients (SShortArray of 18 each).
  LateralCA_0 = 0x0340,
  LateralCA_1 = 0x0341,
  LateralCA_2 = 0x0342,
  LateralCA_3 = 0x0343,
  /// Tone/gamma curve LUT (ShortArray of 63 elements).
  ToneCurve = 0x0400,
  OrfBlackLevels = 0x0600,
  ValidBits = 0x0611,
  CropLeft = 0x0612,
  CropTop = 0x0613,
  CropWidth = 0x0614,
  CropHeight = 0x0615,
  /// DC7 gamma unpacking data (LongArray) for newer OM System cameras.
  DC7Gamma0 = 0x0635,
  DC7Gamma1 = 0x0636,
  /// Noise calibration feature flag (Short, 0 = disabled).
  NoiseCalFlag = 0x1406,
  /// Noise model signed coefficients (SShortArray).
  NoiseModel_Signed = 0x1400,
  /// Noise model unsigned coefficients (ShortArray).
  NoiseModel_Unsigned = 0x1401,
  /// Additional noise model signed coefficients.
  NoiseModel_Signed2 = 0x1402,
  /// Additional noise model unsigned coefficients.
  NoiseModel_Unsigned2 = 0x1403,
  /// Noise reduction parameter.
  NoiseParam1404 = 0x1404,
  /// Noise reduction parameter.
  NoiseParam1405 = 0x1405,
  /// Additional noise model signed coefficients.
  NoiseModel_Signed3 = 0x1408,
  /// Noise model gamma LUT (ShortArray of 15).
  NoiseGammaLUT = 0x1409,
  /// Float calibration data (FloatArray of 4).
  FloatCalibration = 0x150a,
  /// Distortion correction valid flag: 1 = coefficients below are valid.
  DistortionCorrectionValid = 0x150f,
  /// Primary radial distortion coefficients: float[4] = [k1, k2, k3, scale].
  /// The correction formula is `Ru = scale * Rd * (1 + k1·Rd² + k2·Rd⁴ + k3·Rd⁶)`
  /// where Rd and Ru are pixel radii normalised to the image half-diagonal.
  DistortionCoefficients = 0x1510,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum OrfEquipmentTags {
  SerialNumber = 0x0101,
  LensType = 0x0201,
  LensSerialNumber = 0x0202,
}

/// Tags in the RawInfo sub-IFD (MakerNote → 0x3000).
#[allow(non_camel_case_types)]
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum OrfRawInfo {
  /// Per-CFA linear correction pair (ShortArray of 2, default [0x100,0x100]).
  LinearCorrection0100 = 0x0100,
  LinearCorrection0120 = 0x0120,
  LinearCorrection0121 = 0x0121,
  LinearCorrection0122 = 0x0122,
  LinearCorrection0123 = 0x0123,
  LinearCorrection0130 = 0x0130,
  LinearCorrection0131 = 0x0131,
  LinearCorrection0132 = 0x0132,
  LinearCorrection0133 = 0x0133,
  /// Color matrix (ShortArray of 9, 3x3 scaled by 256).
  ColorMatrix1 = 0x0200,
  /// Second color matrix (ShortArray of 9, 3x3 scaled by 256).
  ColorMatrix2 = 0x0240,
  /// Large calibration array (ShortArray of 99 elements) — noise/tone calibration.
  CalibrationArray0 = 0x0250,
  CalibrationArray1 = 0x0251,
  CalibrationArray2 = 0x0252,
}

/// Build a DNG WarpRectilinear OpcodeList3 blob from the Olympus ImageProcessing IFD.
///
/// Reads tags 0x150f (validity flag) and 0x1510 (float[4] = [k1, k2, k3, scale]).
///
/// The Olympus correction model is:
///   `Ru = scale * Rd * (1 + k1·Rd² + k2·Rd⁴ + k3·Rd⁶)`
///
/// which expands to the DNG WarpRectilinear polynomial:
///   `Ru = scale·Rd + scale·k1·Rd³ + scale·k2·Rd⁵ + scale·k3·Rd⁷`
///
/// Both use the image half-diagonal as the normalisation radius, so
/// the coefficients can be passed directly to the DNG encoder as:
///   kr0 = scale,  kr1 = scale·k1,  kr2 = scale·k2,  kr3 = scale·k3
///
/// Returns `None` if the flag is absent/zero or the coefficient tag is missing.
fn build_orf_warp_rectilinear(makernote: &IFD) -> Option<Vec<u8>> {
  // Look up the ImageProcessing sub-IFD specifically.  Olympus reuses tag
  // numbers across sub-IFDs, so a generic recursive `find_ifds_with_tag`
  // search would pick whichever sub-IFD comes first by traversal order
  // (BTreeMap key order — EquipmentIFD 0x2010 before ImageProcessingIFD
  // 0x2040), which is the bug that caused intermittent pink rendering on
  // BlackLevel.  See decoding_bugs.md for the broader pattern.
  let imgproc = makernote.get_sub_ifd(OrfMakernotes::ImageProcessingIFD)?;

  // Check validity flag (tag 0x150f) — value 1 means the data is valid
  let valid = imgproc
    .get_entry(OrfImageProcessing::DistortionCorrectionValid)
    .map(|e| e.force_u8(0))
    .unwrap_or(0);
  if valid == 0 {
    log::debug!("ORF DistortionCorrectionValid = 0, skipping WarpRectilinear");
    return None;
  }

  // Tag 0x1510: float[4] = [k1, k2, k3, scale]
  let entry = imgproc.get_entry(OrfImageProcessing::DistortionCoefficients)?;
  if entry.count() < 4 {
    return None;
  }
  let k1 = entry.force_f32(0) as f64;
  let k2 = entry.force_f32(1) as f64;
  let k3 = entry.force_f32(2) as f64;
  let scale = entry.force_f32(3) as f64;

  if scale == 0.0 {
    return None;
  }

  let kr0 = scale;
  let kr1 = scale * k1;
  let kr2 = scale * k2;
  let kr3 = scale * k3;

  log::debug!("ORF WarpRectilinear: kr0={:.6} kr1={:.6} kr2={:.6} kr3={:.6}", kr0, kr1, kr2, kr3);

  let kr = [[kr0, kr1, kr2, kr3]];
  let kt = [[0.0_f64, 0.0_f64]];
  let opcode = opcodes::encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
  Some(opcodes::encode_opcode_list(&[opcode]))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bits::Endian;
  use crate::formats::tiff::Entry;
  use std::collections::BTreeMap;

  /// Build a minimal IFD containing the given entries (no sub-IFDs).
  fn make_ifd(entries: Vec<(u16, Value)>) -> IFD {
    let mut tags = BTreeMap::new();
    for (tag, value) in entries {
      tags.insert(tag, Entry { tag, value, embedded: None });
    }
    IFD {
      offset: 0,
      base: 0,
      corr: 0,
      next_ifd: 0,
      entries: tags,
      endian: Endian::Little,
      sub: BTreeMap::new(),
      chain: Vec::new(),
    }
  }

  /// Regression: `build_orf_warp_rectilinear` must read DistortionCoefficients
  /// from the ImageProcessing sub-IFD specifically, not from any sub-IFD that
  /// happens to contain tag 0x1510.
  ///
  /// Olympus makernotes contain several sub-IFDs (Equipment, CameraSettings,
  /// ImageProcessing, RawInfo).  Tag numbers are namespaced per sub-IFD, so
  /// the *same* numeric tag ID can mean different things in different
  /// sub-IFDs.  Using `find_ifds_with_tag(0x1510).first()` picks whichever
  /// sub-IFD comes first by traversal order (BTreeMap key order), which is
  /// `EquipmentIFD = 0x2010` BEFORE `ImageProcessingIFD = 0x2040`.
  #[test]
  fn build_orf_warp_rectilinear_uses_imageprocessing_subifd() {
    // ImageProcessingIFD: valid distortion data with scale=2.0 — must be picked
    let imgproc = make_ifd(vec![
      (OrfImageProcessing::DistortionCorrectionValid as u16, Value::Byte(vec![1])),
      (
        OrfImageProcessing::DistortionCoefficients as u16,
        Value::Float(vec![0.1, 0.2, 0.3, 2.0]),
      ),
    ]);

    // EquipmentIFD: a confounder that ALSO has tag 0x1510 with a different
    // scale. (In real ORFs the same tag number is reused for unrelated data
    // in different sub-IFDs — that is the bug we are guarding against.)
    let equipment = make_ifd(vec![
      (OrfImageProcessing::DistortionCorrectionValid as u16, Value::Byte(vec![1])),
      (
        OrfImageProcessing::DistortionCoefficients as u16,
        Value::Float(vec![9.9, 9.9, 9.9, 9.9]),
      ),
    ]);

    // Build the makernote with both sub-IFDs.  EquipmentIFD (0x2010) sorts
    // before ImageProcessingIFD (0x2040), so a naive `find_ifds_with_tag`
    // search would return Equipment first.
    let mut makernote = make_ifd(vec![]);
    makernote.sub.insert(OrfMakernotes::EquipmentIFD as u16, vec![equipment]);
    makernote.sub.insert(OrfMakernotes::ImageProcessingIFD as u16, vec![imgproc]);

    let result = build_orf_warp_rectilinear(&makernote);
    assert!(result.is_some(), "Expected WarpRectilinear opcode to be produced");
    let bytes = result.unwrap();

    // The opcode bytes encode kr0=scale, kr1=scale*k1, etc.  Re-encode the
    // expected output, using the same f32→f64 promotion the production code
    // does so that float rounding matches exactly.
    let to_f64 = |v: f32| v as f64;
    let expected = {
      let scale = to_f64(2.0_f32);
      let kr = [[scale, scale * to_f64(0.1_f32), scale * to_f64(0.2_f32), scale * to_f64(0.3_f32)]];
      let kt = [[0.0_f64, 0.0_f64]];
      let op = opcodes::encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
      opcodes::encode_opcode_list(&[op])
    };
    let wrong = {
      let scale = to_f64(9.9_f32);
      let kr = [[scale, scale * to_f64(9.9_f32), scale * to_f64(9.9_f32), scale * to_f64(9.9_f32)]];
      let kt = [[0.0_f64, 0.0_f64]];
      let op = opcodes::encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
      opcodes::encode_opcode_list(&[op])
    };

    assert_eq!(
      bytes, expected,
      "build_orf_warp_rectilinear picked the wrong sub-IFD's data \
       (got the EquipmentIFD confounder instead of ImageProcessingIFD)"
    );
    assert_ne!(bytes, wrong, "Unexpected match against confounder data");
  }
}
