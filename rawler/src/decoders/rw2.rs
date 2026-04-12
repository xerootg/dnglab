use std::rc::Rc;

use image::DynamicImage;

use crate::CFA;
use crate::RawImage;
use crate::RawLoader;
use crate::RawlerError;
use crate::Result;
use crate::analyze::FormatDump;
use crate::decompressors::packed::decompress_12le_unpacked_left_aligned;
use crate::decompressors::packed::decompress_12le_wcontrol;
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
use crate::lens::LensDescription;
use crate::lens::LensResolver;
use crate::pixarray::PixU16;
use crate::rawimage::CFAConfig;
use crate::rawimage::RawPhotometricInterpretation;
use crate::rawsource::RawSource;
use crate::tags::DngTag;
use crate::tags::ExifTag;
use crate::tags::TiffCommonTag;
use crate::tags::tiff_tag_enum;

use self::v4decompressor::decode_panasonic_v4;
use self::v5decompressor::decode_panasonic_v5;
use self::v6decompressor::decode_panasonic_v6;
use self::v7decompressor::decode_panasonic_v7;
use self::v8decompressor::decode_panasonic_v8;

use super::BlackLevel;
use super::Camera;
use super::Decoder;
use super::FormatHint;
use super::RawDecodeParams;
use super::RawMetadata;
use super::WellKnownIFD;

pub(crate) mod v4decompressor;
pub(crate) mod v5decompressor;
pub(crate) mod v6decompressor;
pub(crate) mod v7decompressor;
pub(crate) mod v8decompressor;

/// Parsed radial distortion parameters from the Panasonic DistortionInfo blob.
///
/// The camera embeds these in the PanasonicRaw IFD (tag 0x0119).
/// The correction formula is: `Ru = scale * (Rd + a·Rd³ + b·Rd⁵ + c·Rd⁷)`
/// where `Rd` / `Ru` are pixel radii normalised to `n` (DistortionN).
#[derive(Debug, Clone)]
struct PanasonicDistortionParams {
  a: f64,     // DistortionParam02 / 32768 — cubic radial coefficient
  b: f64,     // DistortionParam04 / 32768 — quintic radial coefficient
  c: f64,     // DistortionParam08 / 32768 — septic radial coefficient
  scale: f64, // 1 / (1 + raw_scale/32768) — overall scale
  n: f64,     // DistortionN — reference radius in pixels
}

#[derive(Debug, Clone)]
pub struct Rw2Decoder<'a> {
  #[allow(unused)]
  rawloader: &'a RawLoader,
  tiff: GenericTiffReader,
  camera_ifd: Option<IFD>,
  camera: Camera,
  dist_params: Option<PanasonicDistortionParams>,
}

impl<'a> Rw2Decoder<'a> {
  pub fn new(_file: &RawSource, tiff: GenericTiffReader, rawloader: &'a RawLoader) -> Result<Rw2Decoder<'a>> {
    let raw = {
      let data = tiff.find_ifds_with_tag(TiffCommonTag::PanaOffsets);
      if !data.is_empty() {
        data[0]
      } else {
        tiff
          .find_first_ifd_with_tag(TiffCommonTag::StripOffsets)
          .ok_or_else(|| RawlerError::DecoderFailed(format!("Failed to find a IFD with StripOffsets tag")))?
      }
    };

    let width = fetch_tiff_tag!(raw, TiffCommonTag::PanaWidth).force_usize(0);
    let height = fetch_tiff_tag!(raw, TiffCommonTag::PanaLength).force_usize(0);

    let mode = {
      let ratio = width * 100 / height;
      if ratio < 125 {
        "1:1"
      } else if ratio < 145 {
        "4:3"
      } else if ratio < 165 {
        "3:2"
      } else {
        "16:9"
      }
    };
    let camera = rawloader.check_supported_with_mode(tiff.root_ifd(), mode)?;

    let camera_ifd = if let Some(ifd) = tiff.get_entry(PanasonicTag::CameraIFD) {
      let buf = ifd.get_data();
      match IFD::new_root(&mut std::io::Cursor::new(buf), 0) {
        Ok(ifd) => Some(ifd),
        Err(_) => None,
      }
    } else {
      None
    };

    let dist_params = tiff
      .get_entry(PanasonicTag::DistortionInfo)
      .and_then(|e| {
        if let Value::Undefined(data) = &e.value {
          parse_panasonic_distortion(data)
        } else {
          None
        }
      });

    Ok(Rw2Decoder {
      rawloader,
      tiff,
      camera_ifd,
      camera,
      dist_params,
    })
  }
}

impl<'a> Decoder for Rw2Decoder<'a> {
  fn raw_image(&self, file: &RawSource, _params: &RawDecodeParams, dummy: bool) -> Result<RawImage> {
    let width;
    let height;

    let (raw, split) = {
      let data = self.tiff.find_ifds_with_tag(TiffCommonTag::PanaOffsets);
      if !data.is_empty() {
        (data[0], true)
      } else {
        (
          self
            .tiff
            .find_first_ifd_with_tag(TiffCommonTag::StripOffsets)
            .ok_or_else(|| RawlerError::DecoderFailed(format!("Failed to find a IFD with StripOffsets tag")))?,
          false,
        )
      }
    };

    let compression = raw.get_entry(PanasonicTag::Compression).map(|entry| entry.force_u16(0)).unwrap_or_default(); // TODO BUG
    //let compression = fetch_tiff_tag!(raw, PanasonicTag::Compression).force_u16(0);

    let raw_format = raw.get_entry(PanasonicTag::RawFormat).map(|entry| entry.force_u16(0)).unwrap_or_default(); // TODO BUG

    let bps = fetch_tiff_tag!(raw, PanasonicTag::BitsPerSample).force_u32(0);
    let multishot = raw.get_entry(PanasonicTag::Multishot).map(|entry| entry.force_u32(0) == 65536).unwrap_or(false);

    let image = {
      let data = self.tiff.find_ifds_with_tag(TiffCommonTag::PanaOffsets);
      if !data.is_empty() {
        let raw = data[0];
        width = fetch_tiff_tag!(raw, TiffCommonTag::PanaWidth).force_usize(0);
        height = fetch_tiff_tag!(raw, TiffCommonTag::PanaLength).force_usize(0);
        let offset = fetch_tiff_tag!(raw, TiffCommonTag::PanaOffsets).force_usize(0);
        //let size = fetch_tiff_tag!(raw, TiffCommonTag::StripByteCounts).force_usize(0);
        log::debug!("PanaOffset: {}", offset);
        let src = file.subview_until_eof_padded(offset as u64)?; // TODO add size and check all samples
        Rw2Decoder::decode_panasonic(file, &src, width, height, split, raw_format, bps, self.tiff.root_ifd(), dummy)?
      } else {
        let raw = self
          .tiff
          .find_first_ifd_with_tag(TiffCommonTag::StripOffsets)
          .ok_or_else(|| RawlerError::DecoderFailed(format!("Failed to find a IFD with StripOffsets tag")))?;
        width = fetch_tiff_tag!(raw, TiffCommonTag::PanaWidth).force_usize(0);
        height = fetch_tiff_tag!(raw, TiffCommonTag::PanaLength).force_usize(0);
        let offset = fetch_tiff_tag!(raw, TiffCommonTag::StripOffsets).force_usize(0);
        //let size = fetch_tiff_tag!(raw, TiffCommonTag::StripByteCounts).force_usize(0);
        log::debug!("StripOffset: {}", offset);
        let src = file.subview_until_eof_padded(offset as u64)?; // TODO add size and check all samples

        if src.len() >= width * height * 2 {
          decompress_12le_unpacked_left_aligned(&src, width, height, dummy)?
        } else if src.len() >= width * height * 3 / 2 {
          decompress_12le_wcontrol(&src, width, height, dummy)?
        } else {
          Rw2Decoder::decode_panasonic(file, &src, width, height, split, raw_format, bps, self.tiff.root_ifd(), dummy)?
        }
      }
    };

    log::debug!(
      "RW2 raw: {}, compression: {}, bps: {}, width: {}, height: {}, multishot: {}",
      raw_format,
      compression,
      bps,
      width,
      height,
      multishot
    );

    let cpp = 1;
    let blacklevel = self.get_blacklevel()?;
    let mut camera = self.camera.clone();
    if let Some(cfa) = self.get_cfa()? {
      camera.cfa = cfa;
    }

    // Compute ISO-dependent NoiseProfile if not already set via TOML.
    // per-camera noise model: variance(x) = S*x + O (normalized),
    // where S scales linearly with ISO gain and O scales with its square.
    if camera.noise_profile.is_none() {
      if let Some(iso_entry) = self.tiff.get_entry(PanasonicTag::ISO) {
        let iso = iso_entry.force_u32(0) as f64;
        if iso > 0.0 {
          const NOISE_S_BASE: f64 = 3.685797665369650e-05;
          const NOISE_O_BASE: f64 = 6.496027680310430e-09;
          let gain = iso / 100.0;
          let s = NOISE_S_BASE * gain;
          let o = NOISE_O_BASE * gain * gain;
          camera.noise_profile = Some(vec![s, o]);
        }
      }
    }

    let photometric = RawPhotometricInterpretation::Cfa(CFAConfig::new_from_camera(&camera));
    let mut img = RawImage::new(camera, image, cpp, normalize_wb(self.get_wb()?), photometric, blacklevel, None, dummy);

    if let Some(area) = self.get_active_area()? {
      img.active_area = Some(area);
      img.crop_area = Some(area);
    } else if let Some(area) = self.get_crop()? {
      img.crop_area = Some(area);
    }

    Ok(img)
  }

  fn preview_image(&self, _file: &RawSource, params: &RawDecodeParams) -> Result<Option<DynamicImage>> {
    if params.image_index != 0 {
      return Ok(None);
    }
    if let Some(data) = self.tiff.get_entry(PanasonicTag::JpegData) {
      let buf = data.get_data();
      let img = image::load_from_memory_with_format(buf, image::ImageFormat::Jpeg)
        .map_err(|e| RawlerError::DecoderFailed(format!("Unable to load jpeg preview: {:?}", e)))?;
      return Ok(Some(img));
    }
    Ok(None)
  }

  fn preview_jpeg(&self, _file: &RawSource, _params: &RawDecodeParams) -> Result<Option<(Vec<u8>, u32, u32)>> {
    if let Some(data) = self.tiff.get_entry(PanasonicTag::JpegData) {
      let buf = data.get_data();
      let (width, height) = super::jpeg_dimensions(buf);
      if width > 0 && height > 0 {
        return Ok(Some((buf.to_vec(), width, height)));
      }
    }
    Ok(None)
  }

  fn format_dump(&self) -> FormatDump {
    todo!()
  }

  fn raw_metadata(&self, _file: &RawSource, _params: &RawDecodeParams) -> Result<RawMetadata> {
    let mut exif = Exif::new(self.tiff.root_ifd())?;
    // The PanasonicRaw EXIF sub-IFD only has ~14 basic entries.
    // The full standard EXIF (Contrast, Saturation, ExposureMode, etc.)
    // lives in the embedded JPEG's EXIF structure. Parse it to fill gaps.
    // Also extract makernote-based metadata (serial numbers, lens type).
    let mut mn_lens_type = None;
    let mut mn_serial = None;
    let mut mn_lens_serial = None;
    let mut mn_utc_timestamp = None;
    if let Some(jpeg_entry) = self.tiff.get_entry(PanasonicTag::JpegData) {
      let jpeg_buf = jpeg_entry.get_data();
      if let Some(exif_start) = jpeg_buf.windows(6).position(|w| w == b"Exif\x00\x00") {
        let tiff_data = &jpeg_buf[exif_start + 6..];
        if let Ok(jpeg_ifd) = IFD::new_root(&mut std::io::Cursor::new(tiff_data), 0) {
          exif.extend_from_ifd(&jpeg_ifd)?;
          if let Some(jpeg_exif_ifd) = jpeg_ifd.get_sub_ifd(ExifTag::ExifOffset) {
            exif.extend_from_ifd(jpeg_exif_ifd)?;
            // MakerNotes live in the JPEG's ExifIFD, not in the raw IFD.
            let (lt, sn, ls, ts) = Self::parse_panasonic_makernotes(jpeg_exif_ifd, tiff_data);
            mn_lens_type = lt;
            mn_serial = sn;
            mn_lens_serial = ls;
            mn_utc_timestamp = ts;
          }
        }
      }
    }
    if exif.iso_speed.unwrap_or(0) == 0 && exif.iso_speed_ratings.unwrap_or(0) == 0 && exif.recommended_exposure_index.unwrap_or(0) == 0 {
      // Use ISO from PanasonicRaw IFD
      if let Some(iso) = self.tiff.get_entry(PanasonicTag::ISO) {
        exif.iso_speed_ratings = Some(iso.force_u16(0));
      }
    }

    if exif.serial_number.is_none() {
      exif.serial_number = mn_serial;
    }
    if exif.lens_serial_number.is_none() {
      exif.lens_serial_number = mn_lens_serial;
    }

    // Compute OffsetTime from makernote UTC timestamp vs DateTimeOriginal (local).
    // TimeStamp (0x00af) is UTC; DateTimeOriginal is local time.
    // Difference = local - UTC = timezone offset.
    if exif.offset_time.is_none() {
      if let (Some(utc_str), Some(local_str)) = (&mn_utc_timestamp, &exif.date_time_original) {
        if let Some(offset) = compute_timezone_offset(local_str, utc_str) {
          exif.offset_time = Some(offset.clone());
          exif.offset_time_original = Some(offset.clone());
          exif.offset_time_digitized = Some(offset);
        }
      }
    }

    let mut mdata = RawMetadata::new_with_lens(&self.camera, exif, self.get_lens_description()?.cloned());

    // If the lens database didn't resolve (older bodies without LensTypeMake/LensTypeModel),
    // fall back to the Panasonic LensType string (tag 0x0051) from makernotes.
    if mdata.exif.lens_model.is_none() {
      if let Some(lens_type) = mn_lens_type {
        mdata.exif.lens_model = Some(lens_type);
      }
    }

    // When lens_spec is still None but we have a lens_model string,
    // try to parse focal/aperture ranges from it.
    // Examples: "LUMIX G VARIO 14-42/F3.5-5.6 II", "LUMIX G 25/F1.7"
    if mdata.exif.lens_spec.is_none() {
      if let Some(ref model) = mdata.exif.lens_model {
        if let Some(spec) = parse_lens_spec_from_name(model) {
          mdata.exif.lens_spec = Some(spec);
        }
      }
    }

    Ok(mdata)
  }

  fn format_hint(&self) -> FormatHint {
    FormatHint::RW2
  }

  fn ifd(&self, wk_ifd: WellKnownIFD) -> crate::Result<Option<Rc<IFD>>> {
    match wk_ifd {
      WellKnownIFD::VirtualDngRootTags => {
        let mut ifd = IFD::default();
        // Panasonic MakerNotes use absolute offsets (preceded by 12-byte
        // "Panasonic\0\0\0" header), so they are safe to copy verbatim.
        ifd.entries.insert(
          DngTag::MakerNoteSafety.into(),
          Entry { tag: DngTag::MakerNoteSafety.into(), value: Value::Short(vec![1]), embedded: None },
        );
        return Ok(Some(Rc::new(ifd)));
      }
      WellKnownIFD::VirtualDngRawTags => {
        let mut ifd = IFD::default();

        ifd.entries.insert(
          DngTag::BayerGreenSplit.into(),
          Entry { tag: DngTag::BayerGreenSplit.into(), value: Value::Long(vec![250]), embedded: None },
        );
        ifd.entries.insert(
          DngTag::AntiAliasStrength.into(),
          Entry {
            tag: DngTag::AntiAliasStrength.into(),
            value: Value::Rational(vec![Rational::new(1, 1)]),
            embedded: None,
          },
        );
        ifd.entries.insert(
          DngTag::NoiseReductionApplied.into(),
          Entry {
            tag: DngTag::NoiseReductionApplied.into(),
            value: Value::Rational(vec![Rational::new(0, 1)]),
            embedded: None,
          },
        );

        // Add WarpRectilinear distortion correction if available.
        if let Some(ref dp) = self.dist_params {
          let w = self.tiff.get_entry(PanasonicTag::PanaWidth).map(|e| e.force_usize(0)).unwrap_or(0);
          let h = self.tiff.get_entry(PanasonicTag::PanaLength).map(|e| e.force_usize(0)).unwrap_or(0);
          if w > 0 && h > 0 {
            // m = half-diagonal of the full sensor image in pixels
            let m = ((w * w + h * h) as f64).sqrt() / 2.0;
            let ratio = m / dp.n;
            let ratio2 = ratio * ratio;

            // Convert Panasonic polynomial to DNG WarpRectilinear coefficients.
            //
            // Panasonic: Ru = scale * (Rd + a·Rd³ + b·Rd⁵ + c·Rd⁷)    (r normalised to N)
            // DNG:       Ru = (kr0·Rd + kr1·Rd³ + kr2·Rd⁵ + kr3·Rd⁷)   (r normalised to m)
            //
            // Substituting Rd_pan = Rd_dng · (m/N) and solving:
            //   kr0 = scale
            //   kr1 = scale · a · (m/N)²
            //   kr2 = scale · b · (m/N)⁴
            //   kr3 = scale · c · (m/N)⁶
            let kr0 = dp.scale;
            let kr1 = dp.scale * dp.a * ratio2;
            let kr2 = dp.scale * dp.b * ratio2 * ratio2;
            let kr3 = dp.scale * dp.c * ratio2 * ratio2 * ratio2;

            log::debug!(
              "RW2 WarpRectilinear: kr0={:.6} kr1={:.6} kr2={:.6} kr3={:.6} (m={:.1} N={:.1})",
              kr0,
              kr1,
              kr2,
              kr3,
              m,
              dp.n
            );

            let kr = [[kr0, kr1, kr2, kr3]];
            let kt = [[0.0_f64, 0.0_f64]];
            let opcode = opcodes::encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, opcodes::FLAG_OPTIONAL);
            let opcode_list3 = opcodes::encode_opcode_list(&[opcode]);

            ifd.entries.insert(
              DngTag::OpcodeList3.into(),
              Entry { tag: DngTag::OpcodeList3.into(), value: Value::Undefined(opcode_list3), embedded: None },
            );
          }
        }

        return Ok(Some(Rc::new(ifd)));
      }
      _ => return Ok(None),
    }
  }
}

impl<'a> Rw2Decoder<'a> {
  fn get_wb(&self) -> Result<[f32; 4]> {
    if self.tiff.has_entry(PanasonicTag::PanaWBsR) && self.tiff.has_entry(PanasonicTag::PanaWBsB) {
      let r = fetch_tiff_tag!(self.tiff, PanasonicTag::PanaWBsR).force_u32(0) as f32;
      let b = fetch_tiff_tag!(self.tiff, PanasonicTag::PanaWBsB).force_u32(0) as f32;
      Ok([r, 256.0, 256.0, b])
    } else if self.tiff.has_entry(PanasonicTag::PanaWBs2R) && self.tiff.has_entry(PanasonicTag::PanaWBs2G) && self.tiff.has_entry(PanasonicTag::PanaWBs2B) {
      let r = fetch_tiff_tag!(self.tiff, PanasonicTag::PanaWBs2R).force_u32(0) as f32;
      let g = fetch_tiff_tag!(self.tiff, PanasonicTag::PanaWBs2G).force_u32(0) as f32;
      let b = fetch_tiff_tag!(self.tiff, PanasonicTag::PanaWBs2B).force_u32(0) as f32;
      Ok([r, g, g, b])
    } else {
      Err(RawlerError::DecoderFailed("RW2: Couldn't find WB".to_string()))
    }
  }

  fn get_cfa(&self) -> Result<Option<CFA>> {
    if self.tiff.has_entry(PanasonicTag::CFAPattern) {
      let pattern = fetch_tiff_tag!(self.tiff, PanasonicTag::CFAPattern).force_u16(0);
      Ok(Some(match pattern {
        1 => CFA::new("RGGB"),
        2 => CFA::new("GRBG"),
        3 => CFA::new("GBRG"),
        4 => CFA::new("BGGR"),
        _ => return Err(format!("RW2: Unknown CFA pattern: {}", pattern).into()),
      }))
    } else {
      Ok(None)
    }
  }

  fn get_blacklevel(&self) -> Result<Option<BlackLevel>> {
    if self.tiff.has_entry(PanasonicTag::BlackLevelRed) {
      let r = fetch_tiff_tag!(self.tiff, PanasonicTag::BlackLevelRed).force_u16(0);
      let g = fetch_tiff_tag!(self.tiff, PanasonicTag::BlackLevelGreen).force_u16(0);
      let b = fetch_tiff_tag!(self.tiff, PanasonicTag::BlackLevelBlue).force_u16(0);
      Ok(Some(BlackLevel::new(&[r, g, g, b], self.camera.cfa.width, self.camera.cfa.height, 1)))
    } else {
      Ok(None)
    }
  }

  /// Get lens description by analyzing TIFF tags and makernotes
  fn get_lens_description(&self) -> Result<Option<&'static LensDescription>> {
    const MFT_MOUNT: &str = "MFT-mount";
    if let Some(ifd) = &self.camera_ifd {
      if ifd.has_entry(CameraIfdTag::LensTypeMake) && ifd.has_entry(CameraIfdTag::LensTypeModel) {
        let make_id = fetch_tiff_tag!(ifd, CameraIfdTag::LensTypeMake);
        let model_id = fetch_tiff_tag!(ifd, CameraIfdTag::LensTypeModel);

        if make_id.value_type() == 3 && model_id.value_type() == 3 {
          let composite_id = format!(
            "{:02X} {:02X} {:02X}",
            make_id.force_u16(0) & 0xFF,
            model_id.force_u16(0) & 0xFF,
            model_id.force_u16(0) >> 8
          );
          log::debug!("RW2 lens composite ID: {}", composite_id);
          let resolver = LensResolver::new()
            .with_camera(&self.camera)
            .with_olympus_id(Some(composite_id))
            .with_focal_len(self.get_focal_len()?)
            .with_mounts(&[MFT_MOUNT.into()]);
          return Ok(resolver.resolve());
        } else {
          log::info!("Unknown value types for lens tags: {}, {}", make_id.value_type(), model_id.value_type());
        }
      }
    }
    log::warn!("No lens data available");
    Ok(None)
  }

  /// Parse information from Panasonic makernotes.
  ///
  /// Returns (lens_type, serial_number, lens_serial_number, utc_timestamp) extracted from:
  /// - 0x0025: InternalSerialNumber, e.g. "(X04) 2014:10:28 no. 0019" → "X041410280019"
  /// - 0x0051: LensType string (primary lens ID on older bodies)
  /// - 0x0052: LensSerialNumber
  /// - 0x00af: TimeStamp (UTC), format "YYYY:MM:DD HH:MM:SS"
  fn parse_panasonic_makernotes(exif_ifd: &IFD, data: &[u8]) -> (Option<String>, Option<String>, Option<String>, Option<String>) {
    let Some(mn) = exif_ifd
      .parse_makernote(&mut std::io::Cursor::new(data), crate::formats::tiff::ifd::OffsetMode::Absolute, &[])
      .ok()
      .flatten()
    else {
      return (None, None, None, None);
    };

    // 0x0051: LensType string
    let lens_type = mn.get_entry(0x0051_u16).and_then(|entry| {
      if let Value::Ascii(data) = &entry.value {
        let s = data.strings().into_iter().next()?.trim().to_string();
        if !s.is_empty() { Some(s) } else { None }
      } else {
        None
      }
    });

    // 0x0025: InternalSerialNumber → camera serial
    // Format: "(X04) 2014:10:28 no. 0019" → strip to "X041410280019"
    // Stored as either Ascii or Undefined bytes.
    let serial_number = mn.get_entry(0x0025_u16).and_then(|entry| {
      let raw_str = match &entry.value {
        Value::Ascii(data) => data.strings().into_iter().next().map(|s| s.trim().to_string()),
        Value::Undefined(data) => {
          // Treat raw bytes as ASCII, strip trailing NULs
          let s = String::from_utf8_lossy(data);
          let trimmed = s.trim_end_matches('\0').trim().to_string();
          if trimmed.is_empty() { None } else { Some(trimmed) }
        }
        _ => None,
      };
      raw_str.and_then(|s| {
        if s.is_empty() {
          return None;
        }
        // Remove parentheses, colons, spaces, and " no. " to build compact serial
        let compact: String = s
          .replace('(', "")
          .replace(')', "")
          .replace(':', "")
          .replace(" no. ", "")
          .replace(' ', "");
        if !compact.is_empty() { Some(compact) } else { None }
      })
    });

    // 0x0052: LensSerialNumber
    let lens_serial = mn.get_entry(0x0052_u16).and_then(|entry| {
      if let Value::Ascii(data) = &entry.value {
        let s = data.strings().into_iter().next()?.trim().to_string();
        if !s.is_empty() { Some(s) } else { None }
      } else {
        None
      }
    });

    // 0x00af: TimeStamp (UTC) — format "YYYY:MM:DD HH:MM:SS"
    let utc_timestamp = mn.get_entry(0x00af_u16).and_then(|entry| {
      let s = match &entry.value {
        Value::Ascii(data) => data.strings().into_iter().next().map(|s| s.trim().to_string()),
        Value::Undefined(data) => {
          let s = String::from_utf8_lossy(data);
          let trimmed = s.trim_end_matches('\0').trim().to_string();
          if trimmed.is_empty() { None } else { Some(trimmed) }
        }
        _ => None,
      };
      s.filter(|ts| ts.len() >= 19)
    });

    (lens_type, serial_number, lens_serial, utc_timestamp)
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

  fn get_crop(&self) -> Result<Option<Rect>> {
    if self.tiff.has_entry(PanasonicTag::CropLeft) {
      let crop_left = fetch_tiff_tag!(self.tiff, PanasonicTag::CropLeft).force_usize(0);
      let crop_top = fetch_tiff_tag!(self.tiff, PanasonicTag::CropTop).force_usize(0);
      let crop_right = fetch_tiff_tag!(self.tiff, PanasonicTag::CropRight).force_usize(0);
      let crop_bottom = fetch_tiff_tag!(self.tiff, PanasonicTag::CropBottom).force_usize(0);
      Ok(Some(Rect::new(
        Point::new(crop_left, crop_top),
        Dim2::new(crop_right - crop_left, crop_bottom - crop_top),
      )))
    } else {
      Ok(None)
    }
  }

  fn get_active_area(&self) -> Result<Option<Rect>> {
    if self.tiff.has_entry(PanasonicTag::SensorLeftBorder) {
      let sensor_left = fetch_tiff_tag!(self.tiff, PanasonicTag::SensorLeftBorder).force_usize(0);
      let sensor_top = fetch_tiff_tag!(self.tiff, PanasonicTag::SensorTopBorder).force_usize(0);
      let sensor_right = fetch_tiff_tag!(self.tiff, PanasonicTag::SensorRightBorder).force_usize(0);
      let sensor_bottom = fetch_tiff_tag!(self.tiff, PanasonicTag::SensorBottomBorder).force_usize(0);
      Ok(Some(Rect::new(
        Point::new(sensor_left, sensor_top),
        Dim2::new(sensor_right - sensor_left, sensor_bottom - sensor_top),
      )))
    } else {
      Ok(None)
    }
  }

  pub(crate) fn decode_panasonic(
    file: &RawSource,
    buf: &[u8],
    width: usize,
    height: usize,
    split: bool,
    raw_format: u16,
    bps: u32,
    ifd: &IFD,
    dummy: bool,
  ) -> Result<PixU16> {
    log::debug!("width: {}, height: {}, bps: {}", width, height, bps);
    Ok(match raw_format {
      3 => decode_panasonic_v4(buf, width, height, split, dummy),
      4 => decode_panasonic_v4(buf, width, height, split, dummy),
      5 => decode_panasonic_v5(buf, width, height, bps, dummy)?,
      6 => decode_panasonic_v6(buf, width, height, bps, dummy)?,
      7 => decode_panasonic_v7(buf, width, height, bps, dummy)?,
      8 => decode_panasonic_v8(file, width, height, bps, ifd, dummy)?,
      _ => todo!("Format {} is not implemented", raw_format), // TODO: return error
    })
  }
}

fn normalize_wb(raw_wb: [f32; 4]) -> [f32; 4] {
  log::debug!("RW2 raw wb: {:?}", raw_wb);
  let div = raw_wb[1];
  let mut norm = raw_wb;
  norm.iter_mut().for_each(|v| {
    if v.is_normal() {
      *v /= div
    }
  });
  [norm[0], (norm[1] + norm[2]) / 2.0, norm[3], f32::NAN]
}

/// Compute timezone offset string (e.g. "+09:00", "-06:00") from local and UTC
/// datetime strings in "YYYY:MM:DD HH:MM:SS" format.
fn compute_timezone_offset(local_str: &str, utc_str: &str) -> Option<String> {
  // Parse "YYYY:MM:DD HH:MM:SS" into (year, month, day, hour, min, sec)
  fn parse_dt(s: &str) -> Option<(i64, i64, i64, i64, i64, i64)> {
    let parts: Vec<&str> = s.splitn(2, ' ').collect();
    if parts.len() != 2 {
      return None;
    }
    let date: Vec<i64> = parts[0].split(':').filter_map(|p| p.parse().ok()).collect();
    let time: Vec<i64> = parts[1].split(':').filter_map(|p| p.parse().ok()).collect();
    if date.len() == 3 && time.len() == 3 {
      Some((date[0], date[1], date[2], time[0], time[1], time[2]))
    } else {
      None
    }
  }

  // Convert to a simple minutes-since-epoch approximation (good enough for offset)
  fn to_minutes(y: i64, mo: i64, d: i64, h: i64, mi: i64, _s: i64) -> i64 {
    // Approximate: treat each month as 30 days, each year as 365 days
    (y * 365 + mo * 30 + d) * 24 * 60 + h * 60 + mi
  }

  let (ly, lmo, ld, lh, lmi, ls) = parse_dt(local_str)?;
  let (uy, umo, ud, uh, umi, us) = parse_dt(utc_str)?;

  let local_min = to_minutes(ly, lmo, ld, lh, lmi, ls);
  let utc_min = to_minutes(uy, umo, ud, uh, umi, us);
  let diff = local_min - utc_min;

  // Sanity check: offset should be between -14 and +14 hours
  if diff.abs() > 14 * 60 {
    return None;
  }

  let sign = if diff >= 0 { '+' } else { '-' };
  let abs_diff = diff.abs();
  let hours = abs_diff / 60;
  let minutes = abs_diff % 60;
  Some(format!("{}{:02}:{:02}", sign, hours, minutes))
}

tiff_tag_enum!(PanasonicTag);
tiff_tag_enum!(CameraIfdTag);

/// Common tags, generally used in root IFD or SubIFDs
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum PanasonicTag {
  PanaWidth = 0x0002,
  PanaLength = 0x0003,
  SensorTopBorder = 0x0004,
  SensorLeftBorder = 0x0005,
  SensorBottomBorder = 0x0006,
  SensorRightBorder = 0x0007,
  SamplesPerPixel = 0x0008,
  CFAPattern = 0x0009,
  BitsPerSample = 0x000a,
  Compression = 0x000b,
  PanaWBsR = 0x0011,
  PanaWBsB = 0x0012,
  ISO = 0x0017,

  BlackLevelRed = 0x001c,
  BlackLevelGreen = 0x001d,
  BlackLevelBlue = 0x001e,

  PanaWBs2R = 0x0024,
  PanaWBs2G = 0x0025,
  PanaWBs2B = 0x0026,
  RawFormat = 0x0002d,
  JpegData = 0x002e,
  CropTop = 0x002f,
  CropLeft = 0x0030,
  CropBottom = 0x0031,
  CropRight = 0x0032,

  CF2StripHeight = 0x0037,
  CF2Unknown1 = 0x0039, // Gamma table CF2_GammaSlope?
  CF2Unknown2 = 0x003a, // Gamma table CF2_GammaPoint?
  CF2ClipVal = 0x003b,  // CF2_GammaClipVal
  CF2HufInitVal0 = 0x003c,
  CF2HufInitVal1 = 0x003d,
  CF2HufInitVal2 = 0x003e,
  CF2HufInitVal3 = 0x003f,
  CF2HufTable = 0x0040,
  CF2HufShiftDown = 0x0041,
  CF2NumberOfStripsH = 0x0042,
  CF2NumberOfStripsV = 0x0043,
  CF2StripByteOffsets = 0x0044,
  CF2StripLineOffsets = 0x0045,
  CF2StripDataSize = 0x0046,
  CF2StripWidths = 0x0047,
  CF2StripHeights = 0x0048,
  CF2StripWidth = 0x0064,

  NoiseReductionParams = 0x001b,
  WBInfo2 = 0x0027,

  CameraIFD = 0x0120,
  Multishot = 0x0121,
  DistortionInfo = 0x0119,
}

/// Common tags, generally used in root IFD or SubIFDs
#[derive(Debug, Copy, Clone, PartialEq, enumn::N)]
#[repr(u16)]
pub enum CameraIfdTag {
  LensTypeMake = 0x1201,
  LensTypeModel = 0x1202,
}

/// Read a little-endian i16 from a byte slice at the given byte offset.
#[inline]
fn read_le_i16(data: &[u8], offset: usize) -> i16 {
  i16::from_le_bytes([data[offset], data[offset + 1]])
}

/// Parse the 32-byte Panasonic DistortionInfo blob (tag 0x0119).
///
/// Layout (little-endian int16s, indices are byte offsets):
/// ```text
///  0-1   Checksum A
///  1-4   Magic "THPF" (ASCII)
///  4-5   DistortionParam02 — cubic radial coefficient  (int16s / 32768)
///  6-7   (unused)
///  8-9   DistortionParam04 — quintic radial coefficient (int16s / 32768)
/// 10-11  DistortionScale   — scale factor raw           (int16s)
/// 12-13  (unused)
/// 14-15  DistortionCorrection flag  (low nibble: 0=off/no-lens 1=on)
/// 16-17  DistortionParam08 — septic radial coefficient  (int16s / 32768)
/// 18-19  DistortionParam09 — (tangential, unused here)
/// 20-21  (unused)
/// 22-23  DistortionParam11 — (higher order, unused here)
/// 24-25  DistortionN       — reference radius in pixels (int16s)
/// 26-27  (unused)
/// 28-29  Checksum B
/// 30-31  Checksum C
/// ```
///
/// Returns `None` if the blob is malformed, too short, or correction is disabled.
fn parse_panasonic_distortion(data: &[u8]) -> Option<PanasonicDistortionParams> {
  if data.len() < 32 {
    return None;
  }
  // Verify the "THPF" marker at bytes 1-4
  if &data[1..5] != b"THPF" {
    log::debug!("RW2 DistortionInfo: missing THPF magic, skipping");
    return None;
  }
  // DistortionCorrection flag: low nibble of the int16s at byte offset 14.
  // 0 = correction off / no lens, 1 = on.
  let correction_flag = read_le_i16(data, 14) & 0x0f;
  if correction_flag == 0 {
    log::debug!("RW2 DistortionInfo: correction disabled (flag=0), skipping");
    return None;
  }

  let raw_a = read_le_i16(data, 4) as f64;   // DistortionParam02
  let raw_b = read_le_i16(data, 8) as f64;   // DistortionParam04
  let raw_s = read_le_i16(data, 10) as f64;  // DistortionScale
  let raw_c = read_le_i16(data, 16) as f64;  // DistortionParam08
  let raw_n = read_le_i16(data, 24) as f64;  // DistortionN

  if raw_n <= 0.0 {
    log::debug!("RW2 DistortionInfo: DistortionN <= 0, skipping");
    return None;
  }

  let params = PanasonicDistortionParams {
    a: raw_a / 32768.0,
    b: raw_b / 32768.0,
    c: raw_c / 32768.0,
    scale: 1.0 / (1.0 + raw_s / 32768.0),
    n: raw_n,
  };

  log::debug!(
    "RW2 DistortionInfo: a={:.6} b={:.6} c={:.6} scale={:.6} N={:.0}",
    params.a,
    params.b,
    params.c,
    params.scale,
    params.n
  );

  Some(params)
}

/// Parse focal and aperture ranges from a Panasonic/Lumix lens name string.
///
/// Handles common formats:
///   - `"LUMIX G VARIO 14-42/F3.5-5.6 II"` → \[14, 42, 3.5, 5.6\]
///   - `"LUMIX G 25/F1.7"` → \[25, 25, 1.7, 1.7\]
///   - `"LEICA DG 12-60/F2.8-4.0"` → \[12, 60, 2.8, 4.0\]
fn parse_lens_spec_from_name(name: &str) -> Option<[Rational; 4]> {
  // Find the "NN-NN/FN.N-N.N" or "NN/FN.N" pattern.
  // Strategy: find the '/' that separates focal from aperture.
  let slash_pos = name.find('/')?;
  let before_slash = &name[..slash_pos];
  let after_slash = &name[slash_pos + 1..];

  // Parse focal range: take the last numeric segment before '/'.
  // E.g. "LUMIX G VARIO 14-42" → "14-42"
  let focal_str = before_slash.rsplit(|c: char| !c.is_ascii_digit() && c != '-').next().unwrap_or("");
  let (fl_min, fl_max) = parse_range(focal_str)?;

  // Parse aperture range: strip leading 'F' or 'f', then parse.
  // E.g. "F3.5-5.6 II" → "3.5-5.6"
  let ap_str = after_slash.trim_start_matches(|c: char| c == 'F' || c == 'f');
  let ap_end = ap_str.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(ap_str.len());
  let (ap_min, ap_max) = parse_range(&ap_str[..ap_end])?;

  Some([float_to_rational(fl_min), float_to_rational(fl_max), float_to_rational(ap_min), float_to_rational(ap_max)])
}

fn parse_range(s: &str) -> Option<(f64, f64)> {
  if s.is_empty() {
    return None;
  }
  if let Some(dash_pos) = s.find('-') {
    let a: f64 = s[..dash_pos].parse().ok()?;
    let b: f64 = s[dash_pos + 1..].parse().ok()?;
    Some((a, b))
  } else {
    let v: f64 = s.parse().ok()?;
    Some((v, v))
  }
}

fn float_to_rational(v: f64) -> Rational {
  // Use denominator 10 for values with one decimal, 100 for two, 1 for integers.
  if (v - v.round()).abs() < 0.001 {
    Rational { n: v.round() as u32, d: 1 }
  } else if (v * 10.0 - (v * 10.0).round()).abs() < 0.001 {
    Rational { n: (v * 10.0).round() as u32, d: 10 }
  } else {
    Rational { n: (v * 100.0).round() as u32, d: 100 }
  }
}
