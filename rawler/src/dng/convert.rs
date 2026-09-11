use std::{
  ffi::OsStr,
  io::{Cursor, Seek, Write},
  path::{Path, PathBuf},
  sync::Arc,
  thread::JoinHandle,
};

use image::DynamicImage;

use crate::{
  RawImage, RawImageData,
  dcp::{DcpProfile, auto_find_dcp, find_dcp},
  decoders::{Decoder, RawDecodeParams, RawPhotometricInterpretation, WellKnownIFD, WhiteLevel},
  dng::{DNG_VERSION_V1_4, PREVIEW_JPEG_QUALITY, original::OriginalCompressed, writer::DngWriter},
  formats::tiff::{Rational, SRational},
  imgop::{
    develop::RawDevelop,
    fuji_rotate::fuji_normalize_rotation,
    sensor::{Demosaic, bayer::ppg::PPGDemosaic},
  },
  pixarray::PixF32,
  rawsource::RawSource,
  tags::{DngTag, ExifTag, TiffCommonTag},
};

use super::{CropMode, DngCompression, DngPhotometricConversion};

/// Parameters for DNG conversion
#[derive(Clone, Debug)]
pub struct ConvertParams {
  pub embedded: bool,
  pub compression: DngCompression,
  pub photometric_conversion: DngPhotometricConversion,
  pub apply_scaling: bool,
  pub crop: CropMode,
  pub predictor: u8,
  pub preview: bool,
  pub thumbnail: bool,
  pub artist: Option<String>,
  pub software: String,
  pub index: usize,
  pub keep_mtime: bool,
  /// Optional path to a directory containing DCP (DNG Camera Profile) files.
  /// When set, the converter looks for `<dcp_dir>/<UniqueCameraModel>.dcp`
  /// and injects ForwardMatrix, HueSatMap, ToneCurve, and related tags.
  pub dcp_dir: Option<PathBuf>,
  /// Explicit DCP file to inject, bypassing the `dcp_dir` auto-match.
  /// Takes precedence over both `dcp_dir` and the system-path
  /// `auto_find_dcp` fallback.
  pub dcp_file: Option<PathBuf>,
}

impl Default for ConvertParams {
  fn default() -> Self {
    Self {
      embedded: true,
      compression: DngCompression::Lossless,
      photometric_conversion: DngPhotometricConversion::Original,
      apply_scaling: false,
      crop: CropMode::Best,
      predictor: 1,
      preview: true,
      thumbnail: true,
      artist: None,
      software: "DNGLab".into(),
      index: 0,
      keep_mtime: false,
      dcp_dir: None,
      dcp_file: None,
    }
  }
}

/// Convert a raw input file into DNG
///
/// We don't accept a DNG file path here, because we don't know
/// how to handle existing target files, buffering, etc.
/// This is up to the caller.
pub fn convert_raw_file<W: Write + Seek + Send>(raw: &Path, dng: &mut W, params: &ConvertParams) -> crate::Result<()> {
  let original_filename = raw.file_name().and_then(OsStr::to_str).unwrap_or_default();
  //let raw_stream = BufReader::new(File::open(raw)?); // TODO: add path hint to error?
  //let rawfile = RawFile::new(PathBuf::from(raw), raw_stream);

  let rawfile = Arc::new(RawSource::new(raw)?);

  let original_compress_thread = if params.embedded {
    let orig_source = rawfile.clone();
    Some(std::thread::spawn(move || OriginalCompressed::compress(&mut orig_source.reader())))
  } else {
    None
  };

  internal_convert(&rawfile, dng, original_filename, original_compress_thread, params)
}

/// Convert a raw input file into DNG
pub fn convert_raw_source<W>(raw_source: &RawSource, dng: &mut W, original_filename: impl AsRef<str>, params: &ConvertParams) -> crate::Result<()>
where
  W: Write + Seek + Send,
{
  let original_compress_thread = if params.embedded {
    let mut original_stream = Cursor::new(raw_source.as_vec()?);
    Some(std::thread::spawn(move || OriginalCompressed::compress(&mut original_stream)))
  } else {
    None
  };

  internal_convert(raw_source, dng, original_filename, original_compress_thread, params)
}

fn internal_convert<W>(
  rawfile: &RawSource,
  dng: &mut W,
  original_filename: impl AsRef<str>,
  original_compress_thread: Option<JoinHandle<Result<OriginalCompressed, std::io::Error>>>,
  params: &ConvertParams,
) -> crate::Result<()>
where
  W: Write + Seek + Send,
{
  let decoder = crate::get_decoder(rawfile)?;
  let raw_params = RawDecodeParams { image_index: params.index };
  let mut rawimage = decoder.raw_image(rawfile, &raw_params, false)?;
  let metadata = decoder.raw_metadata(rawfile, &raw_params)?;

  log::info!(
    "DNG conversion: '{}', make: {}, model: {}, raw-image-count: {}",
    original_filename.as_ref(),
    rawimage.clean_make,
    rawimage.clean_model,
    decoder.raw_image_count()?
  );
  log::debug!("Raw image WB coeff: {:?}", rawimage.wb_coeffs);

  if rawimage.camera.find_hint("fuji_rotation") || rawimage.camera.find_hint("fuji_rotation_alt") {
    // if the raw image needs to be rotated, we do this before
    // writing the image to DNG. This requires scaling and debayer
    // to be applied.
    rawimage.apply_scaling()?;
    log::debug!("Raw image requires fuji_rotation before writing to DNG");
    let pixels = PixF32::new_with(rawimage.data.as_f32().into_owned(), rawimage.width, rawimage.height);
    let roi = rawimage.active_area.unwrap_or(pixels.rect());
    let demosaic = PPGDemosaic::new();
    let mut rgb = demosaic.demosaic(&pixels, &rawimage.camera.cfa, &rawimage.camera.plane_color, roi);
    let fuji_rotation_width = rawimage.fuji_rotation_width.expect("fuji_rotate: no rotation width found");
    let extra_rotate = rawimage.camera.find_hint("fuji_rotate_90cw");
    rgb = fuji_normalize_rotation(&rgb, fuji_rotation_width, extra_rotate);
    rawimage.width = rgb.width;
    rawimage.height = rgb.height;
    rawimage.active_area = None;
    rawimage.cpp = 3;
    rawimage.whitelevel = WhiteLevel::new([1, 1, 1]); // Already scaled up to 0.0 .. 1.0
    rawimage.photometric = RawPhotometricInterpretation::LinearRaw;
    rawimage.data = RawImageData::Float(rgb.into_flatten());
  } else if params.apply_scaling {
    rawimage.apply_scaling()?;
  }

  let mut dng = DngWriter::new(dng, DNG_VERSION_V1_4)?;

  // Write RAW image for subframe type 0
  // If no thumbnail should be written to root IFD, we need to put the raw image into
  // root IFD instead.
  let mut raw = if params.thumbnail { dng.subframe(0) } else { dng.subframe_on_root(0) };
  raw.raw_image(&rawimage, params.crop, params.compression, params.photometric_conversion, params.predictor)?;
  // Check for DNG raw IFD related tags
  if let Some(dng_raw_ifd) = decoder.ifd(WellKnownIFD::VirtualDngRawTags)? {
    raw.ifd_mut().copy(dng_raw_ifd.value_iter());
  }

  // Centralized LCP lens correction fallback.
  // If the decoder didn't provide OpcodeList3 (distortion correction via
  // VirtualDngRawTags), try looking up the lens in the Adobe LCP database.
  if !raw.ifd().contains(DngTag::OpcodeList3) {
    if let Some(ref lens) = metadata.lens {
      let focal_mm = metadata
        .exif
        .focal_length
        .map(|r| r.n as f64 / r.d as f64)
        .unwrap_or(0.0);
      let aperture = metadata
        .exif
        .fnumber
        .map(|r| r.n as f64 / r.d as f64)
        .unwrap_or(0.0);

      if focal_mm > 0.0 && rawimage.width > 0 && rawimage.height > 0 {
        log::debug!(
          "LCP fallback: lens='{}', focal={:.1}mm, f/{:.1}, {}x{}",
          lens.lens_name, focal_mm, aperture, rawimage.width, rawimage.height
        );
        if let Some(opcodes) = crate::lens_profiles::lookup_lens_opcodes(
          &lens.lens_name,
          focal_mm,
          aperture,
          rawimage.width as u32,
          rawimage.height as u32,
        ) {
          log::info!("Using Adobe LCP correction for '{}'", lens.lens_name);
          if !opcodes.opcode_list3.is_empty() {
            raw.ifd_mut().add_tag_undefined(DngTag::OpcodeList3, opcodes.opcode_list3);
          }
          if !opcodes.opcode_list1.is_empty() && !raw.ifd().contains(DngTag::OpcodeList1) {
            raw.ifd_mut().add_tag_undefined(DngTag::OpcodeList1, opcodes.opcode_list1);
          }
        }
      }
    }
  }

  raw.finalize()?;

  // Write preview and thumbnail if requested.
  // Prefer raw JPEG passthrough (preview_jpeg) to avoid decode+re-encode overhead.
  if params.preview || params.thumbnail {
    let mut preview_written = false;
    if params.preview {
      match decoder.preview_jpeg(rawfile, &raw_params) {
        Ok(Some((jpeg_data, width, height))) if width > 0 && height > 0 => {
          let mut preview = dng.subframe(1);
          preview.preview_jpeg(&jpeg_data, width, height)?;
          preview.finalize()?;
          if params.thumbnail {
            // Decode JPEG for thumbnail since we need a resized image
            if let Ok(img) = image::load_from_memory_with_format(&jpeg_data, image::ImageFormat::Jpeg) {
              dng.thumbnail(&img)?;
            }
          }
          preview_written = true;
        }
        _ => {}
      }
    }
    if !preview_written {
      match generate_preview(rawfile, decoder.as_ref(), &rawimage, &raw_params) {
        Ok(image) => {
          if params.preview {
            let mut preview = dng.subframe(1);
            preview.preview(&image, PREVIEW_JPEG_QUALITY)?;
            preview.finalize()?;
          }
          if params.thumbnail {
            dng.thumbnail(&image)?;
          }
        }
        Err(err) => log::warn!("Failed to get review image, continue anyway: {:?}", err),
      }
    }
  }
  // Write metadata
  dng.load_base_tags(&rawimage)?;
  dng.load_metadata(&metadata)?;

  // Write DNG-specific metadata tags
  // ColorimetricReference: 0 = scene-referred (standard for raw files)
  dng.colorimetric_reference(0);

  // CameraSerialNumber from EXIF
  if let Some(serial) = &metadata.exif.serial_number {
    dng.camera_serial_number(serial);
  }

  // Baseline tags from camera definition
  if let Some(be) = rawimage.camera.baseline_exposure {
    dng.baseline_exposure(SRational::new((be * 100.0) as i32, 100));
  }
  if let Some(bn) = rawimage.camera.baseline_noise {
    dng.baseline_noise(Rational::new((bn * 100.0) as u32, 100));
  }
  if let Some(bs) = rawimage.camera.baseline_sharpness {
    dng.baseline_sharpness(Rational::new((bs * 100.0) as u32, 100));
  }
  if let Some(lrl) = rawimage.camera.linear_response_limit {
    dng.linear_response_limit(Rational::new((lrl * 100.0) as u32, 100));
  }
  if let Some(ref np) = rawimage.camera.noise_profile {
    dng.noise_profile(np);
  }
  if let Some(aas) = rawimage.camera.anti_alias_strength {
    dng.anti_alias_strength(Rational::new((aas * 100.0) as u32, 100));
  }
  if let Some(bgs) = rawimage.camera.bayer_green_split {
    dng.bayer_green_split(bgs);
  }
  // CameraCalibration: dynamic cc_reference_wb takes precedence over static camera_calibration
  if let Some([ref_r, ref_b]) = rawimage.camera.cc_reference_wb {
    let wb = &rawimage.wb_coeffs;
    let wb_r = if wb[0].is_nan() || wb[0] <= 0.0 { 1.0 } else { wb[0] as f64 };
    let wb_b = if wb[2].is_nan() || wb[2] <= 0.0 { 1.0 } else { wb[2] as f64 };
    let cc_r = (ref_r / wb_r).clamp(0.8, 1.25);
    let cc_b = (ref_b / wb_b).clamp(0.8, 1.25);
    let matrix = [
      SRational::new((cc_r * 10000.0) as i32, 10000), SRational::new(0, 10000), SRational::new(0, 10000),
      SRational::new(0, 10000), SRational::new(10000, 10000), SRational::new(0, 10000),
      SRational::new(0, 10000), SRational::new(0, 10000), SRational::new((cc_b * 10000.0) as i32, 10000),
    ];
    dng.camera_calibration(1, &matrix);
    dng.camera_calibration(2, &matrix);
    log::debug!("Dynamic CameraCalibration: diag({:.4}, 1.0, {:.4}) from ref_wb({:.4}, {:.4}) / as_shot({:.4}, {:.4})", cc_r, cc_b, ref_r, ref_b, wb_r, wb_b);
  } else if let Some(cc) = rawimage.camera.camera_calibration {
    let matrix = [
      SRational::new((cc[0] * 10000.0) as i32, 10000), SRational::new(0, 10000), SRational::new(0, 10000),
      SRational::new(0, 10000), SRational::new((cc[1] * 10000.0) as i32, 10000), SRational::new(0, 10000),
      SRational::new(0, 10000), SRational::new(0, 10000), SRational::new((cc[2] * 10000.0) as i32, 10000),
    ];
    dng.camera_calibration(1, &matrix);
    dng.camera_calibration(2, &matrix);
  }
  if let Some(duc) = rawimage.camera.default_user_crop {
    dng.default_user_crop(&[
      Rational::new((duc[0] * 1_000_000.0) as u32, 1_000_000),
      Rational::new((duc[1] * 1_000_000.0) as u32, 1_000_000),
      Rational::new((duc[2] * 1_000_000.0) as u32, 1_000_000),
      Rational::new((duc[3] * 1_000_000.0) as u32, 1_000_000),
    ]);
  }

  // Compute missing EXIF APEX values from exposure parameters
  if metadata.exif.shutter_speed_value.is_none() {
    if let Some(et) = metadata.exif.exposure_time {
      let time_sec = et.n as f64 / et.d as f64;
      if time_sec > 0.0 {
        let apex = -time_sec.log2();
        dng.exif_ifd_mut().add_tag(ExifTag::ShutterSpeedValue, SRational::new((apex * 1_000_000.0).round() as i32, 1_000_000));
      }
    }
  }
  if metadata.exif.aperture_value.is_none() {
    if let Some(f) = metadata.exif.fnumber {
      let fnum = f.n as f64 / f.d as f64;
      if fnum > 0.0 {
        let apex = 2.0 * fnum.log2();
        dng.exif_ifd_mut().add_tag(ExifTag::ApertureValue, Rational::new((apex * 1_000_000.0).round() as u32, 1_000_000));
      }
    }
  }

  // Compute FocalLengthIn35mmFormat and FocalPlane resolution from crop_factor
  if let Some(cf) = rawimage.camera.crop_factor {
    if metadata.exif.focal_len_in_35mm_format.is_none() {
      if let Some(fl) = metadata.exif.focal_length {
        let mm = fl.n as f64 / fl.d as f64;
        let eq35 = (mm * cf).round() as u16;
        if eq35 > 0 {
          dng.exif_ifd_mut().add_tag(ExifTag::FocalLengthIn35mmFormat, eq35);
        }
      }
    }

    let (active_w, active_h) = if let Some(area) = rawimage.active_area {
      (area.d.w, area.d.h)
    } else {
      (rawimage.width, rawimage.height)
    };
    if active_w > 0 && active_h > 0 {
      let diag_35mm: f64 = 43.2666;
      let sensor_diag = diag_35mm / cf;
      let pixel_diag = ((active_w as f64).powi(2) + (active_h as f64).powi(2)).sqrt();
      let sensor_w_mm = sensor_diag * active_w as f64 / pixel_diag;
      let sensor_h_mm = sensor_diag * active_h as f64 / pixel_diag;
      // pixels per centimeter
      let fp_xres = active_w as f64 / sensor_w_mm * 10.0;
      let fp_yres = active_h as f64 / sensor_h_mm * 10.0;
      dng.exif_ifd_mut().add_tag(ExifTag::FocalPlaneXResolution2, Rational::new((fp_xres * 100.0).round() as u32, 100));
      dng.exif_ifd_mut().add_tag(ExifTag::FocalPlaneYResolution2, Rational::new((fp_yres * 100.0).round() as u32, 100));
      dng.exif_ifd_mut().add_tag(ExifTag::FocalPlaneResolutionUnit2, 3u16); // centimeters
    }
  }

  // Apply DCP color profile — from explicit override file, then --dcp-dir,
  // then auto-discovered from system paths.
  let style_hint = decoder.picture_style_hint();
  let auto_dcp_allowed = decoder.auto_dcp_profile_allowed();
  let unique_model = format!("{} {}", rawimage.clean_make, rawimage.clean_model);
  let dcp_path = if let Some(explicit) = &params.dcp_file {
    if explicit.exists() {
      Some(explicit.clone())
    } else if !auto_dcp_allowed {
      log::warn!("DCP override file not found: {} — automatic DCP matching disabled for this custom camera look", explicit.display());
      None
    } else {
      log::warn!("DCP override file not found: {} — falling back to auto-match", explicit.display());
      params.dcp_dir.as_ref().and_then(|d| find_dcp(d, &unique_model, style_hint.as_deref()))
        .or_else(|| auto_find_dcp(&unique_model, style_hint.as_deref()))
    }
  } else if !auto_dcp_allowed {
    log::debug!("Automatic DCP matching disabled for custom camera look: {}", unique_model);
    None
  } else if let Some(dcp_dir) = &params.dcp_dir {
    find_dcp(dcp_dir, &unique_model, style_hint.as_deref())
  } else {
    auto_find_dcp(&unique_model, style_hint.as_deref())
  };
  if let Some(dcp_path) = &dcp_path {
    match DcpProfile::load(dcp_path) {
      Ok(profile) => {
        // Bake BaselineExposureOffset from the DCP into the BaselineExposure tag.
        // Adobe combines the camera's static BE + the DCP offset into a single
        // BaselineExposure value — no separate BaselineExposureOffset tag is written.
        if let Some(beo) = profile.baseline_exposure_offset() {
          let base_be = rawimage.camera.baseline_exposure.unwrap_or(0.0) as f64;
          let combined = base_be + beo;
          dng.baseline_exposure(SRational::new((combined * 100.0) as i32, 100));
          log::debug!("BaselineExposure adjusted: {:.2} + DCP offset {:.2} = {:.2}", base_be, beo, combined);
        }
        dng.root_ifd_mut().copy(profile.copy_tags_iter());
        log::debug!("Applied DCP profile from: {}", dcp_path.display());
      }
      Err(err) => log::warn!("Failed to load DCP profile '{}': {:?}", dcp_path.display(), err),
    }
  }
  if !dng.root_ifd().contains(ExifTag::Orientation) {
    dng.root_ifd_mut().add_tag(ExifTag::Orientation, rawimage.orientation.to_u16());
  }

  // Check for DNG root IFD related tags
  if let Some(dng_root_ifd) = decoder.ifd(WellKnownIFD::VirtualDngRootTags)? {
    dng.root_ifd_mut().copy(dng_root_ifd.value_iter());
  }

  // Check for TIFF root IFD related tags
  if let Some(tiff_root) = decoder.ifd(WellKnownIFD::Root)? {
    dng.root_ifd_mut().copy(tiff_root.value_iter().filter(|(tag, _)| {
      [
        // Tags from CinemaDNG files
        TiffCommonTag::TimeCodes as u16,
        TiffCommonTag::FrameFrate as u16,
        TiffCommonTag::TStop as u16,
      ]
      .contains(tag)
    }));
  }

  // Remove MakerNotes unless the decoder explicitly marks them safe (MakerNoteSafety == 1).
  // Default is to strip them; relative offsets would be invalid in the re-written DNG.
  let makernote_is_safe = decoder
    .ifd(WellKnownIFD::VirtualDngRootTags)?
    .and_then(|ifd| ifd.get_entry(DngTag::MakerNoteSafety).cloned())
    .and_then(|entry| {
      if let crate::formats::tiff::Value::Short(v) = entry.value {
        v.into_iter().next()
      } else {
        None
      }
    })
    .unwrap_or(0)
    == 1;
  if !makernote_is_safe {
    dng.exif_ifd_mut().remove_tag(ExifTag::MakerNotes);
  }

  {
    let mut xpacket = decoder.xpacket(rawfile, &raw_params)?;
    // If we injected a DCP profile, update crd:CameraProfile in the XMP to
    // match the profile name so that ACR/Lightroom picks up the right profile.
    if let Some(dcp_path) = &dcp_path {
      if let Ok(profile) = DcpProfile::load(dcp_path) {
        if let Some(name) = profile.profile_name() {
          if let Some(pkt) = xpacket.take() {
            xpacket = Some(patch_xmp_camera_profile(pkt, &name));
          }
        }
      }
    }
    // Inject the normalized camera recipe (Nikon Picture Control, …) as
    // lb:recipe JSON so the editor can seed slider defaults on open. Synthesizes
    // a minimal XMP packet when the source carries none. See docs/camera-recipes.md.
    //
    // colorCurves is deliberately NOT fit here anymore. It comes exclusively
    // from the server-side DB seed (rust-renderer/src/jobs.rs::run_look_fit),
    // which fits against the renderer's real neutral (post DCP profile +
    // BaselineExposure). Fitting it here used a bare RawDevelop neutral,
    // missing both of those, so it produced a worse curve that raced the DB
    // seed and made the editor's look flip once the correct job landed. See
    // docs/learned-look-fit.md.
    if let Some(recipe) = decoder.recipe(rawfile, &raw_params)? {
      xpacket = inject_recipe_xmp(xpacket, &recipe);
    }
    if let Some(pkt) = xpacket {
      dng.xpacket(&pkt)?;
    }
  }

  if let Some(handle) = original_compress_thread {
    let original = handle
      .join()
      .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to join compression thread: {:?}", err)))??;
    dng.original_file(&original, original_filename)?;
  }

  if let Some(artist) = &params.artist {
    dng.root_ifd_mut().add_tag(TiffCommonTag::Artist, artist);
  }
  dng.root_ifd_mut().add_tag(TiffCommonTag::Software, &params.software);

  dng
    .root_ifd_mut()
    .add_tag(ExifTag::ModifyDate, chrono::Local::now().format("%Y:%m:%d %H:%M:%S").to_string());

  dng.close()?;

  Ok(())
}

/// Fit per-channel display-space R/G/B curves mapping a *neutral* develop to the
/// camera's own embedded JPEG preview. With the editor's look sliders at default,
/// the pristine-open render reduces to `srgb_encode(neutral linear)` followed by
/// these per-channel RGB curves (the shader's row-1 lane), so seeding them makes
/// the editor open matching the in-body JPEG — capturing tone + colour + WB +
/// Active D-Lighting's global component in one measured, portable primitive.
///
/// Both inputs are display-referred sRGB. Returns `None` when the inputs are
/// unusable or the fit is degenerate. See `docs/camera-recipes.md` and the ADL
/// design note. Method (validated offline, MAE ~2-4 vs preview): downsample both
/// to a common grid, pair pixels, quantile-bin by the neutral value, take the
/// mean preview value per bin, enforce monotonicity, anchor the endpoints toward
/// identity so out-of-sample highlights don't clip.
pub fn fit_color_curves(neutral: &DynamicImage, preview: &DynamicImage) -> Option<[crate::recipe::ToneCurve; 3]> {
  use crate::recipe::{CurveType, ToneCurve};
  // Common low-res grid: cheap, robust, and statistically ample (~44k samples).
  const FW: u32 = 256;
  const FH: u32 = 171;
  const NB: usize = 64; // quantile bins per channel (was 24; finer = closer fit)
  // Display-sRGB threshold (~5/255) below which a neutral tone is "crushed":
  // the develop has clipped that region to ~0 and lost all tonal gradation, so
  // its per-pixel mapping to the preview is no longer reliable. Bins whose mean
  // neutral value falls under this are remapped to the preview's shadow floor —
  // see the crushed-black guard in `fit_channel`.
  const BLACK_EPS: f32 = 0.02;
  let n = neutral.resize_exact(FW, FH, image::imageops::FilterType::Triangle).to_rgb8();
  let p = preview.resize_exact(FW, FH, image::imageops::FilterType::Triangle).to_rgb8();
  let np = n.as_raw();
  let pp = p.as_raw();
  if np.len() != pp.len() || np.len() < (NB * 3 * 8) {
    return None;
  }
  let px = (FW * FH) as usize;

  let fit_channel = |c: usize| -> ToneCurve {
    // Collect (base, prev) pairs in [0,1] for this channel.
    let mut pairs: Vec<(f32, f32)> = Vec::with_capacity(px);
    for i in 0..px {
      let b = np[i * 3 + c] as f32 / 255.0;
      let q = pp[i * 3 + c] as f32 / 255.0;
      pairs.push((b, q));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Quantile bins: equal sample counts, so dark frames still get resolution.
    let mut ctrl: Vec<(f32, f32)> = Vec::with_capacity(NB + 2);
    let per = pairs.len() / NB;
    if per == 0 {
      return ToneCurve { curve_type: CurveType::Empty, points: vec![] };
    }
    // Preview SHADOW FLOOR for this channel: the darkest tone the camera
    // actually renders, estimated as a robust low percentile (2nd) of the WHOLE
    // preview channel. This is a *global* statistic — deliberately independent of
    // which preview pixels happen to fall in a given neutral bin — so it does not
    // track the per-bin scatter that inflates a crushed bin's mean/p10 (see the
    // crushed-black guard below). For well-exposed frames it is ~0; for frames
    // with a genuine lifted toe it is the true floor (e.g. ~0.05-0.10).
    let floor_y = {
      let mut qs: Vec<f32> = pairs.iter().map(|pr| pr.1).collect();
      qs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
      let idx = ((qs.len() as f32 * 0.02) as usize).min(qs.len().saturating_sub(1));
      qs[idx]
    };
    let mut last_y = 0.0f32;
    for bin in 0..NB {
      let lo = bin * per;
      let hi = if bin == NB - 1 { pairs.len() } else { (bin + 1) * per };
      let cnt = (hi - lo) as f32;
      let mut sx = 0.0f32;
      let mut sy = 0.0f32;
      for pr in &pairs[lo..hi] {
        sx += pr.0;
        sy += pr.1;
      }
      let cx = sx / cnt;
      let mut cy = sy / cnt;
      // Crushed-black guard. When the neutral tone is at/near black
      // (`cx < BLACK_EPS`) the develop has clipped that region to ~0 and lost all
      // tonal gradation, so this bin's pixels pair with a *scattered* slice of the
      // preview — true shadows AND lifted/noisy mid-shadows that also crushed to
      // 0. Their MEAN (and even a per-bin low percentile) is dragged up by that
      // bright tail and, once the cummax below propagates it forward and the
      // dark-end extrapolation carries it to x=0, pins the black point to a lifted
      // value: the ~0.22 black-lift washout the editor then over-applies. The
      // per-bin spread is exactly the scatter, so a per-bin statistic cannot
      // recover the floor; map crushed black to the *global* preview shadow floor
      // (`floor_y`) instead. That is scatter-independent, so it lands the toe at
      // the camera's true darkest tone regardless of how badly the bin scattered.
      // No-op for well-exposed frames (their darkest bin sits above BLACK_EPS) and
      // for genuinely-black crushed frames (floor_y ≈ bin mean ≈ 0). Complementary
      // to the `id_err > 0.30` regression guard below: that one *drops* grossly
      // mismatched neutral/preview pairs (e.g. a vignette-black develop vs a
      // corrected JPEG), while this one keeps faithful-but-crushed fits from
      // over-lifting. It can only lower the toe, never raise it, so it never
      // introduces a washout.
      if cx < BLACK_EPS {
        cy = floor_y;
      }
      // Monotone (cummax) so the curve never inverts.
      if cy < last_y {
        cy = last_y;
      }
      last_y = cy;
      // Skip near-duplicate x (keeps spline well-conditioned).
      if let Some(&(px_, _)) = ctrl.last() {
        if cx - px_ < 1.0 / 512.0 {
          if let Some(lastp) = ctrl.last_mut() {
            lastp.1 = cy;
          }
          continue;
        }
      }
      ctrl.push((cx, cy));
    }
    if ctrl.len() < 2 {
      return ToneCurve { curve_type: CurveType::Empty, points: vec![] };
    }
    // The control points so far are at QUANTILE x positions (dense in the
    // populated tonal region). The editor interpolates the stored knots with a
    // NATURAL CUBIC SPLINE, which overshoots/rings between closely-spaced x
    // knots — so we must hand it UNIFORMLY-spaced knots instead. Resample the
    // monotone quantile fit (piecewise-linear between the quantile knots, which
    // is ring-free and what we validated against) onto NK uniform x in [0,1].
    // Below the first / above the last observed tone we extrapolate toward
    // identity so out-of-sample shadows/highlights stay sane on other scenes.
    const NK: usize = 32; // uniform knots (was 16); validated through the editor's
                          // natural-cubic spline to gain ~0.15-0.4 MAE with no ringing
                          // (uniform spacing + monotonicity), see ml/look-fit.
    let (xf, yf) = ctrl[0];
    let (xl, yl) = *ctrl.last().unwrap();
    let interp = |x: f32| -> f32 {
      if x <= xf {
        // toward (0,0)-relative identity: keep the measured offset at xf.
        return (yf - (xf - x)).clamp(0.0, 1.0);
      }
      if x >= xl {
        // identity slope (+1) above the brightest measured tone.
        return (yl + (x - xl)).clamp(0.0, 1.0);
      }
      // piecewise-linear lookup within the monotone quantile knots.
      let mut j = 0;
      while j + 1 < ctrl.len() && ctrl[j + 1].0 < x {
        j += 1;
      }
      let (xa, ya) = ctrl[j];
      let (xb, yb) = ctrl[j + 1];
      let t = if xb > xa { (x - xa) / (xb - xa) } else { 0.0 };
      (ya + t * (yb - ya)).clamp(0.0, 1.0)
    };
    let mut points = Vec::with_capacity(NK * 2);
    let mut last_y = 0.0f32;
    for k in 0..NK {
      let x = k as f32 / (NK - 1) as f32;
      let mut y = interp(x);
      if y < last_y {
        y = last_y; // keep monotone after resample
      }
      last_y = y;
      points.push(x);
      points.push(y);
    }
    ToneCurve { curve_type: CurveType::Spline, points }
  };

  let curves = [fit_channel(0), fit_channel(1), fit_channel(2)];
  if curves.iter().all(|t| t.is_empty()) {
    return None;
  }

  // Regression guard: keep the curves only if they actually improve the match.
  // When the neutral already matches the JPEG (flat picture control) or the
  // difference is spatially-varying/local (not a per-channel function), the
  // fitted curve is near-identity and can slightly HURT — open at neutral
  // instead. Evaluate the uniform-knot curve by linear interpolation on the grid
  // (the editor splines it; linear is close enough for the keep/drop decision).
  let eval = |pts: &[f32], x: f32| -> f32 {
    let nk = pts.len() / 2;
    if nk < 2 {
      return x;
    }
    let t = x.clamp(0.0, 1.0) * (nk - 1) as f32;
    let i = (t.floor() as usize).min(nk - 2);
    let f = t - i as f32;
    pts[2 * i + 1] * (1.0 - f) + pts[2 * (i + 1) + 1] * f
  };
  let (mut fit_err, mut id_err) = (0.0f64, 0.0f64);
  for i in 0..px {
    for c in 0..3 {
      let b = np[i * 3 + c] as f32 / 255.0;
      let q = pp[i * 3 + c] as f32 / 255.0;
      let f = if curves[c].is_empty() { b } else { eval(&curves[c].points, b) };
      fit_err += (f - q).abs() as f64;
      id_err += (b - q).abs() as f64;
    }
  }
  // Sanity gates (mean absolute error in [0,1], display domain):
  //  - fit_err >= id_err     : the curve can't beat identity (no global look).
  //  - id_err  > 0.30        : neutral and preview are grossly mismatched — the
  //                            fit's inputs don't correspond (e.g. a broken/dark
  //                            neutral render, or wrong orientation/crop). A real
  //                            in-body tone gap is < ~0.15; 0.30 only triggers on
  //                            mismatched inputs. Fitting these yields garbage
  //                            curves (e.g. black -> 0.83) that wash the image out.
  //  - fit_err > 0.22        : the fitted match is still poor — don't seed it.
  // Any of these -> return None so the editor opens at neutral instead.
  let n = (px * 3) as f64;
  if fit_err >= id_err || id_err / n > 0.30 || fit_err / n > 0.22 {
    return None;
  }
  Some(curves)
}

/// Cross-channel look-residual matrix for the Panasonic Lumix DMC-GM5
/// (neutral-linear -> preview-linear), validated by leave-one-out
/// cross-validation across 38 real GM5 frames — see
/// `ml/look-fit/analyze_panasonic.py` and its saved output
/// `ml/look-fit/out/panasonic_analysis.json` (`pooled_matrix`).
///
/// `fit_color_curves` alone (a per-channel display tone/color curve) only
/// captures about half of this camera's in-body Photo Style look. A
/// per-frame cross-channel matrix fit on top of the colorCurves residual
/// scatters across frames (scene-content overfit — NOT usable), but the
/// single matrix POOLED across all 38 frames generalizes to held-out
/// frames: mean held-out look MAE improved from 5.44 (colorCurves only) to
/// 4.46 (this matrix applied before colorCurves), 37/38 frames improved.
/// That stability is what makes it safe to ship as a fixed constant rather
/// than a per-photo fit.
const PANASONIC_GM5_LOOK_MATRIX: [[f32; 3]; 3] = [
  [1.164_212_6, -0.172_941_78, -0.126_606_12],
  [0.066_754_55, 0.869_146_57, -0.041_138_878],
  [-0.098_209_05, -0.160_346_99, 1.171_807_7],
];

/// # STATUS: wired into production — as a POST-GAMMA GPU shader stage, NOT
/// a decode-time colour-pipeline matrix
///
/// This matrix and its gate are real, evidence-backed work (see
/// [`PANASONIC_GM5_LOOK_MATRIX`]'s doc comment and the
/// `gm5_pooled_matrix_beats_colorcurves_alone_on_real_frames` corpus test
/// below). A first attempt wired it into `rust-renderer/src/jobs.rs::run_look_fit`
/// (fitting `colorCurves` against a matrix-corrected neutral, via
/// [`apply_look_matrix_srgb`]), but that was reverted: the fitted
/// `colorCurves` gets persisted to `photos.edit_values` and is later
/// *applied at render time* to the plain, UNCORRECTED neutral — so the
/// curve would have been fit against `curve(matrix(neutral))` but applied
/// as `curve(raw_neutral)`, an unvalidated mismatch.
///
/// **The domain fact that earlier guidance here got wrong:** this matrix
/// was derived and cross-validated (`ml/look-fit/analyze_panasonic.py`'s
/// `look_after()`) against `srgb_decode(neutral) @ M.T`, where `neutral` is
/// the FULLY RENDERED, `EditValues=0`, DISPLAY-referred sRGB neutral image —
/// i.e. it is a correction on display-referred pixels, taken *after* the
/// entire scene-linear decode/WB/exposure/profile-tone-curve/highlight-
/// rolloff/HPMINDE/gamma-encode chain. It does **not** belong at the same
/// architectural slot as a DCP `ColorMatrix` (which operates on
/// scene-referred linear camera data, very early in the pipeline, before
/// white balance — see `docs/camera-profiles.md`), and it is not composed
/// into any shared decode-time colour pipeline (native `rust-renderer` or
/// the browser's `wasm-utif`/`glfx-es6` decoder).
///
/// **Where it actually lives:** `rust-renderer/shaders/mega_shader.wgsl`,
/// as its own stage ("6d.") immediately after "6c. Gamma encode" (the
/// scene-linear → display-sRGB transition) and before "7. Contrast" — i.e.
/// under every user edit lane, since this is camera-inherent colour
/// science, not a user adjustment. It decodes the gamma-encoded colour back
/// to linear, applies the matrix (mirroring [`apply_look_matrix_srgb`]'s
/// decode → matmul → clamp-negatives → re-encode exactly), and is gated
/// per-photo by a `MegaUniforms` flag (`rust-renderer/src/pipeline.rs`)
/// rather than per-pixel branching — a pure passthrough (no matrix multiply
/// performed) for every camera the gate below rejects. The flag + matrix
/// are set by `rust-renderer/src/render.rs::render_decoded_image`, which
/// calls THIS function with the camera identity threaded onto
/// `decode::DecodedImage::clean_make`/`clean_model` by the decode paths
/// (`decode_direct`'s DNG branches, `decode_via_dng` — via its temp-DNG
/// round trip, and `decode_raw_neutral_native`).
///
/// Because `render_photo_by_id` (server export/download/thumbnail/share
/// links), the Tauri desktop app's local export/share command
/// (`frontend/src-tauri/src/render_cmds.rs::raw_render_with_edits`, which
/// calls the identical compiled `decode_raw_full` → `render_decoded_image`),
/// AND `render_photo_neutral_native` (the look_fit job's fit-time neutral)
/// all funnel into the same `render_decoded_image` → GPU pipeline, this
/// wiring makes the fit-time neutral and every real render agree
/// automatically — no GM5-specific code needed in
/// `rust-renderer/src/jobs.rs::run_look_fit`, which still calls
/// `fit_color_curves` directly on the plain neutral for every camera, GM5
/// included, unchanged. Regression-pinned by
/// `rust-renderer/tests/look_matrix_tests.rs` (real-GPU, real GM5 + Nikon
/// DNG fixtures under `ml/look-fit/data/`, `#[ignore]`d — needs local
/// fixtures, same convention as `decode.rs::lookfit_native_matches_via_dng`).
///
/// [`apply_look_matrix_srgb`] itself is still not called by any production
/// Rust path — the WGSL stage above reimplements its exact math so the
/// correction can run per-pixel on the GPU — but it remains real, tested
/// (CPU reference + parity target) and is what the WGSL stage's doc
/// comment points back to for the algorithm.
///
/// **Live-preview parity gap (both web AND the Tauri app's interactive
/// editor — they share the same JS):** NOT implemented. The Tauri shell
/// wraps the same React/JS frontend as the web deployment for its
/// interactive editing session; only its local export command drops down
/// to native Rust. So while every *finished render* (server or Tauri
/// export) now carries this correction, dragging sliders live in the
/// editor — in a browser OR inside the Tauri app — still renders through
/// `frontend/src/lib/glfx-es6/filters/adjust/megaShader.js`, which does not.
/// Investigated and deliberately scoped out rather than forced: that GLSL
/// shader's stage structure maps cleanly (it already has `srgb_decode`/
/// `srgb_encode` helpers and the identical `color.rgb = srgb_encode(lin)`
/// → contrast boundary at line ~664), but no source of the *normalized*
/// `clean_make`/`clean_model` identity this gate requires reaches the
/// browser today — `GET /api/photos`' `camera` field
/// (`backend/utils/exif_utils.py::_build_camera_name`) is a free-text
/// `"{raw EXIF Make} {raw EXIF Model}"` string built from a completely
/// separate Python/exiftool EXIF read, not rawler's camera-database
/// `clean_make`/`clean_model` (which can alias/normalize away from the raw
/// tags — see `decoders/camera.rs`), and `frontend/wasm-utif` (the
/// browser's own decoder) has no camera-identity matching at all, only raw
/// FM/CM/AsShotNeutral tag reads. Parsing `camera` client-side to guess
/// make/model would be exactly the "second, possibly-inconsistent
/// camera-identity comparison" this gate is designed to avoid. Making it
/// safe needs new plumbing — e.g. a backend field carrying rawler's actual
/// `clean_make`/`clean_model` through ingestion into the API response — not
/// a client-side string split; see `docs/learned-look-fit.md`.
///
/// Returns the validated [`PANASONIC_GM5_LOOK_MATRIX`] when `clean_make` /
/// `clean_model` identify the EXACT camera body it was derived and
/// cross-validated for — Panasonic Lumix DMC-GM5 — or `None` for every
/// other camera, including every other Panasonic body.
///
/// `clean_make` / `clean_model` must be the same normalized strings this
/// codebase already uses for exact camera-model matching (`RawImage`'s /
/// `RawMetadata`'s `clean_make/clean_model` — see the `unique_model` built
/// from them just above in [`internal_convert`] and consumed by
/// [`crate::dcp::find_dcp`]). Comparison here is a plain string match on
/// those already-normalized values, same as e.g. `raf.rs`'s
/// `self.camera.clean_model == "DBP for GX680"` — deliberately NOT
/// broadened to a prefix/substring/case-insensitive match on the raw
/// vendor Make/Model, so an unrelated Panasonic body can never silently
/// pick this up.
///
/// This gate is intentionally narrow. Only the GM5 has been validated this
/// way (see [`PANASONIC_GM5_LOOK_MATRIX`]'s doc comment); applying this
/// matrix to any other camera — even another Panasonic RW2 body — would be
/// an unvalidated correction with no evidence it helps, which is exactly
/// the failure mode this feature exists to avoid repeating. Adding another
/// body means validating its own matrix and adding its own guarded arm
/// here, not loosening this one.
pub fn panasonic_gm5_look_matrix(clean_make: &str, clean_model: &str) -> Option<[[f32; 3]; 3]> {
  if clean_make == "Panasonic" && clean_model == "DMC-GM5" {
    Some(PANASONIC_GM5_LOOK_MATRIX)
  } else {
    None
  }
}

/// Apply a 3x3 matrix to a display-referred sRGB image in LINEAR light:
/// decode sRGB -> linear (`srgb_invert_gamma`), matrix-multiply each pixel
/// (`out[row] = sum_col matrix[row][col] * lin[col]`), clamp negative
/// results (an off-diagonal-heavy matrix like [`PANASONIC_GM5_LOOK_MATRIX`]
/// can produce them), then re-encode linear -> sRGB (`srgb_apply_gamma`).
///
/// This mirrors `ml/look-fit/analyze_panasonic.py`'s `look_after()`
/// methodology (`PL.srgb_decode(neutral) @ M.T` in linear, then re-encode)
/// using rawler's own sRGB gamma helpers rather than reimplementing the
/// gamma math. This exact algorithm — decode → matmul → clamp-negatives →
/// re-encode — is what production actually runs, but as its own GPU shader
/// stage (`rust-renderer/shaders/mega_shader.wgsl`, right after gamma
/// encode, before any user edit lane) rather than by calling this Rust
/// function directly; see [`panasonic_gm5_look_matrix`]'s STATUS doc for
/// where the correction is wired in and why. This CPU implementation is
/// kept as the validated reference / test fixture (see the corpus test
/// below), not because a production caller invokes it at runtime.
pub fn apply_look_matrix_srgb(img: &DynamicImage, matrix: &[[f32; 3]; 3]) -> DynamicImage {
  let mut rgb = img.to_rgb8();
  for px in rgb.pixels_mut() {
    let lin = [
      crate::imgop::srgb::srgb_invert_gamma(px[0] as f32 / 255.0),
      crate::imgop::srgb::srgb_invert_gamma(px[1] as f32 / 255.0),
      crate::imgop::srgb::srgb_invert_gamma(px[2] as f32 / 255.0),
    ];
    for (c, row) in matrix.iter().enumerate() {
      let v = row[0] * lin[0] + row[1] * lin[1] + row[2] * lin[2];
      let enc = crate::imgop::srgb::srgb_apply_gamma(v.max(0.0));
      px[c] = (enc.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
  }
  DynamicImage::ImageRgb8(rgb)
}

/// Build the `<rdf:Description>` XMP fragment carrying the normalized camera
/// recipe as `lb:recipe` JSON. See `docs/camera-recipes.md`.
fn recipe_xmp_fragment(recipe: &crate::recipe::Recipe) -> Option<String> {
  let json = serde_json::to_string(recipe).ok()?;
  let esc = json.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
  Some(format!(
    "<rdf:Description rdf:about=\"\" xmlns:lb=\"https://lightbox.photo/ns/recipe/1.0/\"><lb:recipe>{}</lb:recipe></rdf:Description>",
    esc
  ))
}

/// Inject the recipe fragment into an existing XMP packet (before the closing
/// `</rdf:RDF>`), or synthesize a minimal packet when `xpacket` is `None`.
/// Returns `None` only if the recipe cannot be serialized.
fn inject_recipe_xmp(xpacket: Option<Vec<u8>>, recipe: &crate::recipe::Recipe) -> Option<Vec<u8>> {
  let fragment = recipe_xmp_fragment(recipe)?;
  match xpacket {
    Some(bytes) => {
      if let Some(s) = std::str::from_utf8(&bytes).ok() {
        if let Some(pos) = s.rfind("</rdf:RDF>") {
          let mut out = String::with_capacity(s.len() + fragment.len());
          out.push_str(&s[..pos]);
          out.push_str(&fragment);
          out.push_str(&s[pos..]);
          return Some(out.into_bytes());
        }
      }
      // Unrecognized packet shape: keep the original (recipe not carried).
      Some(bytes)
    }
    None => {
      let packet = format!(
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">{}</rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>",
        fragment
      );
      Some(packet.into_bytes())
    }
  }
}


/// Replace the value of `crd:CameraProfile` in an XMP byte packet.
///
/// The XMP spec allows both attribute form (`crd:CameraProfile="…"`) and
/// element form (`<crd:CameraProfile>…</crd:CameraProfile>`).  Nikon cameras
/// use the element form, but we handle both for robustness.  The replacement
/// is a plain byte-level find-and-replace on UTF-8; no XML parser needed
/// because `crd:CameraProfile` is always a simple scalar string in practice.
fn patch_xmp_camera_profile(mut xmp: Vec<u8>, profile_name: &str) -> Vec<u8> {
  let src = std::str::from_utf8(&xmp).ok().and_then(|s| {
    // Element form: <crd:CameraProfile>VALUE</crd:CameraProfile>
    let open = "<crd:CameraProfile>";
    let close = "</crd:CameraProfile>";
    if let (Some(start), Some(end)) = (s.find(open), s.find(close)) {
      let value_start = start + open.len();
      if value_start <= end {
        return Some((value_start, end, s[value_start..end].to_owned()));
      }
    }
    // Attribute form: crd:CameraProfile="VALUE"
    let attr = "crd:CameraProfile=\"";
    if let Some(pos) = s.find(attr) {
      let value_start = pos + attr.len();
      if let Some(end_quote) = s[value_start..].find('"') {
        return Some((value_start, value_start + end_quote, s[value_start..value_start + end_quote].to_owned()));
      }
    }
    None
  });

  if let Some((start, end, old_value)) = src {
    let new_bytes = profile_name.as_bytes();
    let old_bytes = old_value.as_bytes();
    // Splice: bytes[..start] + new_value + bytes[end..]
    let mut result = Vec::with_capacity(xmp.len() - old_bytes.len() + new_bytes.len());
    result.extend_from_slice(&xmp[..start]);
    result.extend_from_slice(new_bytes);
    result.extend_from_slice(&xmp[end..]);
    log::debug!("XMP crd:CameraProfile: '{}' → '{}'", old_value, profile_name);
    xmp = result;
  }
  xmp
}

fn generate_preview(rawfile: &RawSource, decoder: &dyn Decoder, rawimage: &RawImage, params: &RawDecodeParams) -> crate::Result<DynamicImage> {
  match decoder.preview_image(rawfile, params)? {
    Some(image) => Ok(image),
    None => {
      log::warn!("Preview image not found, try to generate sRGB from RAW");
      let dev = RawDevelop::default();
      let image = dev.develop_intermediate(rawimage)?;
      /*
      let params = rawimage.develop_params()?;
      let (srgbf, dim) = develop_raw_srgb(&rawimage.data, &params)?;
      let output = convert_from_f32_scaled_u16(&srgbf, 0, u16::MAX);
      let image = if srgbf.len() == dim.w * dim.h {
        DynamicImage::ImageLuma16(ImageBuffer::from_raw(dim.w as u32, dim.h as u32, output).expect("Invalid ImageBuffer size"))
      } else {
        DynamicImage::ImageRgb16(ImageBuffer::from_raw(dim.w as u32, dim.h as u32, output).expect("Invalid ImageBuffer size"))
      };
       */
      Ok(image.to_dynamic_image().unwrap())
    }
  }
}

#[cfg(test)]
mod fit_color_curves_tests {
  use super::fit_color_curves;
  use image::{DynamicImage, RgbImage};

  // Build a (neutral, preview) pair from a per-row generator. `genf(t)` receives the
  // normalized row position in [0,1] and returns ((nr,ng,nb),(pr,pg,pb)) in [0,1].
  // Sized at the fit's own 256x171 grid so resize_exact is ~identity.
  fn pair(genf: impl Fn(f32) -> ((f32, f32, f32), (f32, f32, f32))) -> (DynamicImage, DynamicImage) {
    const W: u32 = 256;
    const H: u32 = 171;
    let mut neu = RgbImage::new(W, H);
    let mut prev = RgbImage::new(W, H);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    for y in 0..H {
      let t = y as f32 / (H - 1) as f32;
      let ((nr, ng, nb), (pr, pg, pb)) = genf(t);
      for x in 0..W {
        neu.put_pixel(x, y, image::Rgb([q(nr), q(ng), q(nb)]));
        prev.put_pixel(x, y, image::Rgb([q(pr), q(pg), q(pb)]));
      }
    }
    (DynamicImage::ImageRgb8(neu), DynamicImage::ImageRgb8(prev))
  }

  fn y0(points: &[f32]) -> f32 {
    // The curve is stored as uniform knots starting at x=0, so points[1] is y(0).
    points.get(1).copied().unwrap_or(0.0)
  }

  /// Regression pin for the washed-out look-fit seed. A "crushed-black" neutral
  /// (a deep-shadow region clipped to ~0 by the develop) paired with a preview
  /// whose matching pixels were lifted/scattered by the in-body tone curve makes
  /// the darkest quantile bin's MEAN preview value inflate; the dark-end
  /// extrapolation then carries that inflation to x=0 and the editor over-applies
  /// it as a ~0.22 black lift. Before the global-floor crushed-black guard this
  /// curve was KEPT by the regression guard (it beats identity) with y(0) ~ 0.26
  /// -> washed out. The guard must now floor the toe to the preview's true shadow
  /// floor (~0.05) while STILL keeping the curve.
  #[test]
  fn floors_crushed_black_kept_washout() {
    let (neutral, preview) = pair(|t| {
      if t < 0.40 {
        // crushed mid-shadow: neutral=0, preview lifted/scattered ~0.15..0.35
        ((0.0, 0.0, 0.0), (0.15 + 0.20 * t, 0.15 + 0.20 * t, 0.15 + 0.20 * t))
      } else if t < 0.50 {
        // genuine deep shadow: neutral=0, preview ~0.05..0.08 -> sets the floor
        ((0.0, 0.0, 0.0), (0.05 + 0.03 * t, 0.05 + 0.03 * t, 0.05 + 0.03 * t))
      } else {
        // matched midtone/highlight ramp: neutral == preview (well fit)
        let v = 0.10 + 0.90 * ((t - 0.50) / 0.50);
        ((v, v, v), (v, v, v))
      }
    });
    let curves = fit_color_curves(&neutral, &preview).expect("crushed-black washout must still be KEPT, not dropped");
    for (c, name) in ["R", "G", "B"].iter().enumerate() {
      let lift = y0(&curves[c].points);
      assert!(
        lift <= 0.10,
        "channel {name}: crushed-black toe must be floored, got y(0)={lift:.3} (washout)"
      );
    }
  }

  /// No-op pin: a NON-crushed dark region with a genuine in-body shadow lift
  /// (the darkest tone is well above BLACK_EPS, so it carries a reliable mapping)
  /// must be preserved, not flattened. This is the "don't regress already-correct
  /// photos / faithful warm-shadow looks" guarantee — the guard only touches bins
  /// crushed below BLACK_EPS.
  #[test]
  fn preserves_genuine_noncrushed_toe() {
    let (neutral, preview) = pair(|t| {
      if t < 0.30 {
        // dark but NOT crushed (~0.05..0.07), genuinely lifted in preview (~0.14)
        ((0.05 + 0.02 * t, 0.05 + 0.02 * t, 0.05 + 0.02 * t), (0.14, 0.14, 0.14))
      } else {
        let v = 0.15 + 0.85 * ((t - 0.30) / 0.70);
        ((v, v, v), (v, v, v))
      }
    });
    let curves = fit_color_curves(&neutral, &preview).expect("a genuine global look must be kept");
    for (c, name) in ["R", "G", "B"].iter().enumerate() {
      let lift = y0(&curves[c].points);
      assert!(
        lift > 0.04,
        "channel {name}: genuine non-crushed toe must be preserved, got y(0)={lift:.3} (over-flattened)"
      );
    }
  }

  // -------------------------------------------------------------------
  // Panasonic DMC-GM5 look-residual matrix: gate + effect.
  // -------------------------------------------------------------------
  use super::{apply_look_matrix_srgb, panasonic_gm5_look_matrix};

  /// THE regression guard: `panasonic_gm5_look_matrix` must return `None`
  /// for everything except the exact validated body, including cameras
  /// that are extremely close by name (a sibling Panasonic GM-series body,
  /// a case-folded GM5 string) and the Nikon bodies actually in this
  /// fleet's test/dev set (`Z f` / `Z 7 2` — see
  /// `dnglab/rawler/data/cameras/nikon/z_f.toml` /
  /// `z7_mk2.toml` clean_make/clean_model). `fit_color_curves` itself takes
  /// no camera identity — it cannot be affected by this gate at all, so
  /// the only place a regression could creep in is this match, which is
  /// exactly what this test pins.
  #[test]
  fn gm5_matrix_gate_is_exact_and_narrow() {
    // The one body that must match.
    assert!(panasonic_gm5_look_matrix("Panasonic", "DMC-GM5").is_some());

    // Nikon bodies from this fleet's own camera database — completely
    // unrelated make, must never match.
    assert!(panasonic_gm5_look_matrix("Nikon", "Z f").is_none());
    assert!(panasonic_gm5_look_matrix("Nikon", "Z 7 2").is_none());

    // A sibling Panasonic body one digit off — must NOT match. This is the
    // exact "generalize to any Panasonic/any RW2" mistake this gate exists
    // to prevent.
    assert!(panasonic_gm5_look_matrix("Panasonic", "DMC-GM1").is_none());
    assert!(panasonic_gm5_look_matrix("Panasonic", "DC-G9").is_none());

    // Right model, wrong (or missing) make.
    assert!(panasonic_gm5_look_matrix("", "DMC-GM5").is_none());
    assert!(panasonic_gm5_look_matrix("Leica", "DMC-GM5").is_none());

    // Case / whitespace variants must NOT loosen the match — clean_make and
    // clean_model are already-normalized values from rawler's own camera
    // database (see gm5.toml), so an exact match is the correct, narrowest
    // reading of "reuse the same normalization" and must not be widened to
    // case-insensitive or substring matching later.
    assert!(panasonic_gm5_look_matrix("panasonic", "dmc-gm5").is_none());
    assert!(panasonic_gm5_look_matrix("Panasonic", "DMC-GM5 ").is_none());
    assert!(panasonic_gm5_look_matrix("Panasonic", " DMC-GM5").is_none());

    // Empty / garbage identity.
    assert!(panasonic_gm5_look_matrix("", "").is_none());
  }

  /// A camera that fails the gate must be *completely* unaffected: the
  /// caller never even has a matrix to apply, so `fit_color_curves`'s
  /// output for a non-GM5 photo is byte-identical whether or not this
  /// feature exists. This exercises the general `gate -> Option<matrix> ->
  /// apply_look_matrix_srgb only if Some` call pattern in isolation — it is
  /// a unit test of `panasonic_gm5_look_matrix`/`apply_look_matrix_srgb`'s
  /// math, not of `run_look_fit` (which never calls either function — the
  /// production correction lives in the mega_shader GPU stage instead; see
  /// the STATUS note on [`panasonic_gm5_look_matrix`]). The equivalent
  /// invariant for the real, wired-in shader stage — a non-GM5 camera's
  /// rendered output is unaffected — is pinned by
  /// `rust-renderer/tests/look_matrix_tests.rs::non_gm5_camera_is_unaffected_by_the_look_matrix_stage`.
  #[test]
  fn non_gm5_camera_fit_is_unaffected_by_the_matrix_feature() {
    let (neutral, preview) = pair(|t| {
      let v = 0.10 + 0.80 * t;
      ((v, v, v), ((v + 0.05).min(1.0), (v * 0.9), (v + 0.02).min(1.0)))
    });
    let plain = fit_color_curves(&neutral, &preview);

    // Simulate exactly what `run_look_fit` does: gate on camera identity,
    // only transform `neutral` when `Some`.
    let gated = match panasonic_gm5_look_matrix("Nikon", "Z f") {
      Some(matrix) => apply_look_matrix_srgb(&neutral, &matrix),
      None => neutral.clone(),
    };
    let gated_curves = fit_color_curves(&gated, &preview);

    assert_eq!(plain, gated_curves, "a non-GM5 camera must produce the exact same fit_color_curves result with or without the matrix feature");
  }

  /// Offline-corpus validation of the underlying matrix math, on real GM5
  /// frames: applying `PANASONIC_GM5_LOOK_MATRIX` (in linear light, via
  /// `apply_look_matrix_srgb`) before `fit_color_curves` must measurably
  /// beat fitting `fit_color_curves` on the uncorrected neutral — mirroring
  /// the leave-one-out result in `ml/look-fit/out/panasonic_analysis.json`
  /// (held-out look MAE 5.44 -> 4.46 across all 38 frames; here on a
  /// handful of them, same direction). This calls
  /// `apply_look_matrix_srgb`/`fit_color_curves` directly, in isolation, as
  /// a CPU-only proof of the matrix's effect — it does NOT exercise
  /// `run_look_fit` or any render path (the production correction is a GPU
  /// shader stage this Rust function is a reference for, not a call target
  /// — see the STATUS note on [`panasonic_gm5_look_matrix`]). The
  /// production-path equivalent — that a real render and the look_fit job's
  /// neutral get the SAME correction — is
  /// `rust-renderer/tests/look_matrix_tests.rs::gm5_fit_time_and_render_time_apply_the_same_correction`,
  /// which renders through the actual `mega_shader.wgsl` stage on GPU.
  /// Needs the real (neutral, preview) fixture pairs that live in the
  /// parent RAW-Manager monorepo (`ml/look-fit/data/`), outside this
  /// submodule — same `samplecheck`-gated pattern as `dng/writer.rs`'s
  /// `convert_canon_cr3_to_dng` (which needs `RAWLER_RAWDB`).
  #[cfg(feature = "samplecheck")]
  #[test]
  fn gm5_pooled_matrix_beats_colorcurves_alone_on_real_frames() {
    let data_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ml/look-fit/data");
    let ids = ["8340806", "8340727", "8350005"];

    let matrix = panasonic_gm5_look_matrix("Panasonic", "DMC-GM5").expect("GM5 must match its own validated matrix");

    let mut base_total = 0.0f64;
    let mut corrected_total = 0.0f64;
    // Content-sniffed load (not extension-based `image::open`): the
    // `neutral_*.png` fixtures are actually JPEG bytes under a `.png` name
    // (confirmed via `file(1)` — their own EXIF even carries
    // manufacturer=Panasonic / model=DMC-GM5), while `preview_*.ppm` is
    // genuine Netpbm. `image::load_from_memory` guesses from the magic
    // bytes, same as `run_look_fit`'s own `image::load_from_memory(&neutral_jpeg)`.
    let load = |path: &std::path::Path| -> DynamicImage {
      let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
      image::load_from_memory(&bytes).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()))
    };
    for id in ids {
      let neutral_path = data_dir.join(format!("neutral_{id}.png"));
      let preview_path = data_dir.join(format!("preview_{id}.ppm"));
      let neutral = load(&neutral_path);
      let preview = load(&preview_path);

      let base_curves = fit_color_curves(&neutral, &preview).expect("baseline colorCurves fit must succeed on a real GM5 frame");
      base_total += curve_mae(&neutral, &preview, &base_curves);

      let corrected_neutral = apply_look_matrix_srgb(&neutral, &matrix);
      let corrected_curves = fit_color_curves(&corrected_neutral, &preview).expect("matrix-corrected colorCurves fit must succeed");
      corrected_total += curve_mae(&corrected_neutral, &preview, &corrected_curves);
    }
    let n = ids.len() as f64;
    let (base_mae, corrected_mae) = (base_total / n, corrected_total / n);
    assert!(
      corrected_mae < base_mae,
      "matrix-corrected fit must beat colorCurves-only on real GM5 frames: base={base_mae:.4} corrected={corrected_mae:.4}"
    );
  }

  /// Same piecewise-linear evaluation `fit_color_curves`'s own internal
  /// regression guard uses (uniform x knots in `[0,1]`), on the same
  /// 256x171 grid, so this measures exactly what the fit optimizes against.
  #[cfg(feature = "samplecheck")]
  fn curve_mae(neutral: &DynamicImage, preview: &DynamicImage, curves: &[crate::recipe::ToneCurve; 3]) -> f64 {
    let n = neutral.resize_exact(256, 171, image::imageops::FilterType::Triangle).to_rgb8();
    let p = preview.resize_exact(256, 171, image::imageops::FilterType::Triangle).to_rgb8();
    let (np, pp) = (n.as_raw(), p.as_raw());
    let eval = |pts: &[f32], x: f32| -> f32 {
      let nk = pts.len() / 2;
      if nk < 2 {
        return x;
      }
      let t = x.clamp(0.0, 1.0) * (nk - 1) as f32;
      let i = (t.floor() as usize).min(nk - 2);
      let f = t - i as f32;
      pts[2 * i + 1] * (1.0 - f) + pts[2 * (i + 1) + 1] * f
    };
    let mut err = 0.0f64;
    let px = np.len() / 3;
    for i in 0..px {
      for c in 0..3 {
        let b = np[i * 3 + c] as f32 / 255.0;
        let q = pp[i * 3 + c] as f32 / 255.0;
        let f = if curves[c].is_empty() { b } else { eval(&curves[c].points, b) };
        err += (f - q).abs() as f64;
      }
    }
    err / (px * 3) as f64
  }
}
