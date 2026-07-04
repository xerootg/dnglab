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
  let unique_model = format!("{} {}", rawimage.clean_make, rawimage.clean_model);
  let dcp_path = if let Some(explicit) = &params.dcp_file {
    if explicit.exists() {
      Some(explicit.clone())
    } else {
      log::warn!("DCP override file not found: {} — falling back to auto-match", explicit.display());
      params.dcp_dir.as_ref().and_then(|d| find_dcp(d, &unique_model, style_hint.as_deref()))
        .or_else(|| auto_find_dcp(&unique_model, style_hint.as_deref()))
    }
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
    if let Some(mut recipe) = decoder.recipe(rawfile, &raw_params)? {
      // Measure the in-body look directly: fit per-channel display-space curves
      // from a neutral develop to the camera's embedded JPEG preview, and carry
      // them in the recipe. This reproduces tone + colour + WB + Active
      // D-Lighting's global component far more faithfully than the static
      // ADL-level EV seed (which it supersedes). Best-effort: a failure here
      // leaves the metadata-only recipe intact. See the ADL design note.
      if recipe.color_curves.is_none() {
        let neutral = RawDevelop::default()
          .develop_intermediate(&rawimage)
          .ok()
          .and_then(|img| img.to_dynamic_image());
        let preview = decoder.preview_image(rawfile, &raw_params).ok().flatten();
        if let (Some(neutral), Some(preview)) = (neutral, preview) {
          if let Some(curves) = fit_color_curves(&neutral, &preview) {
            recipe.color_curves = Some(curves);
            // The measured curves carry the brightening; drop the static EV seed
            // so the two don't stack (one-lane rule).
            recipe.exposure = None;
            // Safety guard: the per-channel colorCurves are the look carrier, so a
            // stale display-space NEF ContrastCurve in `recipe.tone` (from nef.rs)
            // must never survive into the emitted recipe — the editor would feed it
            // to the linear lumPoints lane and double-encode it. Clear it here.
            recipe.tone = None;
          }
        }
      }
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
}
