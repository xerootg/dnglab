// SPDX-License-Identifier: LGPL-2.1
//! Safe Rust API behind the C ABI.
//!
//! Everything here is plain Rust so it can be unit tested on the host and
//! reused by other bindings. `lib.rs` only adds the `extern "C"` plumbing.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};

use image::{DynamicImage, RgbImage};
use jpeg_encoder::{ColorType, Encoder, SamplingFactor};
use rawler::decoders::{Decoder, RawDecodeParams, RawMetadata};
use rawler::dng::convert::{ConvertParams, convert_raw_file};
use rawler::dng::{CropMode, DngCompression, DngPhotometricConversion};
use rawler::imgop::develop::{Intermediate, ProcessingStep, RawDevelop};
use rawler::imgop::srgb::{srgb_apply_gamma, srgb_apply_gamma_n};
use rawler::rawsource::RawSource;
use rawler::{Orientation, RawImage};

use crate::error::{Error, Result};
use crate::exif_jpeg;

/// Software tag written into authored files.
pub const DEFAULT_SOFTWARE: &str = "dnglab-ffi";

/// Longest edge used when the caller does not constrain previews.
pub const DEFAULT_PREVIEW_QUALITY: u8 = 90;

// ---------------------------------------------------------------------------
// Options and outcomes
// ---------------------------------------------------------------------------

/// Chroma subsampling for JPEGs rendered from sensor data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaSubsampling {
  /// 4:4:4 - no subsampling
  S444,
  /// 4:2:2 - horizontal only
  S422,
  /// 4:2:0 - horizontal and vertical
  S420,
}

impl ChromaSubsampling {
  pub fn from_i32(v: i32) -> Self {
    match v {
      1 => Self::S422,
      2 => Self::S420,
      _ => Self::S444,
    }
  }

  fn sampling_factor(self) -> SamplingFactor {
    match self {
      Self::S444 => SamplingFactor::F_1_1,
      Self::S422 => SamplingFactor::F_2_1,
      Self::S420 => SamplingFactor::F_2_2,
    }
  }
}

/// When to reuse the JPEG the camera embedded in the RAW instead of rendering.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddedJpegPolicy {
  /// Always render from sensor data.
  Never,
  /// Reuse the embedded JPEG only when it is (nearly) full sensor resolution.
  FullResolutionOnly,
  /// Reuse whatever embedded JPEG exists, however small.
  Always,
}

impl EmbeddedJpegPolicy {
  pub fn from_i32(v: i32) -> Self {
    match v {
      0 => Self::Never,
      2 => Self::Always,
      _ => Self::FullResolutionOnly,
    }
  }
}

/// DNG authoring options (a subset of [`ConvertParams`] with app-friendly defaults).
#[derive(Clone, Debug)]
pub struct DngOptions {
  /// Embed the original RAW file (DNG OriginalRawFileData).
  pub embed_original: bool,
  /// Include a JPEG preview (embedded camera JPEG is passed through when available).
  pub preview: bool,
  /// Include a thumbnail.
  pub thumbnail: bool,
  /// Lossless JPEG-92 compression; `false` writes uncompressed data.
  pub lossless: bool,
  pub artist: Option<String>,
  /// Directory containing DNG camera profiles (`<UniqueCameraModel>.dcp`).
  pub dcp_dir: Option<PathBuf>,
  /// Explicit DCP file, takes precedence over `dcp_dir`.
  pub dcp_file: Option<PathBuf>,
  pub software: String,
  /// Sub-image index for multi-image RAW containers.
  pub index: usize,
}

impl Default for DngOptions {
  fn default() -> Self {
    Self {
      embed_original: false,
      preview: true,
      thumbnail: true,
      lossless: true,
      artist: None,
      dcp_dir: None,
      dcp_file: None,
      software: DEFAULT_SOFTWARE.to_string(),
      index: 0,
    }
  }
}

/// JPEG export options.
#[derive(Clone, Debug)]
pub struct JpegOptions {
  /// 1..=100, used for the rendered path only.
  pub quality: u8,
  pub chroma: ChromaSubsampling,
  pub optimize_huffman: bool,
  pub progressive: bool,
  pub embedded_policy: EmbeddedJpegPolicy,
  /// Embedded JPEG must be at least this fraction of the sensor's default crop
  /// on both edges to count as "full resolution" (0.0..=1.0).
  pub full_resolution_threshold: f32,
  /// Write EXIF (camera, exposure, lens, GPS, orientation) into the JPEG.
  pub embed_exif: bool,
  /// Rendered path: dcraw-style automatic brightness (scale so that
  /// `auto_bright_threshold` of the samples clip). `false` applies the
  /// camera's DNG BaselineExposure instead.
  pub auto_bright: bool,
  /// Fraction of samples allowed to clip for `auto_bright` (dcraw default 0.01).
  pub auto_bright_threshold: f32,
  pub software: String,
}

impl Default for JpegOptions {
  fn default() -> Self {
    Self {
      quality: 95,
      chroma: ChromaSubsampling::S444,
      optimize_huffman: true,
      progressive: false,
      embedded_policy: EmbeddedJpegPolicy::FullResolutionOnly,
      full_resolution_threshold: 0.9,
      embed_exif: true,
      auto_bright: true,
      auto_bright_threshold: 0.01,
      software: DEFAULT_SOFTWARE.to_string(),
    }
  }
}

/// How the rendered path sets overall brightness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brightness {
  /// Apply the camera's DNG BaselineExposure (2^EV), nothing else.
  BaselineExposure,
  /// dcraw/LibRaw style: scale so this fraction of samples clips (e.g. 0.01).
  Auto(f32),
}

/// Result of [`convert_to_jpeg`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JpegOutcome {
  /// The camera's embedded JPEG was copied instead of rendering.
  pub used_embedded: bool,
  pub width: u32,
  pub height: u32,
  /// EXIF orientation (1..8) of the pixels in the file. Rendered output is
  /// always rotated upright (`1`); passthrough keeps sensor orientation and
  /// relies on the EXIF tag.
  pub orientation: u16,
  /// EXIF was written into the file.
  pub exif_embedded: bool,
}

/// Result of [`extract_preview`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreviewOutcome {
  /// EXIF orientation (1..8) the caller still has to apply. `1` when the
  /// pixels were rotated upright by this library.
  pub orientation: u16,
  pub from_embedded: bool,
  pub width: u32,
  pub height: u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run `f`, converting a panic inside rawler into an [`Error`] instead of
/// unwinding across the FFI boundary.
pub fn guard<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
  match catch_unwind(AssertUnwindSafe(f)) {
    Ok(r) => r,
    Err(payload) => {
      let msg = if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
      } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
      } else {
        "unknown panic".to_string()
      };
      Err(Error::new(format!("Internal decoder failure: {msg}")))
    }
  }
}

/// A RAW file opened for reading together with its decoder.
pub struct Opened {
  pub source: RawSource,
  pub decoder: Box<dyn Decoder>,
  pub params: RawDecodeParams,
}

impl Opened {
  pub fn open(path: &Path) -> Result<Self> {
    if !path.is_file() {
      return Err(Error::new(format!("File not found: {}", path.display())));
    }
    let source = RawSource::new(path).map_err(|e| Error::new(format!("Cannot open {}: {e}", path.display())))?;
    let decoder = rawler::get_decoder(&source)?;
    Ok(Self {
      source,
      decoder,
      params: RawDecodeParams::default(),
    })
  }

  pub fn metadata(&self) -> Result<RawMetadata> {
    Ok(self.decoder.raw_metadata(&self.source, &self.params)?)
  }

  pub fn raw_image(&self, dummy: bool) -> Result<RawImage> {
    Ok(self.decoder.raw_image(&self.source, &self.params, dummy)?)
  }

  /// Embedded JPEG bytes plus their dimensions, if the format carries one.
  pub fn embedded_jpeg(&self) -> Result<Option<(Vec<u8>, u32, u32)>> {
    match self.decoder.preview_jpeg(&self.source, &self.params) {
      Ok(Some((bytes, mut w, mut h))) => {
        if w == 0 || h == 0 {
          let (pw, ph) = exif_jpeg::jpeg_dimensions(&bytes);
          w = pw;
          h = ph;
        }
        if bytes.len() < 4 || w == 0 || h == 0 {
          return Ok(None);
        }
        Ok(Some((bytes, w, h)))
      }
      Ok(None) => Ok(None),
      Err(e) => {
        log::warn!("preview_jpeg failed, continuing without embedded JPEG: {e}");
        Ok(None)
      }
    }
  }

  /// Orientation of the sensor data from the RAW metadata (1 when unknown).
  pub fn orientation(&self, md: &RawMetadata) -> u16 {
    match md.exif.orientation {
      Some(o) if (1..=8).contains(&o) => o,
      _ => 1,
    }
  }
}

/// Dimensions of the "default crop" (what a viewer shows) and of the full
/// sensor frame. Cheap: uses rawler's dummy decode which parses structure but
/// skips pixel decompression. Returns `None` when the decoder cannot tell
/// without decoding.
pub fn sensor_dimensions(opened: &Opened) -> Option<((usize, usize), (usize, usize))> {
  let raw = guard(|| opened.raw_image(true)).ok()?;
  Some((default_crop_dims(&raw), (raw.width, raw.height)))
}

pub fn default_crop_dims(raw: &RawImage) -> (usize, usize) {
  raw
    .crop_area
    .or(raw.active_area)
    .map(|r| (r.d.w, r.d.h))
    .filter(|(w, h)| *w > 0 && *h > 0)
    .unwrap_or((raw.width, raw.height))
}

/// Decide whether an embedded JPEG of `jpeg` pixels counts as full resolution
/// for a sensor whose default crop is `sensor` pixels. Compares long and short
/// edges so rotated previews are judged correctly.
pub fn is_full_resolution(jpeg: (u32, u32), sensor: (usize, usize), threshold: f32) -> bool {
  let (jw, jh) = (jpeg.0 as f32, jpeg.1 as f32);
  let (sw, sh) = (sensor.0 as f32, sensor.1 as f32);
  if sw <= 0.0 || sh <= 0.0 || jw <= 0.0 || jh <= 0.0 {
    return false;
  }
  let t = threshold.clamp(0.0, 1.0);
  jw.max(jh) >= sw.max(sh) * t && jw.min(jh) >= sw.min(sh) * t
}

/// Rotate/flip an image so that it displays upright for the given EXIF orientation.
pub fn apply_orientation(img: DynamicImage, orientation: u16) -> DynamicImage {
  match orientation {
    2 => img.fliph(),
    3 => img.rotate180(),
    4 => img.flipv(),
    5 => img.rotate90().fliph(),
    6 => img.rotate90(),
    7 => img.rotate270().fliph(),
    8 => img.rotate270(),
    _ => img,
  }
}

/// Downscale so that the longest edge is at most `max_dim` (0 = no limit).
pub fn fit_within(img: DynamicImage, max_dim: u32) -> DynamicImage {
  if max_dim == 0 || (img.width() <= max_dim && img.height() <= max_dim) {
    return img;
  }
  // Triangle is a good speed/quality compromise for large downscales of
  // already-processed (gamma encoded, sharpened) camera JPEGs.
  img.resize(max_dim, max_dim, image::imageops::FilterType::Triangle)
}

/// Encode an 8-bit RGB image as JPEG into memory.
pub fn encode_jpeg(rgb: &RgbImage, opts: &JpegOptions, exif_payload: Option<&[u8]>) -> Result<Vec<u8>> {
  let (w, h) = rgb.dimensions();
  if w == 0 || h == 0 || w > u16::MAX as u32 || h > u16::MAX as u32 {
    return Err(Error::new(format!("Image dimensions {w}x{h} not encodable as JPEG")));
  }
  let mut out = Vec::with_capacity((w as usize * h as usize) / 2);
  let mut enc = Encoder::new(&mut out, opts.quality.clamp(1, 100));
  enc.set_sampling_factor(opts.chroma.sampling_factor());
  enc.set_optimized_huffman_tables(opts.optimize_huffman);
  enc.set_progressive(opts.progressive);
  if let Some(payload) = exif_payload {
    // payload already starts with "Exif\0\0"
    enc.add_app_segment(1, payload.to_vec())?;
  }
  enc.encode(rgb.as_raw(), w as u16, h as u16, ColorType::Rgb)?;
  Ok(out)
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
  let mut f = File::create(path).map_err(|e| Error::new(format!("Cannot create {}: {e}", path.display())))?;
  f.write_all(bytes)?;
  f.flush()?;
  Ok(())
}

fn remove_partial(path: &Path) {
  let _ = std::fs::remove_file(path);
}

// ---------------------------------------------------------------------------
// Rendering (demosaic + colour) for formats without a usable embedded JPEG
// ---------------------------------------------------------------------------

/// dcraw-style auto brightness gain: the smallest gain such that `threshold`
/// of the samples (per channel, worst channel wins) end up clipped at 1.0.
/// Samples are expected in [0, 1]; the gain is clamped to [1, 16].
pub fn auto_bright_gain<'a>(channels: impl Iterator<Item = &'a [f32]>, threshold: f32) -> f32 {
  const BINS: usize = 4096;
  let mut white = 0.0f32;
  for samples in channels {
    if samples.is_empty() {
      continue;
    }
    let mut hist = vec![0u32; BINS];
    for v in samples {
      let b = ((v.clamp(0.0, 1.0)) * (BINS - 1) as f32) as usize;
      hist[b] += 1;
    }
    let allowed = (samples.len() as f64 * threshold.clamp(0.0, 0.5) as f64) as u64;
    let mut seen = 0u64;
    let mut level = BINS - 1;
    while level > 0 {
      seen += hist[level] as u64;
      if seen > allowed {
        break;
      }
      level -= 1;
    }
    white = white.max((level as f32 + 1.0) / BINS as f32);
  }
  if white <= 0.0 { 1.0 } else { (1.0 / white).clamp(1.0, 16.0) }
}

/// Develop the sensor data into an upright, display-referred sRGB image.
///
/// Pipeline: rescale to [0,1] -> demosaic (PPG for Bayer, bilinear for
/// X-Trans / 4-colour) -> Fuji rotation -> active area crop -> white balance ->
/// camera matrix to sRGB -> default crop -> brightness gain -> sRGB gamma
/// -> EXIF orientation applied to the pixels.
pub fn render_upright(raw: &RawImage, brightness: Brightness) -> Result<DynamicImage> {
  let develop = RawDevelop::new_with(&[
    ProcessingStep::Rescale,
    ProcessingStep::Demosaic,
    ProcessingStep::FujiRotate,
    ProcessingStep::CropActiveArea,
    ProcessingStep::WhiteBalance,
    ProcessingStep::Calibrate,
    ProcessingStep::CropDefault,
  ]);
  let mut inter = develop.develop_intermediate(raw)?;

  // rawler's linear develop lands at "exposure as recorded"; cameras and raw
  // processors brighten from there. Either use the DNG BaselineExposure the
  // camera database knows, or measure the frame like dcraw does.
  let gain = match brightness {
    Brightness::BaselineExposure => 2f32.powf(raw.camera.baseline_exposure.unwrap_or(0.0)),
    Brightness::Auto(threshold) => {
      let g = match &inter {
        Intermediate::Monochrome(p) => auto_bright_gain(std::iter::once(p.data.as_slice()), threshold),
        Intermediate::ThreeColor(p) => {
          let flat = p.flatten();
          let n = flat.len() / 3;
          let mut ch = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
          for px in flat.chunks_exact(3) {
            ch[0].push(px[0]);
            ch[1].push(px[1]);
            ch[2].push(px[2]);
          }
          auto_bright_gain(ch.iter().map(|c| c.as_slice()), threshold)
        }
        Intermediate::FourColor(p) => {
          let flat = p.flatten();
          auto_bright_gain(std::iter::once(flat.as_slice()), threshold)
        }
      };
      log::debug!("auto brightness gain: {g:.3}");
      g
    }
  };
  let tone = move |v: f32| srgb_apply_gamma((v * gain).clamp(0.0, 1.0));
  match &mut inter {
    Intermediate::Monochrome(p) => p.for_each(tone),
    Intermediate::ThreeColor(p) => {
      p.for_each(move |px| srgb_apply_gamma_n([(px[0] * gain).clamp(0.0, 1.0), (px[1] * gain).clamp(0.0, 1.0), (px[2] * gain).clamp(0.0, 1.0)]))
    }
    Intermediate::FourColor(p) => p.for_each(move |px| {
      srgb_apply_gamma_n([
        (px[0] * gain).clamp(0.0, 1.0),
        (px[1] * gain).clamp(0.0, 1.0),
        (px[2] * gain).clamp(0.0, 1.0),
        (px[3] * gain).clamp(0.0, 1.0),
      ])
    }),
  }

  let img = inter.to_dynamic_image().ok_or_else(|| Error::new("Developed image has invalid dimensions"))?;
  Ok(apply_orientation(img, raw.orientation.to_u16()))
}

// ---------------------------------------------------------------------------
// Public operations
// ---------------------------------------------------------------------------

/// Convert a RAW file into a DNG at `output`.
pub fn convert_to_dng(input: &Path, output: &Path, opts: &DngOptions) -> Result<()> {
  if !input.is_file() {
    return Err(Error::new(format!("File not found: {}", input.display())));
  }
  let params = ConvertParams {
    embedded: opts.embed_original,
    compression: if opts.lossless {
      DngCompression::Lossless
    } else {
      DngCompression::Uncompressed
    },
    photometric_conversion: DngPhotometricConversion::Original,
    apply_scaling: false,
    crop: CropMode::Best,
    predictor: 1,
    preview: opts.preview,
    thumbnail: opts.thumbnail,
    artist: opts.artist.clone(),
    software: opts.software.clone(),
    index: opts.index,
    keep_mtime: false,
    dcp_dir: opts.dcp_dir.clone(),
    dcp_file: opts.dcp_file.clone(),
  };

  let file = File::create(output).map_err(|e| Error::new(format!("Cannot create {}: {e}", output.display())))?;
  let mut writer = BufWriter::new(file);
  let result = guard(|| {
    convert_raw_file(input, &mut writer, &params)?;
    writer.flush()?;
    Ok(())
  });
  if result.is_err() {
    drop(writer);
    remove_partial(output);
  }
  result
}

/// Export a RAW file as JPEG at `output`.
pub fn convert_to_jpeg(input: &Path, output: &Path, opts: &JpegOptions) -> Result<JpegOutcome> {
  let result = guard(|| convert_to_jpeg_inner(input, output, opts));
  if result.is_err() {
    remove_partial(output);
  }
  result
}

fn convert_to_jpeg_inner(input: &Path, output: &Path, opts: &JpegOptions) -> Result<JpegOutcome> {
  let opened = Opened::open(input)?;
  let md = opened.metadata()?;
  let raw_orientation = opened.orientation(&md);

  // Fast path: camera JPEG passthrough.
  if opts.embedded_policy != EmbeddedJpegPolicy::Never {
    if let Some((bytes, w, h)) = opened.embedded_jpeg()? {
      let accept = match opts.embedded_policy {
        EmbeddedJpegPolicy::Always => true,
        EmbeddedJpegPolicy::FullResolutionOnly => match sensor_dimensions(&opened) {
          Some((crop, full)) => {
            let ok = is_full_resolution((w, h), crop, opts.full_resolution_threshold);
            log::info!(
              "Embedded JPEG {w}x{h}, default crop {}x{}, sensor {}x{}: full resolution = {ok}",
              crop.0,
              crop.1,
              full.0,
              full.1
            );
            ok
          }
          None => {
            log::warn!("Could not determine sensor dimensions; not trusting embedded JPEG as full resolution");
            false
          }
        },
        EmbeddedJpegPolicy::Never => false,
      };
      if accept {
        // Embedded JPEGs are stored in sensor orientation unless the camera already
        // rotated them (then their own EXIF says so and the sensor orientation does
        // not apply).
        let orientation = match exif_jpeg::jpeg_exif_orientation(&bytes) {
          Some(o) if o != 1 && o != 0 => o,
          _ => raw_orientation,
        };
        let (bytes, exif_embedded) = if opts.embed_exif {
          match exif_jpeg::build_exif_app1_payload(&md, orientation, &opts.software).and_then(|p| exif_jpeg::splice_exif_app1(&bytes, &p)) {
            Ok(spliced) => (spliced, true),
            Err(e) => {
              log::warn!("Writing EXIF into embedded JPEG failed, copying verbatim: {e}");
              (bytes, false)
            }
          }
        } else {
          (bytes, false)
        };
        write_file(output, &bytes)?;
        return Ok(JpegOutcome {
          used_embedded: true,
          width: w,
          height: h,
          orientation,
          exif_embedded,
        });
      }
    } else {
      log::info!("No embedded JPEG in {}, rendering from sensor data", input.display());
    }
  }

  // Slow path: demosaic and render.
  let raw = opened.raw_image(false)?;
  let brightness = if opts.auto_bright {
    Brightness::Auto(opts.auto_bright_threshold)
  } else {
    Brightness::BaselineExposure
  };
  let img = render_upright(&raw, brightness)?;
  let rgb = img.into_rgb8();
  let (w, h) = rgb.dimensions();

  let exif_payload = if opts.embed_exif {
    match exif_jpeg::build_exif_app1_payload(&md, 1, &opts.software) {
      Ok(p) => Some(p),
      Err(e) => {
        log::warn!("Building EXIF failed, writing JPEG without metadata: {e}");
        None
      }
    }
  } else {
    None
  };
  let exif_embedded = exif_payload.is_some();
  let bytes = encode_jpeg(&rgb, opts, exif_payload.as_deref())?;
  write_file(output, &bytes)?;
  Ok(JpegOutcome {
    used_embedded: false,
    width: w,
    height: h,
    orientation: 1,
    exif_embedded,
  })
}

/// Extract (or render) a JPEG preview with the longest edge limited to
/// `max_dim` pixels (0 = unlimited). Prefers the embedded camera JPEG; when it
/// fits and no rotation is needed the bytes are copied verbatim.
pub fn extract_preview(input: &Path, output: &Path, max_dim: u32) -> Result<PreviewOutcome> {
  let result = guard(|| extract_preview_inner(input, output, max_dim));
  if result.is_err() {
    remove_partial(output);
  }
  result
}

fn extract_preview_inner(input: &Path, output: &Path, max_dim: u32) -> Result<PreviewOutcome> {
  let opened = Opened::open(input)?;
  let md = opened.metadata()?;
  let orientation = opened.orientation(&md);

  let encode_opts = JpegOptions {
    quality: DEFAULT_PREVIEW_QUALITY,
    chroma: ChromaSubsampling::S420,
    optimize_huffman: true,
    progressive: false,
    embedded_policy: EmbeddedJpegPolicy::Always,
    full_resolution_threshold: 0.0,
    embed_exif: false,
    auto_bright: true,
    auto_bright_threshold: 0.01,
    software: DEFAULT_SOFTWARE.to_string(),
  };

  // Writes `img` rotated upright and resized; returns the outcome.
  let finish = |img: DynamicImage, from_embedded: bool| -> Result<PreviewOutcome> {
    let img = fit_within(img, max_dim);
    let img = apply_orientation(img, orientation);
    let rgb = img.into_rgb8();
    let (w, h) = rgb.dimensions();
    let bytes = encode_jpeg(&rgb, &encode_opts, None)?;
    write_file(output, &bytes)?;
    Ok(PreviewOutcome {
      orientation: 1,
      from_embedded,
      width: w,
      height: h,
    })
  };

  // 1. Embedded camera JPEG.
  if let Some((bytes, w, h)) = opened.embedded_jpeg()? {
    let fits = max_dim == 0 || (w <= max_dim && h <= max_dim);
    if fits {
      // Verbatim copy; the caller rotates if needed.
      write_file(output, &bytes)?;
      return Ok(PreviewOutcome {
        orientation,
        from_embedded: true,
        width: w,
        height: h,
      });
    }
    match image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg) {
      Ok(img) => return finish(img, true),
      Err(e) => log::warn!("Embedded JPEG could not be decoded ({e}), trying other previews"),
    }
  }

  // 2. Decoder provided preview / thumbnail images (uncompressed strips, old-JPEG, ...).
  for (name, img) in [
    ("preview_image", opened.decoder.preview_image(&opened.source, &opened.params)),
    ("thumbnail_image", opened.decoder.thumbnail_image(&opened.source, &opened.params)),
  ] {
    match img {
      Ok(Some(img)) if img.width() > 0 && img.height() > 0 => return finish(img, true),
      Ok(_) => {}
      Err(e) => log::debug!("{name} failed: {e}"),
    }
  }

  // 3. Last resort: develop the sensor data.
  log::info!("No embedded preview in {}, developing sensor data", input.display());
  let raw = opened.raw_image(false)?;
  let img = render_upright(&raw, Brightness::Auto(0.01))?; // already upright
  let img = fit_within(img, max_dim);
  let rgb = img.into_rgb8();
  let (w, h) = rgb.dimensions();
  let bytes = encode_jpeg(&rgb, &encode_opts, None)?;
  write_file(output, &bytes)?;
  Ok(PreviewOutcome {
    orientation: 1,
    from_embedded: false,
    width: w,
    height: h,
  })
}

/// Lower-case extensions rawler can decode.
pub fn supported_extensions() -> Vec<String> {
  rawler::decoders::supported_extensions().iter().map(|e| e.to_ascii_lowercase()).collect()
}

pub fn is_supported_extension(ext: &str) -> bool {
  let ext = ext.trim().trim_start_matches('.').to_ascii_uppercase();
  rawler::decoders::supported_extensions().contains(&ext.as_str())
}

/// Version of the library (crate version, identical to the rawler version it wraps).
pub fn version() -> &'static str {
  env!("CARGO_PKG_VERSION")
}

/// Helper for callers that want the orientation enum.
pub fn orientation_from_u16(o: u16) -> Orientation {
  Orientation::from_u16(o)
}

#[cfg(test)]
mod tests {
  use super::*;
  use image::GenericImageView;

  #[test]
  fn full_resolution_decision() {
    // Nikon style: 4928x3264 sensor, 4928x3264 embedded
    assert!(is_full_resolution((4928, 3264), (4928, 3264), 0.9));
    // Sony style: 1616x1080 preview on a 6000x4000 sensor
    assert!(!is_full_resolution((1616, 1080), (6000, 4000), 0.9));
    // Rotated preview counts by edges
    assert!(is_full_resolution((3264, 4928), (4928, 3264), 0.9));
    // Slightly cropped camera JPEG still full res
    assert!(is_full_resolution((5952, 3968), (6000, 4000), 0.9));
    assert!(!is_full_resolution((0, 0), (6000, 4000), 0.9));
    assert!(!is_full_resolution((6000, 4000), (0, 0), 0.9));
  }

  #[test]
  fn policy_and_chroma_mapping() {
    assert_eq!(EmbeddedJpegPolicy::from_i32(0), EmbeddedJpegPolicy::Never);
    assert_eq!(EmbeddedJpegPolicy::from_i32(1), EmbeddedJpegPolicy::FullResolutionOnly);
    assert_eq!(EmbeddedJpegPolicy::from_i32(2), EmbeddedJpegPolicy::Always);
    assert_eq!(EmbeddedJpegPolicy::from_i32(99), EmbeddedJpegPolicy::FullResolutionOnly);
    assert_eq!(ChromaSubsampling::from_i32(0), ChromaSubsampling::S444);
    assert_eq!(ChromaSubsampling::from_i32(1), ChromaSubsampling::S422);
    assert_eq!(ChromaSubsampling::from_i32(2), ChromaSubsampling::S420);
  }

  #[test]
  fn orientation_transforms_dimensions() {
    let img = DynamicImage::new_rgb8(4, 2);
    assert_eq!(apply_orientation(img.clone(), 1).dimensions(), (4, 2));
    assert_eq!(apply_orientation(img.clone(), 3).dimensions(), (4, 2));
    for o in [5u16, 6, 7, 8] {
      assert_eq!(apply_orientation(img.clone(), o).dimensions(), (2, 4), "orientation {o}");
    }
  }

  #[test]
  fn auto_bright_gain_targets_percentile() {
    // 1000 samples linearly spread over [0, 0.5]: 1% clipping => white ~ 0.5
    let samples: Vec<f32> = (0..1000).map(|i| i as f32 / 2000.0).collect();
    let g = auto_bright_gain(std::iter::once(samples.as_slice()), 0.01);
    assert!((g - 2.0).abs() < 0.1, "gain {g}");
    // Already bright: no gain
    let bright: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
    assert!((auto_bright_gain(std::iter::once(bright.as_slice()), 0.01) - 1.0).abs() < 0.05);
    // Empty input is a no-op
    assert_eq!(auto_bright_gain(std::iter::empty(), 0.01), 1.0);
  }

  #[test]
  fn fit_within_limits_longest_edge() {
    let img = DynamicImage::new_rgb8(400, 200);
    assert_eq!(fit_within(img.clone(), 0).dimensions(), (400, 200));
    assert_eq!(fit_within(img.clone(), 1000).dimensions(), (400, 200));
    assert_eq!(fit_within(img, 100).dimensions(), (100, 50));
  }

  #[test]
  fn encode_jpeg_roundtrip_with_exif_segment() {
    // Optimized Huffman tables are disabled here on purpose: the zune-jpeg 0.4
    // decoder pinned by this workspace rejects the very short code tables a
    // 16x8 test image produces (real-size output decodes fine, and zune-jpeg
    // 0.5 accepts the tiny case too).
    let rgb = RgbImage::from_fn(16, 8, |x, y| image::Rgb([(x * 16) as u8, (y * 32) as u8, ((x + y) * 10) as u8]));
    let payload = b"Exif\0\0II*\0\x08\0\0\0\0\0".to_vec();
    let opts = JpegOptions {
      quality: 90,
      chroma: ChromaSubsampling::S444,
      optimize_huffman: false,
      ..JpegOptions::default()
    };
    let bytes = encode_jpeg(&rgb, &opts, Some(&payload)).unwrap();
    assert_eq!(exif_jpeg::jpeg_dimensions(&bytes), (16, 8));
    assert_eq!(bytes.windows(6).filter(|w| *w == b"Exif\0\0").count(), 1);
    let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Jpeg).unwrap();
    assert_eq!(decoded.dimensions(), (16, 8));
  }

  #[test]
  fn extensions_are_lowercase_and_nonempty() {
    let exts = supported_extensions();
    assert!(exts.contains(&"nef".to_string()));
    assert!(exts.iter().all(|e| e.chars().all(|c| !c.is_ascii_uppercase())));
    assert!(is_supported_extension("NEF"));
    assert!(is_supported_extension(".cr3"));
    assert!(!is_supported_extension("jpg"));
  }
}
