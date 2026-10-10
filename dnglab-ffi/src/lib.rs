// SPDX-License-Identifier: LGPL-2.1
//! # dnglab-ffi
//!
//! C ABI over dnglab's `rawler` crate for embedding in native applications
//! (the Android Raw2DNG app being the first consumer). It exposes exactly the
//! operations a RAW converter front-end needs:
//!
//! * RAW -> DNG authoring (`dnglab_convert_to_dng`)
//! * RAW -> JPEG export, reusing the camera's embedded full-resolution JPEG
//!   when available and desired, otherwise rendering the sensor data
//!   (`dnglab_convert_to_jpeg`)
//! * preview/thumbnail extraction with orientation handling
//!   (`dnglab_extract_preview`)
//! * backend-neutral metadata JSON (`dnglab_metadata_json`)
//!
//! All functions are panic-safe (panics become error codes), thread-safe and
//! never unwind across the FFI boundary. Strings returned to the caller are
//! owned by the caller and must be released with `dnglab_string_free`.
//! See `include/dnglab_ffi.h` for the C declarations.

pub mod api;
pub mod error;
pub mod exif_jpeg;
pub mod metadata;

use std::ffi::{CStr, CString, OsString, c_char};
use std::path::PathBuf;
use std::ptr;

use crate::api::{ChromaSubsampling, DngOptions, EmbeddedJpegPolicy, JpegOptions};
use crate::error::{Error, Result};

/// Operation succeeded.
pub const DNGLAB_OK: i32 = 0;
/// Operation failed; `error_out` (when given) holds a message.
pub const DNGLAB_ERROR: i32 = 1;
/// A required pointer argument was NULL or a string was not valid UTF-8.
pub const DNGLAB_INVALID_ARGUMENT: i32 = 2;

/// DNG authoring options. All strings may be NULL.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct dnglab_dng_options {
  /// Embed the original RAW file (non-zero = yes).
  pub embed_original: i32,
  /// Include a JPEG preview (non-zero = yes).
  pub preview: i32,
  /// Include a thumbnail (non-zero = yes).
  pub thumbnail: i32,
  /// Lossless JPEG-92 compression (non-zero) or uncompressed (zero).
  pub lossless: i32,
  /// Sub-image index for multi-image containers (usually 0).
  pub index: u32,
  pub artist: *const c_char,
  pub dcp_dir: *const c_char,
  pub dcp_file: *const c_char,
  pub software: *const c_char,
}

/// JPEG export options.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct dnglab_jpeg_options {
  /// 1..=100
  pub quality: i32,
  /// 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0
  pub chroma_subsampling: i32,
  /// Optimized Huffman tables (non-zero = yes).
  pub optimize_huffman: i32,
  /// Progressive JPEG (non-zero = yes).
  pub progressive: i32,
  /// 0 = never use embedded JPEG, 1 = only when full resolution, 2 = always.
  pub embedded_policy: i32,
  /// Minimum embedded/sensor edge ratio for "full resolution" (0..1, e.g. 0.9).
  pub full_resolution_threshold: f32,
  /// Write EXIF into the JPEG (non-zero = yes).
  pub embed_exif: i32,
  /// Rendered path: dcraw style auto brightness (non-zero) or DNG
  /// BaselineExposure only (zero).
  pub auto_bright: i32,
  /// Software tag, may be NULL.
  pub software: *const c_char,
}

/// Outcome of `dnglab_convert_to_jpeg`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct dnglab_jpeg_result {
  /// Non-zero when the camera's embedded JPEG was copied.
  pub used_embedded: i32,
  pub width: u32,
  pub height: u32,
  /// EXIF orientation (1..8) of the stored pixels.
  pub orientation: i32,
  /// Non-zero when EXIF was written into the file.
  pub exif_embedded: i32,
}

/// Outcome of `dnglab_extract_preview`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct dnglab_preview_result {
  /// EXIF orientation (1..8) the caller still has to apply (1 = upright).
  pub orientation: i32,
  /// Non-zero when the camera's embedded JPEG was used.
  pub from_embedded: i32,
  pub width: u32,
  pub height: u32,
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Marker for argument errors so the C return code can be chosen without
/// inspecting message text.
const ARG_ERROR_PREFIX: &str = "invalid argument: ";

fn arg_error(msg: impl std::fmt::Display) -> Error {
  Error::new(format!("{ARG_ERROR_PREFIX}{msg}"))
}

/// Convert a C string to a path. On Unix (Android included) paths are
/// arbitrary bytes, so no UTF-8 validation is applied; elsewhere UTF-8 is
/// required.
fn bytes_to_path(bytes: &[u8], name: &str) -> Result<PathBuf> {
  #[cfg(unix)]
  {
    use std::os::unix::ffi::OsStringExt;
    let _ = name;
    Ok(PathBuf::from(OsString::from_vec(bytes.to_vec())))
  }
  #[cfg(not(unix))]
  {
    std::str::from_utf8(bytes)
      .map(PathBuf::from)
      .map_err(|_| arg_error(format!("{name} is not valid UTF-8")))
  }
}

/// Required path argument.
unsafe fn path_arg(p: *const c_char, name: &str) -> Result<PathBuf> {
  if p.is_null() {
    return Err(arg_error(format!("{name} must not be NULL")));
  }
  // SAFETY: caller guarantees a NUL terminated string.
  let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
  if bytes.is_empty() {
    return Err(arg_error(format!("{name} must not be empty")));
  }
  bytes_to_path(bytes, name)
}

/// Optional path argument (NULL or empty = none).
unsafe fn path_opt(p: *const c_char) -> Option<PathBuf> {
  if p.is_null() {
    return None;
  }
  // SAFETY: caller guarantees a NUL terminated string.
  let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
  if bytes.is_empty() { None } else { bytes_to_path(bytes, "path").ok() }
}

/// Optional text argument (NULL, empty or non-UTF-8 = none).
unsafe fn cstr_opt(p: *const c_char) -> Option<String> {
  if p.is_null() {
    None
  } else {
    // SAFETY: caller guarantees a NUL terminated string.
    unsafe { CStr::from_ptr(p) }.to_str().ok().map(|s| s.to_string()).filter(|s| !s.is_empty())
  }
}

fn into_c_string(s: String) -> *mut c_char {
  // Replace interior NULs so the conversion cannot fail.
  let cleaned: String = s.chars().map(|c| if c == '\0' { ' ' } else { c }).collect();
  CString::new(cleaned).map(CString::into_raw).unwrap_or(ptr::null_mut())
}

unsafe fn set_error(error_out: *mut *mut c_char, err: &Error) {
  if !error_out.is_null() {
    // SAFETY: error_out is a valid out-pointer per the C contract.
    unsafe { *error_out = into_c_string(err.message().to_string()) };
  }
}

fn code_for(err: &Error) -> i32 {
  if err.message().starts_with(ARG_ERROR_PREFIX) {
    DNGLAB_INVALID_ARGUMENT
  } else {
    DNGLAB_ERROR
  }
}

unsafe fn finish<T>(result: Result<T>, error_out: *mut *mut c_char, on_ok: impl FnOnce(T)) -> i32 {
  match result {
    Ok(v) => {
      on_ok(v);
      DNGLAB_OK
    }
    Err(e) => {
      log::error!("{e}");
      // SAFETY: forwarded from the extern fn.
      unsafe { set_error(error_out, &e) };
      code_for(&e)
    }
  }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

/// Initialise logging. On Android this routes `log` output to logcat (tag
/// "dnglab"); elsewhere it only sets the maximum level. `level`: 0 = off,
/// 1 = error, 2 = warn, 3 = info, 4 = debug, 5 = trace. Safe to call multiple
/// times; on Android only the first call's level takes effect because the
/// logger can be installed once per process.
#[unsafe(no_mangle)]
pub extern "C" fn dnglab_init_logging(level: i32) {
  let filter = match level {
    i32::MIN..=0 => log::LevelFilter::Off,
    1 => log::LevelFilter::Error,
    2 => log::LevelFilter::Warn,
    3 => log::LevelFilter::Info,
    4 => log::LevelFilter::Debug,
    _ => log::LevelFilter::Trace,
  };
  #[cfg(target_os = "android")]
  {
    android_logger::init_once(android_logger::Config::default().with_max_level(filter).with_tag("dnglab"));
  }
  #[cfg(not(target_os = "android"))]
  {
    log::set_max_level(filter);
  }
}

/// Library version, static string (do not free).
#[unsafe(no_mangle)]
pub extern "C" fn dnglab_version() -> *const c_char {
  static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
  VERSION.get_or_init(|| CString::new(api::version()).expect("version string")).as_ptr()
}

/// Comma separated, lower-case list of supported extensions (without dots).
/// Free with `dnglab_string_free`.
#[unsafe(no_mangle)]
pub extern "C" fn dnglab_supported_extensions() -> *mut c_char {
  into_c_string(api::supported_extensions().join(","))
}

/// Non-zero when `ext` (with or without leading dot, any case) is decodable.
///
/// # Safety
/// `ext` must be NULL or a valid NUL terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnglab_is_supported_extension(ext: *const c_char) -> i32 {
  // SAFETY: documented contract.
  match unsafe { cstr_opt(ext) } {
    Some(e) if api::is_supported_extension(&e) => 1,
    _ => 0,
  }
}

/// Release a string returned by this library. NULL is ignored.
///
/// # Safety
/// `s` must be NULL or a pointer previously returned by this library and not
/// yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnglab_string_free(s: *mut c_char) {
  if !s.is_null() {
    // SAFETY: s came from CString::into_raw in this library.
    drop(unsafe { CString::from_raw(s) });
  }
}

/// Convert `input` RAW into a DNG at `output`. `options` may be NULL for defaults.
///
/// # Safety
/// Pointer arguments must be NULL or valid NUL terminated strings / structs
/// for the duration of the call; `error_out` must be NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnglab_convert_to_dng(
  input: *const c_char,
  output: *const c_char,
  options: *const dnglab_dng_options,
  error_out: *mut *mut c_char,
) -> i32 {
  let result = (|| -> Result<()> {
    // SAFETY: documented contract.
    let input = unsafe { path_arg(input, "input") }?;
    let output = unsafe { path_arg(output, "output") }?;
    let mut opts = DngOptions::default();
    if !options.is_null() {
      // SAFETY: non-null, caller provided struct.
      let o = unsafe { &*options };
      opts.embed_original = o.embed_original != 0;
      opts.preview = o.preview != 0;
      opts.thumbnail = o.thumbnail != 0;
      opts.lossless = o.lossless != 0;
      opts.index = o.index as usize;
      opts.artist = unsafe { cstr_opt(o.artist) };
      opts.dcp_dir = unsafe { path_opt(o.dcp_dir) };
      opts.dcp_file = unsafe { path_opt(o.dcp_file) };
      if let Some(sw) = unsafe { cstr_opt(o.software) } {
        opts.software = sw;
      }
    }
    api::convert_to_dng(&input, &output, &opts)
  })();
  // SAFETY: forwarded out-pointer.
  unsafe { finish(result, error_out, |_| {}) }
}

/// Export `input` RAW as JPEG at `output`. `options` may be NULL for defaults,
/// `result_out` may be NULL.
///
/// # Safety
/// Pointer arguments must be NULL or valid NUL terminated strings / structs
/// for the duration of the call; out-pointers must be NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnglab_convert_to_jpeg(
  input: *const c_char,
  output: *const c_char,
  options: *const dnglab_jpeg_options,
  result_out: *mut dnglab_jpeg_result,
  error_out: *mut *mut c_char,
) -> i32 {
  let result = (|| -> Result<api::JpegOutcome> {
    // SAFETY: documented contract.
    let input = unsafe { path_arg(input, "input") }?;
    let output = unsafe { path_arg(output, "output") }?;
    let mut opts = JpegOptions::default();
    if !options.is_null() {
      // SAFETY: non-null, caller provided struct.
      let o = unsafe { &*options };
      opts.quality = o.quality.clamp(1, 100) as u8;
      opts.chroma = ChromaSubsampling::from_i32(o.chroma_subsampling);
      opts.optimize_huffman = o.optimize_huffman != 0;
      opts.progressive = o.progressive != 0;
      opts.embedded_policy = EmbeddedJpegPolicy::from_i32(o.embedded_policy);
      if o.full_resolution_threshold.is_finite() && o.full_resolution_threshold > 0.0 {
        opts.full_resolution_threshold = o.full_resolution_threshold.min(1.0);
      }
      opts.embed_exif = o.embed_exif != 0;
      opts.auto_bright = o.auto_bright != 0;
      if let Some(sw) = unsafe { cstr_opt(o.software) } {
        opts.software = sw;
      }
    }
    api::convert_to_jpeg(&input, &output, &opts)
  })();
  // SAFETY: forwarded out-pointers.
  unsafe {
    finish(result, error_out, |r| {
      if !result_out.is_null() {
        *result_out = dnglab_jpeg_result {
          used_embedded: r.used_embedded as i32,
          width: r.width,
          height: r.height,
          orientation: r.orientation as i32,
          exif_embedded: r.exif_embedded as i32,
        };
      }
    })
  }
}

/// Write a JPEG preview of `input` to `output`, longest edge <= `max_dimension`
/// (0 = unlimited). `result_out` may be NULL.
///
/// # Safety
/// Pointer arguments must be NULL or valid NUL terminated strings for the
/// duration of the call; out-pointers must be NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnglab_extract_preview(
  input: *const c_char,
  output: *const c_char,
  max_dimension: u32,
  result_out: *mut dnglab_preview_result,
  error_out: *mut *mut c_char,
) -> i32 {
  let result = (|| -> Result<api::PreviewOutcome> {
    // SAFETY: documented contract.
    let input = unsafe { path_arg(input, "input") }?;
    let output = unsafe { path_arg(output, "output") }?;
    api::extract_preview(&input, &output, max_dimension)
  })();
  // SAFETY: forwarded out-pointers.
  unsafe {
    finish(result, error_out, |r| {
      if !result_out.is_null() {
        *result_out = dnglab_preview_result {
          orientation: r.orientation as i32,
          from_embedded: r.from_embedded as i32,
          width: r.width,
          height: r.height,
        };
      }
    })
  }
}

/// Metadata of `input` as a JSON document (see `metadata.rs` for the schema).
/// Returns NULL on failure with `error_out` set. Free with `dnglab_string_free`.
///
/// # Safety
/// `input` must be a valid NUL terminated string; `error_out` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn dnglab_metadata_json(input: *const c_char, error_out: *mut *mut c_char) -> *mut c_char {
  let result = (|| -> Result<String> {
    // SAFETY: documented contract.
    let input = unsafe { path_arg(input, "input") }?;
    metadata::metadata_json(&input)
  })();
  match result {
    Ok(json) => into_c_string(json),
    Err(e) => {
      log::error!("{e}");
      // SAFETY: forwarded out-pointer.
      unsafe { set_error(error_out, &e) };
      ptr::null_mut()
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn null_arguments_are_rejected() {
    let mut err: *mut c_char = ptr::null_mut();
    let rc = unsafe { dnglab_convert_to_dng(ptr::null(), ptr::null(), ptr::null(), &mut err) };
    assert_eq!(rc, DNGLAB_INVALID_ARGUMENT);
    assert!(!err.is_null());
    let msg = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
    assert!(msg.contains("input"));
    unsafe { dnglab_string_free(err) };
  }

  #[test]
  fn missing_file_is_an_error() {
    let input = CString::new("/nonexistent/file.nef").unwrap();
    let output = CString::new("/nonexistent/out.dng").unwrap();
    let mut err: *mut c_char = ptr::null_mut();
    let rc = unsafe { dnglab_convert_to_dng(input.as_ptr(), output.as_ptr(), ptr::null(), &mut err) };
    assert_eq!(rc, DNGLAB_ERROR);
    assert!(!err.is_null());
    unsafe { dnglab_string_free(err) };

    let json = unsafe { dnglab_metadata_json(input.as_ptr(), &mut err) };
    assert!(json.is_null());
    assert!(!err.is_null());
    unsafe { dnglab_string_free(err) };
  }

  #[cfg(unix)]
  #[test]
  fn non_utf8_paths_are_accepted_as_bytes() {
    // Not valid UTF-8, but a legal Unix path: must be treated as "file not
    // found", not "invalid argument".
    let input = CString::new(b"/nonexistent/\xff\xfe.nef".to_vec()).unwrap();
    let mut err: *mut c_char = ptr::null_mut();
    let json = unsafe { dnglab_metadata_json(input.as_ptr(), &mut err) };
    assert!(json.is_null());
    let msg = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_string();
    assert!(msg.contains("File not found"), "{msg}");
    unsafe { dnglab_string_free(err) };
  }

  #[test]
  fn extensions_and_version() {
    let v = unsafe { CStr::from_ptr(dnglab_version()) }.to_str().unwrap();
    assert!(!v.is_empty());
    let exts = dnglab_supported_extensions();
    let s = unsafe { CStr::from_ptr(exts) }.to_str().unwrap().to_string();
    assert!(s.split(',').any(|e| e == "nef"));
    unsafe { dnglab_string_free(exts) };
    let nef = CString::new("NEF").unwrap();
    assert_eq!(unsafe { dnglab_is_supported_extension(nef.as_ptr()) }, 1);
    assert_eq!(unsafe { dnglab_is_supported_extension(ptr::null()) }, 0);
  }
}
