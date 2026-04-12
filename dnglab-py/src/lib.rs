// SPDX-License-Identifier: LGPL-2.1
// Python bindings for dnglab RAW-to-DNG conversion via PyO3.

use std::io::Cursor;
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use rawler::decoders::RawDecodeParams;
use rawler::dng::convert::{ConvertParams, convert_raw_file};
use rawler::dng::{CropMode, DngCompression, DngPhotometricConversion};
use rawler::rawsource::RawSource;

/// Convert a RAW file to DNG bytes in memory.
///
/// Parameters
/// ----------
/// raw_path : str
///     Path to the input RAW file.
/// embed_raw : bool, default False
///     Embed the original RAW data inside the DNG.
/// preview : bool, default False
///     Generate a JPEG preview inside the DNG.
/// thumbnail : bool, default False
///     Generate a thumbnail inside the DNG.
/// compression : str, default "lossless"
///     Compression mode: "lossless" or "uncompressed".
/// crop : str, default "best"
///     Crop mode: "best", "activearea", or "none".
/// artist : str or None, default None
///     Artist EXIF tag value.
/// dcp_dir : str or None, default None
///     Directory containing DCP color profiles.
/// index : int, default 0
///     Sub-image index for multi-image RAW files.
///
/// Returns
/// -------
/// bytes
///     The complete DNG file as a bytes object.
#[pyfunction]
#[pyo3(signature = (
    raw_path,
    embed_raw = false,
    preview = false,
    thumbnail = false,
    compression = "lossless",
    crop = "best",
    artist = None,
    dcp_dir = None,
    index = 0,
))]
fn convert_to_dng(
    py: Python<'_>,
    raw_path: &str,
    embed_raw: bool,
    preview: bool,
    thumbnail: bool,
    compression: &str,
    crop: &str,
    artist: Option<String>,
    dcp_dir: Option<&str>,
    index: usize,
) -> PyResult<Py<PyBytes>> {
    let compression = match compression {
        "lossless" => DngCompression::Lossless,
        "uncompressed" => DngCompression::Uncompressed,
        other => return Err(PyRuntimeError::new_err(format!("Unknown compression: {other}"))),
    };

    let crop = match crop {
        "best" => CropMode::Best,
        "activearea" => CropMode::ActiveArea,
        "none" => CropMode::None,
        other => return Err(PyRuntimeError::new_err(format!("Unknown crop mode: {other}"))),
    };

    let params = ConvertParams {
        embedded: embed_raw,
        compression,
        photometric_conversion: DngPhotometricConversion::Original,
        apply_scaling: false,
        crop,
        predictor: 1,
        preview,
        thumbnail,
        artist,
        software: "dnglab-py".into(),
        index,
        keep_mtime: false,
        dcp_dir: dcp_dir.map(PathBuf::from),
    };

    let raw = Path::new(raw_path);
    if !raw.exists() {
        return Err(PyRuntimeError::new_err(format!("File not found: {raw_path}")));
    }

    // Release the GIL during conversion so Python threads aren't blocked.
    let dng_bytes = py
        .detach(|| {
            let mut buf = Cursor::new(Vec::new());
            convert_raw_file(raw, &mut buf, &params)?;
            Ok::<Vec<u8>, rawler::RawlerError>(buf.into_inner())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Conversion failed: {e}")))?;

    Ok(PyBytes::new(py, &dng_bytes).into())
}

/// Check whether a file extension is supported by rawler.
///
/// Parameters
/// ----------
/// raw_path : str
///     Path (or just a filename with extension) to check.
///
/// Returns
/// -------
/// bool
#[pyfunction]
fn is_supported(raw_path: &str) -> bool {
    let ext = Path::new(raw_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_uppercase();
    rawler::decoders::supported_extensions().contains(&ext.as_str())
}

/// Return the list of supported camera make/model strings.
///
/// Each entry is "Make Model" (with optional mode suffix).
#[pyfunction]
fn supported_cameras() -> Vec<String> {
    let loader = rawler::global_loader();
    let cameras = loader.get_cameras();
    cameras
        .keys()
        .map(|(make, model, mode)| {
            if mode.is_empty() {
                format!("{make} {model}")
            } else {
                format!("{make} {model} ({mode})")
            }
        })
        .collect()
}

/// Extract metadata from a RAW file as a Python dict.
///
/// Returns camera make/model, EXIF fields (exposure, ISO, dates, lens,
/// GPS, orientation, etc.) and resolved lens description when available.
/// This is equivalent to the metadata that exiftool would extract.
///
/// Parameters
/// ----------
/// raw_path : str
///     Path to the RAW file.
///
/// Returns
/// -------
/// dict
///     Nested dict with keys: make, model, lens, rating, unique_image_id,
///     and exif (containing all standard EXIF/GPS sub-fields).
#[pyfunction]
fn raw_metadata(py: Python<'_>, raw_path: &str) -> PyResult<Py<PyDict>> {
    let raw = Path::new(raw_path);
    if !raw.exists() {
        return Err(PyRuntimeError::new_err(format!("File not found: {raw_path}")));
    }

    let md = py
        .detach(|| {
            let rawfile = RawSource::new(raw)?;
            let decoder = rawler::get_decoder(&rawfile)?;
            decoder.raw_metadata(&rawfile, &RawDecodeParams::default())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Metadata extraction failed: {e}")))?;

    let json_str = serde_json::to_string(&md)
        .map_err(|e| PyRuntimeError::new_err(format!("Serialization failed: {e}")))?;

    // Parse JSON string into a Python dict via the json stdlib module.
    let json_mod = py.import("json")?;
    let dict = json_mod.call_method1("loads", (&json_str,))?;
    Ok(dict.cast_into::<PyDict>()?.unbind())
}

/// Return the list of supported RAW file extensions (upper-case).
#[pyfunction]
fn supported_extensions() -> Vec<String> {
    rawler::decoders::supported_extensions()
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Extract the embedded JPEG preview from a RAW file without re-encoding.
///
/// This is the fastest way to get a preview — it returns the raw JPEG bytes
/// stored by the camera.  Returns ``None`` if the format has no embedded
/// JPEG (rare; mainly DNG files).  When ``max_dimension`` is given the
/// image is decoded, resized so its longest side fits, and re-encoded at
/// the requested ``quality``.
///
/// Parameters
/// ----------
/// raw_path : str
///     Path to the RAW file.
/// max_dimension : int or None, default None
///     If set, resize so the longest side is at most this many pixels.
/// quality : int, default 85
///     JPEG quality (1-100) used when resizing or when the embedded
///     preview is not available and a fallback must be generated.
///
/// Returns
/// -------
/// tuple[bytes, int, int] | None
///     ``(jpeg_bytes, width, height)`` or ``None`` if no preview exists
///     and the fallback pipeline also fails.
#[pyfunction]
#[pyo3(signature = (raw_path, max_dimension = None, quality = 85))]
fn extract_preview(
    py: Python<'_>,
    raw_path: &str,
    max_dimension: Option<u32>,
    quality: u8,
) -> PyResult<Option<(Py<PyBytes>, u32, u32)>> {
    let raw = Path::new(raw_path);
    if !raw.exists() {
        return Err(PyRuntimeError::new_err(format!("File not found: {raw_path}")));
    }

    let result = py
        .detach(|| -> Result<Option<(Vec<u8>, u32, u32)>, String> {
            let rawfile = RawSource::new(raw).map_err(|e| e.to_string())?;
            let decoder = rawler::get_decoder(&rawfile).map_err(|e| e.to_string())?;
            let params = RawDecodeParams::default();

            // Fast path: try to get embedded JPEG bytes without decoding.
            if max_dimension.is_none() {
                if let Some((jpeg_bytes, w, h)) =
                    decoder.preview_jpeg(&rawfile, &params).map_err(|e| e.to_string())?
                {
                    return Ok(Some((jpeg_bytes, w, h)));
                }
            }

            // Slow path: decode to pixels (preview → full → raw develop).
            let img = match decoder.preview_image(&rawfile, &params).map_err(|e| e.to_string())? {
                Some(img) => img,
                None => {
                    // Try the embedded JPEG as a decoded image fallback.
                    match decoder.preview_jpeg(&rawfile, &params).map_err(|e| e.to_string())? {
                        Some((jpeg_bytes, _, _)) => {
                            image::load_from_memory_with_format(&jpeg_bytes, image::ImageFormat::Jpeg)
                                .map_err(|e| e.to_string())?
                        }
                        None => return Ok(None),
                    }
                }
            };

            // Resize if requested.
            let img = match max_dimension {
                Some(max_dim) => {
                    let (w, h) = (img.width(), img.height());
                    if w > max_dim || h > max_dim {
                        img.resize(max_dim, max_dim, image::imageops::FilterType::Lanczos3)
                    } else {
                        img
                    }
                }
                None => img,
            };

            // Encode to JPEG at the requested quality.
            let (w, h) = (img.width(), img.height());
            let rgb = img.to_rgb8();
            let mut jpeg_buf = Cursor::new(Vec::new());
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(
                &mut jpeg_buf,
                quality,
            );
            encoder
                .encode_image(&rgb)
                .map_err(|e| e.to_string())?;
            Ok(Some((jpeg_buf.into_inner(), w, h)))
        })
        .map_err(|e| PyRuntimeError::new_err(format!("Preview extraction failed: {e}")))?;

    match result {
        Some((bytes, w, h)) => Ok(Some((PyBytes::new(py, &bytes).into(), w, h))),
        None => Ok(None),
    }
}

#[pymodule]
fn dnglab_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_to_dng, m)?)?;
    m.add_function(wrap_pyfunction!(is_supported, m)?)?;
    m.add_function(wrap_pyfunction!(supported_cameras, m)?)?;
    m.add_function(wrap_pyfunction!(supported_extensions, m)?)?;
    m.add_function(wrap_pyfunction!(raw_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(extract_preview, m)?)?;
    Ok(())
}
