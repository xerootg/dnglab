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
use rawler::formats::tiff::writer::{DirectoryWriter, TiffWriter};
use rawler::rawsource::RawSource;
use rawler::tags::TiffCommonTag;

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
        dcp_file: None,
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

/// Embed EXIF metadata from a RAW file into JPEG bytes.
///
/// Reads all EXIF/GPS metadata from the RAW file, builds a standards-compliant
/// EXIF APP1 segment, and splices it into the JPEG byte stream.
/// Orientation is overridden to the specified value (default 1, meaning
/// rotation is already baked into the pixels).
///
/// Parameters
/// ----------
/// raw_path : str
///     Path to the original RAW file (metadata source).
/// jpeg_bytes : bytes
///     The JPEG image data to embed EXIF into.
/// orientation : int, default 1
///     EXIF orientation value to write (1 = normal / already rotated).
///
/// Returns
/// -------
/// bytes
///     Modified JPEG bytes with EXIF APP1 segment embedded.
#[pyfunction]
#[pyo3(signature = (raw_path, jpeg_bytes, orientation = 1))]
fn embed_exif(
    py: Python<'_>,
    raw_path: &str,
    jpeg_bytes: &[u8],
    orientation: u16,
) -> PyResult<Py<PyBytes>> {
    let raw = Path::new(raw_path);
    if !raw.exists() {
        return Err(PyRuntimeError::new_err(format!("File not found: {raw_path}")));
    }

    let result = py
        .detach(|| -> Result<Vec<u8>, String> {
            // Extract metadata from the RAW file.
            let rawfile = RawSource::new(raw).map_err(|e| e.to_string())?;
            let decoder = rawler::get_decoder(&rawfile).map_err(|e| e.to_string())?;
            let mut md = decoder
                .raw_metadata(&rawfile, &RawDecodeParams::default())
                .map_err(|e| e.to_string())?;

            // Override orientation (rotation is baked into rendered pixels).
            md.exif.orientation = Some(orientation);

            // Strip MakerNotes — they're proprietary blobs often >100KB that
            // exceed the 64KB APP1 limit and aren't useful for display.
            md.exif.makernotes = None;

            // Build the TIFF/EXIF structure using the existing writer infrastructure.
            let mut tiff_buf = Cursor::new(Vec::new());
            let mut tiff = TiffWriter::new(&mut tiff_buf).map_err(|e| e.to_string())?;
            let mut root_ifd = DirectoryWriter::new();
            let mut exif_ifd = DirectoryWriter::new();

            // Write Make/Model to root IFD.
            if !md.make.is_empty() {
                root_ifd.add_tag(TiffCommonTag::Make, md.make.as_str());
            }
            if !md.model.is_empty() {
                root_ifd.add_tag(TiffCommonTag::Model, md.model.as_str());
            }

            // Write all EXIF tags (root IFD fields, Exif sub-IFD, GPS sub-IFD).
            md.write_exif_tags(&mut tiff, &mut root_ifd, &mut exif_ifd)
                .map_err(|e| e.to_string())?;

            // Build Exif sub-IFD first, then link from root.
            if exif_ifd.entry_count() > 0 {
                let exif_offset = exif_ifd.build(&mut tiff).map_err(|e| e.to_string())?;
                root_ifd.add_tag(TiffCommonTag::ExifIFDPointer, exif_offset);
            }

            // Finalize the TIFF structure.
            tiff.build(root_ifd).map_err(|e| e.to_string())?;
            let tiff_bytes = tiff_buf.into_inner();

            // Build the EXIF APP1 segment:
            //   FF E1  (APP1 marker)
            //   NN NN  (length: 2 + 6 + tiff_bytes.len())
            //   "Exif\x00\x00"  (6-byte header)
            //   <TIFF data>
            let exif_header = b"Exif\x00\x00";
            let app1_data_len = exif_header.len() + tiff_bytes.len();
            // APP1 length field includes itself (2 bytes) + data.
            let app1_length = (2 + app1_data_len) as u16;

            if app1_data_len + 2 > 0xFFFF {
                return Err("EXIF data too large for a single APP1 segment".into());
            }

            let mut app1 = Vec::with_capacity(2 + 2 + app1_data_len);
            app1.extend_from_slice(&[0xFF, 0xE1]); // APP1 marker
            app1.extend_from_slice(&app1_length.to_be_bytes()); // Length (big-endian)
            app1.extend_from_slice(exif_header);
            app1.extend_from_slice(&tiff_bytes);

            // Splice into the JPEG: insert after SOI (FF D8), replacing any
            // existing APP1 EXIF segment.
            let jpeg = jpeg_bytes;
            if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
                return Err("Input is not a valid JPEG (missing SOI marker)".into());
            }

            let mut output = Vec::with_capacity(jpeg.len() + app1.len());
            output.extend_from_slice(&jpeg[..2]); // SOI

            // Skip over any existing APP1 EXIF segments.
            let mut pos = 2;
            while pos + 4 <= jpeg.len() && jpeg[pos] == 0xFF {
                let marker = jpeg[pos + 1];
                // APP1 = 0xE1
                if marker == 0xE1 {
                    let seg_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
                    // Check if this APP1 is EXIF (starts with "Exif\0\0")
                    if pos + 4 + 6 <= jpeg.len() && &jpeg[pos + 4..pos + 4 + 4] == b"Exif" {
                        // Skip this segment entirely.
                        pos += 2 + seg_len;
                        continue;
                    }
                }
                break;
            }

            // Insert our new APP1 segment.
            output.extend_from_slice(&app1);
            // Append the rest of the JPEG.
            output.extend_from_slice(&jpeg[pos..]);

            Ok(output)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("EXIF embedding failed: {e}")))?;

    Ok(PyBytes::new(py, &result).into())
}

#[pymodule]
fn dnglab_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert_to_dng, m)?)?;
    m.add_function(wrap_pyfunction!(is_supported, m)?)?;
    m.add_function(wrap_pyfunction!(supported_cameras, m)?)?;
    m.add_function(wrap_pyfunction!(supported_extensions, m)?)?;
    m.add_function(wrap_pyfunction!(raw_metadata, m)?)?;
    m.add_function(wrap_pyfunction!(extract_preview, m)?)?;
    m.add_function(wrap_pyfunction!(embed_exif, m)?)?;
    Ok(())
}
