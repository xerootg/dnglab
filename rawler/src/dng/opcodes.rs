// SPDX-License-Identifier: LGPL-2.1
// Copyright 2024 RAW-Manager contributors

//! DNG OpcodeList binary encoder.
//!
//! Format (all big-endian):
//! ```text
//! u32: numOpcodes
//! for each opcode:
//!   u32: opcodeID
//!   u32: version  (4 bytes packed: major.minor.patch.build)
//!   u32: flags    (bit 0 = optional, bit 1 = skip_if_preview)
//!   u32: paramLength
//!   [paramLength bytes: opcode-specific parameters, big-endian]
//! ```
//!
//! Opcode IDs used here:
//!   1 = WarpRectilinear  (DNG 1.3)
//!   3 = FixVignetteRadial (DNG 1.3)

/// Version 1.3.0.0 packed as u32
const VERSION_1_3_0_0: u32 = 0x01_03_00_00;

/// Opcode is optional (readers may skip it)
pub const FLAG_OPTIONAL: u32 = 1;

fn write_u32_be(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn write_f64_be(buf: &mut Vec<u8>, v: f64) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn opcode_header(opcode_id: u32, flags: u32, params: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + params.len());
    write_u32_be(&mut buf, opcode_id);
    write_u32_be(&mut buf, VERSION_1_3_0_0);
    write_u32_be(&mut buf, flags);
    write_u32_be(&mut buf, params.len() as u32);
    buf.extend_from_slice(params);
    buf
}

/// Encode a **FixVignetteRadial** opcode (ID=3, DNG 1.3).
///
/// The opcode multiplies each pixel by:
/// `g(r) = 1 + k0·r² + k1·r⁴ + k2·r⁶ + k3·r⁸ + k4·r¹⁰`
/// where `r` is the normalised radius (1.0 at the image corner).
///
/// For Nikon Z-series files, the three VignetteCoefficient values map
/// directly to `k0`, `k1`, `k2` (positive values brighten edges to
/// compensate for lens fall-off).  `k3` and `k4` should be set to 0.
///
/// `cx` / `cy`: normalised centre coordinates (0.5 = image centre).
pub fn encode_fix_vignette_radial(k0: f64, k1: f64, k2: f64, k3: f64, k4: f64, cx: f64, cy: f64, flags: u32) -> Vec<u8> {
    let mut params = Vec::with_capacity(56);
    write_f64_be(&mut params, k0);
    write_f64_be(&mut params, k1);
    write_f64_be(&mut params, k2);
    write_f64_be(&mut params, k3);
    write_f64_be(&mut params, k4);
    write_f64_be(&mut params, cx);
    write_f64_be(&mut params, cy);
    opcode_header(3, flags, &params)
}

/// Encode a **WarpRectilinear** opcode (ID=1, DNG 1.3).
///
/// The opcode warps pixel coordinates using:
/// ```text
/// dx = (x - cx) / m
/// dy = (y - cy) / m
/// r  = sqrt(dx² + dy²)
/// x' = cx + m · (kr0 + kr1·r² + kr2·r⁴ + kr3·r⁶) · dx  +  tangential
/// y' = cy + m · (kr0 + kr1·r² + kr2·r⁴ + kr3·r⁶) · dy  +  tangential
/// ```
/// where `m` = half-diagonal of the image and `kr0 = 1.0` means no scaling.
///
/// For Nikon Z-series RadialDistortionCoefficient values (d1, d2, d3):
///   `kr0 = 1.0, kr1 = d1, kr2 = d2, kr3 = d3, kt0 = kt1 = 0`
///
/// # Panics
/// Panics if `kr.len() != num_planes` or `kt.len() != num_planes`.
pub fn encode_warp_rectilinear(kr: &[[f64; 4]], kt: &[[f64; 2]], cx: f64, cy: f64, flags: u32) -> Vec<u8> {
    assert_eq!(kr.len(), kt.len());
    let num_planes = kr.len() as u32;
    // 4 bytes numPlanes + num_planes * (4 + 2) * 8 bytes + 16 bytes center
    let param_size = 4 + num_planes as usize * 6 * 8 + 2 * 8;
    let mut params = Vec::with_capacity(param_size);
    write_u32_be(&mut params, num_planes);
    for i in 0..kr.len() {
        write_f64_be(&mut params, kr[i][0]);
        write_f64_be(&mut params, kr[i][1]);
        write_f64_be(&mut params, kr[i][2]);
        write_f64_be(&mut params, kr[i][3]);
        write_f64_be(&mut params, kt[i][0]);
        write_f64_be(&mut params, kt[i][1]);
    }
    write_f64_be(&mut params, cx);
    write_f64_be(&mut params, cy);
    opcode_header(1, flags, &params)
}

/// Encode a **FixBadPixelsList** opcode (ID=6, DNG 1.3).
///
/// Corrects known defective pixels and columns by interpolation.
/// `bayer_phase`: the Bayer mosaic phase (0-3) of the top-left pixel.
/// `bad_points`: list of individual bad pixel (row, column) pairs.
/// `bad_columns`: list of entirely defective column indices.
pub fn encode_fix_bad_pixels_list(bayer_phase: u32, bad_points: &[(u32, u32)], bad_columns: &[u32], flags: u32) -> Vec<u8> {
    let param_size = 4 + 4 + 4 + bad_points.len() * 8 + bad_columns.len() * 4;
    let mut params = Vec::with_capacity(param_size);
    write_u32_be(&mut params, bayer_phase);
    write_u32_be(&mut params, bad_points.len() as u32);
    write_u32_be(&mut params, bad_columns.len() as u32);
    for &(row, col) in bad_points {
        write_u32_be(&mut params, row);
        write_u32_be(&mut params, col);
    }
    for &col in bad_columns {
        write_u32_be(&mut params, col);
    }
    opcode_header(6, flags, &params)
}

/// Wrap one or more encoded opcodes into a complete OpcodeList blob.
///
/// The blob begins with a big-endian u32 count, followed by the
/// individual opcode bytes (each already including its own 16-byte
/// header and parameters).
pub fn encode_opcode_list(opcodes: &[Vec<u8>]) -> Vec<u8> {
    let total = 4 + opcodes.iter().map(|o| o.len()).sum::<usize>();
    let mut buf = Vec::with_capacity(total);
    write_u32_be(&mut buf, opcodes.len() as u32);
    for op in opcodes {
        buf.extend_from_slice(op);
    }
    buf
}
