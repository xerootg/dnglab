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

/// True if a `FixVignetteRadial` gain `g(r) = 1 + Σ kᵢ·r^(2(i+1))` stays positive
/// across the image radius `r ∈ [0, 1]` (`r = 1` = farthest corner). A valid
/// brightening correction is ≥ 1 everywhere; coefficients (from a camera
/// MakerNote or an LCP fit) that make the polynomial dip toward/below zero black
/// out the corners — a faithful renderer clamps negative gain to 0. Callers skip
/// the opcode when this is false (render uncorrected rather than render black).
pub fn vignette_gain_valid(k: &[f64; 5]) -> bool {
    let mut r = 0.0_f64;
    while r <= 1.0001 {
        let r2 = r * r;
        let g = 1.0 + k[0] * r2 + k[1] * r2.powi(2) + k[2] * r2.powi(3) + k[3] * r2.powi(4) + k[4] * r2.powi(5);
        if g < 0.5 {
            return false;
        }
        r += 0.05;
    }
    true
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
    // (validity is the caller's responsibility — see `vignette_gain_valid`)
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Read a big-endian u32 from a byte slice at the given offset.
    fn read_u32_be(buf: &[u8], off: usize) -> u32 {
        u32::from_be_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
    }

    /// Read a big-endian f64 from a byte slice at the given offset.
    fn read_f64_be(buf: &[u8], off: usize) -> f64 {
        f64::from_be_bytes([
            buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
            buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7],
        ])
    }

    // ── OpcodeList envelope ─────────────────────────────────────────────

    #[test]
    fn opcode_list_empty() {
        let blob = encode_opcode_list(&[]);
        assert_eq!(blob.len(), 4);
        assert_eq!(read_u32_be(&blob, 0), 0, "empty list count must be 0");
    }

    #[test]
    fn opcode_list_count_matches_entries() {
        let op1 = encode_fix_vignette_radial(0.1, 0.2, 0.3, 0.0, 0.0, 0.5, 0.5, FLAG_OPTIONAL);
        let op2 = encode_warp_rectilinear(
            &[[1.0, -0.01, 0.002, 0.0]],
            &[[0.0, 0.0]],
            0.5, 0.5, FLAG_OPTIONAL,
        );
        let blob = encode_opcode_list(&[op1.clone(), op2.clone()]);
        assert_eq!(read_u32_be(&blob, 0), 2, "count must equal number of opcodes");
        assert_eq!(blob.len(), 4 + op1.len() + op2.len());
    }

    // ── Opcode header structure (DNG spec §7) ───────────────────────────

    /// Verify the 16-byte opcode header: opcodeID, version, flags, paramLength.
    fn assert_opcode_header(buf: &[u8], expected_id: u32, expected_flags: u32) {
        assert!(buf.len() >= 16, "opcode must be at least 16 bytes");
        let id = read_u32_be(buf, 0);
        let version = read_u32_be(buf, 4);
        let flags = read_u32_be(buf, 8);
        let param_len = read_u32_be(buf, 12);
        assert_eq!(id, expected_id, "opcode ID mismatch");
        assert_eq!(version, 0x01_03_00_00, "version must be DNG 1.3.0.0");
        assert_eq!(flags, expected_flags, "flags mismatch");
        assert_eq!(
            param_len as usize,
            buf.len() - 16,
            "paramLength must equal remaining bytes after header"
        );
    }

    // ── WarpRectilinear (opcode ID 1) ───────────────────────────────────

    #[test]
    fn warp_rectilinear_single_plane_header() {
        let kr = [[1.0_f64, -0.01, 0.002, -0.0003]];
        let kt = [[0.0_f64, 0.0]];
        let opcode = encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, FLAG_OPTIONAL);
        assert_opcode_header(&opcode, 1, FLAG_OPTIONAL);
    }

    #[test]
    fn warp_rectilinear_single_plane_layout() {
        // DNG spec §7 WarpRectilinear layout:
        //   header(16) + u32 numPlanes(4) + 1 plane × 6 × f64(48) + 2 × f64 center(16) = 84
        let kr = [[1.0, -0.05, 0.003, 0.0]];
        let kt = [[0.0, 0.0]];
        let cx = 0.5;
        let cy = 0.5;
        let opcode = encode_warp_rectilinear(&kr, &kt, cx, cy, 0);
        let params = &opcode[16..]; // skip header

        // numPlanes
        assert_eq!(read_u32_be(params, 0), 1);

        // Plane 0: kr0, kr1, kr2, kr3, kt0, kt1
        let off = 4; // after numPlanes
        assert_eq!(read_f64_be(params, off), 1.0);       // kr0
        assert_eq!(read_f64_be(params, off + 8), -0.05);  // kr1
        assert_eq!(read_f64_be(params, off + 16), 0.003); // kr2
        assert_eq!(read_f64_be(params, off + 24), 0.0);   // kr3
        assert_eq!(read_f64_be(params, off + 32), 0.0);   // kt0
        assert_eq!(read_f64_be(params, off + 40), 0.0);   // kt1

        // Center (after all plane data)
        let center_off = 4 + 6 * 8;
        assert_eq!(read_f64_be(params, center_off), cx);       // ĉx
        assert_eq!(read_f64_be(params, center_off + 8), cy);   // ĉy

        // Total: header(16) + numPlanes(4) + 1×6×8(48) + 2×8(16) = 84
        assert_eq!(opcode.len(), 84);
    }

    #[test]
    fn warp_rectilinear_identity_is_no_op() {
        // Per DNG spec: kr0=1, kr1=kr2=kr3=0, kt0=kt1=0 → identity warp
        let opcode = encode_warp_rectilinear(
            &[[1.0, 0.0, 0.0, 0.0]],
            &[[0.0, 0.0]],
            0.5, 0.5, 0,
        );
        let params = &opcode[16..];
        let off = 4;
        assert_eq!(read_f64_be(params, off), 1.0, "kr0 must be 1 for identity");
        assert_eq!(read_f64_be(params, off + 8), 0.0, "kr1 must be 0 for identity");
        assert_eq!(read_f64_be(params, off + 16), 0.0, "kr2 must be 0 for identity");
        assert_eq!(read_f64_be(params, off + 24), 0.0, "kr3 must be 0 for identity");
    }

    #[test]
    fn warp_rectilinear_three_planes_tca() {
        // 3-plane WarpRectilinear encodes per-channel TCA correction
        let kr = [
            [1.0005, -0.012, 0.001, 0.0],   // R plane
            [1.0,    -0.010, 0.001, 0.0],    // G plane (reference)
            [0.9995, -0.008, 0.001, 0.0],    // B plane
        ];
        let kt = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let opcode = encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, FLAG_OPTIONAL);
        let params = &opcode[16..];

        assert_eq!(read_u32_be(params, 0), 3, "numPlanes must be 3 for TCA");

        // Verify each plane's kr0
        let plane_size = 6 * 8; // 6 coefficients × 8 bytes each
        for (i, expected_kr0) in [1.0005, 1.0, 0.9995].iter().enumerate() {
            let off = 4 + i * plane_size;
            let kr0 = read_f64_be(params, off);
            assert!(
                (kr0 - expected_kr0).abs() < 1e-10,
                "plane {} kr0 mismatch: got {} expected {}", i, kr0, expected_kr0
            );
        }

        // Center at end (after 3 planes × 6 × 8 = 144 bytes + 4 numPlanes)
        let center_off = 4 + 3 * plane_size;
        assert_eq!(read_f64_be(params, center_off), 0.5);
        assert_eq!(read_f64_be(params, center_off + 8), 0.5);

        // Total: header(16) + numPlanes(4) + 3×6×8(144) + 2×8(16) = 180
        assert_eq!(opcode.len(), 180);
    }

    #[test]
    fn warp_rectilinear_nikon_style_coefficients() {
        // Nikon Z-series: kr0=1.0, kr1=d1, kr2=d2, kr3=d3, kt=0
        // Typical barrel distortion values from a wide-angle Z lens
        let d1 = -0.0234;
        let d2 = 0.0056;
        let d3 = -0.0012;
        let opcode = encode_warp_rectilinear(
            &[[1.0, d1, d2, d3]],
            &[[0.0, 0.0]],
            0.5, 0.5, FLAG_OPTIONAL,
        );
        let params = &opcode[16..];
        let off = 4;

        assert_eq!(read_f64_be(params, off), 1.0, "Nikon kr0 must be 1.0");
        assert_eq!(read_f64_be(params, off + 8), d1);
        assert_eq!(read_f64_be(params, off + 16), d2);
        assert_eq!(read_f64_be(params, off + 24), d3);
        assert_eq!(read_f64_be(params, off + 32), 0.0, "Nikon kt0 must be 0");
        assert_eq!(read_f64_be(params, off + 40), 0.0, "Nikon kt1 must be 0");
    }

    #[test]
    fn warp_rectilinear_off_center() {
        let cx = 0.48;
        let cy = 0.52;
        let opcode = encode_warp_rectilinear(
            &[[1.0, -0.01, 0.0, 0.0]],
            &[[0.0, 0.0]],
            cx, cy, 0,
        );
        let params = &opcode[16..];
        let center_off = 4 + 6 * 8;
        assert_eq!(read_f64_be(params, center_off), cx);
        assert_eq!(read_f64_be(params, center_off + 8), cy);
    }

    #[test]
    fn warp_rectilinear_with_tangential() {
        // Tangential distortion is rare but the spec supports it
        let kt0 = 0.0001;
        let kt1 = -0.0002;
        let opcode = encode_warp_rectilinear(
            &[[1.0, -0.01, 0.0, 0.0]],
            &[[kt0, kt1]],
            0.5, 0.5, 0,
        );
        let params = &opcode[16..];
        let off = 4;
        assert_eq!(read_f64_be(params, off + 32), kt0, "kt0");
        assert_eq!(read_f64_be(params, off + 40), kt1, "kt1");
    }

    #[test]
    #[should_panic]
    fn warp_rectilinear_mismatched_planes_panics() {
        // kr and kt must have the same number of planes
        encode_warp_rectilinear(
            &[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            &[[0.0, 0.0]],
            0.5, 0.5, 0,
        );
    }

    // ── FixVignetteRadial (opcode ID 3) ─────────────────────────────────

    #[test]
    fn fix_vignette_radial_header() {
        let opcode = encode_fix_vignette_radial(0.1, 0.2, 0.3, 0.0, 0.0, 0.5, 0.5, FLAG_OPTIONAL);
        assert_opcode_header(&opcode, 3, FLAG_OPTIONAL);
    }

    #[test]
    fn fix_vignette_radial_layout() {
        // DNG spec §7 FixVignetteRadial: 7 × f64 = 56 bytes of parameters
        // g(r) = 1 + k0·r² + k1·r⁴ + k2·r⁶ + k3·r⁸ + k4·r¹⁰
        let k0 = 0.15;
        let k1 = -0.08;
        let k2 = 0.02;
        let k3 = 0.0;
        let k4 = 0.0;
        let cx = 0.5;
        let cy = 0.5;
        let opcode = encode_fix_vignette_radial(k0, k1, k2, k3, k4, cx, cy, 0);
        let params = &opcode[16..];

        assert_eq!(params.len(), 56, "FixVignetteRadial params must be 7 × f64 = 56 bytes");
        assert_eq!(read_f64_be(params, 0), k0);
        assert_eq!(read_f64_be(params, 8), k1);
        assert_eq!(read_f64_be(params, 16), k2);
        assert_eq!(read_f64_be(params, 24), k3);
        assert_eq!(read_f64_be(params, 32), k4);
        assert_eq!(read_f64_be(params, 40), cx);
        assert_eq!(read_f64_be(params, 48), cy);

        // Total: header(16) + params(56) = 72
        assert_eq!(opcode.len(), 72);
    }

    #[test]
    fn fix_vignette_radial_identity_all_zeros() {
        // Per DNG spec: if all k terms are zero, gain function is identity (g=1)
        let opcode = encode_fix_vignette_radial(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0);
        let params = &opcode[16..];
        for i in 0..5 {
            assert_eq!(
                read_f64_be(params, i * 8), 0.0,
                "k{} must be 0 for identity vignette", i
            );
        }
    }

    #[test]
    fn fix_vignette_radial_nikon_style() {
        // Nikon Z-series: three VignetteCoefficients → k0, k1, k2; k3=k4=0
        let k0 = 0.25;
        let k1 = -0.12;
        let k2 = 0.04;
        let opcode = encode_fix_vignette_radial(k0, k1, k2, 0.0, 0.0, 0.5, 0.5, FLAG_OPTIONAL);
        let params = &opcode[16..];

        assert_eq!(read_f64_be(params, 0), k0);
        assert_eq!(read_f64_be(params, 8), k1);
        assert_eq!(read_f64_be(params, 16), k2);
        assert_eq!(read_f64_be(params, 24), 0.0, "Nikon k3 must be 0");
        assert_eq!(read_f64_be(params, 32), 0.0, "Nikon k4 must be 0");
    }

    // ── FixBadPixelsList (opcode ID 6) ──────────────────────────────────

    #[test]
    fn fix_bad_pixels_list_header() {
        let opcode = encode_fix_bad_pixels_list(0, &[(10, 20)], &[5], FLAG_OPTIONAL);
        assert_opcode_header(&opcode, 6, FLAG_OPTIONAL);
    }

    #[test]
    fn fix_bad_pixels_list_layout() {
        let bad_points = vec![(100, 200), (300, 400)];
        let bad_columns = vec![50, 75];
        let bayer_phase = 1;
        let opcode = encode_fix_bad_pixels_list(bayer_phase, &bad_points, &bad_columns, 0);
        let params = &opcode[16..];

        // Layout: u32 bayerPhase + u32 badPointCount + u32 badColumnCount
        //       + badPointCount × (u32 row + u32 col) + badColumnCount × u32 col
        assert_eq!(read_u32_be(params, 0), bayer_phase);
        assert_eq!(read_u32_be(params, 4), 2, "badPointCount");
        assert_eq!(read_u32_be(params, 8), 2, "badColumnCount");

        // Bad points
        assert_eq!(read_u32_be(params, 12), 100); // point 0 row
        assert_eq!(read_u32_be(params, 16), 200); // point 0 col
        assert_eq!(read_u32_be(params, 20), 300); // point 1 row
        assert_eq!(read_u32_be(params, 24), 400); // point 1 col

        // Bad columns
        assert_eq!(read_u32_be(params, 28), 50);
        assert_eq!(read_u32_be(params, 32), 75);

        // Total: header(16) + 3×4(12) + 2×8(16) + 2×4(8) = 52
        assert_eq!(opcode.len(), 52);
    }

    #[test]
    fn fix_bad_pixels_list_empty() {
        let opcode = encode_fix_bad_pixels_list(0, &[], &[], 0);
        let params = &opcode[16..];

        assert_eq!(read_u32_be(params, 0), 0, "bayer phase");
        assert_eq!(read_u32_be(params, 4), 0, "no bad points");
        assert_eq!(read_u32_be(params, 8), 0, "no bad columns");
        assert_eq!(params.len(), 12);
    }

    // ── Roundtrip: encode then parse as a consumer would ────────────────

    #[test]
    fn roundtrip_opcode_list_with_all_types() {
        // Build a realistic OpcodeList3 containing WarpRectilinear,
        // then an OpcodeList1 containing FixVignetteRadial — wrap each
        // into an OpcodeList blob and verify the consumer can walk it.
        let kr = [1.0_f64, -0.015, 0.004, -0.0008];
        let cx = 0.5;
        let cy = 0.5;
        let warp = encode_warp_rectilinear(&[kr], &[[0.0, 0.0]], cx, cy, FLAG_OPTIONAL);
        let vig_k = [0.2, -0.1, 0.03, 0.0, 0.0];
        let vig = encode_fix_vignette_radial(vig_k[0], vig_k[1], vig_k[2], vig_k[3], vig_k[4], cx, cy, FLAG_OPTIONAL);
        let bad = encode_fix_bad_pixels_list(0, &[(10, 20)], &[], 0);

        let blob = encode_opcode_list(&[warp, vig, bad]);

        // Walk the blob like a DNG reader would (per DNG spec §7)
        let count = read_u32_be(&blob, 0);
        assert_eq!(count, 3);

        let mut pos = 4usize;
        let mut found_ids = Vec::new();
        for _ in 0..count {
            let opcode_id = read_u32_be(&blob, pos);
            let _version = read_u32_be(&blob, pos + 4);
            let _flags = read_u32_be(&blob, pos + 8);
            let param_len = read_u32_be(&blob, pos + 12) as usize;

            found_ids.push(opcode_id);
            pos += 16 + param_len;
        }
        assert_eq!(pos, blob.len(), "consumed all bytes");
        assert_eq!(found_ids, vec![1, 3, 6], "opcodes in order: WarpRectilinear, FixVignetteRadial, FixBadPixelsList");
    }

    #[test]
    fn roundtrip_warp_rectilinear_coefficients() {
        // Encode and then read back exact coefficient values
        let kr = [1.0_f64, -0.0234, 0.0056, -0.0012];
        let kt = [0.0001_f64, -0.0002];
        let cx = 0.48;
        let cy = 0.52;

        let opcode = encode_warp_rectilinear(&[kr], &[kt], cx, cy, 0);
        let blob = encode_opcode_list(&[opcode]);

        // Skip list header (4) and opcode header (16)
        let params = &blob[4 + 16..];
        let num_planes = read_u32_be(params, 0);
        assert_eq!(num_planes, 1);

        let off = 4;
        let got_kr = [
            read_f64_be(params, off),
            read_f64_be(params, off + 8),
            read_f64_be(params, off + 16),
            read_f64_be(params, off + 24),
        ];
        let got_kt = [
            read_f64_be(params, off + 32),
            read_f64_be(params, off + 40),
        ];
        let center_off = 4 + 6 * 8;
        let got_cx = read_f64_be(params, center_off);
        let got_cy = read_f64_be(params, center_off + 8);

        assert_eq!(got_kr, kr, "radial coefficients must roundtrip exactly");
        assert_eq!(got_kt, kt, "tangential coefficients must roundtrip exactly");
        assert_eq!(got_cx, cx, "center x must roundtrip exactly");
        assert_eq!(got_cy, cy, "center y must roundtrip exactly");
    }

    #[test]
    fn roundtrip_vignette_coefficients() {
        let k = [0.25_f64, -0.12, 0.04, -0.005, 0.001];
        let cx = 0.49;
        let cy = 0.51;

        let opcode = encode_fix_vignette_radial(k[0], k[1], k[2], k[3], k[4], cx, cy, FLAG_OPTIONAL);
        let blob = encode_opcode_list(&[opcode]);

        let params = &blob[4 + 16..];
        let got_k: Vec<f64> = (0..5).map(|i| read_f64_be(params, i * 8)).collect();
        let got_cx = read_f64_be(params, 40);
        let got_cy = read_f64_be(params, 48);

        assert_eq!(got_k, k.to_vec(), "vignette coefficients must roundtrip exactly");
        assert_eq!(got_cx, cx);
        assert_eq!(got_cy, cy);
    }

    // ── Flag encoding ───────────────────────────────────────────────────

    #[test]
    fn flags_optional_bit() {
        let opcode = encode_warp_rectilinear(&[[1.0, 0.0, 0.0, 0.0]], &[[0.0, 0.0]], 0.5, 0.5, FLAG_OPTIONAL);
        assert_eq!(read_u32_be(&opcode, 8), 1, "bit 0 = optional");
    }

    #[test]
    fn flags_zero_means_required() {
        let opcode = encode_warp_rectilinear(&[[1.0, 0.0, 0.0, 0.0]], &[[0.0, 0.0]], 0.5, 0.5, 0);
        assert_eq!(read_u32_be(&opcode, 8), 0, "flags=0 means opcode is required");
    }

    #[test]
    fn flags_skip_preview_bit() {
        // Bit 1 = skip when doing preview quality processing
        let opcode = encode_fix_vignette_radial(0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 2);
        assert_eq!(read_u32_be(&opcode, 8), 2, "bit 1 = skip_if_preview");
    }

    #[test]
    fn flags_combined_optional_and_skip_preview() {
        let opcode = encode_fix_vignette_radial(0.1, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, FLAG_OPTIONAL | 2);
        assert_eq!(read_u32_be(&opcode, 8), 3, "optional + skip_if_preview");
    }

    // ── Big-endian encoding sanity checks ───────────────────────────────

    #[test]
    fn endianness_u32_is_big_endian() {
        let opcode = encode_warp_rectilinear(&[[1.0, 0.0, 0.0, 0.0]], &[[0.0, 0.0]], 0.5, 0.5, 0);
        // Opcode ID 1 in big-endian: 0x00 0x00 0x00 0x01
        assert_eq!(opcode[0..4], [0x00, 0x00, 0x00, 0x01]);
        // Version 1.3.0.0 in big-endian: 0x01 0x03 0x00 0x00
        assert_eq!(opcode[4..8], [0x01, 0x03, 0x00, 0x00]);
    }

    #[test]
    fn endianness_f64_is_big_endian() {
        // IEEE 754 f64 1.0 is 0x3FF0000000000000 in big-endian
        let opcode = encode_warp_rectilinear(&[[1.0, 0.0, 0.0, 0.0]], &[[0.0, 0.0]], 0.5, 0.5, 0);
        // kr0=1.0 starts at offset 20 (header=16, numPlanes=4)
        assert_eq!(
            opcode[20..28],
            [0x3F, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            "f64 1.0 must be big-endian IEEE 754"
        );
    }

    // ── DNG SDK cross-reference ─────────────────────────────────────────
    // These tests verify byte-level compatibility with the DNG SDK's
    // dng_opcode_WarpRectilinear and dng_opcode_FixVignetteRadial
    // by checking sizes match the spec formula.

    #[test]
    fn warp_rectilinear_size_formula() {
        // Per DNG spec: param size = 4 + N × 6 × 8 + 16
        for num_planes in 1..=3 {
            let kr: Vec<[f64; 4]> = vec![[1.0, 0.0, 0.0, 0.0]; num_planes];
            let kt: Vec<[f64; 2]> = vec![[0.0, 0.0]; num_planes];
            let opcode = encode_warp_rectilinear(&kr, &kt, 0.5, 0.5, 0);
            let expected_params = 4 + num_planes * 6 * 8 + 16;
            let expected_total = 16 + expected_params;
            assert_eq!(
                opcode.len(), expected_total,
                "size mismatch for {} plane(s)", num_planes
            );
        }
    }

    #[test]
    fn fix_vignette_radial_size_is_fixed() {
        // Per DNG spec: always 7 × f64 = 56 bytes of parameters
        let opcode = encode_fix_vignette_radial(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0);
        assert_eq!(opcode.len(), 16 + 56);
    }
}
