// SPDX-License-Identifier: LGPL-2.1
// Copyright 2026 xerootg <4009802+xerootg@users.noreply.github.com>

//! ISO-adaptive NoiseProfile calibration data for Olympus/OM Digital Solutions cameras.
//!
//! Calibration values derived from Adobe DNG Converter reference output and
//! extrapolated using the standard CMOS noise model:
//!   S(ISO) = S_base × (ISO / base_ISO)     — shot noise scales linearly with gain
//!   O(ISO) = O_base × (ISO / base_ISO)²    — read noise scales with gain squared
//!
//! Each entry is `(ISO, [S_r, O_r, S_g, O_g, S_b, O_b])` where S is
//! the shot noise scale and O is the read noise offset for each color plane.
//!
//! Interpolation between calibration points is performed in log-log space
//! (linear interpolation of ln(value) vs ln(ISO)), matching the physical
//! scaling behavior of sensor noise.

/// Calibration entry: (ISO, [S_r, O_r, S_g, O_g, S_b, O_b])
type NoiseCal = (f64, [f64; 6]);

// ---------------------------------------------------------------------------
// OM-5 / OM-5 Mark II — 20.4 MP Live MOS (shared sensor platform)
//
// Reference: Adobe DNG Converter 18.2 output for OM-5 at ISO 1600.
// Base ISO 200. Calibration points generated via standard CMOS noise model
// and anchored to the Adobe reference at ISO 1600.
// ---------------------------------------------------------------------------
const OM5_NOISE_CAL: &[NoiseCal] = &[
    (  200.0, [4.023594e-05, 1.559536e-08, 4.057279e-05, 1.431850e-08, 3.323053e-05, 1.462363e-08]),
    (  400.0, [8.047189e-05, 6.238145e-08, 8.114557e-05, 5.727401e-08, 6.646105e-05, 5.849452e-08]),
    (  800.0, [1.609438e-04, 2.495258e-07, 1.622911e-04, 2.290960e-07, 1.329221e-04, 2.339781e-07]),
    ( 1600.0, [3.218875e-04, 9.981033e-07, 3.245823e-04, 9.163842e-07, 2.658442e-04, 9.359123e-07]), // Adobe reference
    ( 3200.0, [6.437751e-04, 3.992413e-06, 6.491646e-04, 3.665537e-06, 5.316884e-04, 3.743649e-06]),
    ( 6400.0, [1.287550e-03, 1.596965e-05, 1.298329e-03, 1.466215e-05, 1.063377e-03, 1.497460e-05]),
    (12800.0, [2.575100e-03, 6.387861e-05, 2.596658e-03, 5.864859e-05, 2.126754e-03, 5.989839e-05]),
    (25600.0, [5.150201e-03, 2.555144e-04, 5.193317e-03, 2.345944e-04, 4.253507e-03, 2.395935e-04]),
];

/// Compute a NoiseProfile for the given camera model and ISO.
///
/// Returns `Some([S_r, O_r, S_g, O_g, S_b, O_b])` if calibration data exists
/// for the model, or `None` for unsupported cameras.
pub fn noise_profile_for_iso(model: &str, iso: u32) -> Option<Vec<f64>> {
    let cal = match model {
        "OM-5" | "OM-5 Mark II" | "OM-5MarkII" => OM5_NOISE_CAL,
        _ => return None,
    };
    Some(interpolate_noise_profile(cal, iso as f64).to_vec())
}

/// Log-log linear interpolation of noise profile values.
///
/// For ISOs below the first calibration point, clamps to the lowest entry.
/// For ISOs above the last calibration point, clamps to the highest entry.
/// Between calibration points, interpolates each of the 6 values independently
/// in ln(value) vs ln(ISO) space.
fn interpolate_noise_profile(cal: &[NoiseCal], iso: f64) -> [f64; 6] {
    let iso = iso.max(1.0);
    let ln_iso = iso.ln();

    let idx = cal.iter().position(|(cal_iso, _)| *cal_iso >= iso);

    match idx {
        None => cal.last().map(|(_, vals)| *vals).unwrap_or([0.0; 6]),
        Some(0) => {
            if (cal[0].0 - iso).abs() < 0.5 {
                cal[0].1
            } else {
                cal[0].1
            }
        }
        Some(i) => {
            let (iso_lo, vals_lo) = &cal[i - 1];
            let (iso_hi, vals_hi) = &cal[i];

            if (iso_hi - iso).abs() < 0.5 {
                return *vals_hi;
            }
            if (iso_lo - iso).abs() < 0.5 {
                return *vals_lo;
            }

            let ln_lo = iso_lo.ln();
            let ln_hi = iso_hi.ln();
            let t = (ln_iso - ln_lo) / (ln_hi - ln_lo);

            let mut result = [0.0_f64; 6];
            for j in 0..6 {
                let ln_v_lo = vals_lo[j].ln();
                let ln_v_hi = vals_hi[j].ln();
                result[j] = (ln_v_lo + t * (ln_v_hi - ln_v_lo)).exp();
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_iso_match() {
        // ISO 1600 should match the Adobe reference exactly
        let np = noise_profile_for_iso("OM-5", 1600).unwrap();
        assert_eq!(np.len(), 6);
        assert!((np[0] - 3.218875e-04).abs() < 1e-10);
        assert!((np[1] - 9.981033e-07).abs() < 1e-13);
    }

    #[test]
    fn test_interpolated_iso() {
        let np = noise_profile_for_iso("OM-5", 1000).unwrap();
        assert_eq!(np.len(), 6);
        // Should be between ISO 800 and ISO 1600 values
        assert!(np[0] > 1.609438e-04); // > S_r at ISO 800
        assert!(np[0] < 3.218875e-04); // < S_r at ISO 1600
    }

    #[test]
    fn test_below_min_iso() {
        let np = noise_profile_for_iso("OM-5", 100).unwrap();
        // Should clamp to ISO 200 values
        assert!((np[0] - 4.023594e-05).abs() < 1e-10);
    }

    #[test]
    fn test_above_max_iso() {
        let np = noise_profile_for_iso("OM-5", 51200).unwrap();
        // Should clamp to ISO 25600 values
        assert!((np[0] - 5.150201e-03).abs() < 1e-09);
    }

    #[test]
    fn test_om5_mark2_shares_data() {
        let np5 = noise_profile_for_iso("OM-5", 1600).unwrap();
        let np5m2 = noise_profile_for_iso("OM-5MarkII", 1600).unwrap();
        assert_eq!(np5, np5m2);
    }

    #[test]
    fn test_unknown_model() {
        assert!(noise_profile_for_iso("E-PL1", 100).is_none());
    }
}
