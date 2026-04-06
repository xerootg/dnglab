// SPDX-License-Identifier: LGPL-2.1
// Copyright 2025 Daniel Vogelbacher <daniel@chaospixel.com>

//! ISO-adaptive NoiseProfile calibration data for Nikon cameras.
//!
//! Calibration values derived from Adobe DNG Converter output across multiple
//! ISOs. Each entry is `(ISO, [S_r, O_r, S_g, O_g, S_b, O_b])` where S is
//! the shot noise scale and O is the read noise offset for each color plane.
//!
//! Interpolation between calibration points is performed in log-log space
//! (linear interpolation of ln(value) vs ln(ISO)), which naturally handles
//! the exponential scaling of noise with ISO.

/// Calibration entry: (ISO, [S_r, O_r, S_g, O_g, S_b, O_b])
type NoiseCal = (f64, [f64; 6]);

/// Nikon Z f noise calibration data from Adobe DNG Converter.
///
/// 19 ISOs spanning 100–16000. The sensor exhibits dual conversion gain (DCG)
/// with a transition around ISO 450–800, causing non-monotonic read noise
/// behavior that log-log interpolation handles naturally.
const NIKON_ZF_NOISE_CAL: &[NoiseCal] = &[
    (100.0,   [1.158e-05, 1.094e-08, 9.168e-06, 6.021e-09, 8.937e-06, 1.055e-08]),
    (220.0,   [2.556e-05, 2.701e-08, 2.023e-05, 2.867e-08, 1.946e-05, 2.641e-08]),
    (280.0,   [3.246e-05, 4.059e-08, 2.540e-05, 4.007e-08, 2.464e-05, 3.858e-08]),
    (360.0,   [4.167e-05, 6.299e-08, 3.229e-05, 5.823e-08, 3.155e-05, 5.837e-08]),
    (400.0,   [4.627e-05, 7.603e-08, 3.574e-05, 6.857e-08, 3.501e-05, 6.980e-08]),
    (450.0,   [5.230e-05, 6.842e-08, 4.050e-05, 6.062e-08, 3.946e-05, 6.266e-08]),
    (500.0,   [5.833e-05, 6.120e-08, 4.526e-05, 5.315e-08, 4.392e-05, 5.590e-08]),
    (560.0,   [6.556e-05, 5.308e-08, 5.097e-05, 4.484e-08, 4.927e-05, 4.829e-08]),
    (1000.0,  [1.155e-04, 3.837e-08, 9.230e-05, 2.844e-08, 9.000e-05, 3.451e-08]),
    (1100.0,  [1.261e-04, 4.522e-08, 1.015e-04, 3.409e-08, 9.965e-05, 4.085e-08]),
    (1250.0,  [1.419e-04, 5.654e-08, 1.154e-04, 4.354e-08, 1.141e-04, 5.136e-08]),
    (1400.0,  [1.576e-04, 6.913e-08, 1.293e-04, 5.413e-08, 1.286e-04, 6.307e-08]),
    (2000.0,  [2.238e-04, 1.295e-07, 1.833e-04, 9.919e-08, 1.843e-04, 1.199e-07]),
    (2800.0,  [3.141e-04, 2.367e-07, 2.544e-04, 1.726e-07, 2.569e-04, 2.218e-07]),
    (4000.0,  [4.442e-04, 4.528e-07, 3.626e-04, 3.300e-07, 3.681e-04, 4.222e-07]),
    (5600.0,  [6.140e-04, 8.442e-07, 5.078e-04, 6.272e-07, 5.177e-04, 7.790e-07]),
    (8000.0,  [8.712e-04, 1.631e-06, 7.234e-04, 1.193e-06, 7.363e-04, 1.498e-06]),
    (12800.0, [1.388e-03, 3.931e-06, 1.152e-03, 2.777e-06, 1.168e-03, 3.605e-06]),
    (16000.0, [1.721e-03, 5.226e-06, 1.435e-03, 3.699e-06, 1.447e-03, 4.885e-06]),
];

/// Compute a NoiseProfile for the given camera model and ISO.
///
/// Returns `Some([S_r, O_r, S_g, O_g, S_b, O_b])` if calibration data exists
/// for the model, or `None` for unsupported cameras.
pub fn noise_profile_for_iso(model: &str, iso: u32) -> Option<Vec<f64>> {
    let cal = match model {
        "NIKON Z f" => NIKON_ZF_NOISE_CAL,
        _ => return None,
    };
    Some(interpolate_noise_profile(cal, iso as f64).to_vec())
}

/// Log-log linear interpolation of noise profile values.
///
/// For ISOs below the first calibration point, uses the lowest entry.
/// For ISOs above the last calibration point, extrapolates from the last two entries.
/// Between calibration points, interpolates each of the 6 values independently
/// in ln(value) vs ln(ISO) space.
fn interpolate_noise_profile(cal: &[NoiseCal], iso: f64) -> [f64; 6] {
    let iso = iso.max(1.0); // guard against zero/negative
    let ln_iso = iso.ln();

    // Find the bracketing pair
    let idx = cal.iter().position(|(cal_iso, _)| *cal_iso >= iso);

    match idx {
        // ISO is at or below the first calibration point
        None => cal.last().map(|(_, vals)| *vals).unwrap_or([0.0; 6]),
        Some(0) => {
            if (cal[0].0 - iso).abs() < 0.5 {
                // Exact match at first point
                cal[0].1
            } else {
                // Below first cal point — clamp to first entry
                cal[0].1
            }
        }
        Some(i) => {
            let (iso_lo, vals_lo) = &cal[i - 1];
            let (iso_hi, vals_hi) = &cal[i];

            // Exact match check
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
        let np = noise_profile_for_iso("NIKON Z f", 100).unwrap();
        assert_eq!(np.len(), 6);
        assert!((np[0] - 1.158e-05).abs() < 1e-10);
        assert!((np[1] - 1.094e-08).abs() < 1e-13);
    }

    #[test]
    fn test_interpolated_iso() {
        let np = noise_profile_for_iso("NIKON Z f", 640).unwrap();
        assert_eq!(np.len(), 6);
        // Should be between ISO 560 and ISO 1000 values
        assert!(np[0] > 6.556e-05); // > S_r at ISO 560
        assert!(np[0] < 1.155e-04); // < S_r at ISO 1000
    }

    #[test]
    fn test_below_min_iso() {
        let np = noise_profile_for_iso("NIKON Z f", 50).unwrap();
        // Should clamp to ISO 100 values
        assert!((np[0] - 1.158e-05).abs() < 1e-10);
    }

    #[test]
    fn test_above_max_iso() {
        let np = noise_profile_for_iso("NIKON Z f", 25600).unwrap();
        // Should use last entry (clamped)
        assert!((np[0] - 1.721e-03).abs() < 1e-8);
    }

    #[test]
    fn test_unknown_model() {
        assert!(noise_profile_for_iso("NIKON D850", 100).is_none());
    }
}
