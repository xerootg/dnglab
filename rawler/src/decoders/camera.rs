use toml::Value;

use crate::CFA;
use crate::cfa::PlaneColor;
use crate::imgop::xyz::FlatColorMatrix;
use crate::imgop::xyz::Illuminant;

use std::collections::HashMap;

use super::BlackLevel;
use super::WhiteLevel;

/// Contains sanitized information about the raw image's properties
#[derive(Debug, Clone, Default)]
pub struct Camera {
  pub make: String,
  pub model: String,
  pub mode: String,
  pub clean_make: String,
  pub clean_model: String,
  pub remark: Option<String>,
  pub filesize: usize,
  pub raw_width: usize,
  pub raw_height: usize,
  //pub orientation: Orientation,
  pub whitelevel: Option<Vec<u32>>,
  pub blacklevel: Option<Vec<u32>>,
  pub blackareah: Option<(usize, usize)>,
  pub blackareav: Option<(usize, usize)>,
  pub xyz_to_cam: [[f32; 3]; 4],
  pub color_matrix: HashMap<Illuminant, FlatColorMatrix>,
  pub forward_matrix: HashMap<Illuminant, FlatColorMatrix>,
  pub cfa: CFA,
  pub plane_color: PlaneColor,
  // Active area relative to sensor size
  pub active_area: Option<[usize; 4]>,
  // Recommended area relative to sensor size
  pub crop_area: Option<[usize; 4]>,
  // Hint/Replacement for EXIF BITDEPTH info
  pub bps: Option<usize>,
  // The BPS of the output after decoding
  pub real_bps: usize,
  pub highres_width: usize,
  pub default_scale: DefaultScale,
  pub best_quality_scale: BestQualityScale,
  pub hints: Vec<String>,
  pub params: HashMap<String, Value>,
  pub baseline_exposure: Option<f32>,
  pub baseline_noise: Option<f32>,
  pub baseline_sharpness: Option<f32>,
  pub linear_response_limit: Option<f32>,
  /// NoiseProfile per DNG spec: pairs of (noise_scale, noise_offset) per color plane.
  /// For a 3-plane sensor: [s0, o0, s1, o1, s2, o2] (6 doubles).
  pub noise_profile: Option<Vec<f64>>,
  /// AntiAliasStrength: 0.0 = no AA filter, 1.0 = standard AA filter.
  pub anti_alias_strength: Option<f32>,
  /// BayerGreenSplit: green channel inter-pixel variance (0 = best, higher = worse).
  pub bayer_green_split: Option<u32>,
  /// DefaultUserCrop: [top, left, bottom, right] as fractions of the active area.
  pub default_user_crop: Option<[f64; 4]>,
  /// CameraCalibration: 3x3 diagonal matrix as [r, g, b] scaling factors.
  /// Written as both CameraCalibration1 and CameraCalibration2 DNG tags.
  pub camera_calibration: Option<[f64; 3]>,
  /// Crop factor relative to 35mm full-frame (diagonal ratio).
  /// Used to compute FocalLengthIn35mmFormat and FocalPlane resolution tags.
  pub crop_factor: Option<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DefaultScale(pub [[u32; 2]; 2]);

impl Default for DefaultScale {
  fn default() -> Self {
    Self([[1, 1], [1, 1]])
  }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BestQualityScale(pub [u32; 2]);

impl Default for BestQualityScale {
  fn default() -> Self {
    Self([1, 1])
  }
}

impl Camera {
  pub fn find_hint(&self, hint: &str) -> bool {
    self.hints.contains(&hint.to_string())
  }

  pub fn param_usize(&self, name: &str) -> Option<usize> {
    self.params.get(name).and_then(|p| p.as_integer()).map(|i| i as usize)
  }

  pub fn param_i32(&self, name: &str) -> Option<i32> {
    self.params.get(name).and_then(|p| p.as_integer()).map(|i| i as i32)
  }

  pub fn param_str(&self, name: &str) -> Option<&str> {
    self.params.get(name).and_then(|p| p.as_str())
  }

  pub fn make_blacklevel(&self, cpp: usize) -> Option<BlackLevel> {
    self.blacklevel.as_ref().map(|x| {
      if x.len() == 1 {
        BlackLevel::new(&vec![x[0]; cpp], 1, 1, cpp)
      } else if x.len() == self.cfa.width * self.cfa.height * cpp {
        BlackLevel::new(x, self.cfa.width, self.cfa.height, cpp)
      } else {
        panic!("Invalid blacklevel data")
      }
    })
  }

  pub fn make_whitelevel(&self, cpp: usize) -> Option<WhiteLevel> {
    self.whitelevel.as_ref().map(|x| {
      if x.len() == 1 {
        WhiteLevel(vec![x[0] as u32; cpp])
      } else if x.len() == cpp {
        WhiteLevel(x.clone())
      } else {
        panic!("Invalid whitelevel data")
      }
    })
  }

  pub fn update_from_toml(&mut self, ct: &toml::value::Table) {
    for (name, val) in ct {
      match name.as_ref() {
        n @ "make" => {
          self.make = val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)).to_string();
        }
        n @ "model" => {
          self.model = val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)).to_string();
        }
        n @ "mode" => {
          self.mode = val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)).to_string();
        }
        n @ "clean_make" => {
          self.clean_make = val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)).to_string();
        }
        n @ "clean_model" => {
          self.clean_model = val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)).to_string();
        }
        n @ "remark" => {
          self.remark = Some(val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)).to_string());
        }
        n @ "whitepoint" => {
          let white = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as u32;
          self.whitelevel = Some(vec![white]);
        }
        n @ "blackpoint" => {
          let black = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as u32;
          self.blacklevel = Some(vec![black]);
        }
        n @ "blackareah" => {
          let vals = val.as_array().unwrap_or_else(|| panic!("{} must be an array", n));
          self.blackareah = Some((vals[0].as_integer().unwrap() as usize, vals[1].as_integer().unwrap() as usize));
        }
        n @ "blackareav" => {
          let vals = val.as_array().unwrap_or_else(|| panic!("{} must be an array", n));
          self.blackareav = Some((vals[0].as_integer().unwrap() as usize, vals[1].as_integer().unwrap() as usize));
        }
        "color_matrix" => {
          if let Some(color_matrix) = val.as_table() {
            for (illu_str, matrix) in color_matrix.into_iter() {
              let illu = Illuminant::new_from_str(illu_str).unwrap();
              let xyz_to_cam = matrix
                .as_array()
                .expect("color matrix must be array")
                .iter()
                .map(|a| a.as_float().expect("color matrix values must be float") as f32)
                .collect();
              self.color_matrix.insert(illu, xyz_to_cam);
            }
          } else {
            eprintln!("Invalid matrix spec for {}", self.clean_model);
          }
          assert!(!self.color_matrix.is_empty());
        }
        n @ "active_area" => {
          let crop_vals = val.as_array().unwrap_or_else(|| panic!("{} must be an array", n));
          let mut crop = [0, 0, 0, 0];
          for (i, val) in crop_vals.iter().enumerate() {
            crop[i] = val.as_integer().unwrap() as usize;
          }
          self.active_area = Some(crop);
        }
        n @ "crop_area" => {
          let crop_vals = val.as_array().unwrap_or_else(|| panic!("{} must be an array", n));
          let mut crop = [0, 0, 0, 0];
          for (i, val) in crop_vals.iter().enumerate() {
            crop[i] = val.as_integer().unwrap() as usize;
          }
          self.crop_area = Some(crop);
        }
        n @ "color_pattern" => {
          self.cfa = CFA::new(val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)));
        }
        n @ "plane_color" => {
          self.plane_color = PlaneColor::new(val.as_str().unwrap_or_else(|| panic!("{} must be a string", n)));
        }
        n @ "bps" => {
          self.bps = Some(val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as usize);
        }
        n @ "real_bps" => {
          self.real_bps = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as usize;
        }
        n @ "filesize" => {
          self.filesize = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as usize;
        }
        n @ "raw_width" => {
          self.raw_width = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as usize;
        }
        n @ "raw_height" => {
          self.raw_height = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as usize;
        }
        n @ "highres_width" => {
          self.highres_width = val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as usize;
        }
        n @ "default_scale" => {
          let scale_vals = val.as_array().unwrap_or_else(|| panic!("{} must be an array", n));
          let scale_h = scale_vals[0].as_array().expect("must be array");
          let scale_v = scale_vals[1].as_array().expect("must be array");
          let scale = [
            [
              scale_h[0].as_integer().expect("must be integer") as u32,
              scale_h[1].as_integer().expect("must be integer") as u32,
            ],
            [
              scale_v[0].as_integer().expect("must be integer") as u32,
              scale_v[1].as_integer().expect("must be integer") as u32,
            ],
          ];
          self.default_scale = DefaultScale(scale);
        }
        n @ "best_quality_scale" => {
          let scale_vals = val.as_array().unwrap_or_else(|| panic!("{} must be an array", n));
          self.best_quality_scale = BestQualityScale([
            scale_vals[0].as_integer().expect("must be integer") as u32,
            scale_vals[1].as_integer().expect("must be integer") as u32,
          ]);
        }
        n @ "hints" => {
          self.hints = Vec::new();
          for hint in val.as_array().unwrap_or_else(|| panic!("{} must be an array", n)) {
            self.hints.push(hint.as_str().expect("hints must be a string").to_string());
          }
        }
        n @ "params" => {
          for (name, val) in val.as_table().unwrap_or_else(|| panic!("{} must be a table", n)) {
            self.params.insert(name.clone(), val.clone());
          }
        }
        "model_aliases" => {}
        "modes" => {} // ignore
        "forward_matrix" => {
          if let Some(forward_matrix) = val.as_table() {
            for (illu_str, matrix) in forward_matrix.into_iter() {
              let illu = Illuminant::new_from_str(illu_str).unwrap();
              let mat = matrix
                .as_array()
                .expect("forward matrix must be array")
                .iter()
                .map(|a| a.as_float().expect("forward matrix values must be float") as f32)
                .collect();
              self.forward_matrix.insert(illu, mat);
            }
          }
        }
        n @ "baseline_exposure" => {
          self.baseline_exposure = Some(val.as_float().unwrap_or_else(|| panic!("{} must be a float", n)) as f32);
        }
        n @ "baseline_noise" => {
          self.baseline_noise = Some(val.as_float().unwrap_or_else(|| panic!("{} must be a float", n)) as f32);
        }
        n @ "baseline_sharpness" => {
          self.baseline_sharpness = Some(val.as_float().unwrap_or_else(|| panic!("{} must be a float", n)) as f32);
        }
        n @ "linear_response_limit" => {
          self.linear_response_limit = Some(val.as_float().unwrap_or_else(|| panic!("{} must be a float", n)) as f32);
        }
        "noise_profile" => {
          let arr = val.as_array().expect("noise_profile must be an array of floats");
          self.noise_profile = Some(arr.iter().map(|v| v.as_float().expect("noise_profile values must be floats")).collect());
        }
        n @ "anti_alias_strength" => {
          self.anti_alias_strength = Some(val.as_float().unwrap_or_else(|| panic!("{} must be a float", n)) as f32);
        }
        n @ "bayer_green_split" => {
          self.bayer_green_split = Some(val.as_integer().unwrap_or_else(|| panic!("{} must be an integer", n)) as u32);
        }
        "default_user_crop" => {
          let arr = val.as_array().expect("default_user_crop must be an array of 4 floats");
          assert_eq!(arr.len(), 4, "default_user_crop must have exactly 4 values");
          self.default_user_crop = Some([
            arr[0].as_float().expect("default_user_crop values must be floats"),
            arr[1].as_float().expect("default_user_crop values must be floats"),
            arr[2].as_float().expect("default_user_crop values must be floats"),
            arr[3].as_float().expect("default_user_crop values must be floats"),
          ]);
        }
        "camera_calibration" => {
          let arr = val.as_array().expect("camera_calibration must be an array of 3 floats");
          assert_eq!(arr.len(), 3, "camera_calibration must have exactly 3 values [r, g, b]");
          self.camera_calibration = Some([
            arr[0].as_float().expect("camera_calibration values must be floats"),
            arr[1].as_float().expect("camera_calibration values must be floats"),
            arr[2].as_float().expect("camera_calibration values must be floats"),
          ]);
        }
        "crop_factor" => {
          self.crop_factor = Some(val.as_float().expect("crop_factor must be a float"));
        }
        key => {
          panic!("Unknown key: {}", key);
        }
      }
    }
  }

  pub fn new() -> Camera {
    Camera {
      make: "".to_string(),
      model: "".to_string(),
      mode: "".to_string(),
      clean_make: "".to_string(),
      clean_model: "".to_string(),
      remark: None,
      filesize: 0,
      raw_width: 0,
      raw_height: 0,
      whitelevel: None,
      blacklevel: None,
      blackareah: None,
      blackareav: None,
      xyz_to_cam: [[0.0; 3]; 4],
      color_matrix: HashMap::new(),
      forward_matrix: HashMap::new(),
      cfa: CFA::new(""),
      plane_color: PlaneColor::default(),
      active_area: None,
      crop_area: None,
      bps: None,
      real_bps: 16,
      highres_width: usize::MAX,
      default_scale: DefaultScale::default(),
      best_quality_scale: BestQualityScale::default(),
      hints: Vec::new(),
      params: HashMap::new(),
      baseline_exposure: None,
      baseline_noise: None,
      baseline_sharpness: None,
      linear_response_limit: None,
      noise_profile: None,
      anti_alias_strength: None,
      bayer_green_split: None,
      default_user_crop: None,
      camera_calibration: None,
      crop_factor: None,
      //orientation: Orientation::Unknown,
    }
  }
}
