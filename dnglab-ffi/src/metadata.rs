// SPDX-License-Identifier: LGPL-2.1
//! Backend-neutral metadata JSON.
//!
//! The flat top-level keys form a stable contract for app front-ends (they
//! mirror what the LibRaw based predecessor exposed); `details` carries the
//! complete rawler [`RawMetadata`] for power users.

use std::path::Path;

use chrono::{FixedOffset, NaiveDateTime, TimeZone};

use rawler::decoders::RawMetadata;
use rawler::formats::tiff::{Rational, SRational};
use rawler::imgop::sensor::SensorType;
use rawler::rawimage::RawPhotometricInterpretation;
use serde_json::{Map, Value, json};

use crate::api::{Opened, default_crop_dims, effective_orientation, guard, is_full_resolution};
use crate::error::Result;

fn rat(r: &Rational) -> f64 {
  if r.d == 0 { 0.0 } else { r.n as f64 / r.d as f64 }
}

fn srat(r: &SRational) -> f64 {
  if r.d == 0 { 0.0 } else { r.n as f64 / r.d as f64 }
}

fn put_str(m: &mut Map<String, Value>, key: &str, v: Option<&String>) {
  if let Some(s) = v {
    let s = s.trim();
    if !s.is_empty() {
      m.insert(key.into(), Value::String(s.to_string()));
    }
  }
}

fn put_f64_pos(m: &mut Map<String, Value>, key: &str, v: f64) {
  if v > 0.0 && v.is_finite() {
    m.insert(key.into(), json!(v));
  }
}

fn put_u64(m: &mut Map<String, Value>, key: &str, v: Option<u64>) {
  if let Some(v) = v {
    m.insert(key.into(), json!(v));
  }
}

/// Format an exposure time like a camera would: "1/250" or "2.5s".
pub fn format_shutter(seconds: f64) -> String {
  if seconds <= 0.0 || !seconds.is_finite() {
    String::new()
  } else if seconds < 1.0 {
    format!("1/{}", (1.0 / seconds).round() as u64)
  } else {
    format!("{seconds:.1}s")
  }
}

/// Convert an EXIF date ("YYYY:MM:DD HH:MM:SS") to display form and unix seconds.
/// `offset` is an EXIF OffsetTime string like "+02:00". Without it the camera's
/// zone is unknown, so no unix timestamp is produced (guessing the device's
/// current zone is wrong whenever the photo was taken elsewhere).
pub fn exif_date(date: &str, offset: Option<&str>) -> (String, Option<i64>) {
  let date = date.trim();
  let Ok(naive) = NaiveDateTime::parse_from_str(date, "%Y:%m:%d %H:%M:%S") else {
    return (date.replacen(':', "-", 2), None);
  };
  let display = naive.format("%Y-%m-%d %H:%M:%S").to_string();
  let ts = offset
    .and_then(|o| o.trim().parse::<FixedOffset>().ok())
    .and_then(|tz| tz.from_local_datetime(&naive).earliest())
    .map(|dt| dt.timestamp());
  (display, ts)
}

/// Build the metadata document for `input`.
pub fn metadata_json(input: &Path) -> Result<String> {
  guard(|| {
    let opened = Opened::open(input)?;
    let md = opened.metadata()?;
    let doc = build_document(&opened, &md);
    Ok(serde_json::to_string(&doc)?)
  })
}

fn build_document(opened: &Opened, md: &RawMetadata) -> Value {
  let mut m = Map::new();
  let exif = &md.exif;

  m.insert("backend".into(), json!("dnglab"));
  m.insert("backend_version".into(), json!(crate::api::version()));
  m.insert("format".into(), json!(format!("{:?}", opened.decoder.format_hint())));

  // Camera
  if !md.make.is_empty() {
    m.insert("make".into(), json!(md.make));
  }
  if !md.model.is_empty() {
    m.insert("model".into(), json!(md.model));
  }
  put_str(&mut m, "body_serial", exif.serial_number.as_ref());
  put_str(&mut m, "owner_name", exif.owner_name.as_ref());

  // Lens: resolved lens database entry first, raw EXIF second.
  if let Some(lens) = &md.lens {
    if !lens.lens_make.is_empty() {
      m.insert("lens_make".into(), json!(lens.lens_make));
    }
    if !lens.lens_model.is_empty() {
      m.insert("lens_model".into(), json!(lens.lens_model));
    } else if !lens.lens_name.is_empty() {
      m.insert("lens_model".into(), json!(lens.lens_name));
    }
    put_f64_pos(&mut m, "min_focal", rat(&lens.focal_range[0]));
    put_f64_pos(&mut m, "max_focal", rat(&lens.focal_range[1]));
  } else {
    put_str(&mut m, "lens_make", exif.lens_make.as_ref());
    put_str(&mut m, "lens_model", exif.lens_model.as_ref());
    if let Some(spec) = &exif.lens_spec {
      put_f64_pos(&mut m, "min_focal", rat(&spec[0]));
      put_f64_pos(&mut m, "max_focal", rat(&spec[1]));
    }
  }
  put_str(&mut m, "lens_serial", exif.lens_serial_number.as_ref());

  // Exposure
  if let Some(fl) = &exif.focal_length {
    put_f64_pos(&mut m, "focal_length", rat(fl));
  }
  if let Some(fl35) = exif.focal_len_in_35mm_format {
    if fl35 > 0 {
      m.insert("focal_length_35mm".into(), json!(fl35));
    }
  }
  if let Some(f) = &exif.fnumber {
    put_f64_pos(&mut m, "aperture", rat(f));
  }
  if let Some(et) = &exif.exposure_time {
    let secs = rat(et);
    if secs > 0.0 {
      m.insert("shutter_raw".into(), json!(secs));
      m.insert("shutter".into(), json!(format_shutter(secs)));
    }
  }
  let iso = exif
    .iso_speed_ratings
    .map(|v| v as u64)
    .filter(|v| *v > 0)
    .or(exif.iso_speed.map(|v| v as u64).filter(|v| *v > 0))
    .or(exif.recommended_exposure_index.map(|v| v as u64).filter(|v| *v > 0));
  put_u64(&mut m, "iso", iso);
  if let Some(ev) = &exif.exposure_bias {
    m.insert("exposure_bias".into(), json!(srat(ev)));
  }
  if let Some(v) = exif.exposure_program {
    m.insert("exposure_program".into(), json!(v));
  }
  if let Some(v) = exif.metering_mode {
    m.insert("metering_mode".into(), json!(v));
  }
  if let Some(v) = exif.flash {
    m.insert("flash".into(), json!(v));
  }
  if let Some(v) = exif.white_balance {
    m.insert("white_balance".into(), json!(v));
  }

  // Date/time
  let date = exif.date_time_original.as_ref().or(exif.create_date.as_ref()).or(exif.modify_date.as_ref());
  if let Some(date) = date {
    let offset = exif.offset_time_original.as_deref().or(exif.offset_time.as_deref());
    let (display, ts) = exif_date(date, offset);
    if !display.is_empty() {
      m.insert("timestamp".into(), json!(display));
    }
    if let Some(ts) = ts {
      m.insert("timestamp_raw".into(), json!(ts));
    }
  }

  // Orientation: EXIF, else what the decoder put on the frame (DNG/QTK), else 1.
  let mut orientation = exif.orientation.filter(|o| (1..=8).contains(o)).unwrap_or(1);

  // Dimensions and sensor layout via a structure-only decode.
  match guard(|| opened.raw_image(true)) {
    Ok(raw) => {
      let (w, h) = default_crop_dims(&raw);
      m.insert("width".into(), json!(w));
      m.insert("height".into(), json!(h));
      m.insert("raw_width".into(), json!(raw.width));
      m.insert("raw_height".into(), json!(raw.height));
      orientation = effective_orientation(md, &raw);
      let (colors, pattern) = match &raw.photometric {
        RawPhotometricInterpretation::Cfa(cfa) => {
          let pattern = match cfa.sensor {
            SensorType::Bayer => cfa.cfa.to_string(),
            SensorType::Xtrans => "X-Trans".to_string(),
          };
          (cfa.cfa.unique_colors(), pattern)
        }
        RawPhotometricInterpretation::LinearRaw => (raw.cpp, String::new()),
        RawPhotometricInterpretation::BlackIsZero => (1, String::new()),
      };
      m.insert("colors".into(), json!(colors));
      if !pattern.is_empty() {
        m.insert("bayer_pattern".into(), json!(pattern));
      }
      m.insert("bits_per_sample".into(), json!(raw.bps));
      m.insert("clean_make".into(), json!(raw.clean_make));
      m.insert("clean_model".into(), json!(raw.clean_model));
      if let Some(be) = raw.camera.baseline_exposure {
        m.insert("baseline_exposure".into(), json!(be));
      }

      // Embedded preview summary (drives the "use embedded JPEG" decision)
      let mut preview = Map::new();
      match opened.embedded_jpeg() {
        Ok(Some((bytes, pw, ph))) => {
          preview.insert("embedded_jpeg".into(), json!(true));
          preview.insert("width".into(), json!(pw));
          preview.insert("height".into(), json!(ph));
          preview.insert("bytes".into(), json!(bytes.len()));
          preview.insert("full_resolution".into(), json!(is_full_resolution((pw, ph), (w, h), 0.9)));
        }
        _ => {
          preview.insert("embedded_jpeg".into(), json!(false));
        }
      }
      m.insert("preview".into(), Value::Object(preview));
    }
    Err(e) => {
      log::warn!("Structure decode failed, dimensions unavailable: {e}");
      match opened.embedded_jpeg() {
        Ok(Some((bytes, pw, ph))) => {
          m.insert(
            "preview".into(),
            json!({ "embedded_jpeg": true, "width": pw, "height": ph, "bytes": bytes.len() }),
          );
        }
        _ => {
          m.insert("preview".into(), json!({ "embedded_jpeg": false }));
        }
      }
    }
  }
  m.insert("orientation".into(), json!(orientation));

  // GPS
  let mut has_gps = false;
  if let Some(gps) = &exif.gps {
    if let (Some(lat), Some(lon)) = (&gps.gps_latitude, &gps.gps_longitude) {
      let nonzero = lat.iter().chain(lon.iter()).any(|r| r.n != 0);
      if nonzero {
        has_gps = true;
        m.insert("gps_lat_deg".into(), json!(rat(&lat[0])));
        m.insert("gps_lat_min".into(), json!(rat(&lat[1])));
        m.insert("gps_lat_sec".into(), json!(rat(&lat[2])));
        let lat_ref = gps.gps_latitude_ref.as_deref().map(str::trim).filter(|s| *s == "S").unwrap_or("N");
        m.insert("gps_lat_ref".into(), json!(lat_ref));
        m.insert("gps_lon_deg".into(), json!(rat(&lon[0])));
        m.insert("gps_lon_min".into(), json!(rat(&lon[1])));
        m.insert("gps_lon_sec".into(), json!(rat(&lon[2])));
        let lon_ref = gps.gps_longitude_ref.as_deref().map(str::trim).filter(|s| *s == "W").unwrap_or("E");
        m.insert("gps_lon_ref".into(), json!(lon_ref));
        if let Some(alt) = &gps.gps_altitude {
          m.insert("gps_altitude".into(), json!(rat(alt)));
        }
        m.insert("gps_alt_ref".into(), json!(gps.gps_altitude_ref.unwrap_or(0)));
        if let Some(ts) = &gps.gps_timestamp {
          m.insert("gps_time_hour".into(), json!(rat(&ts[0])));
          m.insert("gps_time_min".into(), json!(rat(&ts[1])));
          m.insert("gps_time_sec".into(), json!(rat(&ts[2])));
        }
      }
    }
  }
  m.insert("has_gps".into(), json!(has_gps));

  // Other
  put_str(&mut m, "description", exif.image_description.as_ref());
  put_str(&mut m, "artist", exif.artist.as_ref());
  put_str(&mut m, "copyright", exif.copyright.as_ref());
  put_str(&mut m, "user_comment", exif.user_comment.as_ref());
  if let Some(r) = md.rating {
    m.insert("rating".into(), json!(r));
  }

  // Full rawler metadata minus the MakerNotes blob.
  let mut details = md.clone();
  details.exif.makernotes = None;
  if let Ok(v) = serde_json::to_value(&details) {
    m.insert("details".into(), v);
  }

  Value::Object(m)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn shutter_formatting() {
    assert_eq!(format_shutter(1.0 / 250.0), "1/250");
    assert_eq!(format_shutter(0.5), "1/2");
    assert_eq!(format_shutter(2.5), "2.5s");
    assert_eq!(format_shutter(0.0), "");
  }

  #[test]
  fn exif_date_parsing() {
    let (d, ts) = exif_date("2024:06:01 12:34:56", Some("+00:00"));
    assert_eq!(d, "2024-06-01 12:34:56");
    assert_eq!(ts, Some(1717245296));
    let (d, ts) = exif_date("2024:06:01 12:34:56", Some("+02:00"));
    assert_eq!(d, "2024-06-01 12:34:56");
    assert_eq!(ts, Some(1717245296 - 7200));
    let (d, ts) = exif_date("garbage", None);
    assert_eq!(d, "garbage");
    assert_eq!(ts, None);
    // No offset: display only, no guessed timestamp
    let (d, ts) = exif_date("2024:06:01 12:34:56", None);
    assert_eq!(d, "2024-06-01 12:34:56");
    assert_eq!(ts, None);
  }
}
