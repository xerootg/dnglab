// SPDX-License-Identifier: LGPL-2.1
//! Error type shared by the safe API and the C ABI.

use std::fmt;

/// A single, message-carrying error type. Everything the FFI surface can fail
/// with is flattened into a human readable message because that is all a C
/// caller can meaningfully consume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl Error {
  pub fn new(msg: impl Into<String>) -> Self {
    Self(msg.into())
  }

  pub fn message(&self) -> &str {
    &self.0
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str(&self.0)
  }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

impl From<rawler::RawlerError> for Error {
  fn from(e: rawler::RawlerError) -> Self {
    Self(e.to_string())
  }
}

impl From<std::io::Error> for Error {
  fn from(e: std::io::Error) -> Self {
    Self(format!("I/O error: {e}"))
  }
}

impl From<image::ImageError> for Error {
  fn from(e: image::ImageError) -> Self {
    Self(format!("Image error: {e}"))
  }
}

impl From<jpeg_encoder::EncodingError> for Error {
  fn from(e: jpeg_encoder::EncodingError) -> Self {
    Self(format!("JPEG encoding error: {e}"))
  }
}

impl From<rawler::formats::tiff::TiffError> for Error {
  fn from(e: rawler::formats::tiff::TiffError) -> Self {
    Self(format!("TIFF error: {e}"))
  }
}

impl From<serde_json::Error> for Error {
  fn from(e: serde_json::Error) -> Self {
    Self(format!("JSON error: {e}"))
  }
}

impl From<String> for Error {
  fn from(s: String) -> Self {
    Self(s)
  }
}

impl From<&str> for Error {
  fn from(s: &str) -> Self {
    Self(s.to_string())
  }
}
