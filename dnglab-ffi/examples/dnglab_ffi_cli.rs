// SPDX-License-Identifier: LGPL-2.1
//! Tiny command line front-end over the safe API, handy for exercising the
//! library on a desktop: `cargo run -p dnglab-ffi --example dnglab_ffi_cli -- <cmd> ...`

use std::path::Path;
use std::process::exit;

use dnglab_ffi::api::{self, EmbeddedJpegPolicy, JpegOptions};

fn usage() -> ! {
  eprintln!(
    "usage:\n  dnglab_ffi_cli dng <in> <out.dng>\n  dnglab_ffi_cli jpeg <in> <out.jpg> [policy 0|1|2] [quality] [auto_bright 0|1]\n  dnglab_ffi_cli preview <in> <out.jpg> [max_dim]\n  dnglab_ffi_cli meta <in>\n  dnglab_ffi_cli exts"
  );
  exit(2)
}

fn main() {
  env_logger::init();
  let args: Vec<String> = std::env::args().collect();
  if args.len() < 2 {
    usage();
  }
  let res: Result<String, dnglab_ffi::error::Error> = match args[1].as_str() {
    "dng" if args.len() >= 4 => api::convert_to_dng(Path::new(&args[2]), Path::new(&args[3]), &api::DngOptions::default()).map(|_| "ok".into()),
    "jpeg" if args.len() >= 4 => {
      let mut opts = JpegOptions::default();
      if let Some(p) = args.get(4) {
        opts.embedded_policy = EmbeddedJpegPolicy::from_i32(p.parse().unwrap_or(1));
      }
      if let Some(q) = args.get(5) {
        opts.quality = q.parse().unwrap_or(95);
      }
      if let Some(ab) = args.get(6) {
        opts.auto_bright = ab != "0";
      }
      api::convert_to_jpeg(Path::new(&args[2]), Path::new(&args[3]), &opts).map(|r| format!("{r:?}"))
    }
    "preview" if args.len() >= 4 => {
      let max = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2048);
      api::extract_preview(Path::new(&args[2]), Path::new(&args[3]), max).map(|r| format!("{r:?}"))
    }
    "meta" if args.len() >= 3 => dnglab_ffi::metadata::metadata_json(Path::new(&args[2])),
    "exts" => Ok(api::supported_extensions().join(",")),
    _ => usage(),
  };
  match res {
    Ok(s) => println!("{s}"),
    Err(e) => {
      eprintln!("error: {e}");
      exit(1);
    }
  }
}
