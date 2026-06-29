//! fit_look — fit per-channel display `colorCurves` from the EDITOR's neutral
//! render to a raw's embedded-JPEG preview (the in-body look).
//!
//! This is the production-CORRECT look fit: it fits against the same neutral the
//! editor renders — the rust-renderer `render-one-raw` output (EditValues=0, with
//! DCP base-tone + BaselineExposure applied) — NOT the bare `RawDevelop`
//! intermediate the in-`convert` fit uses, which is degenerate (black) on dark
//! scenes and mismatched even when it isn't. See `docs/learned-look-fit.md` and
//! `ml/look-fit/`.
//!
//! Usage:
//!   render-one-raw <raw|dng> neutral.png            # the editor's neutral
//!   cargo run --release --example fit_look -- neutral.png <raw> [out.json]
//!
//! Prints the fitted `[ToneCurve; 3]` as JSON (the `lb:recipe.colorCurves` payload
//! the editor's `applyRecipe` seeds), or `null` when the regression guard fires
//! (no global look to recover → open at neutral).

use rawler::decoders::RawDecodeParams;
use rawler::dng::convert::fit_color_curves;
use rawler::rawsource::RawSource;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: fit_look <neutral.png> <raw> [out.json]");
        std::process::exit(2);
    }
    // The renderer neutral is JPEG content (render-one-raw encodes JPEG regardless
    // of the file extension), so sniff the format from bytes, not the name.
    let neutral = image::ImageReader::open(&args[1])?.with_guessed_format()?.decode()?;
    let rawfile = RawSource::new(Path::new(&args[2]))?;
    let decoder = rawler::get_decoder(&rawfile)?;
    let preview = decoder
        .preview_image(&rawfile, &RawDecodeParams::default())?
        .ok_or_else(|| anyhow::anyhow!("no embedded preview in {}", args[2]))?;

    match fit_color_curves(&neutral, &preview) {
        Some(curves) => {
            let json = serde_json::to_string_pretty(&curves)?;
            match args.get(3) {
                Some(out) => {
                    std::fs::write(out, &json)?;
                    eprintln!("wrote {out}");
                }
                None => println!("{json}"),
            }
        }
        None => {
            eprintln!("lookCaptured=false — no global look to recover; open at neutral");
            println!("null");
        }
    }
    Ok(())
}
