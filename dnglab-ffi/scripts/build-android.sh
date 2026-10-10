#!/usr/bin/env bash
# Cross-compile dnglab-ffi as a static library for Android.
#
# Usage:
#   build-android.sh --ndk <path> --abi <abi> [--api <level>] [--profile android|release|debug]
#                    [--target-dir <dir>] [--manifest <Cargo.toml>]
#
# Prints the path of the resulting libdnglab_ffi.a on the last line of stdout.
# Set CARGO to use a specific cargo binary (defaults to `cargo` on PATH).
# Requires the matching rustup targets:
#   rustup target add aarch64-linux-android armv7-linux-androideabi \
#                     i686-linux-android x86_64-linux-android
set -euo pipefail

NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK:-}}"
ABI=""
API="${ANDROID_API_LEVEL:-24}"
PROFILE="android"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/../Cargo.toml"
TARGET_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ndk) NDK="$2"; shift 2 ;;
    --abi) ABI="$2"; shift 2 ;;
    --api) API="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --target-dir) TARGET_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$NDK" ]] || { echo "--ndk (or ANDROID_NDK_HOME) is required" >&2; exit 2; }
[[ -n "$ABI" ]] || { echo "--abi is required" >&2; exit 2; }

case "$ABI" in
  arm64-v8a)   TRIPLE="aarch64-linux-android";   CLANG_PREFIX="aarch64-linux-android" ;;
  armeabi-v7a) TRIPLE="armv7-linux-androideabi"; CLANG_PREFIX="armv7a-linux-androideabi" ;;
  x86)         TRIPLE="i686-linux-android";      CLANG_PREFIX="i686-linux-android" ;;
  x86_64)      TRIPLE="x86_64-linux-android";    CLANG_PREFIX="x86_64-linux-android" ;;
  *) echo "unsupported ABI: $ABI" >&2; exit 2 ;;
esac

case "$(uname -s)" in
  Linux)  HOST_TAG="linux-x86_64" ;;
  Darwin) HOST_TAG="darwin-x86_64" ;;
  *) echo "unsupported host: $(uname -s)" >&2; exit 2 ;;
esac

TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/${HOST_TAG}"
BIN="${TOOLCHAIN}/bin"
CLANG="${BIN}/${CLANG_PREFIX}${API}-clang"
[[ -x "$CLANG" ]] || { echo "NDK clang not found: $CLANG" >&2; exit 1; }

# Cargo reads linker settings from CARGO_TARGET_<TRIPLE>_LINKER; cc-rs based
# build scripts (if any dependency needs one) read CC_<triple>/AR_<triple>
# (underscore form, hyphens are not valid in shell variable names).
TRIPLE_ENV="$(echo "$TRIPLE" | tr 'a-z-' 'A-Z_')"
TRIPLE_US="$(echo "$TRIPLE" | tr '-' '_')"
export "CARGO_TARGET_${TRIPLE_ENV}_LINKER=${CLANG}"
export "CARGO_TARGET_${TRIPLE_ENV}_AR=${BIN}/llvm-ar"
export "CC_${TRIPLE_US}=${CLANG}"
export "CXX_${TRIPLE_US}=${CLANG}++"
export "AR_${TRIPLE_US}=${BIN}/llvm-ar"
export "RANLIB_${TRIPLE_US}=${BIN}/llvm-ranlib"

ARGS=(build --manifest-path "$MANIFEST" --lib --target "$TRIPLE")
case "$PROFILE" in
  debug) ;;
  release) ARGS+=(--release) ;;
  *) ARGS+=(--profile "$PROFILE") ;;
esac
[[ -n "$TARGET_DIR" ]] && ARGS+=(--target-dir "$TARGET_DIR")

CARGO_BIN="${CARGO:-cargo}"
echo "dnglab-ffi: $CARGO_BIN ${ARGS[*]} (linker: $CLANG)" >&2
"$CARGO_BIN" "${ARGS[@]}"

if [[ -z "$TARGET_DIR" ]]; then
  # Default cargo target directory of the workspace that owns the manifest
  # (honours CARGO_TARGET_DIR like cargo itself does).
  if [[ -n "${CARGO_TARGET_DIR:-}" ]]; then
    TARGET_DIR="$CARGO_TARGET_DIR"
  else
    TARGET_DIR="$(cd "$(dirname "$MANIFEST")/.." && pwd)/target"
  fi
fi
OUT="${TARGET_DIR}/${TRIPLE}/${PROFILE}/libdnglab_ffi.a"
[[ -f "$OUT" ]] || { echo "expected output missing: $OUT" >&2; exit 1; }
echo "$OUT"
