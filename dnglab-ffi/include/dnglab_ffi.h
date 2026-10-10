/* SPDX-License-Identifier: LGPL-2.1 */
/*
 * dnglab-ffi: C ABI over dnglab's rawler crate.
 *
 * All functions are thread-safe and panic-safe. Functions returning int32_t
 * use DNGLAB_OK (0) for success; on failure a message is stored in *error_out
 * (when error_out is not NULL) which must be released with dnglab_string_free.
 * Strings returned as char* are owned by the caller (dnglab_string_free);
 * const char* results are static.
 *
 * Keep the struct layouts in sync with dnglab-ffi/src/lib.rs.
 */
#ifndef DNGLAB_FFI_H
#define DNGLAB_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DNGLAB_OK 0
#define DNGLAB_ERROR 1
#define DNGLAB_INVALID_ARGUMENT 2

/* Chroma subsampling for rendered JPEGs */
#define DNGLAB_CHROMA_444 0
#define DNGLAB_CHROMA_422 1
#define DNGLAB_CHROMA_420 2

/* Embedded JPEG policy for dnglab_convert_to_jpeg */
#define DNGLAB_EMBEDDED_NEVER 0
#define DNGLAB_EMBEDDED_FULL_RESOLUTION_ONLY 1
#define DNGLAB_EMBEDDED_ALWAYS 2

/* Log levels for dnglab_init_logging */
#define DNGLAB_LOG_OFF 0
#define DNGLAB_LOG_ERROR 1
#define DNGLAB_LOG_WARN 2
#define DNGLAB_LOG_INFO 3
#define DNGLAB_LOG_DEBUG 4
#define DNGLAB_LOG_TRACE 5

typedef struct dnglab_dng_options {
    int32_t embed_original;      /* embed the original RAW (non-zero = yes) */
    int32_t preview;             /* include JPEG preview */
    int32_t thumbnail;           /* include thumbnail */
    int32_t lossless;            /* lossless JPEG-92 (non-zero) or uncompressed (zero) */
    uint32_t index;              /* sub-image index, usually 0 */
    const char* artist;          /* may be NULL */
    const char* dcp_dir;         /* may be NULL */
    const char* dcp_file;        /* may be NULL */
    const char* software;        /* may be NULL */
} dnglab_dng_options;

typedef struct dnglab_jpeg_options {
    int32_t quality;                 /* 1..100 */
    int32_t chroma_subsampling;      /* DNGLAB_CHROMA_* */
    int32_t optimize_huffman;        /* non-zero = yes */
    int32_t progressive;             /* non-zero = yes */
    int32_t embedded_policy;         /* DNGLAB_EMBEDDED_* */
    float full_resolution_threshold; /* e.g. 0.9; <= 0 keeps the default */
    int32_t embed_exif;              /* non-zero = write EXIF into the JPEG */
    int32_t auto_bright;             /* rendered path: dcraw style auto brightness (non-zero) */
    const char* software;            /* may be NULL */
} dnglab_jpeg_options;

typedef struct dnglab_jpeg_result {
    int32_t used_embedded;  /* camera JPEG was copied */
    uint32_t width;
    uint32_t height;
    int32_t orientation;    /* EXIF orientation (1..8) of the stored pixels */
    int32_t exif_embedded;  /* EXIF was written into the file */
} dnglab_jpeg_result;

typedef struct dnglab_preview_result {
    int32_t orientation;    /* EXIF orientation the caller still has to apply (1 = upright) */
    int32_t from_embedded;  /* camera JPEG was used */
    uint32_t width;
    uint32_t height;
} dnglab_preview_result;

/* Initialise logging (Android: logcat tag "dnglab"). Safe to call repeatedly. */
void dnglab_init_logging(int32_t level);

/* Static version string. */
const char* dnglab_version(void);

/* Comma separated lower-case extensions without dots. Free with dnglab_string_free. */
char* dnglab_supported_extensions(void);

/* Non-zero when the extension (any case, optional leading dot) is decodable. */
int32_t dnglab_is_supported_extension(const char* ext);

void dnglab_string_free(char* s);

/* RAW -> DNG. options may be NULL for defaults. */
int32_t dnglab_convert_to_dng(const char* input,
                              const char* output,
                              const dnglab_dng_options* options,
                              char** error_out);

/* RAW -> JPEG (embedded passthrough or rendered). options/result_out may be NULL. */
int32_t dnglab_convert_to_jpeg(const char* input,
                               const char* output,
                               const dnglab_jpeg_options* options,
                               dnglab_jpeg_result* result_out,
                               char** error_out);

/* JPEG preview with the longest edge limited to max_dimension (0 = unlimited). */
int32_t dnglab_extract_preview(const char* input,
                               const char* output,
                               uint32_t max_dimension,
                               dnglab_preview_result* result_out,
                               char** error_out);

/* Metadata JSON document, or NULL on failure. Free with dnglab_string_free. */
char* dnglab_metadata_json(const char* input, char** error_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* DNGLAB_FFI_H */
