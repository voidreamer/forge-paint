//! Shared output stage for the static model -> USD converters
//! (`obj_to_usd`, `gltf_to_usd`).
//!
//! The converters generate a `#usda 1.0` document in memory; this
//! module decides how it lands on disk. `.usda` is written as plain
//! text straight from Rust — no USD runtime involved, which keeps the
//! converter unit tests and the `tools/obj_to_usd.py` parity path
//! dependency-free. `.usd` / `.usdc` are re-encoded into USD's crate
//! binary format through rust-usd (`SdfLayer::Export` under the hood),
//! which is what the app's save dialog now defaults to: ASCII scales
//! terribly with vertex count, the crate format is the production
//! choice.

use anyhow::{Context, Result, bail};
use std::path::Path;

pub fn write_usda_document(text: &str, dest: &Path) -> Result<()> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let ext = dest
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    match ext.as_deref() {
        Some("usd") | Some("usdc") => {
            if !rust_usd::write_usda_text_as_usd(text, dest) {
                bail!(
                    "USD crate encoding failed for {} (the generated usda did not parse, or the destination is not writable)",
                    dest.display()
                );
            }
            Ok(())
        }
        _ => std::fs::write(dest, text).with_context(|| format!("write {}", dest.display())),
    }
}

/// Compact float formatting for usda output: trims trailing zeros so
/// the text stays readable and diff-friendly.
pub fn fmt_f32(value: f32) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.9}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string()
    }
}

/// Clamp an arbitrary name into a legal USD prim identifier
/// (`[A-Za-z_][A-Za-z0-9_]*`).
pub fn sanitize_identifier(input: &str) -> String {
    let mut out = String::new();
    for (i, c) in input.chars().enumerate() {
        let valid = c == '_' || c.is_ascii_alphanumeric();
        if i == 0 && c.is_ascii_digit() {
            out.push('_');
        }
        out.push(if valid { c } else { '_' });
    }
    out
}
