// rust-usd's build script emits @rpath link args for its own examples,
// but `cargo:rustc-link-arg` doesn't propagate to downstream crates.
// rust-usd-build's helper re-emits them for our bins and examples so
// the dynamic loader finds libpxr_*.dylib without DYLD_LIBRARY_PATH.

fn main() {
    rust_usd_build::emit_runtime_rpath();
}
