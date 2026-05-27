# forge-paint release packaging

The release zips are self-contained for the default renderer set:

- `forge-paint(.exe)`
- `usd/` with OpenUSD runtime libraries, plugInfo files, file-format plugins, and Hydra/Storm
- `assets/` with starter meshes, materials, HDRI, stencils, and displacement textures
- `README.txt`

## Build

Use the GitHub Actions workflows:

- `.github/workflows/windows-build.yml`
- `.github/workflows/macos-build.yml`

Manual dispatch creates downloadable artifacts. Pushing a `v*` tag also attaches
the zip to the GitHub Release.

The workflows build OpenUSD with imaging enabled because `hydra-rs` links Hydra
and `UsdImagingGL`, and the bundled app must include the Storm render delegate.
They still disable Python, tests, examples, tutorials, docs, and `usdview`.

## 3Delight

3Delight is not bundled in the default zip. It is proprietary external software,
even though the `HydraNSI` delegate source is Apache-2.0.

For testers who need it:

1. Install 3Delight separately.
2. Build or copy an `hdNSI` plugin that was compiled against the same OpenUSD
   ABI as the bundled `usd/` tree.
3. Place the plugin under `usd/plugin/usd/` or `plugins/usd/` in the bundle, or
   provide a launcher that prepends its plugInfo directory to
   `PXR_PLUGINPATH_NAME`.
4. Ensure the 3Delight runtime directory is on `PATH`/`DYLD_LIBRARY_PATH`, or
   use the vendor environment setup scripts.

The app should work without 3Delight. In that case the delegate picker shows
Storm only.

## Size

OpenUSD with Hydra is large. A stripped runtime bundle can still be hundreds of
MB compressed and may exceed 1 GB uncompressed depending on platform and build
type. Keep removing:

- headers
- import libraries
- PDB/dSYM debug files
- CMake exports
- USD command-line tools
- generated `.auto.tdl` texture caches

## Starter Assets

The repo tracks a minimal starter pack:

- `assets/default_mesh/default.usda`
- `assets/hdri/forge_studio_4x2.hdr`
- `assets/stencils/forge_soft_round.png`
- `assets/displacement/forge_checker_height.png`
- `assets/materials/*.usda`

Larger HDRIs, Poly Haven downloads, local stencils, and `.auto.tdl` caches stay
ignored. Add third-party assets only with a clear license note.
