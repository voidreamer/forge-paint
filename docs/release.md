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

CI uses the published `rust-usd` and `hydra-rs` crates. The repository keeps a
local `[patch.crates-io]` block for day-to-day development against
`../rust-usd`; the workflows strip that block before building so they do not
need access to a private sibling checkout.

## 3Delight

3Delight itself is not bundled. It is proprietary external software, even
though the `HydraNSI` delegate source is Apache-2.0.

The intended shipping model is:

1. The forge-paint zip ships `plugins/usd/hdNSI` when CI built it.
2. The tester installs 3Delight normally.
3. forge-paint discovers both pieces at startup.

The app should work without 3Delight. In that case the delegate picker shows
Storm only.

The Windows workflow has an optional `include_hdnsi` dispatch input. When it is
enabled, CI runs `.github/scripts/package-hdnsi-windows.ps1` after OpenUSD and
forge-paint are built. The script clones HydraNSI, configures it with
`pxr_DIR` pointing at the just-built OpenUSD package, builds the delegate, and
stages the delegate under `plugins/usd/hdNSI`. It strips common 3Delight
runtime files from the staged delegate folder; the release zip must not
redistribute 3Delight itself.

That optional step still needs 3Delight at build time because HydraNSI compiles
against the 3Delight NSI SDK. Use either:

- a self-hosted runner with `DELIGHT` pointing at the 3Delight install, or
- a repository secret named `DELIGHT_WINDOWS_ARCHIVE_URL` pointing at a private
  zip that contains a 3Delight install tree with `bin/renderdl.exe`.

At runtime on Windows, forge-paint searches for 3Delight in this order:

- `FORGE_PAINT_3DELIGHT_DIR`
- `DELIGHT`
- a sibling `3Delight/` folder next to `forge-paint.exe`
- common `Program Files` folders whose names contain `3Delight`

When it finds a root containing `bin/renderdl.exe`, it sets `DELIGHT` if needed,
prepends the root's `bin/` and `lib/` folders to `PATH`, and scans the root for
an `hdNSI/resources/plugInfo.json` in case the user-installed package already
contains a compatible Hydra delegate. The bundled `plugins/usd/**/plugInfo.json`
folders are also added to `PXR_PLUGINPATH_NAME`.

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
