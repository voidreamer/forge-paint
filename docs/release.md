# forge-paint release packaging

The release zips are self-contained for the default renderer set:

- `forge-paint(.exe)`
- Windows only: root-level OpenUSD DLL copies for direct EXE launch, and a
  second `usd/lib/forge-paint.exe` copy the root EXE relaunches at startup
  (see below)
- `usd/` with OpenUSD runtime libraries, plugInfo files, file-format plugins, and Hydra/Storm
- `assets/` with starter meshes, materials, HDRI, stencils, and displacement textures
- `tools/` with small utility scripts such as `obj_to_usd.py`
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

On Windows, DLLs imported by `forge-paint.exe` must be discoverable before
`main()` runs, so the workflow copies the top-level OpenUSD runtime DLLs from
`usd/bin` and `usd/lib` beside the EXE. The full `usd/` tree is still bundled
because plugInfo discovery and plug-in-relative library paths depend on that
layout. There is no launcher script: environment variables set in the shell
before starting the EXE (`DELIGHT`, `FORGE_PAINT_3DELIGHT_DIR`,
`PXR_PLUGINPATH_NAME`, `PATH`) are merged into the bundle environment, which
covers the explicit-setup / debugging use case the old `forge-paint.bat`
served (now retired to `attic/`).

The app must not actually *run* from the bundle root, though: USD's plug
registry later `LoadLibrary()`s `usd/lib/usd_*.dll` by absolute path, which
would coexist with the import-loaded root-level copies — two instances of
every USD module in one process, which deadlocks inside
`Hgi::CreatePlatformDefaultHgi` on the first Hydra render. USD also captures
`PXR_PLUGINPATH_NAME` in a static constructor while `usd_plug.dll` loads
(before `main()`), so in-process `set_var` can never feed plugin discovery.
The workflow therefore stages a second EXE copy at `usd/lib/forge-paint.exe`;
the root EXE detects the bundle layout, computes the environment, relaunches
that copy with the environment applied, and forwards its exit code. Imports
and plug loads then resolve to the same `usd/lib` files, and the environment
exists before USD initializes — for double-clicks and manual probe runs
alike.

Storm is bundled with OpenUSD and shows in the delegate picker by default.
On Windows, every Hydra delegate switch is guarded by a short out-of-process
startup probe, so a machine where no usable GL context can come up (remote
desktop, missing driver) gets a viewport overlay naming the failure instead
of a hang or crash. Set `FORGE_PAINT_ENABLE_STORM=0` to hide Storm
explicitly.

## OBJ Import

forge-paint is still USD-first. When the user opens or drops an `.obj`, the app
asks where to save a converted `.usda`, runs the built-in static OBJ converter,
and opens the result. FBX and other interchange formats are intentionally not
handled yet; convert those externally to USD.

The built-in converter supports positions, normals, UVs, polygon faces, negative
OBJ indices, and fan triangulation. It ignores `.mtl` material libraries for
now. The same minimal converter is available as:

```bash
python3 tools/obj_to_usd.py model.obj model.usda
```

## 3Delight

3Delight itself is not bundled. It is proprietary external software, even
though the `HydraNSI` delegate source is Apache-2.0.

The intended shipping model is:

1. The forge-paint zip ships `plugins/usd/hdNSI` when CI built it.
2. The tester installs 3Delight normally.
3. forge-paint discovers both pieces at startup.

The app should work without 3Delight. The delegate picker then shows Storm
only (on all platforms).

The Windows workflow has an optional `include_hdnsi` dispatch input. Manual
builds use that input, and `v*` tag builds attempt the same optional packaging
automatically. CI runs `.github/scripts/package-hdnsi-windows.ps1` after
OpenUSD and forge-paint are built. The script clones HydraNSI, configures it
with `pxr_DIR` pointing at the just-built OpenUSD package, builds the
delegate, runs `cmake --install`, and stages the *installed* layout
(`hdNSI/hdNSI.dll` + `hdNSI/resources/plugInfo.json`, plus `usdNSI/` when
built) under `plugins/usd/`. Staging the installed layout matters: the
plugInfo.json relative paths only resolve from there, and both the script and
the workflow fail the build if the layout is wrong. The script strips common
3Delight runtime files from the staged folder; the release zip must not
redistribute 3Delight itself.

That optional step still needs 3Delight at build time because HydraNSI compiles
against the 3Delight NSI SDK. Manual builds with `include_hdnsi` enabled fail
loudly when none of the inputs below is available, because otherwise the
uploaded artifact looks successful but has no `plugins/usd/hdNSI` folder. Tag
builds still attempt hdNSI packaging opportunistically and skip it if 3Delight
is unavailable. Use one of:

- a self-hosted runner with `DELIGHT` pointing at the 3Delight install, or
- a repo archive at `.github/3delight-windows.zip`, or
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
