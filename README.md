# forge-paint

A USD-centric 3D texture painter, written in Rust — a prototype
Substance-Painter-style tool that opens USD stages, paints PBR texture
sets directly on the mesh, bakes mesh maps, and writes everything back
as USD.

- **USD-native**: opens `.usd` / `.usda` / `.usdc` / `.usdz` stages (plus
  OBJ, glTF/GLB, and Alembic via built-in conversion), authors
  UsdPreviewSurface materials, and resolves `forge://` URIs through a
  custom Ar resolver when the pipeline provides one.
- **Layered PBR painting**: base color, roughness, metallic, normal, and
  displacement channels; paint and fill layers with masks, blend modes,
  and smart-mask presets; UDIM tile support.
- **Viewport**: wgpu PBR renderer with image-based lighting, FXAA,
  tonemapping, and wireframe — or a Hydra viewport (Storm, and 3Delight
  hdNSI where installed) rendering the actual USD stage.
- **Baking**: AO, normal, world normal, curvature, position, height,
  thickness, bent normals, and ID maps via the vendored
  [`texture-baker`](crates/texture-baker) crate (GPU-accelerated where
  possible), pulled into paint layers as mesh maps.
- **Material graph**: node editor for wiring textures into material
  bindings, mirrored into the USD stage.

**Status**: personal prototype under active development. macOS
(Apple Silicon) is the primary development target, Windows x64 ships as
a packaged zip, Linux is CI-checked but not a supported runtime yet.

## Getting a build

Prebuilt zips for macOS and Windows are produced by the
[release workflows](docs/release.md) (`v*` tags attach them to GitHub
Releases). The zips are self-contained: OpenUSD runtime, Storm delegate,
starter assets, and conversion tools included.

## Building from source

forge-paint links OpenUSD through the [`rust-usd`] / [`hydra-rs`]
bindings, so you need a **prebuilt OpenUSD 25.05** (imaging enabled;
Python optional) before `cargo build` works:

```sh
python OpenUSD/build_scripts/build_usd.py \
  --no-python --no-tests --no-examples --no-tutorials --no-docs --no-usdview \
  --alembic ~/USD
```

Then point rust-usd's build script at it:

| Variable | Meaning | Example |
| --- | --- | --- |
| `USD_INSTALL_DIR` | OpenUSD install prefix | `~/USD` |
| `USD_LIB_PREFIX` | Library name prefix USD was built with | `usd_` |
| `USD_LINK_PYTHON` | `none` for a `--no-python` USD; `framework` when USD links a macOS Python.framework | `none` |

With a `--no-python` USD (what CI uses), `USD_LINK_PYTHON=none` is the
whole story. A Python-enabled USD additionally needs
`USD_PYTHON_FRAMEWORK` / `USD_PYTHON_FRAMEWORK_DIR` /
`USD_PYTHON_INCLUDE_DIR` — see [`anvil/forge-paint.yaml`](anvil/forge-paint.yaml)
for a working macOS example. The exact CI recipes live in
[`.github/workflows/macos-build.yml`](.github/workflows/macos-build.yml)
and [`windows-build.yml`](.github/workflows/windows-build.yml).

```sh
cargo build --release
```

### Developing against rust-usd sources

`Cargo.toml` resolves `rust-usd` / `hydra-rs` from crates.io, so a fresh
clone builds without any sibling checkout. **If you develop against a
local `../rust-usd` tree, opt in** by creating an untracked
`.cargo/config.toml` (gitignored) next to this README:

```toml
[patch.crates-io]
rust-usd = { path = "../rust-usd" }
hydra-rs = { path = "../rust-usd/hydra-rs" }
```

Without that file, builds silently use the published crates — if a local
binding change "isn't taking effect", check that this file exists. With
the patch active, Cargo rewrites the two `Cargo.lock` entries to
path-flavored form; don't commit that flip.

### Working without OpenUSD

The `texture-baker` crate has no USD dependency, so on any machine you
can still run:

```sh
cargo fmt --all --check
cargo clippy -p texture-baker --all-targets -- -D warnings
cargo test -p texture-baker
```

That trio is exactly what [CI](.github/workflows/ci.yml) gates every
push and pull request on.

## Running

```sh
forge-paint [file.usd]         # GUI; also accepts .obj/.gltf/.glb/.abc and forge:// URIs
forge-paint bake --help        # headless mesh-map baking (texture-baker CLI)
forge-paint convert in.obj out.usdc   # headless model -> USD conversion
```

Optional env vars: `FORGE_PAINT_WORK_DIR` (sidecar save/load dir),
`FORGE_PAINT_RESOLUTION` (default tile resolution),
`FORGE_PAINT_LOG_FILE` (file log), `FORGE_PAINT_3DELIGHT_DIR` /
`DELIGHT` (3Delight discovery on Windows).

## Repository layout

| Path | What it is |
| --- | --- |
| `src/` | The application: UI (`app.rs`), viewport/renderer, paint stack, bake integration, USD load/save, converters |
| `crates/texture-baker/` | Vendored GPU/CPU mesh-map baker (library-only; no USD dependency) |
| `assets/` | Tiny starter assets; bring-your-own HDRIs/stencils ([policy](assets/README.md)) |
| `anvil/` | Package manifest for the forge/anvil pipeline (env + deps for `forge://` resolution) |
| `docs/` | [Release packaging](docs/release.md), [improvement roadmap](docs/improvement-roadmap.md), postmortems |
| `tools/` | Standalone helper scripts bundled into release zips |
| `attic/` | Retired code kept for reference ([why](attic/README.md)) |

## Development notes

- One-time repo-wide `rustfmt` commit is listed in
  [`.git-blame-ignore-revs`](.git-blame-ignore-revs); enable it locally with
  `git config blame.ignoreRevsFile .git-blame-ignore-revs`.
- Improvement plan and audit: [`docs/improvement-roadmap.md`](docs/improvement-roadmap.md).

## License

[MIT](LICENSE). The 3Delight SDK/runtime and OpenUSD have their own
licenses; neither is redistributed in this repository.

[`rust-usd`]: https://crates.io/crates/rust-usd
[`hydra-rs`]: https://crates.io/crates/hydra-rs
