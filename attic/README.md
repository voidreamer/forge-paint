# attic/

Code that has been removed from the build but is worth keeping nearby
for future reference. Files here are NOT compiled — they live outside
`src/` and are never declared as modules. They survive in-tree (rather
than only in git history) so a curious reader can find the prior
attempt without spelunking `git log`.

## Current contents

### `hydra_view.rs`

The Hydra (Storm) preview viewport wrapper. It rendered the active USD
stage through `hydra_rs::Renderer` into a side panel next to the wgpu
painter — a "production reference" second view rather than a paint
target. It built and rendered, but the integration was unreliable
enough (lighting routing, projection conventions, single-threaded
renderer constraints, GPU requirements) that keeping it in the build
was costing more than it bought us. Parked rather than deleted because
the matrix-bridge and lighting-model notes inside the file are
expensive to re-derive.

What was removed alongside the move:

- `hydra-rs = "0.0.2"` dependency in `Cargo.toml`
- `mod hydra_view;` in `src/main.rs`
- `App::show_hydra_view` / `App::hydra` / `App::hydra_egui_tex` fields
- The "Hydra preview" View-menu checkbox
- `App::show_hydra_window`, the side panel + render path

The file's top-of-module comment has the recipe for reviving it.
