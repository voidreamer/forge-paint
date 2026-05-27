# forge-paint starter assets

Tracked assets in this directory are intentionally tiny and redistributable.

- `default_mesh/default.usda`: UV cube used as the startup mesh.
- `hdri/forge_studio_4x2.hdr`: generated studio-gradient HDRI for first launch.
- `stencils/forge_soft_round.png`: generated soft round stencil.
- `displacement/forge_checker_height.png`: generated checker height texture.
- `materials/*.usda`: small USD material presets.

Local production assets can live in these folders, but most image and mesh
formats are ignored by default to keep release artifacts and git history sane.
Do not commit third-party assets without adding their license information here.
