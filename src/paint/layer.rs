//! A single paint layer: owns per-channel `texture_2d_array`s plus per-tile
//! views. Brushes stamp into these; the compositor samples them into the
//! shared display `PaintTarget` that the PBR shader reads.

use egui_wgpu::wgpu;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FillParams {
    /// sRGB-authored color. Written to the sRGB texture as-is (byte = sRGB
    /// value × 255); the sampler decodes it back to linear at read time.
    pub base_color_srgb: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
}

impl Default for FillParams {
    fn default() -> Self {
        Self {
            base_color_srgb: [0.5, 0.5, 0.5],
            roughness: 0.5,
            metallic: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerKind {
    /// Artist paints into full-resolution textures — the current behavior.
    Paint,
    /// Layer is a uniform material fill; no brush strokes. Textures are 1×1
    /// placeholders that the compositor samples the same way as Paint layers;
    /// `FillParams` is the source of truth.
    Fill(FillParams),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Multiply,
    Screen,
    Add,
}

impl BlendMode {
    pub fn label(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Add => "Add",
        }
    }

    pub const ALL: &'static [BlendMode] = &[
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Add,
    ];
}

pub struct Layer {
    pub name: String,
    pub opacity: f32,
    pub visible: bool,
    pub blend_mode: BlendMode,
    pub kind: LayerKind,

    pub base_color: wgpu::Texture,
    pub base_color_view: wgpu::TextureView,               // full array view
    pub base_color_layer_views: Vec<wgpu::TextureView>,    // per tile — stamp / sample

    pub roughness: wgpu::Texture,
    pub roughness_view: wgpu::TextureView,
    pub roughness_layer_views: Vec<wgpu::TextureView>,

    pub metallic: wgpu::Texture,
    pub metallic_view: wgpu::TextureView,
    pub metallic_layer_views: Vec<wgpu::TextureView>,

    pub normal: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub normal_layer_views: Vec<wgpu::TextureView>,

    /// Optional R8 mask (0 = hidden, 1 = fully visible). Absent = fully visible.
    pub mask: Option<Mask>,
}

pub struct Mask {
    pub texture: wgpu::Texture,
    pub array_view: wgpu::TextureView,
    pub layer_views: Vec<wgpu::TextureView>,
}

impl Mask {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        tile_count: u32,
    ) -> Self {
        let tile_count = tile_count.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("layer.mask"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: tile_count,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("layer.mask.array_view"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views: Vec<_> = (0..tile_count)
            .map(|t| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("layer.mask.tile_view"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: t,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Seed to 1.0 everywhere — "fully visible" default.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mask.init_fill"),
        });
        for t in 0..tile_count as usize {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mask.init_fill_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &layer_views[t],
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 1.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
        }
        queue.submit(Some(encoder.finish()));

        Self {
            texture,
            array_view,
            layer_views,
        }
    }
}

impl Layer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: impl Into<String>,
        resolution: u32,
        tile_count: u32,
    ) -> Self {
        let tile_count = tile_count.max(1);

        let base_color = make_array(
            device,
            "layer.base_color",
            resolution,
            tile_count,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let roughness = make_array(
            device,
            "layer.roughness",
            resolution,
            tile_count,
            wgpu::TextureFormat::R8Unorm,
        );
        let metallic = make_array(
            device,
            "layer.metallic",
            resolution,
            tile_count,
            wgpu::TextureFormat::R8Unorm,
        );
        let normal = make_array(
            device,
            "layer.normal",
            resolution,
            tile_count,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let base_color_view = array_view(&base_color, "layer.base_color.array_view");
        let roughness_view = array_view(&roughness, "layer.roughness.array_view");
        let metallic_view = array_view(&metallic, "layer.metallic.array_view");
        let normal_view = array_view(&normal, "layer.normal.array_view");

        let base_color_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&base_color, t, "layer.base_color.tile_view"))
            .collect();
        let roughness_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&roughness, t, "layer.roughness.tile_view"))
            .collect();
        let metallic_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&metallic, t, "layer.metallic.tile_view"))
            .collect();
        let normal_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&normal, t, "layer.normal.tile_view"))
            .collect();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer.init_fill"),
        });
        for t in 0..tile_count {
            for (view, clear) in [
                (
                    &base_color_layer_views[t as usize],
                    crate::paint::target::defaults::base_color_clear(),
                ),
                (
                    &roughness_layer_views[t as usize],
                    crate::paint::target::defaults::roughness_clear(),
                ),
                (
                    &metallic_layer_views[t as usize],
                    crate::paint::target::defaults::metallic_clear(),
                ),
                (
                    &normal_layer_views[t as usize],
                    crate::paint::target::defaults::normal_clear(),
                ),
            ] {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("layer.init_fill_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(clear),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });
            }
        }
        queue.submit(Some(encoder.finish()));

        Self {
            name: name.into(),
            opacity: 1.0,
            visible: true,
            blend_mode: BlendMode::Normal,
            kind: LayerKind::Paint,
            base_color,
            base_color_view,
            base_color_layer_views,
            roughness,
            roughness_view,
            roughness_layer_views,
            metallic,
            metallic_view,
            metallic_layer_views,
            normal,
            normal_view,
            normal_layer_views,
            mask: None,
        }
    }

    pub fn add_mask(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        tile_count: u32,
    ) {
        if self.mask.is_none() {
            self.mask = Some(Mask::new(device, queue, resolution, tile_count));
        }
    }

    pub fn remove_mask(&mut self) {
        self.mask = None;
    }

    pub fn is_fill(&self) -> bool {
        matches!(self.kind, LayerKind::Fill(_))
    }

    pub fn fill_params(&self) -> Option<FillParams> {
        if let LayerKind::Fill(p) = self.kind {
            Some(p)
        } else {
            None
        }
    }

    /// Construct a Fill layer. Its channel textures are 1×1 per tile —
    /// effectively free — and hold bytes derived from `FillParams::default()`.
    /// Use `set_fill_params` to update the stored values (and re-upload the
    /// 1×1 tiles) when the user drags sliders.
    pub fn new_fill(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        name: impl Into<String>,
        tile_count: u32,
    ) -> Self {
        let tile_count = tile_count.max(1);

        let base_color = make_array(
            device,
            "layer.fill.base_color",
            1,
            tile_count,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let roughness = make_array(
            device,
            "layer.fill.roughness",
            1,
            tile_count,
            wgpu::TextureFormat::R8Unorm,
        );
        let metallic = make_array(
            device,
            "layer.fill.metallic",
            1,
            tile_count,
            wgpu::TextureFormat::R8Unorm,
        );
        let normal = make_array(
            device,
            "layer.fill.normal",
            1,
            tile_count,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let base_color_view = array_view(&base_color, "layer.fill.base_color.array_view");
        let roughness_view = array_view(&roughness, "layer.fill.roughness.array_view");
        let metallic_view = array_view(&metallic, "layer.fill.metallic.array_view");
        let normal_view = array_view(&normal, "layer.fill.normal.array_view");

        let base_color_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&base_color, t, "layer.fill.base_color.tile_view"))
            .collect();
        let roughness_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&roughness, t, "layer.fill.roughness.tile_view"))
            .collect();
        let metallic_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&metallic, t, "layer.fill.metallic.tile_view"))
            .collect();
        let normal_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&normal, t, "layer.fill.normal.tile_view"))
            .collect();

        let layer = Self {
            name: name.into(),
            opacity: 1.0,
            visible: true,
            blend_mode: BlendMode::Normal,
            kind: LayerKind::Fill(FillParams::default()),
            base_color,
            base_color_view,
            base_color_layer_views,
            roughness,
            roughness_view,
            roughness_layer_views,
            metallic,
            metallic_view,
            metallic_layer_views,
            normal,
            normal_view,
            normal_layer_views,
            mask: None,
        };
        layer.upload_fill_bytes(queue);
        layer
    }

    /// Replace the fill parameters and re-upload the 1×1 tiles. No-op for
    /// Paint layers.
    pub fn set_fill_params(&mut self, queue: &wgpu::Queue, params: FillParams) {
        match &mut self.kind {
            LayerKind::Fill(stored) => {
                if *stored == params {
                    return;
                }
                *stored = params;
            }
            LayerKind::Paint => return,
        }
        self.upload_fill_bytes(queue);
    }

    fn upload_fill_bytes(&self, queue: &wgpu::Queue) {
        let LayerKind::Fill(params) = self.kind else {
            return;
        };
        // sRGB: write byte = srgb × 255. Sampler decodes on read.
        let bc = [
            (params.base_color_srgb[0].clamp(0.0, 1.0) * 255.0) as u8,
            (params.base_color_srgb[1].clamp(0.0, 1.0) * 255.0) as u8,
            (params.base_color_srgb[2].clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ];
        let r = [(params.roughness.clamp(0.0, 1.0) * 255.0) as u8];
        let m = [(params.metallic.clamp(0.0, 1.0) * 255.0) as u8];
        // Flat tangent-space normal — fill layers don't perturb normals.
        let nm = [128u8, 128, 255, 255];

        let tiles = self.base_color_layer_views.len() as u32;
        for tile in 0..tiles {
            for (tex, bytes, bpp) in [
                (&self.base_color, &bc[..], 4u32),
                (&self.roughness, &r[..], 1u32),
                (&self.metallic, &m[..], 1u32),
                (&self.normal, &nm[..], 4u32),
            ] {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d {
                            x: 0,
                            y: 0,
                            z: tile,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytes,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bpp),
                        rows_per_image: Some(1),
                    },
                    wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
}

/// Stack of paint layers. Phase 3.1.1 always has exactly one layer; 3.1.2
/// adds / deletes / reorders.
pub struct LayerStack {
    pub layers: Vec<Layer>,
    pub active: usize,
    pub resolution: u32,
    pub tile_count: u32,
}

impl LayerStack {
    pub fn new_with_initial_layer(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        resolution: u32,
        tile_count: u32,
    ) -> Self {
        let initial = Layer::new(device, queue, "Layer 1", resolution, tile_count);
        Self {
            layers: vec![initial],
            active: 0,
            resolution,
            tile_count,
        }
    }

    pub fn active_layer(&self) -> &Layer {
        &self.layers[self.active]
    }

    pub fn active_layer_mut(&mut self) -> &mut Layer {
        &mut self.layers[self.active]
    }

    /// Append a fresh Paint layer on top of the stack and make it active.
    pub fn add_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let name = self.unique_name("Layer");
        let layer = Layer::new(device, queue, name, self.resolution, self.tile_count);
        self.layers.push(layer);
        self.active = self.layers.len() - 1;
    }

    /// Append a fresh Fill layer on top of the stack and make it active.
    pub fn add_fill_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let name = self.unique_name("Fill");
        let layer = Layer::new_fill(device, queue, name, self.tile_count);
        self.layers.push(layer);
        self.active = self.layers.len() - 1;
    }

    /// Remove the layer at `idx`. Always keeps at least one layer — no-op if
    /// the stack would empty.
    pub fn remove_at(&mut self, idx: usize) {
        if self.layers.len() <= 1 || idx >= self.layers.len() {
            return;
        }
        self.layers.remove(idx);
        if self.active >= self.layers.len() {
            self.active = self.layers.len() - 1;
        }
    }

    pub fn set_active(&mut self, idx: usize) {
        if idx < self.layers.len() {
            self.active = idx;
        }
    }

    pub fn add_mask_to(&mut self, idx: usize, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(l) = self.layers.get_mut(idx) {
            l.add_mask(device, queue, self.resolution, self.tile_count);
        }
    }

    pub fn remove_mask_from(&mut self, idx: usize) {
        if let Some(l) = self.layers.get_mut(idx) {
            l.remove_mask();
        }
    }

    fn unique_name(&self, base: &str) -> String {
        let mut n = self.layers.len() + 1;
        loop {
            let candidate = format!("{base} {n}");
            if !self.layers.iter().any(|l| l.name == candidate) {
                return candidate;
            }
            n += 1;
        }
    }
}

fn make_array(
    device: &wgpu::Device,
    label: &str,
    resolution: u32,
    layers: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

fn array_view(tex: &wgpu::Texture, label: &str) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn tile_view(tex: &wgpu::Texture, layer: u32, label: &str) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_array_layer: layer,
        array_layer_count: Some(1),
        ..Default::default()
    })
}
