//! A single paint layer: owns per-channel `texture_2d_array`s plus per-tile
//! views. Brushes stamp into these; the compositor samples them into the
//! shared display `PaintTarget` that the PBR shader reads.

use egui_wgpu::wgpu;

pub struct Layer {
    pub name: String,
    pub opacity: f32,
    pub visible: bool,

    pub base_color: wgpu::Texture,
    pub base_color_view: wgpu::TextureView,               // full array view
    pub base_color_layer_views: Vec<wgpu::TextureView>,    // per tile — stamp / sample

    pub rough_metal: wgpu::Texture,
    pub rough_metal_view: wgpu::TextureView,
    pub rough_metal_layer_views: Vec<wgpu::TextureView>,

    pub normal: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub normal_layer_views: Vec<wgpu::TextureView>,
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
        let rough_metal = make_array(
            device,
            "layer.rough_metal",
            resolution,
            tile_count,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let normal = make_array(
            device,
            "layer.normal",
            resolution,
            tile_count,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        let base_color_view = array_view(&base_color, "layer.base_color.array_view");
        let rough_metal_view = array_view(&rough_metal, "layer.rough_metal.array_view");
        let normal_view = array_view(&normal, "layer.normal.array_view");

        let base_color_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&base_color, t, "layer.base_color.tile_view"))
            .collect();
        let rough_metal_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&rough_metal, t, "layer.rough_metal.tile_view"))
            .collect();
        let normal_layer_views: Vec<_> = (0..tile_count)
            .map(|t| tile_view(&normal, t, "layer.normal.tile_view"))
            .collect();

        // Seed neutral defaults via GPU clear-fill passes — same technique as
        // the display PaintTarget uses. No host-side buffers.
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
                    &rough_metal_layer_views[t as usize],
                    crate::paint::target::defaults::rough_metal_clear(),
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
            base_color,
            base_color_view,
            base_color_layer_views,
            rough_metal,
            rough_metal_view,
            rough_metal_layer_views,
            normal,
            normal_view,
            normal_layer_views,
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

    /// Append a fresh layer on top of the stack and make it active.
    pub fn add_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let name = self.unique_name("Layer");
        let layer = Layer::new(device, queue, name, self.resolution, self.tile_count);
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
