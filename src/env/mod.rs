//! HDRI environment: load equirectangular HDR image → GPU texture + sampler,
//! plus bind-group plumbing and a small procedural "neutral studio" fallback
//! so there's always *something* lit before the user picks an asset.
//!
//! Phase 3.2.1: direct equirect sampling in the fragment shader.
//! Phase 3.2.2a: added BRDF integration LUT for Karis split-sum specular IBL.

pub mod brdf_lut;
pub mod skybox;
pub use brdf_lut::BrdfLut;
pub use skybox::SkyboxPipeline;

use anyhow::{anyhow, Context, Result};
use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu;
use half::f16;
use std::path::Path;

/// Matches WGSL `struct Env` in pbr.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct EnvUniforms {
    pub intensity: f32,
    pub rotation_y: f32,
    pub skybox_visible: u32,
    pub mip_count: f32, // used to compute specular LOD from roughness
}

impl Default for EnvUniforms {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            rotation_y: 0.0,
            skybox_visible: 0,
            mip_count: 1.0,
        }
    }
}

pub struct Environment {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub uniform_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub bgl: wgpu::BindGroupLayout,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
    pub name: String,
}

impl Environment {
    pub fn new_procedural(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        brdf_lut: &BrdfLut,
    ) -> Self {
        // 256×128 sky/ground gradient so the default IBL isn't just a directional
        // light. Small and still good for low-frequency bounce.
        const W: u32 = 256;
        const H: u32 = 128;
        let mut pixels: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
        for y in 0..H {
            // v goes 0..1 top→bottom; remap to latitude −π/2..π/2 with top = +π/2.
            let v = (y as f32 + 0.5) / H as f32;
            let lat = (1.0 - v) * std::f32::consts::PI - std::f32::consts::FRAC_PI_2;
            let s = (lat.sin() + 1.0) * 0.5; // 0 at ground, 1 at sky

            // Warm low-intensity sky (bright above horizon, warm dim below)
            let sky = [1.1_f32, 1.2, 1.4];
            let ground = [0.35_f32, 0.28, 0.22];
            let rgb = [
                ground[0] * (1.0 - s) + sky[0] * s,
                ground[1] * (1.0 - s) + sky[1] * s,
                ground[2] * (1.0 - s) + sky[2] * s,
            ];
            let row = [
                f16::from_f32(rgb[0]),
                f16::from_f32(rgb[1]),
                f16::from_f32(rgb[2]),
                f16::from_f32(1.0),
            ];
            for _ in 0..W {
                pixels.push(row);
            }
        }
        Self::from_equirect_rgba16f(
            device,
            queue,
            brdf_lut,
            "procedural_studio",
            W,
            H,
            &pixels,
        )
    }

    pub fn load_hdr(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        brdf_lut: &BrdfLut,
        path: &Path,
    ) -> Result<Self> {
        let img = image::open(path)
            .with_context(|| format!("open HDRI {}", path.display()))?;
        // `image::open` on a .hdr returns an Rgb32F image via the HdrDecoder path.
        let rgb32 = img.to_rgb32f();
        let (w, h) = (rgb32.width(), rgb32.height());
        let pixels: Vec<[f16; 4]> = rgb32
            .pixels()
            .map(|p| {
                [
                    f16::from_f32(p[0]),
                    f16::from_f32(p[1]),
                    f16::from_f32(p[2]),
                    f16::from_f32(1.0),
                ]
            })
            .collect();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("hdri")
            .to_string();
        Ok(Self::from_equirect_rgba16f(
            device, queue, brdf_lut, &name, w, h, &pixels,
        ))
    }

    /// Build an `Environment` from already-decoded Rgba16F equirectangular
    /// pixels. Generates a full mip chain so roughness → LOD works for cheap
    /// specular IBL.
    pub fn from_equirect_rgba16f(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        brdf_lut: &BrdfLut,
        name: &str,
        width: u32,
        height: u32,
        pixels: &[[f16; 4]],
    ) -> Self {
        let mip_count = mip_count_for(width.max(height));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("env.{name}")),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let bytes_per_row = width * 8; // Rgba16 = 8 bytes/texel
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Box-filter downsample each mip level on the CPU. Cheap for 2k input.
        let mut prev_level: Vec<[f16; 4]> = pixels.to_vec();
        let mut prev_w = width;
        let mut prev_h = height;
        for level in 1..mip_count {
            let w = (prev_w / 2).max(1);
            let h = (prev_h / 2).max(1);
            let mut next = Vec::with_capacity((w * h) as usize);
            for y in 0..h {
                for x in 0..w {
                    // Average 2x2 block from prev_level
                    let x0 = x * 2;
                    let y0 = y * 2;
                    let x1 = (x0 + 1).min(prev_w - 1);
                    let y1 = (y0 + 1).min(prev_h - 1);
                    let a = prev_level[(y0 * prev_w + x0) as usize];
                    let b = prev_level[(y0 * prev_w + x1) as usize];
                    let c = prev_level[(y1 * prev_w + x0) as usize];
                    let d = prev_level[(y1 * prev_w + x1) as usize];
                    let avg = |i: usize| -> f16 {
                        let s = a[i].to_f32() + b[i].to_f32() + c[i].to_f32() + d[i].to_f32();
                        f16::from_f32(s * 0.25)
                    };
                    next.push([avg(0), avg(1), avg(2), avg(3)]);
                }
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&next),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 8),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            prev_level = next;
            prev_w = w;
            prev_h = h;
        }

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(&format!("env.{name}.view")),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env.sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,    // longitude wraps
            address_mode_v: wgpu::AddressMode::ClampToEdge, // latitude clamps
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bgl = env_bgl(device);
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("env.uniform_buf"),
            size: std::mem::size_of::<EnvUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let u = EnvUniforms {
            mip_count: mip_count as f32,
            ..Default::default()
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("env.bind_group"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&brdf_lut.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&brdf_lut.sampler),
                },
            ],
        });

        Self {
            texture,
            view,
            sampler,
            uniform_buf,
            bind_group,
            bgl,
            width,
            height,
            mip_count,
            name: name.to_string(),
        }
    }

    pub fn write_uniforms(&self, queue: &wgpu::Queue, u: &EnvUniforms) {
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(u));
    }
}

pub fn env_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("env_bgl"),
        entries: &[
            // env uniforms
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // equirect env texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // shared BRDF integration LUT (RG16Float)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn mip_count_for(dim: u32) -> u32 {
    32 - dim.max(1).leading_zeros()
}

/// Attempt to load any bundled HDRIs from `assets/hdri/`. Returns `(name, path)`.
pub fn discover_bundled_hdris(root: &Path) -> Vec<(String, std::path::PathBuf)> {
    let dir = root.join("assets").join("hdri");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let is_hdr = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("hdr") || e.eq_ignore_ascii_case("exr"))
            .unwrap_or(false);
        if !is_hdr {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("hdri")
            .to_string();
        out.push((name, path));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[allow(dead_code)]
fn _unused(_: anyhow::Error) {}
