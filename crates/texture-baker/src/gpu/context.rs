use wgpu;

/// Shared GPU context: device + queue.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Initialize GPU. Returns None if no suitable adapter found.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // wgpu 24: request_adapter returns Option, request_device takes a
        // separate `trace_path` argument and DeviceDescriptor has no
        // `trace` field. Upstream texture-baker is on wgpu 25 — patches
        // adapt the API surface so we can share the workspace's wgpu.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        log::info!(
            "GPU: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("texture-baker"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_storage_buffer_binding_size: 512 * 1024 * 1024,
                        max_buffer_size: 512 * 1024 * 1024,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        Some(GpuContext { device, queue })
    }
}
