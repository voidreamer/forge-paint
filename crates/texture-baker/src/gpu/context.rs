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

        // Texture-baker's GPU paths need fat storage buffers — at 8K
        // resolution the AO texel buffer alone is ~2 GB. Pull whatever
        // the adapter actually advertises rather than hard-coding 512
        // MB; on Apple Silicon / RTX this gets us into the multi-GB
        // range. The defaults stay at wgpu::Limits::default() for every
        // other knob.
        let adapter_limits = adapter.limits();
        let limits = wgpu::Limits {
            max_buffer_size: adapter_limits.max_buffer_size,
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            ..wgpu::Limits::default()
        };

        log::info!(
            "GPU buffer limits: max_buffer={} MB, max_storage_binding={} MB",
            limits.max_buffer_size / (1024 * 1024),
            limits.max_storage_buffer_binding_size / (1024 * 1024),
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("texture-baker"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;

        Some(GpuContext { device, queue })
    }
}
