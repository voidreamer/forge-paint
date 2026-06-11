pub mod brush;
pub mod composite;
pub mod layer;
pub mod presets;
pub mod projection;
pub mod smart_mask;
pub mod smart_mask_pipeline;
pub mod target;
pub mod udim;

pub use brush::{BrushPipeline, BrushUniforms, PaintChannel};
pub use composite::Compositor;
pub use layer::{BlendMode, ChannelMask, FillParams, Layer, LayerKind, LayerStack, Mask};
pub use presets::SmartMaterialPreset;
pub use projection::{ProjBrushUniforms, ProjectionBrushPipeline};
pub use smart_mask::{SmartMaskParams, SmartMaskSource};
pub use smart_mask_pipeline::SmartMaskPipeline;
pub use target::{MaterialUniforms, PaintTarget, MAX_TILES};
