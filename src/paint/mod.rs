pub mod brush;
pub mod composite;
pub mod layer;
pub mod projection;
pub mod target;
pub mod udim;

pub use brush::{BrushPipeline, BrushUniforms, PaintChannel};
pub use composite::Compositor;
pub use layer::{BlendMode, FillParams, Layer, LayerKind, LayerStack, Mask};
pub use projection::{ProjBrushUniforms, ProjectionBrushPipeline};
pub use target::{MaterialUniforms, PaintTarget, MAX_TILES};
