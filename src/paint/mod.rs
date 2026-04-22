pub mod brush;
pub mod composite;
pub mod layer;
pub mod target;
pub mod udim;

pub use brush::{BrushPipeline, BrushUniforms, PaintChannel};
pub use composite::Compositor;
pub use layer::{Layer, LayerStack, Mask};
pub use target::{MaterialUniforms, PaintTarget, MAX_TILES};
