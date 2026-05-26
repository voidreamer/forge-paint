//! egui-snarl-based material network editor.
//!
//! Renders the currently-bound library material as a node graph in the
//! Material properties tab. v1 scope:
//!
//! - **Shader** node — the bound material, one input pin per editable
//!   slot (Base Color, Metallic, Roughness, …) with the existing
//!   inline slider / colour-picker controls. Slots the shader doesn't
//!   expose for its kind render disabled.
//! - **Output** node — sink for the shader's surface terminal.
//! - **Texture** nodes — file picker + UV-scale knobs, single RGB
//!   output. Added through the right-click "Add → Texture" menu;
//!   wires from a texture's RGB output to a shader-input pin are
//!   captured visually and persisted via the project sidecar.
//!
//! Texture connections are visual-only in v1 — hooking them up to
//! Hydra requires a `set_external_material_input_texture` entry
//! point on hydra-rs which lands in a follow-up. Scalar / colour
//! pin edits work today; they mutate the shared `MaterialInputs`
//! buffer that the per-frame `draw_hydra_central` override push
//! already consumes, so live preview keeps working unchanged.

use std::path::PathBuf;

use eframe::egui;
use egui_snarl::{
    ui::{PinInfo, SnarlViewer},
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
};

use crate::assets::{MaterialInputs, ShaderInputNames};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MaterialNode {
    Shader,
    Texture(TextureNode),
    Output,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TextureNode {
    pub path: PathBuf,
    pub uv_scale: [f32; 2],
}

impl Default for TextureNode {
    fn default() -> Self {
        Self {
            path: PathBuf::new(),
            uv_scale: [1.0, 1.0],
        }
    }
}

/// Per-material-binding graph state. Persisted via the project
/// sidecar so node positions + texture wiring round-trip.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaterialGraph {
    pub snarl: Snarl<MaterialNode>,
    pub shader_node: Option<NodeId>,
    pub output_node: Option<NodeId>,
}

impl Default for MaterialGraph {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            shader_node: None,
            output_node: None,
        }
    }
}

impl MaterialGraph {
    /// Wipe and recreate the graph for a freshly-bound material:
    /// a Shader node wired through to an Output node. Texture
    /// nodes the user adds afterwards stay in `snarl`.
    pub fn rebuild_for_material(&mut self) {
        self.snarl = Snarl::new();
        let shader = self
            .snarl
            .insert_node(egui::pos2(40.0, 80.0), MaterialNode::Shader);
        let output = self
            .snarl
            .insert_node(egui::pos2(420.0, 80.0), MaterialNode::Output);
        let _ = self.snarl.connect(
            OutPinId {
                node: shader,
                output: 0,
            },
            InPinId {
                node: output,
                input: 0,
            },
        );
        self.shader_node = Some(shader);
        self.output_node = Some(output);
    }
}

/// User-facing slot order on the Shader node. Indexed by input pin
/// number, so `show_input` reads `SHADER_PINS[pin.id.input]`.
#[derive(Copy, Clone, Debug)]
pub enum ShaderPin {
    DiffuseColor,
    Metallic,
    Roughness,
    Opacity,
    Clearcoat,
    ClearcoatRoughness,
    EmissionColor,
    EmissionIntensity,
}

pub const SHADER_PINS: &[ShaderPin] = &[
    ShaderPin::DiffuseColor,
    ShaderPin::Metallic,
    ShaderPin::Roughness,
    ShaderPin::Opacity,
    ShaderPin::Clearcoat,
    ShaderPin::ClearcoatRoughness,
    ShaderPin::EmissionColor,
    ShaderPin::EmissionIntensity,
];

impl ShaderPin {
    pub fn label(self) -> &'static str {
        match self {
            ShaderPin::DiffuseColor => "Base Color",
            ShaderPin::Metallic => "Metallic",
            ShaderPin::Roughness => "Roughness",
            ShaderPin::Opacity => "Opacity",
            ShaderPin::Clearcoat => "Clearcoat",
            ShaderPin::ClearcoatRoughness => "Clearcoat Rough",
            ShaderPin::EmissionColor => "Emission Color",
            ShaderPin::EmissionIntensity => "Emission Intensity",
        }
    }

    pub fn shader_input_name<'a>(self, names: &'a ShaderInputNames) -> Option<&'a str> {
        match self {
            ShaderPin::DiffuseColor => names.diffuse_color,
            ShaderPin::Metallic => names.metallic,
            ShaderPin::Roughness => names.roughness,
            ShaderPin::Opacity => names.opacity,
            ShaderPin::Clearcoat => names.clearcoat,
            ShaderPin::ClearcoatRoughness => names.clearcoat_roughness,
            ShaderPin::EmissionColor => names.emission_color,
            ShaderPin::EmissionIntensity => names.emission_intensity,
        }
    }
}

pub struct GraphViewer<'a> {
    pub material_title: String,
    pub shader_inputs: ShaderInputNames,
    pub inputs: &'a mut MaterialInputs,
}

impl<'a> SnarlViewer<MaterialNode> for GraphViewer<'a> {
    fn title(&mut self, node: &MaterialNode) -> String {
        match node {
            MaterialNode::Shader => self.material_title.clone(),
            MaterialNode::Texture(t) => t
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("texture")
                .to_string(),
            MaterialNode::Output => "Output".to_string(),
        }
    }

    fn inputs(&mut self, node: &MaterialNode) -> usize {
        match node {
            MaterialNode::Shader => SHADER_PINS.len(),
            MaterialNode::Texture(_) => 0,
            MaterialNode::Output => 1,
        }
    }

    fn outputs(&mut self, node: &MaterialNode) -> usize {
        match node {
            MaterialNode::Shader => 1,
            MaterialNode::Texture(_) => 1,
            MaterialNode::Output => 0,
        }
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<MaterialNode>,
    ) -> PinInfo {
        match &snarl[pin.id.node] {
            MaterialNode::Shader => {
                let slot = SHADER_PINS
                    .get(pin.id.input)
                    .copied()
                    .unwrap_or(ShaderPin::DiffuseColor);
                let supported = slot.shader_input_name(&self.shader_inputs).is_some();
                ui.add_enabled_ui(supported, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(slot.label());
                        match slot {
                            ShaderPin::DiffuseColor => {
                                ui.color_edit_button_rgb(&mut self.inputs.diffuse_color);
                            }
                            ShaderPin::Metallic => {
                                ui.add(
                                    egui::Slider::new(&mut self.inputs.metallic, 0.0..=1.0)
                                        .show_value(false),
                                );
                            }
                            ShaderPin::Roughness => {
                                ui.add(
                                    egui::Slider::new(&mut self.inputs.roughness, 0.0..=1.0)
                                        .show_value(false),
                                );
                            }
                            ShaderPin::Opacity => {
                                ui.add(
                                    egui::Slider::new(&mut self.inputs.opacity, 0.0..=1.0)
                                        .show_value(false),
                                );
                            }
                            ShaderPin::Clearcoat => {
                                ui.add(
                                    egui::Slider::new(&mut self.inputs.clearcoat, 0.0..=2.0)
                                        .show_value(false),
                                );
                            }
                            ShaderPin::ClearcoatRoughness => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.inputs.clearcoat_roughness,
                                        0.0..=1.0,
                                    )
                                    .show_value(false),
                                );
                            }
                            ShaderPin::EmissionColor => {
                                ui.color_edit_button_rgb(&mut self.inputs.emission_color);
                            }
                            ShaderPin::EmissionIntensity => {
                                ui.add(
                                    egui::Slider::new(
                                        &mut self.inputs.emission_intensity,
                                        0.0..=10.0,
                                    )
                                    .show_value(false),
                                );
                            }
                        }
                    });
                });
                pin_color(supported)
            }
            MaterialNode::Output => {
                ui.label("Surface");
                PinInfo::square().with_fill(egui::Color32::from_rgb(120, 200, 120))
            }
            MaterialNode::Texture(_) => PinInfo::circle(),
        }
    }

    fn show_output(
        &mut self,
        pin: &OutPin,
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<MaterialNode>,
    ) -> PinInfo {
        match &snarl[pin.id.node] {
            MaterialNode::Shader => {
                ui.label("Surface");
                PinInfo::square().with_fill(egui::Color32::from_rgb(120, 200, 120))
            }
            MaterialNode::Texture(_) => {
                ui.label("RGB");
                PinInfo::circle().with_fill(egui::Color32::from_rgb(240, 200, 120))
            }
            MaterialNode::Output => PinInfo::circle(),
        }
    }

    /// Right-click on empty graph background → "Add" menu.
    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<MaterialNode>) -> bool {
        true
    }

    fn show_graph_menu(
        &mut self,
        pos: egui::Pos2,
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<MaterialNode>,
    ) {
        ui.label("Add");
        if ui.button("Texture").clicked() {
            let path = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg", "exr", "hdr", "tif", "tiff"])
                .pick_file()
                .unwrap_or_default();
            snarl.insert_node(
                pos,
                MaterialNode::Texture(TextureNode {
                    path,
                    uv_scale: [1.0, 1.0],
                }),
            );
            ui.close_menu();
        }
    }

    /// Restrict wiring: only Texture-RGB → Shader-input is meaningful
    /// in v1. Anything else (texture→output, shader→shader, …) gets
    /// silently rejected.
    fn connect(
        &mut self,
        from: &OutPin,
        to: &InPin,
        snarl: &mut Snarl<MaterialNode>,
    ) {
        let from_is_texture = matches!(snarl[from.id.node], MaterialNode::Texture(_));
        let to_is_shader = matches!(snarl[to.id.node], MaterialNode::Shader);
        let surface_terminal = matches!(snarl[from.id.node], MaterialNode::Shader)
            && matches!(snarl[to.id.node], MaterialNode::Output);
        if (from_is_texture && to_is_shader) || surface_terminal {
            // Replace any existing wire on the destination input —
            // shader inputs are single-valued.
            for existing in &to.remotes {
                snarl.disconnect(*existing, to.id);
            }
            snarl.connect(from.id, to.id);
        }
    }
}

fn pin_color(supported: bool) -> PinInfo {
    let fill = if supported {
        egui::Color32::WHITE
    } else {
        egui::Color32::DARK_GRAY
    };
    PinInfo::circle().with_fill(fill)
}
