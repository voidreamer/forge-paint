//! egui-snarl-based material network editor.
//!
//! C2b model: each `Shader` node in the graph represents one
//! `MaterialBindingInstance` from the App's `material_bindings`
//! Vec — `binding_id` is the back-reference. Clicking a chip in
//! the gallery spawns a new Shader node (unassigned); right-clicking
//! a Shader node opens an assignment menu ("Assign to selection",
//! "Assign to stage-wide", "Unassign", "Remove"). The per-frame
//! draw_hydra_central loop reads the bindings Vec back out and
//! pushes only the assigned ones through hydra-rs.
//!
//! Texture node connections are represented in the editor graph. Imported USDZ
//! materials also author those connections into generated UsdPreviewSurface
//! sources so Hydra can consume them through the existing external-material
//! binding path.

use std::path::PathBuf;

use eframe::egui;
use egui_snarl::{
    InPin, InPinId, NodeId, OutPin, OutPinId, Snarl,
    ui::{PinInfo, SnarlViewer},
};

use crate::assets::ShaderInputNames;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MaterialNode {
    /// Backed by `App.material_bindings[binding_id]`. Inputs are
    /// rendered as port rows with inline sliders / colour pickers
    /// that mutate the binding's `MaterialInputs` directly.
    Shader { binding_id: u64 },
    /// Texture node — visual only in v1.
    Texture(TextureNode),
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

/// Per-stage material network. Lives on `App`; persists across
/// renderer switches and binding edits within a session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaterialGraph {
    pub snarl: Snarl<MaterialNode>,
    /// Auto-placement cursor for new Shader nodes — advances on
    /// each spawn so chips dropped in quick succession don't all
    /// pile up at the same position.
    next_node_pos: [f32; 2],
    /// snarl's viewport scale as of the last draw, captured by the
    /// viewer's `draw_background`. Runtime-only — used to convert
    /// screen-space pan deltas into graph space.
    #[serde(skip)]
    pub last_scale: Option<f32>,
}

impl Default for MaterialGraph {
    fn default() -> Self {
        Self {
            snarl: Snarl::new(),
            next_node_pos: [40.0, 40.0],
            last_scale: None,
        }
    }
}

impl MaterialGraph {
    /// Spawn a Shader node for the given binding at the next auto-
    /// layout position. Caller is responsible for adding the
    /// matching MaterialBindingInstance to the App's Vec first.
    pub fn spawn_shader_node(&mut self, binding_id: u64) -> NodeId {
        let pos = egui::pos2(self.next_node_pos[0], self.next_node_pos[1]);
        let id = self
            .snarl
            .insert_node(pos, MaterialNode::Shader { binding_id });
        self.next_node_pos[0] += 220.0;
        if self.next_node_pos[0] > 800.0 {
            self.next_node_pos[0] = 40.0;
            self.next_node_pos[1] += 260.0;
        }
        id
    }

    pub fn spawn_texture_node_at(&mut self, path: PathBuf, pos: egui::Pos2) -> NodeId {
        self.snarl.insert_node(
            pos,
            MaterialNode::Texture(TextureNode {
                path,
                uv_scale: [1.0, 1.0],
            }),
        )
    }

    pub fn connect_texture_to_shader(
        &mut self,
        texture_node: NodeId,
        shader_node: NodeId,
        pin: ShaderPin,
    ) {
        let Some(input) = SHADER_PINS.iter().position(|&p| p == pin) else {
            return;
        };
        self.snarl.connect(
            OutPinId {
                node: texture_node,
                output: 0,
            },
            InPinId {
                node: shader_node,
                input,
            },
        );
    }

    /// Drop the Shader node backing `binding_id`, if any. Used when
    /// the user explicitly removes a binding from the graph's
    /// right-click menu.
    pub fn remove_shader_node(&mut self, binding_id: u64) {
        let ids: Vec<NodeId> = self
            .snarl
            .node_ids()
            .filter_map(|(id, node)| match node {
                MaterialNode::Shader { binding_id: b } if *b == binding_id => Some(id),
                _ => None,
            })
            .collect();
        for id in ids {
            self.snarl.remove_node(id);
        }
    }

    /// Pan the visible graph by moving every node together. egui-snarl's
    /// built-in viewport transform is private, so this gives us reliable
    /// middle-mouse canvas panning even when the cursor is over a node body.
    pub fn pan_nodes_by(&mut self, delta: egui::Vec2) {
        if delta == egui::Vec2::ZERO {
            return;
        }
        for node in self.snarl.nodes_info_mut() {
            node.pos += delta;
        }
        self.next_node_pos[0] += delta.x;
        self.next_node_pos[1] += delta.y;
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShaderPin {
    DiffuseColor,
    Metallic,
    Roughness,
    Normal,
    Opacity,
    Clearcoat,
    ClearcoatRoughness,
    Occlusion,
    EmissionColor,
    EmissionIntensity,
}

pub const SHADER_PINS: &[ShaderPin] = &[
    ShaderPin::DiffuseColor,
    ShaderPin::Metallic,
    ShaderPin::Roughness,
    ShaderPin::Normal,
    ShaderPin::Opacity,
    ShaderPin::Clearcoat,
    ShaderPin::ClearcoatRoughness,
    ShaderPin::Occlusion,
    ShaderPin::EmissionColor,
    ShaderPin::EmissionIntensity,
];

impl ShaderPin {
    pub fn label(self) -> &'static str {
        match self {
            ShaderPin::DiffuseColor => "Base Color",
            ShaderPin::Metallic => "Metallic",
            ShaderPin::Roughness => "Roughness",
            ShaderPin::Normal => "Normal",
            ShaderPin::Opacity => "Opacity",
            ShaderPin::Clearcoat => "Clearcoat",
            ShaderPin::ClearcoatRoughness => "Clearcoat Rough",
            ShaderPin::Occlusion => "Occlusion",
            ShaderPin::EmissionColor => "Emission Color",
            ShaderPin::EmissionIntensity => "Emission Intensity",
        }
    }

    pub fn shader_input_name<'a>(self, names: &'a ShaderInputNames) -> Option<&'a str> {
        match self {
            ShaderPin::DiffuseColor => names.diffuse_color,
            ShaderPin::Metallic => names.metallic,
            ShaderPin::Roughness => names.roughness,
            ShaderPin::Normal => names.normal,
            ShaderPin::Opacity => names.opacity,
            ShaderPin::Clearcoat => names.clearcoat,
            ShaderPin::ClearcoatRoughness => names.clearcoat_roughness,
            ShaderPin::Occlusion => names.occlusion,
            ShaderPin::EmissionColor => names.emission_color,
            ShaderPin::EmissionIntensity => names.emission_intensity,
        }
    }
}

/// Action the graph viewer emits as the user interacts with the
/// right-click menu on a Shader node. App drains the Vec after
/// snarl.show returns.
#[derive(Debug, Clone)]
pub enum GraphAction {
    AssignToSelection(u64),
    AssignToStage(u64),
    Unassign(u64),
    Remove(u64),
}

pub struct GraphViewer<'a> {
    pub bindings: &'a mut Vec<crate::app::MaterialBindingInstance>,
    pub browser_selection: &'a std::collections::HashSet<String>,
    pub pending_actions: &'a mut Vec<GraphAction>,
    /// Written every frame from `draw_background` — snarl's viewport
    /// scale, which the show() API doesn't otherwise expose.
    pub scale_out: &'a mut f32,
}

impl<'a> GraphViewer<'a> {
    fn binding_for(&self, binding_id: u64) -> Option<&crate::app::MaterialBindingInstance> {
        self.bindings.iter().find(|b| b.id == binding_id)
    }

    fn binding_for_mut(
        &mut self,
        binding_id: u64,
    ) -> Option<&mut crate::app::MaterialBindingInstance> {
        self.bindings.iter_mut().find(|b| b.id == binding_id)
    }
}

impl<'a> SnarlViewer<MaterialNode> for GraphViewer<'a> {
    /// Default background drawing plus a scale capture — snarl passes
    /// its live viewport here and nowhere else we can reach.
    fn draw_background(
        &mut self,
        background: Option<&egui_snarl::ui::BackgroundPattern>,
        viewport: &egui_snarl::ui::Viewport,
        snarl_style: &egui_snarl::ui::SnarlStyle,
        style: &egui::Style,
        painter: &egui::Painter,
        _snarl: &Snarl<MaterialNode>,
    ) {
        *self.scale_out = viewport.scale;
        if let Some(background) = background {
            background.draw(viewport, snarl_style, style, painter);
        }
    }

    fn title(&mut self, node: &MaterialNode) -> String {
        match node {
            MaterialNode::Shader { binding_id } => {
                let b = self.binding_for(*binding_id);
                let stem = b
                    .map(|b| {
                        b.source
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| format!("material#{binding_id}"))
                    })
                    .unwrap_or_else(|| format!("material#{binding_id} (missing)"));
                let scope = b
                    .map(|b| {
                        if !b.assigned {
                            "unassigned".to_string()
                        } else if b.target_prims.is_empty() {
                            "stage".to_string()
                        } else {
                            format!("{} prim(s)", b.target_prims.len())
                        }
                    })
                    .unwrap_or_default();
                if scope.is_empty() {
                    stem
                } else {
                    format!("{stem} · {scope}")
                }
            }
            MaterialNode::Texture(t) => t
                .path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("texture")
                .to_string(),
        }
    }

    fn inputs(&mut self, node: &MaterialNode) -> usize {
        match node {
            MaterialNode::Shader { .. } => SHADER_PINS.len(),
            MaterialNode::Texture(_) => 0,
        }
    }

    fn outputs(&mut self, node: &MaterialNode) -> usize {
        match node {
            MaterialNode::Shader { .. } => 1,
            MaterialNode::Texture(_) => 1,
        }
    }

    fn has_body(&mut self, node: &MaterialNode) -> bool {
        matches!(node, MaterialNode::Shader { .. })
    }

    fn show_body(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<MaterialNode>,
    ) {
        let binding_id = match &snarl[node] {
            MaterialNode::Shader { binding_id } => *binding_id,
            MaterialNode::Texture(_) => return,
        };
        if let Some(binding) = self.binding_for(binding_id) {
            let (rect, _) = ui.allocate_exact_size(egui::vec2(76.0, 76.0), egui::Sense::hover());
            crate::assets::paint_material_preview_ball(ui, rect, binding.inputs);
        }
    }

    fn show_input(
        &mut self,
        pin: &InPin,
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<MaterialNode>,
    ) -> PinInfo {
        match snarl[pin.id.node].clone() {
            MaterialNode::Shader { binding_id } => {
                let slot = SHADER_PINS
                    .get(pin.id.input)
                    .copied()
                    .unwrap_or(ShaderPin::DiffuseColor);
                let mut supported = false;
                if let Some(b) = self.binding_for_mut(binding_id) {
                    let names = b.kind.input_names();
                    supported = slot.shader_input_name(&names).is_some();
                    ui.add_enabled_ui(supported, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(slot.label());
                            match slot {
                                ShaderPin::DiffuseColor => {
                                    ui.color_edit_button_rgb(&mut b.inputs.diffuse_color);
                                }
                                ShaderPin::Metallic => {
                                    ui.add(
                                        egui::Slider::new(&mut b.inputs.metallic, 0.0..=1.0)
                                            .show_value(false),
                                    );
                                }
                                ShaderPin::Roughness => {
                                    ui.add(
                                        egui::Slider::new(&mut b.inputs.roughness, 0.0..=1.0)
                                            .show_value(false),
                                    );
                                }
                                ShaderPin::Normal => {
                                    ui.weak("texture");
                                }
                                ShaderPin::Opacity => {
                                    ui.add(
                                        egui::Slider::new(&mut b.inputs.opacity, 0.0..=1.0)
                                            .show_value(false),
                                    );
                                }
                                ShaderPin::Clearcoat => {
                                    ui.add(
                                        egui::Slider::new(&mut b.inputs.clearcoat, 0.0..=2.0)
                                            .show_value(false),
                                    );
                                }
                                ShaderPin::ClearcoatRoughness => {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut b.inputs.clearcoat_roughness,
                                            0.0..=1.0,
                                        )
                                        .show_value(false),
                                    );
                                }
                                ShaderPin::Occlusion => {
                                    ui.weak("texture");
                                }
                                ShaderPin::EmissionColor => {
                                    ui.color_edit_button_rgb(&mut b.inputs.emission_color);
                                }
                                ShaderPin::EmissionIntensity => {
                                    ui.add(
                                        egui::Slider::new(
                                            &mut b.inputs.emission_intensity,
                                            0.0..=10.0,
                                        )
                                        .show_value(false),
                                    );
                                }
                            }
                        });
                    });
                } else {
                    ui.weak("(binding gone)");
                }
                pin_color(supported)
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
            MaterialNode::Shader { .. } => {
                ui.label("Surface");
                PinInfo::square().with_fill(egui::Color32::from_rgb(120, 200, 120))
            }
            MaterialNode::Texture(_) => {
                ui.label("RGB");
                PinInfo::circle().with_fill(egui::Color32::from_rgb(240, 200, 120))
            }
        }
    }

    fn has_node_menu(&mut self, _node: &MaterialNode) -> bool {
        true
    }

    fn show_node_menu(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut egui::Ui,
        _scale: f32,
        snarl: &mut Snarl<MaterialNode>,
    ) {
        match &snarl[node] {
            MaterialNode::Shader { binding_id } => {
                let id = *binding_id;
                let sel_n = self.browser_selection.len();
                let label = if sel_n > 0 {
                    format!("Assign to selection ({sel_n})")
                } else {
                    "Assign to selection".to_string()
                };
                if ui
                    .add_enabled(sel_n > 0, egui::Button::new(label))
                    .clicked()
                {
                    self.pending_actions
                        .push(GraphAction::AssignToSelection(id));
                    ui.close_menu();
                }
                if ui.button("Assign to entire stage").clicked() {
                    self.pending_actions.push(GraphAction::AssignToStage(id));
                    ui.close_menu();
                }
                if ui.button("Unassign").clicked() {
                    self.pending_actions.push(GraphAction::Unassign(id));
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Remove").clicked() {
                    self.pending_actions.push(GraphAction::Remove(id));
                    ui.close_menu();
                }
            }
            MaterialNode::Texture(_) => {
                if ui.button("Remove").clicked() {
                    snarl.remove_node(node);
                    ui.close_menu();
                }
            }
        }
    }

    /// Right-click on empty graph background → "Add Texture".
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
                .add_filter(
                    "Image",
                    &["png", "jpg", "jpeg", "exr", "hdr", "tif", "tiff"],
                )
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

    fn connect(&mut self, from: &OutPin, to: &InPin, snarl: &mut Snarl<MaterialNode>) {
        let from_is_texture = matches!(snarl[from.id.node], MaterialNode::Texture(_));
        let to_is_shader = matches!(snarl[to.id.node], MaterialNode::Shader { .. });
        if from_is_texture && to_is_shader {
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
