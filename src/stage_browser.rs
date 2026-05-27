//! Stage browser — left-side dockable tree pane for the loaded USD
//! stage. Read-only in C1: lazy expand, hover-highlight, multi-select.
//! C2 will add right-click "Assign material" + per-prim bindings.
//!
//! Tree is snapshotted eagerly on stage load (path / name / type_name
//! / child paths only — light enough for ~thousands of prims). We
//! don't keep `rust_usd::Stage` or `Prim` handles between frames,
//! so the browser stays storable across egui repaints without
//! UniquePtr lifetime gymnastics. Re-opening the stage for live
//! attribute queries in later slices is cheap — USD's layer cache
//! short-circuits the disk read.
//!
//! UI state (expansion, selection) is keyed by SdfPath strings so it
//! survives a stage reload as long as the prim paths haven't moved.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use eframe::egui;

#[derive(Debug, Clone)]
pub struct PrimNode {
    /// Absolute SdfPath ("/World/asset/Body"). Unique within the
    /// stage; doubles as our HashMap key.
    pub path: String,
    /// Leaf name only ("Body") — what the row shows.
    pub name: String,
    /// USD type token ("Mesh", "Xform", "Scope", ""). Empty for
    /// pseudo-root and untyped over prims.
    pub type_name: String,
    pub children: Vec<PrimNode>,
}

pub struct StageBrowser {
    /// Path of the stage we cached `root` from. Used to detect when
    /// the user opened a new file and we need to rebuild.
    stage_path: Option<PathBuf>,
    /// Prim hierarchy snapshot. None means no stage loaded yet (or
    /// the last load failed).
    root: Option<PrimNode>,

    /// Expansion state — set of SdfPaths currently open. Persisted
    /// across reloads when the path matches.
    expanded: HashSet<String>,
    /// Multi-select set. Click toggles, ctrl/cmd+click adds.
    selected: HashSet<String>,
    /// Last single-clicked path, for shift-select range anchor in a
    /// later slice. Tracked now so the binding API can default to it.
    pub last_focused: Option<String>,

    /// Search filter — case-insensitive substring match against
    /// prim names. Empty string = no filter.
    filter: String,
}

impl Default for StageBrowser {
    fn default() -> Self {
        Self {
            stage_path: None,
            root: None,
            expanded: HashSet::new(),
            selected: HashSet::new(),
            last_focused: None,
            filter: String::new(),
        }
    }
}

impl StageBrowser {
    /// Build / rebuild the cached tree if the stage path changed.
    /// No-op if `path` matches what we already have.
    pub fn ensure_loaded(&mut self, path: &Path) {
        if self.stage_path.as_deref() == Some(path) {
            return;
        }
        match rust_usd::Stage::open(path) {
            Ok(stage) => {
                self.root = Some(snapshot_prim(&stage.pseudo_root()));
                self.stage_path = Some(path.to_path_buf());
                // Auto-expand pseudo-root and its immediate children
                // so the user lands looking at the top of the asset
                // rather than a single root row.
                self.expanded.clear();
                if let Some(root) = self.root.as_ref() {
                    self.expanded.insert(root.path.clone());
                    for child in &root.children {
                        self.expanded.insert(child.path.clone());
                    }
                }
                // Selection / focus reset — last stage's paths are
                // probably nonsense in the new one.
                self.selected.clear();
                self.last_focused = None;
            }
            Err(e) => {
                log::warn!(
                    "stage_browser: open {} failed: {}",
                    path.display(),
                    e.what()
                );
                self.root = None;
                self.stage_path = None;
            }
        }
    }

    pub fn selected(&self) -> &HashSet<String> {
        &self.selected
    }

    /// Apply the same selection semantics as clicking a row in the
    /// hierarchy, but driven by another surface such as viewport picking.
    pub fn select_path(&mut self, path: &str, multi: bool) -> bool {
        let Some(root) = self.root.as_ref() else {
            return false;
        };
        if find_node(root, path).is_none() {
            return false;
        }

        let was_selected = self.selected.contains(path);
        if !multi {
            self.selected.clear();
        }
        if was_selected && multi {
            self.selected.remove(path);
        } else {
            self.selected.insert(path.to_string());
        }
        self.last_focused = Some(path.to_string());
        expand_ancestors(path, &mut self.expanded);
        true
    }

    /// Expand the directly-clicked selection set with every
    /// descendant prim of each selected ancestor. Mirrors what the
    /// wgpu mask-build does via prefix matching, but produces an
    /// explicit list — what Hydra's `SetSelected(SdfPathVector)`
    /// needs, since Storm doesn't auto-cascade from an Xform down to
    /// its leaves. Returns the union (selected + descendants).
    pub fn effective_selection(&self) -> HashSet<String> {
        let mut out = self.selected.clone();
        if let Some(root) = self.root.as_ref() {
            for sel in &self.selected {
                if let Some(node) = find_node(root, sel) {
                    collect_descendants(node, &mut out);
                }
            }
        }
        out
    }

    /// Render the browser's body. Header (filter input + dock toggle)
    /// then the tree. Caller wraps in SidePanel::left or Window.
    pub fn show(&mut self, ui: &mut egui::Ui, undocked: &mut bool) {
        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.filter);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (label, tip) = if *undocked {
                    ("⮌ Dock", "Dock the Stage browser back into the main layout")
                } else {
                    (
                        "⮎ Undock",
                        "Pop out the Stage browser into a floating window",
                    )
                };
                if ui.button(label).on_hover_text(tip).clicked() {
                    *undocked = !*undocked;
                }
            });
        });
        ui.separator();

        if let Some(root) = self.root.clone() {
            let filter = self.filter.to_lowercase();
            egui::ScrollArea::vertical()
                .id_salt("stage_browser_scroll")
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    // Skip the pseudo-root row itself (it's just "/");
                    // jump straight to its children so the visible
                    // tree feels like a usable outliner.
                    for child in &root.children {
                        self.draw_node(ui, child, &filter, 0);
                    }
                });
        } else {
            ui.weak("No stage loaded.");
        }
    }

    fn draw_node(&mut self, ui: &mut egui::Ui, node: &PrimNode, filter: &str, depth: usize) {
        // Substring filter — when active, only show prims whose name
        // matches OR have a descendant that matches. Cheap recursive
        // scan; cached `descendant_matches` could optimise later.
        if !filter.is_empty()
            && !node.name.to_lowercase().contains(filter)
            && !subtree_matches(node, filter)
        {
            return;
        }

        let has_children = !node.children.is_empty();
        let is_expanded = self.expanded.contains(&node.path);
        let is_selected = self.selected.contains(&node.path);

        ui.horizontal(|ui| {
            ui.add_space(depth as f32 * 12.0);
            // Disclosure triangle — clickable area separate from the
            // row label so clicking the name selects without
            // collapsing.
            let chevron = if has_children {
                if is_expanded {
                    "▼"
                } else {
                    "▶"
                }
            } else {
                "  "
            };
            if has_children {
                if ui.add(egui::Button::new(chevron).frame(false)).clicked() {
                    if is_expanded {
                        self.expanded.remove(&node.path);
                    } else {
                        self.expanded.insert(node.path.clone());
                    }
                }
            } else {
                ui.label(chevron);
            }

            let row_label = if node.type_name.is_empty() {
                node.name.clone()
            } else {
                format!("{}  ⟨{}⟩", node.name, node.type_name)
            };
            let resp = ui.selectable_label(is_selected, row_label);
            if resp.clicked() {
                let multi = ui.input(|i| i.modifiers.command || i.modifiers.ctrl);
                if !multi {
                    self.selected.clear();
                }
                if is_selected && multi {
                    self.selected.remove(&node.path);
                } else {
                    self.selected.insert(node.path.clone());
                }
                self.last_focused = Some(node.path.clone());
            }
            resp.on_hover_text(&node.path);
        });

        if is_expanded {
            for child in &node.children {
                self.draw_node(ui, child, filter, depth + 1);
            }
        }
    }
}

fn subtree_matches(node: &PrimNode, filter: &str) -> bool {
    node.children
        .iter()
        .any(|c| c.name.to_lowercase().contains(filter) || subtree_matches(c, filter))
}

fn snapshot_prim(prim: &rust_usd::Prim) -> PrimNode {
    let path = prim.path();
    let name = sdf_path_leaf(&path);
    let type_name = prim.type_name();
    let children = prim
        .children()
        .iter()
        .map(snapshot_prim)
        .collect::<Vec<_>>();
    PrimNode {
        path,
        name,
        type_name,
        children,
    }
}

fn find_node<'a>(node: &'a PrimNode, target: &str) -> Option<&'a PrimNode> {
    if node.path == target {
        return Some(node);
    }
    // Prune: if `target` isn't under `node`'s subtree, skip recursion.
    if !target.starts_with(&node.path) {
        return None;
    }
    for child in &node.children {
        if let Some(found) = find_node(child, target) {
            return Some(found);
        }
    }
    None
}

fn collect_descendants(node: &PrimNode, out: &mut HashSet<String>) {
    for child in &node.children {
        out.insert(child.path.clone());
        collect_descendants(child, out);
    }
}

fn expand_ancestors(path: &str, expanded: &mut HashSet<String>) {
    expanded.insert("/".to_string());
    let mut start = 1;
    while start < path.len() {
        let Some(rel_idx) = path[start..].find('/') else {
            break;
        };
        let idx = start + rel_idx;
        if idx > 0 {
            expanded.insert(path[..idx].to_string());
        }
        start = idx + 1;
    }
}

fn sdf_path_leaf(path: &str) -> String {
    if path == "/" {
        return "/".to_string();
    }
    path.rsplit('/').next().unwrap_or(path).to_string()
}
