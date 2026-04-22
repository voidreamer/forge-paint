//! Stroke-level undo / redo for paint layer textures.
//!
//! A "stroke" is a continuous brush drag from primary button down to up.
//! On stroke-start we snapshot the targeted channel of the active layer into
//! a sibling texture via `copy_texture_to_texture`. Undo copies the snapshot
//! back; redo is symmetric.
//!
//! Memory cost per snapshot = `resolution² × 4 × N_tiles` bytes (R8 mask is
//! /4 that). A 2k single-tile base_color is ~16 MB; default ring-buffer depth
//! 16 gives ~256 MB worst case for typical assets.

use egui_wgpu::wgpu;
use std::collections::VecDeque;

use crate::paint::{Layer, LayerStack};

const DEFAULT_DEPTH: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotKind {
    BaseColor,
    RoughMetal,
    Mask,
}

pub struct UndoSnapshot {
    pub layer_index: usize,
    pub kind: SnapshotKind,
    pub texture: wgpu::Texture,
}

pub struct UndoStack {
    undo: VecDeque<UndoSnapshot>,
    redo: VecDeque<UndoSnapshot>,
    max_depth: usize,
}

impl Default for UndoStack {
    fn default() -> Self {
        Self::new(DEFAULT_DEPTH)
    }
}

impl UndoStack {
    pub fn new(max_depth: usize) -> Self {
        Self {
            undo: VecDeque::new(),
            redo: VecDeque::new(),
            max_depth: max_depth.max(1),
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_depth(&self) -> usize {
        self.redo.len()
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// Snapshot the current state of `(layer_index, kind)` in preparation for
    /// the stroke that's about to land on it. New activity invalidates the
    /// redo history.
    pub fn push_pre_stroke(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        stack: &LayerStack,
        layer_index: usize,
        kind: SnapshotKind,
    ) {
        let Some(src_tex) = source_texture(stack, layer_index, kind) else {
            return;
        };
        let snapshot = clone_texture(device, src_tex, "undo.pre_stroke");
        copy_full(encoder, src_tex, &snapshot);

        if self.undo.len() >= self.max_depth {
            self.undo.pop_front();
        }
        self.undo.push_back(UndoSnapshot {
            layer_index,
            kind,
            texture: snapshot,
        });
        self.redo.clear();
    }

    /// Pop the most recent stroke and restore its pre-state. Current live
    /// state is captured onto the redo stack first.
    pub fn undo(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        stack: &mut LayerStack,
    ) -> bool {
        let Some(entry) = self.undo.pop_back() else {
            return false;
        };
        // Snapshot the *current* state so redo can restore it.
        if let Some(live_tex) = source_texture(stack, entry.layer_index, entry.kind) {
            let redo_snap = clone_texture(device, live_tex, "undo.for_redo");
            copy_full(encoder, live_tex, &redo_snap);
            // Copy the pre-stroke snapshot back into the live texture.
            copy_full(encoder, &entry.texture, live_tex);
            self.redo.push_back(UndoSnapshot {
                layer_index: entry.layer_index,
                kind: entry.kind,
                texture: redo_snap,
            });
            true
        } else {
            // Layer/mask vanished since then. Drop the entry.
            false
        }
    }

    pub fn redo(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        stack: &mut LayerStack,
    ) -> bool {
        let Some(entry) = self.redo.pop_back() else {
            return false;
        };
        if let Some(live_tex) = source_texture(stack, entry.layer_index, entry.kind) {
            let undo_snap = clone_texture(device, live_tex, "redo.for_undo");
            copy_full(encoder, live_tex, &undo_snap);
            copy_full(encoder, &entry.texture, live_tex);
            self.undo.push_back(UndoSnapshot {
                layer_index: entry.layer_index,
                kind: entry.kind,
                texture: undo_snap,
            });
            true
        } else {
            false
        }
    }
}

fn source_texture(stack: &LayerStack, layer_index: usize, kind: SnapshotKind) -> Option<&wgpu::Texture> {
    let layer = stack.layers.get(layer_index)?;
    match kind {
        SnapshotKind::BaseColor => Some(&layer.base_color),
        SnapshotKind::RoughMetal => Some(&layer.rough_metal),
        SnapshotKind::Mask => layer.mask.as_ref().map(|m| &m.texture),
    }
}

fn clone_texture(device: &wgpu::Device, src: &wgpu::Texture, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: src.size(),
        mip_level_count: src.mip_level_count(),
        sample_count: src.sample_count(),
        dimension: src.dimension(),
        format: src.format(),
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn copy_full(encoder: &mut wgpu::CommandEncoder, src: &wgpu::Texture, dst: &wgpu::Texture) {
    let size = src.size();
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: dst,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
}

/// Which channel of the active layer a given brush configuration will stamp.
pub fn snapshot_kind_for_stamp(
    channel: crate::paint::PaintChannel,
    mask_edit: bool,
    active_has_mask: bool,
) -> SnapshotKind {
    use crate::paint::PaintChannel;
    if mask_edit && active_has_mask {
        SnapshotKind::Mask
    } else {
        match channel {
            PaintChannel::BaseColor => SnapshotKind::BaseColor,
            PaintChannel::Roughness | PaintChannel::Metallic => SnapshotKind::RoughMetal,
            PaintChannel::Mask => SnapshotKind::Mask,
        }
    }
}

#[allow(dead_code)]
impl Layer {
    // Helpers to quiet `Layer` is unused warning if persistence calls above drop it.
    fn _touch_for_docs() {}
}
