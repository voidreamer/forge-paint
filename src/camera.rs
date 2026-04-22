use eframe::egui;
use glam::{Mat4, Vec3};

pub struct OrbitCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
    pub fov_y_deg: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.35,
            distance: 3.5,
            target: Vec3::ZERO,
            fov_y_deg: 45.0,
            z_near: 0.01,
            z_far: 1000.0,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        self.target
            + Vec3::new(
                self.distance * self.pitch.cos() * self.yaw.sin(),
                self.distance * self.pitch.sin(),
                self.distance * self.pitch.cos() * self.yaw.cos(),
            )
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let proj = Mat4::perspective_rh(
            self.fov_y_deg.to_radians(),
            aspect.max(1e-4),
            self.z_near,
            self.z_far,
        );
        proj * view
    }

    /// Modifier-based nav, frees plain LMB for painting.
    ///
    /// - Ctrl (or Cmd) + LMB drag: orbit
    /// - Shift + LMB drag OR MMB drag: pan
    /// - Scroll wheel: zoom
    pub fn handle_input(&mut self, response: &egui::Response, scroll_dy: f32) -> bool {
        let mut changed = false;
        let drag = response.drag_delta();
        let modifiers = response.ctx.input(|i| i.modifiers);
        let primary_drag = response.dragged_by(egui::PointerButton::Primary);
        let middle_drag = response.dragged_by(egui::PointerButton::Middle);

        // Treat ctrl and cmd (mac_cmd) as interchangeable for orbit; `command` is
        // egui's platform-abstracted modifier (cmd on mac, ctrl elsewhere).
        let is_ctrl_or_cmd = modifiers.ctrl || modifiers.mac_cmd || modifiers.command;
        let orbit = primary_drag && is_ctrl_or_cmd && !modifiers.shift && !modifiers.alt;
        let pan = (primary_drag && modifiers.shift && !is_ctrl_or_cmd) || middle_drag;

        if orbit && (drag.x != 0.0 || drag.y != 0.0) {
            self.yaw -= drag.x * 0.01;
            self.pitch = (self.pitch + drag.y * 0.01).clamp(-1.5, 1.5);
            changed = true;
        }

        if pan && (drag.x != 0.0 || drag.y != 0.0) {
            // Build camera basis: right = forward × up, then up = right × forward.
            let forward = (self.target - self.eye()).normalize_or_zero();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();
            let speed = self.distance * 0.0015;
            self.target -= right * drag.x * speed;
            self.target += up * drag.y * speed;
            changed = true;
        }

        if scroll_dy != 0.0 {
            self.distance = (self.distance * (1.0 - scroll_dy * 0.0015)).clamp(0.05, 10_000.0);
            changed = true;
        }

        changed
    }
}
