use std::cell::RefCell;
use std::rc::Rc;

pub struct SimulationState {
    pub is_running: bool,
    pub single_step: bool,
    pub ticks_per_frame: u32,
    pub total_ticks: u64,
    pub selected_aperture_type: i32, // -1: Sink, 1: Source
    pub active_scale_level: u32,      // 2^N scale hierarchy level
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            is_running: true,
            single_step: false,
            ticks_per_frame: 1,
            total_ticks: 0,
            selected_aperture_type: -1,
            active_scale_level: 0,
        }
    }
}

pub struct GuiOverlay {
    pub state: Rc<RefCell<SimulationState>>,
}

impl GuiOverlay {
    pub fn new(state: Rc<RefCell<SimulationState>>) -> Self {
        Self { state }
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        let mut state = self.state.borrow_mut();

        egui::Window::new("V.A.S.T. 7 Controls")
            .default_pos([15.0, 15.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Curvilinear Lattice Engine");
                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button(if state.is_running { "Pause" } else { "Play" }).clicked() {
                        state.is_running = !state.is_running;
                    }
                    if ui.button("Step Compton Tick").clicked() {
                        state.single_step = true;
                    }
                });

                ui.add(egui::Slider::new(&mut state.ticks_per_frame, 1..=20).text("Ticks / Frame"));
                ui.label(format!("Total Compton Ticks: {}", state.total_ticks));

                ui.separator();
                ui.heading("Aperture Control");
                ui.radio_value(&mut state.selected_aperture_type, -1, "ID 1: Sink (0 VPU)");
                ui.radio_value(&mut state.selected_aperture_type, 1, "ID 0: Source (2^31 VPU)");

                ui.separator();
                ui.heading("2^N Scale Hierarchy");
                ui.add(egui::Slider::new(&mut state.active_scale_level, 0..=3).text("Scale Level (N)"));
                ui.label(format!("Resolution: 2^{} Planck Volume", state.active_scale_level));

                ui.separator();
                ui.label("Navigation:");
                ui.label("• Drag Left Mouse: Orbit Camera");
                ui.label("• Drag Right Mouse: Pan Target");
                ui.label("• Scroll Wheel: Zoom");
            });
    }
}
