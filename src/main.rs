//! Perceptual audio visualizer: offline per-sample GCFB v2.34 analysis of
//! audio files (builder) and a memory-mapped spectrogram viewer synced to
//! playback.

mod analysis;
mod builder_ui;
mod viewer_ui;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1150.0, 720.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Perceptual Audio Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

#[derive(Default, PartialEq, Clone, Copy)]
enum Tab {
    #[default]
    Builder,
    Viewer,
}

#[derive(Default)]
struct App {
    tab: Tab,
    builder: builder_ui::BuilderTab,
    viewer: viewer_ui::ViewerTab,
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("tabs").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.tab, Tab::Builder, "Analysis builder");
                ui.selectable_value(&mut self.tab, Tab::Viewer, "Analysis viewer");
            });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                Tab::Builder => self.builder.ui(ui),
                Tab::Viewer => self.viewer.ui(ui),
            });
        });
    }
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {}
}
