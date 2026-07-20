//! Analysis builder tab: pick an audio file, configure the GCFB v2.34
//! per-sample analysis, and stream the result to a `.gca` file on a worker
//! thread with progress and cancellation.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use eframe::egui;
use gammachirp_rs::gcfb_v234::ControlMode;

use crate::analysis::{
    AnalysisError, AnalysisHeader, AudioProbe, BuilderParams, probe_audio, run_analysis,
};

const CONTROL_MODES: [(&str, ControlMode); 3] = [
    ("Dynamic", ControlMode::Dynamic),
    ("Static", ControlMode::Static),
    ("Level", ControlMode::Level),
];

enum BuilderMsg {
    Progress(u64),
    Done(AnalysisHeader),
    Failed(String),
    Cancelled,
}

struct RunningState {
    cancel: Arc<AtomicBool>,
    rx: Receiver<BuilderMsg>,
}

pub struct BuilderTab {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    probe: Option<AudioProbe>,
    f_low: f64,
    f_high: f64,
    num_channels: u32,
    control_idx: usize,
    running: Option<RunningState>,
    done_samples: u64,
    status: String,
    status_is_error: bool,
}

impl Default for BuilderTab {
    fn default() -> Self {
        let defaults = BuilderParams::default();
        Self {
            input: None,
            output: None,
            probe: None,
            f_low: defaults.f_range[0],
            f_high: defaults.f_range[1],
            num_channels: defaults.num_channels as u32,
            control_idx: 0,
            running: None,
            done_samples: 0,
            status: String::new(),
            status_is_error: false,
        }
    }
}

impl BuilderTab {
    fn set_status(&mut self, text: impl Into<String>, is_error: bool) {
        self.status = text.into();
        self.status_is_error = is_error;
    }

    fn params(&self) -> BuilderParams {
        BuilderParams {
            num_channels: self.num_channels as usize,
            f_range: [self.f_low, self.f_high],
            control: CONTROL_MODES[self.control_idx].1,
        }
    }

    fn validation_error(&self) -> Option<String> {
        if self.input.is_none() {
            return Some("choose an input audio file".into());
        }
        if self.output.is_none() {
            return Some("choose an output file".into());
        }
        if !(self.f_low > 0.0 && self.f_low < self.f_high) {
            return Some("frequency range must satisfy 0 < low < high".into());
        }
        if let Some(probe) = &self.probe {
            let nyquist = probe.sample_rate as f64 / 2.0;
            if self.f_high >= nyquist {
                return Some(format!(
                    "max frequency {:.0} Hz must be below Nyquist {nyquist:.0} Hz \
                     for this {:.0} Hz file",
                    self.f_high, probe.sample_rate
                ));
            }
        }
        None
    }

    fn estimated_bytes(&self) -> Option<u64> {
        let probe = self.probe.as_ref()?;
        let duration = probe.total_duration?;
        let samples = duration.as_secs_f64() * f64::from(probe.sample_rate);
        Some(samples as u64 * u64::from(self.num_channels) * 4)
    }

    fn start(&mut self) {
        let (Some(input), Some(output)) = (self.input.clone(), self.output.clone()) else {
            return;
        };
        let params = self.params();
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let thread_cancel = cancel.clone();
        thread::spawn(move || {
            let result = run_analysis(
                &input,
                &output,
                &params,
                |done| {
                    let _ = tx.send(BuilderMsg::Progress(done));
                },
                &thread_cancel,
            );
            let msg = match result {
                Ok(header) => BuilderMsg::Done(header),
                Err(AnalysisError::Cancelled) => BuilderMsg::Cancelled,
                Err(error) => BuilderMsg::Failed(error.to_string()),
            };
            let _ = tx.send(msg);
        });
        self.done_samples = 0;
        self.set_status("analysis running…", false);
        self.running = Some(RunningState { cancel, rx });
    }

    fn poll(&mut self) {
        let Some(state) = &self.running else { return };
        let mut finished = None;
        while let Ok(msg) = state.rx.try_recv() {
            match msg {
                BuilderMsg::Progress(done) => self.done_samples = done,
                other => finished = Some(other),
            }
        }
        if let Some(msg) = finished {
            self.running = None;
            match msg {
                BuilderMsg::Done(header) => self.set_status(
                    format!(
                        "done: {} samples × {} channels ({:.1} s) written",
                        header.num_samples,
                        header.num_channels,
                        header.duration()
                    ),
                    false,
                ),
                BuilderMsg::Failed(error) => self.set_status(format!("failed: {error}"), true),
                BuilderMsg::Cancelled => self.set_status("cancelled; partial file deleted", false),
                BuilderMsg::Progress(_) => unreachable!(),
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        self.poll();
        let running = self.running.is_some();
        if running {
            ui.ctx().request_repaint_after(Duration::from_millis(100));
        }

        ui.heading("Analysis builder");
        ui.add_space(4.0);
        ui.label(
            "Offline per-sample GCFB v2.34 analysis. The audio is decoded, downmixed to \
             mono, and streamed through the filterbank one sample at a time; results are \
             written straight to disk, so files larger than RAM are fine.",
        );
        ui.add_space(8.0);

        let enabled = !running;
        ui.add_enabled_ui(enabled, |ui| {
            file_row(ui, "Audio input", &mut self.input, false);
            if ui
                .button("Probe / re-read file info")
                .on_hover_text("Reads sample rate, channel count, and duration without decoding everything")
                .clicked()
                && let Some(path) = &self.input
            {
                match probe_audio(path) {
                    Ok(probe) => {
                        self.set_status(
                            format!(
                                "{} Hz, {} channel(s){}",
                                probe.sample_rate,
                                probe.channels,
                                probe
                                    .total_duration
                                    .map(|d| format!(", {:.1} s", d.as_secs_f64()))
                                    .unwrap_or_default()
                            ),
                            false,
                        );
                        self.probe = Some(probe);
                    }
                    Err(error) => {
                        self.probe = None;
                        self.set_status(format!("cannot read audio: {error}"), true);
                    }
                }
            }
            if self.input.is_some() && self.probe.is_none() {
                // Auto-probe once a file is chosen.
                if let Some(path) = &self.input {
                    match probe_audio(path) {
                        Ok(probe) => self.probe = Some(probe),
                        Err(error) => self.set_status(format!("cannot read audio: {error}"), true),
                    }
                }
            }

            file_row(ui, "Output (.gca)", &mut self.output, true);
            if self.output.is_none()
                && let Some(input) = &self.input
            {
                self.output = Some(input.with_extension("gca"));
            }

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label("Frequency range (Hz)");
                ui.add(
                    egui::DragValue::new(&mut self.f_low)
                        .range(1.0..=96_000.0)
                        .speed(1.0),
                );
                ui.label("to");
                ui.add(
                    egui::DragValue::new(&mut self.f_high)
                        .range(1.0..=96_000.0)
                        .speed(10.0),
                );
                ui.label("Channels");
                ui.add(egui::DragValue::new(&mut self.num_channels).range(2..=512));
                ui.label("Control");
                egui::ComboBox::from_id_salt("control_mode")
                    .selected_text(CONTROL_MODES[self.control_idx].0)
                    .show_ui(ui, |ui| {
                        for (idx, (label, _)) in CONTROL_MODES.iter().enumerate() {
                            ui.selectable_value(&mut self.control_idx, idx, *label);
                        }
                    });
            });
            ui.label(
                "Dynamic is the full level-dependent dcGC (slowest, several × realtime). \
                 Static and Level are cheaper. Outer/middle-ear correction is off; \
                 hearing-loss characteristics are normal hearing.",
            );
        });

        ui.add_space(8.0);
        if let Some(bytes) = self.estimated_bytes() {
            ui.label(format!(
                "Estimated output size: {} ({} ch × 4 B × per-sample)",
                human_bytes(bytes),
                self.num_channels
            ));
        } else if let Some(probe) = &self.probe {
            ui.label(format!(
                "Output size: ~{}/s of audio at {} channels",
                human_bytes(u64::from(probe.sample_rate) * u64::from(self.num_channels) * 4),
                self.num_channels
            ));
        }

        let error = self.validation_error();
        ui.horizontal(|ui| {
            if running {
                if ui.button("Cancel").clicked()
                    && let Some(state) = &self.running
                {
                    state.cancel.store(true, Ordering::Relaxed);
                    self.set_status("cancelling…", false);
                }
            } else {
                let can_run = error.is_none();
                if ui
                    .add_enabled(can_run, egui::Button::new("Run analysis"))
                    .clicked()
                {
                    self.start();
                }
                if let Some(error) = &error {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            }
        });

        if running {
            let total = self.probe.as_ref().and_then(|p| {
                p.total_duration
                    .map(|d| (d.as_secs_f64() * f64::from(p.sample_rate)) as u64)
            });
            let bar = match total {
                Some(total) if total > 0 => egui::ProgressBar::new(
                    (self.done_samples as f32 / total as f32).min(1.0),
                )
                .show_percentage(),
                _ => egui::ProgressBar::new(0.0)
                    .animate(true)
                    .text(format!("{} samples", self.done_samples)),
            };
            ui.add(bar.desired_width(ui.available_width() * 0.6));
        }

        if !self.status.is_empty() {
            let color = if self.status_is_error {
                egui::Color32::LIGHT_RED
            } else {
                ui.style().visuals.text_color()
            };
            ui.colored_label(color, &self.status);
        }
    }
}

fn file_row(ui: &mut egui::Ui, label: &str, path: &mut Option<PathBuf>, save: bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        let text = path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(none)".into());
        ui.monospace(text);
        if ui.button("Browse…").clicked() {
            let mut dialog = rfd::FileDialog::new();
            if save {
                dialog = dialog.add_filter("gammachirp analysis", &["gca"]);
            } else {
                dialog = dialog
                    .add_filter("audio", &["wav", "flac", "mp3", "ogg", "m4a", "mp4", "aac"]);
            }
            let picked = if save { dialog.save_file() } else { dialog.pick_file() };
            if let Some(picked) = picked {
                *path = Some(picked);
            }
        }
    });
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
