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
use gammachirp_rs::breebaart2001::EiDelayConvention;
use gammachirp_rs::gcfb_v234::{ControlMode, GainReference, GcParam};
use ndarray::Array1;

use crate::analysis::{
    AnalysisError, AnalysisHeader, AnalysisMode, AudioProbe, BuilderParams, probe_audio,
    run_analysis,
};

const CONTROL_MODES: [(&str, ControlMode); 3] = [
    ("Dynamic", ControlMode::Dynamic),
    ("Static", ControlMode::Static),
    ("Level", ControlMode::Level),
];

/// Analysis modes offered by the builder (see `AnalysisMode`).
const ANALYSIS_MODES: [(&str, AnalysisMode); 2] = [
    ("Mono downmix", AnalysisMode::Mono),
    ("Binaural (Breebaart 2001)", AnalysisMode::Binaural),
];

/// EI characteristic-delay conventions (see `EiDelayConvention`).
const DELAY_CONVENTIONS: [(&str, EiDelayConvention); 2] = [
    ("Paper symmetric", EiDelayConvention::PaperSymmetric),
    ("AMT one-sided", EiDelayConvention::AmtOneSidedInteger),
];

/// Valid `out_mid_crct` values (see `mk_filter_field2cochlea` in gammachirp-rs).
const OUT_MID_CRCT_OPTIONS: [&str; 6] =
    ["No", "ELC", "FreeField", "DiffuseField", "ITU", "EarDrum"];

/// Valid `hloss_type` values (see `hearing_pattern` in gammachirp-rs).
const HLOSS_TYPES: [&str; 10] = [
    "NH", "HL0", "HL1", "HL2", "HL3", "HL4", "HL5", "HL6", "HL7", "HL8",
];

/// Audiogram frequencies for the manual (HL0) hearing-level editor.
const AUDIOGRAM_FREQS: [f64; 7] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];

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

#[derive(Default)]
pub struct BuilderTab {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    probe: Option<AudioProbe>,
    params: BuilderParams,
    running: Option<RunningState>,
    done_samples: u64,
    status: String,
    status_is_error: bool,
}

impl BuilderTab {
    fn set_status(&mut self, text: impl Into<String>, is_error: bool) {
        self.status = text.into();
        self.status_is_error = is_error;
    }

    fn validation_error(&self) -> Option<String> {
        if self.input.is_none() {
            return Some("choose an input audio file".into());
        }
        if self.output.is_none() {
            return Some("choose an output file".into());
        }
        let f_range = self.params.gc.f_range;
        if !(f_range[0] > 0.0 && f_range[0] < f_range[1]) {
            return Some("frequency range must satisfy 0 < low < high".into());
        }
        if let Some(probe) = &self.probe {
            let nyquist = probe.sample_rate as f64 / 2.0;
            if f_range[1] >= nyquist {
                return Some(format!(
                    "max frequency {:.0} Hz must be below Nyquist {nyquist:.0} Hz \
                     for this {:.0} Hz file",
                    f_range[1], probe.sample_rate
                ));
            }
        }
        if self.params.mode == AnalysisMode::Binaural {
            let bin = &self.params.binaural;
            if !(bin.tau_max_seconds.is_finite() && bin.tau_max_seconds > 0.0) {
                return Some("ITD range must be positive".into());
            }
            if bin.num_tau == 0 || bin.num_iid == 0 {
                return Some("EI population must have at least one unit per dimension".into());
            }
            if !(bin.iid_max_db.is_finite() && bin.iid_max_db > 0.0) {
                return Some("IID range must be positive".into());
            }
            if let Some(probe) = &self.probe {
                if probe.channels != 2 {
                    return Some("binaural analysis requires a stereo input file".into());
                }
                let nyquist = probe.sample_rate as f64 / 2.0;
                if bin.peripheral.ihc_cutoff_hz >= nyquist {
                    return Some(format!(
                        "IHC cutoff {:.0} Hz must be below Nyquist {nyquist:.0} Hz \
                         for this {:.0} Hz file",
                        bin.peripheral.ihc_cutoff_hz, probe.sample_rate
                    ));
                }
            }
        }
        None
    }

    fn estimated_bytes(&self) -> Option<u64> {
        let probe = self.probe.as_ref()?;
        let duration = probe.total_duration?;
        let samples = duration.as_secs_f64() * f64::from(probe.sample_rate);
        Some(samples as u64 * self.values_per_sample() as u64 * 4)
    }

    /// f32 values written per input sample: the GCFB channels in mono, the
    /// two-ear dcGC mean plus the lowest-EI-unit IID, ITD, and activity in
    /// binaural mode.
    fn values_per_sample(&self) -> usize {
        let num_ch = self.params.gc.num_ch;
        match self.params.mode {
            AnalysisMode::Mono => num_ch,
            AnalysisMode::Binaural => 4 * num_ch,
        }
    }

    fn start(&mut self) {
        let (Some(input), Some(output)) = (self.input.clone(), self.output.clone()) else {
            return;
        };
        let params = self.params.clone();
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

    /// Controls for every customizable `GcParam` item, grouped into collapsible
    /// sections. Fields the builder manages (`fs`, `dyn_hpaf.str_prc`) or the
    /// library computes (`hloss`, `fr1`, derived `lvl_est` values) are
    /// intentionally not shown.
    fn params_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Analysis mode");
            egui::ComboBox::from_id_salt("analysis_mode")
                .selected_text(analysis_mode_name(self.params.mode))
                .show_ui(ui, |ui| {
                    for (label, mode) in ANALYSIS_MODES {
                        ui.selectable_value(&mut self.params.mode, mode, label);
                    }
                });
        });

        let gc = &mut self.params.gc;

        egui::CollapsingHeader::new("Filterbank")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Frequency range (Hz)");
                    ui.add(
                        egui::DragValue::new(&mut gc.f_range[0])
                            .range(1.0..=96_000.0)
                            .speed(1.0),
                    );
                    ui.label("to");
                    ui.add(
                        egui::DragValue::new(&mut gc.f_range[1])
                            .range(1.0..=96_000.0)
                            .speed(10.0),
                    );
                    ui.label("Channels");
                    ui.add(egui::DragValue::new(&mut gc.num_ch).range(2..=512));
                });
                ui.horizontal(|ui| {
                    ui.label("Control");
                    egui::ComboBox::from_id_salt("control_mode")
                        .selected_text(control_mode_name(gc.ctrl))
                        .show_ui(ui, |ui| {
                            for (label, mode) in CONTROL_MODES {
                                ui.selectable_value(&mut gc.ctrl, mode, label);
                            }
                        });
                    ui.label("Outer/middle-ear correction");
                    egui::ComboBox::from_id_salt("out_mid_crct")
                        .selected_text(gc.out_mid_crct.as_str())
                        .show_ui(ui, |ui| {
                            for option in OUT_MID_CRCT_OPTIONS {
                                ui.selectable_value(
                                    &mut gc.out_mid_crct,
                                    option.to_owned(),
                                    option,
                                );
                            }
                        });
                });
                ui.label(
                    "Dynamic is the full level-dependent dcGC (slowest, several × realtime). \
                     Static and Level are cheaper.",
                );
            });

        egui::CollapsingHeader::new("Gammachirp coefficients")
            .default_open(false)
            .show(ui, |ui| {
                ui.label(
                    "Each pair is [value at 1 kHz, ERB slope]: per channel it becomes \
                     value + slope × (ERB(f) / ERB(1 kHz) − 1).",
                );
                egui::Grid::new("gammachirp_coef_grid")
                    .num_columns(3)
                    .show(ui, |ui| {
                        ui.label("n (filter order)");
                        ui.add(egui::DragValue::new(&mut gc.n).speed(0.01));
                        ui.end_row();
                        coef_pair_row(ui, "b1", &mut gc.b1);
                        coef_pair_row(ui, "c1", &mut gc.c1);
                        coef_pair_row(ui, "frat0", &mut gc.frat[0]);
                        coef_pair_row(ui, "frat1", &mut gc.frat[1]);
                        coef_pair_row(ui, "b2", &mut gc.b2[0]);
                        coef_pair_row(ui, "c2", &mut gc.c2[0]);
                        coef_pair_row(ui, "b2[1] (unused)", &mut gc.b2[1]);
                        coef_pair_row(ui, "c2[1] (unused)", &mut gc.c2[1]);
                    });
                ui.label("The second rows of b2 and c2 are validated but unused by GCFB v2.34.");
            });

        egui::CollapsingHeader::new("Gain & levels")
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("gain_level_grid")
                    .num_columns(4)
                    .show(ui, |ui| {
                        ui.label("Gain compensation (dB)");
                        ui.add(egui::DragValue::new(&mut gc.gain_cmpnst_db).speed(0.1));
                        ui.label("Gain reference");
                        gain_ref_editor(ui, gc);
                        ui.end_row();

                        ui.label("Static-mode level (dB)");
                        ui.add(egui::DragValue::new(&mut gc.level_db_scgcfb).speed(1.0))
                            .on_hover_text(
                                "Presentation level at which the passive gammachirp \
                                 response is fixed in Static control mode",
                            );
                        ui.label("Asym. comp. update interval (samples)");
                        ui.add(egui::DragValue::new(&mut gc.num_update_asym_cmp).range(1..=65_536));
                        ui.end_row();

                        ui.label("Meddis HC RMS 0 dB ↔ SPL (dB)");
                        ui.add(
                            egui::DragValue::new(&mut gc.meddis_hc_level_rms0db_spldb).speed(0.1),
                        );
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("Level estimation")
            .default_open(false)
            .show(ui, |ui| {
                ui.label("Level estimate feeding the Dynamic and Level control modes.");
                let lvl = &mut gc.lvl_est;
                egui::Grid::new("lvl_est_grid")
                    .num_columns(4)
                    .show(ui, |ui| {
                        ui.label("Location (ERB)");
                        ui.add(egui::DragValue::new(&mut lvl.lct_erb).speed(0.05));
                        ui.label("Half-life decay");
                        ui.add(
                            egui::DragValue::new(&mut lvl.decay_hl)
                                .range(0.000_1..=120.0)
                                .speed(0.01),
                        );
                        ui.end_row();

                        ui.label("b2");
                        ui.add(egui::DragValue::new(&mut lvl.b2).speed(0.01));
                        ui.label("c2");
                        ui.add(egui::DragValue::new(&mut lvl.c2).speed(0.01));
                        ui.end_row();

                        ui.label("frat");
                        ui.add(egui::DragValue::new(&mut lvl.frat).speed(0.01));
                        ui.label("Weight");
                        ui.add(egui::DragValue::new(&mut lvl.weight).speed(0.01));
                        ui.end_row();

                        ui.label("RMS→SPL (dB)");
                        ui.add(egui::DragValue::new(&mut lvl.rms2spldb).speed(0.1));
                        ui.label("Reference level (dB)");
                        ui.add(egui::DragValue::new(&mut lvl.ref_db).speed(1.0));
                        ui.end_row();

                        ui.label("pwr[0]");
                        ui.add(egui::DragValue::new(&mut lvl.pwr[0]).speed(0.01));
                        ui.label("pwr[1]");
                        ui.add(egui::DragValue::new(&mut lvl.pwr[1]).speed(0.01));
                        ui.end_row();
                    });
            });

        egui::CollapsingHeader::new("Hearing loss")
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Hearing-loss type");
                    egui::ComboBox::from_id_salt("hloss_type")
                        .selected_text(gc.hloss_type.as_str())
                        .show_ui(ui, |ui| {
                            for kind in HLOSS_TYPES {
                                ui.selectable_value(&mut gc.hloss_type, kind.to_owned(), kind);
                            }
                        });
                });
                if gc.hloss_type == "HL0" {
                    let audiogram = gc
                        .hloss_hearing_level_db
                        .get_or_insert_with(|| Array1::zeros(7));
                    ui.label("Manual audiogram (dB HL):");
                    egui::Grid::new("audiogram_grid")
                        .num_columns(7)
                        .show(ui, |ui| {
                            for freq in AUDIOGRAM_FREQS {
                                ui.label(format!("{freq:.0} Hz"));
                            }
                            ui.end_row();
                            for value in audiogram.iter_mut() {
                                ui.add(egui::DragValue::new(value).range(0.0..=120.0).speed(0.5));
                            }
                            ui.end_row();
                        });
                } else {
                    gc.hloss_hearing_level_db = None;
                }
                ui.horizontal(|ui| {
                    let mut override_health = gc.hloss_compression_health.is_some();
                    if ui
                        .checkbox(&mut override_health, "Override compression health")
                        .changed()
                    {
                        gc.hloss_compression_health = override_health.then_some(1.0);
                    }
                    if let Some(health) = &mut gc.hloss_compression_health {
                        ui.add(egui::DragValue::new(health).range(0.0..=1.0).speed(0.01))
                            .on_hover_text(
                                "0 = fully impaired, 1 = healthy; defaults to 1 for NH \
                                 and 0.5 for HL types",
                            );
                    }
                });
            });

        if self.params.mode == AnalysisMode::Binaural {
            let bin = &mut self.params.binaural;

            egui::CollapsingHeader::new("EI population")
                .default_open(false)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("ITD range ± (ms)");
                        let mut tau_ms = bin.tau_max_seconds * 1e3;
                        if ui
                            .add(
                                egui::DragValue::new(&mut tau_ms)
                                    .range(0.05..=10.0)
                                    .speed(0.05),
                            )
                            .changed()
                        {
                            bin.tau_max_seconds = tau_ms * 1e-3;
                        }
                        ui.label("ITD units");
                        ui.add(egui::DragValue::new(&mut bin.num_tau).range(1..=41));
                    });
                    ui.horizontal(|ui| {
                        ui.label("IID range ± (dB)");
                        ui.add(
                            egui::DragValue::new(&mut bin.iid_max_db)
                                .range(1.0..=40.0)
                                .speed(0.5),
                        );
                        ui.label("IID units");
                        ui.add(egui::DragValue::new(&mut bin.num_iid).range(1..=21));
                    });
                    ui.label(
                        "Units form a linear ITD × IID grid; per channel and sample only \
                         the lowest-activity unit's characteristic IID, ITD, and activity \
                         are stored, so the population size does not affect file size.",
                    );
                });

            egui::CollapsingHeader::new("EI stage (Breebaart 2001)")
                .default_open(false)
                .show(ui, |ui| {
                    let ei = &mut bin.ei;
                    egui::Grid::new("ei_stage_grid")
                        .num_columns(4)
                        .show(ui, |ui| {
                            ui.label("Integration time constant (ms)");
                            let mut tau_ms = ei.integration_time_constant_seconds * 1e3;
                            if ui
                                .add(egui::DragValue::new(&mut tau_ms).range(1.0..=200.0))
                                .changed()
                            {
                                ei.integration_time_constant_seconds = tau_ms * 1e-3;
                            }
                            ui.label("Compression a");
                            ui.add(
                                egui::DragValue::new(&mut ei.compression_a)
                                    .range(0.001..=1.0)
                                    .speed(0.01),
                            );
                            ui.end_row();

                            ui.label("Compression b");
                            ui.add(
                                egui::DragValue::new(&mut ei.compression_b)
                                    .range(1e-7..=1e-2)
                                    .speed(1e-6),
                            );
                            ui.label("Delay-weight time constant (ms)");
                            let mut weight_ms = ei.delay_weight_time_constant_seconds * 1e3;
                            if ui
                                .add(
                                    egui::DragValue::new(&mut weight_ms)
                                        .range(0.1..=20.0)
                                        .speed(0.1),
                                )
                                .changed()
                            {
                                ei.delay_weight_time_constant_seconds = weight_ms * 1e-3;
                            }
                            ui.end_row();

                            ui.label("Delay convention");
                            egui::ComboBox::from_id_salt("ei_delay_convention")
                                .selected_text(delay_convention_name(ei.delay_convention))
                                .show_ui(ui, |ui| {
                                    for (label, convention) in DELAY_CONVENTIONS {
                                        ui.selectable_value(
                                            &mut ei.delay_convention,
                                            convention,
                                            label,
                                        );
                                    }
                                });
                            ui.label("Internal noise σ (MU)");
                            ui.add(
                                egui::DragValue::new(&mut ei.internal_noise_std_mu)
                                    .range(0.0..=10.0)
                                    .speed(0.1),
                            );
                            ui.end_row();

                            ui.label("Noise seed");
                            ui.add(egui::DragValue::new(&mut ei.noise_seed));
                            ui.end_row();
                        });
                    ui.label("Integration boundary: causal zero-state (required for streaming)");
                });

            egui::CollapsingHeader::new("Peripheral (IHC & adaptation)")
                .default_open(false)
                .show(ui, |ui| {
                    let per = &mut bin.peripheral;
                    ui.horizontal(|ui| {
                        ui.label("IHC lowpass cutoff (Hz)");
                        ui.add(
                            egui::DragValue::new(&mut per.ihc_cutoff_hz)
                                .range(100.0..=4000.0)
                                .speed(10.0),
                        );
                        ui.label("Minimum level (dB SPL)");
                        ui.add(
                            egui::DragValue::new(&mut per.minimum_level_db_spl)
                                .range(-20.0..=40.0)
                                .speed(1.0),
                        );
                    });
                    ui.label("Adaptation time constants (s):");
                    egui::Grid::new("adaptation_tc_grid")
                        .num_columns(5)
                        .show(ui, |ui| {
                            for loop_index in 1..=per.adaptation_time_constants_seconds.len() {
                                ui.label(format!("Loop {loop_index}"));
                            }
                            ui.end_row();
                            for tc in per.adaptation_time_constants_seconds.iter_mut() {
                                ui.add(
                                    egui::DragValue::new(tc).range(0.001..=2.0).speed(0.001),
                                );
                            }
                            ui.end_row();
                        });
                    ui.horizontal(|ui| {
                        let mut limited = per.overshoot_limit.is_some();
                        if ui.checkbox(&mut limited, "Overshoot limit").changed() {
                            per.overshoot_limit = limited.then_some(10.0);
                        }
                        if let Some(limit) = &mut per.overshoot_limit {
                            ui.add(egui::DragValue::new(limit).range(1.0..=100.0).speed(0.5))
                                .on_hover_text(
                                    "AMT smooth limiter applied inside each adaptation loop; \
                                     unchecked keeps the paper's unlimited response",
                                );
                        }
                    });
                    ui.horizontal(|ui| {
                        let mut calibrated = per.amplitude_one_db_spl.is_some();
                        if ui
                            .checkbox(&mut calibrated, "Amplitude-one level")
                            .on_hover_text("dcGC amplitude 1 ↔ dB SPL")
                            .changed()
                        {
                            per.amplitude_one_db_spl = calibrated.then_some(100.0);
                        }
                        if let Some(db) = &mut per.amplitude_one_db_spl {
                            ui.add(egui::DragValue::new(db).range(60.0..=120.0).speed(1.0));
                        }
                    });
                    ui.horizontal(|ui| {
                        let mut noise_on = per.absolute_threshold_noise_level_db_spl.is_some();
                        if ui
                            .checkbox(&mut noise_on, "Absolute-threshold noise")
                            .changed()
                        {
                            per.absolute_threshold_noise_level_db_spl = noise_on.then_some(9.54);
                        }
                        if let Some(db) = &mut per.absolute_threshold_noise_level_db_spl {
                            ui.add(
                                egui::DragValue::new(db)
                                    .range(-20.0..=40.0)
                                    .speed(0.5)
                                    .suffix(" dB SPL"),
                            );
                        }
                        ui.label("Noise seed");
                        ui.add(egui::DragValue::new(
                            &mut per.absolute_threshold_noise_seed,
                        ));
                    });
                });
        }

        ui.add_space(4.0);
        if ui.button("Reset parameters to defaults").clicked() {
            self.params = BuilderParams::default();
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
        let mode_blurb = match self.params.mode {
            AnalysisMode::Mono => {
                "decoded, downmixed to mono, and streamed through the filterbank"
            }
            AnalysisMode::Binaural => {
                "decoded as stereo and streamed through per-ear filterbanks plus a \
                 Breebaart-2001 EI population"
            }
        };
        ui.label(format!(
            "Offline per-sample GCFB v2.34 analysis. The audio is {mode_blurb} one sample \
             at a time; results are written straight to disk, so files larger than RAM are \
             fine."
        ));
        ui.add_space(8.0);

        let enabled = !running;
        ui.add_enabled_ui(enabled, |ui| {
            file_row(ui, "Audio input", &mut self.input, false);
            if ui
                .button("Probe / re-read file info")
                .on_hover_text(
                    "Reads sample rate, channel count, and duration without decoding everything",
                )
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
            self.params_ui(ui);
        });

        ui.add_space(8.0);
        if let Some(bytes) = self.estimated_bytes() {
            ui.label(format!(
                "Estimated output size: {} ({} values × 4 B × per-sample)",
                human_bytes(bytes),
                self.values_per_sample()
            ));
        } else if let Some(probe) = &self.probe {
            ui.label(format!(
                "Output size: ~{}/s of audio at {} values per sample",
                human_bytes(u64::from(probe.sample_rate) * self.values_per_sample() as u64 * 4),
                self.values_per_sample()
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
                Some(total) if total > 0 => {
                    egui::ProgressBar::new((self.done_samples as f32 / total as f32).min(1.0))
                        .show_percentage()
                }
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

fn control_mode_name(mode: ControlMode) -> &'static str {
    CONTROL_MODES
        .iter()
        .find(|(_, candidate)| *candidate == mode)
        .map(|(label, _)| *label)
        .unwrap_or("Unknown")
}

fn analysis_mode_name(mode: AnalysisMode) -> &'static str {
    ANALYSIS_MODES
        .iter()
        .find(|(_, candidate)| *candidate == mode)
        .map(|(label, _)| *label)
        .unwrap_or("Unknown")
}

fn delay_convention_name(convention: EiDelayConvention) -> &'static str {
    DELAY_CONVENTIONS
        .iter()
        .find(|(_, candidate)| *candidate == convention)
        .map(|(label, _)| *label)
        .unwrap_or("Unknown")
}

/// One labelled grid row editing a two-element coefficient pair.
fn coef_pair_row(ui: &mut egui::Ui, label: &str, pair: &mut [f64; 2]) {
    ui.label(label);
    ui.add(egui::DragValue::new(&mut pair[0]).speed(0.001));
    ui.add(egui::DragValue::new(&mut pair[1]).speed(0.001));
    ui.end_row();
}

/// Combo box for the gain-reference mode plus a dB editor when a fixed level
/// is selected. `GainReference::Db` is a reference presentation level; 50 dB
/// matches the library's static-mode default.
fn gain_ref_editor(ui: &mut egui::Ui, gc: &mut GcParam) {
    let mut fixed_db = matches!(gc.gain_ref, GainReference::Db(_));
    egui::ComboBox::from_id_salt("gain_ref")
        .selected_text(if fixed_db {
            "Fixed level (dB)"
        } else {
            "Normalize IO function"
        })
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut fixed_db, false, "Normalize IO function");
            ui.selectable_value(&mut fixed_db, true, "Fixed level (dB)");
        });
    gc.gain_ref = match (fixed_db, gc.gain_ref) {
        (true, GainReference::Db(db)) => GainReference::Db(db),
        (true, GainReference::NormalizeIoFunction) => GainReference::Db(50.0),
        (false, _) => GainReference::NormalizeIoFunction,
    };
    if let GainReference::Db(ref mut db) = gc.gain_ref {
        ui.add(egui::DragValue::new(db).speed(1.0).suffix(" dB"));
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
                dialog =
                    dialog.add_filter("audio", &["wav", "flac", "mp3", "ogg", "m4a", "mp4", "aac"]);
            }
            let picked = if save {
                dialog.save_file()
            } else {
                dialog.pick_file()
            };
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
