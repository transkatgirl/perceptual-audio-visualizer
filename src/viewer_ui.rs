//! Analysis viewer tab: memory-mapped `.gca` heatmap synced to rodio playback,
//! with seeking, pausing, zooming, and panning. Only the visible time window is
//! ever read from disk; the audio decoder streams.

use std::time::Duration;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use eframe::egui;
use rodio::{DeviceSinkBuilder, Player};

use crate::analysis::{
    AnalysisReader, VALUE_AMPLITUDE, VALUE_SALIENCE, control_mode_label, mode_label, probe_audio,
    streaming_decoder, value_kind_label,
};

/// Displayed dB for one stored value: reassigned energy and consensus
/// salience are power-like quantities (10·log10), dcGC amplitude is an
/// amplitude (20·log10).
fn value_db(value_kind: u32, value: f32) -> f32 {
    if value_kind == VALUE_AMPLITUDE {
        20.0 * (value.abs() + 1e-12).log10()
    } else {
        10.0 * (value.abs() + 1e-12).log10()
    }
}

/// Spectrogram texture width in columns; the visible window is always exactly
/// this many columns wide, so zoom level maps to samples-per-column.
const TEX_W: usize = 4096;
const MIN_SPAN: f64 = 0.05;
const MAX_SPAN: f64 = 10.0;

pub struct ViewerTab {
    audio_path: Option<PathBuf>,
    analysis_path: Option<PathBuf>,
    status: String,
    status_is_error: bool,
    loaded: Option<Loaded>,
    follow: bool,
    auto_range: bool,
    view_start: f64,
    view_span: f64,
    floor_db: f32,
    ceil_db: f32,
    last_auto_range: Option<Instant>,
}

struct Loaded {
    reader: AnalysisReader,
    audio_path: PathBuf,
    playback: Option<Playback>,
    spec: Spectrogram,
    duration: f64,
}

struct Playback {
    _handle: rodio::MixerDeviceSink,
    player: Player,
}

impl Default for ViewerTab {
    fn default() -> Self {
        Self {
            audio_path: None,
            analysis_path: None,
            status: String::new(),
            status_is_error: false,
            loaded: None,
            follow: true,
            auto_range: false,
            view_start: 0.0,
            view_span: 5.0,
            floor_db: -100.0,
            ceil_db: -55.0,
            last_auto_range: None,
        }
    }
}

impl ViewerTab {
    fn set_status(&mut self, text: impl Into<String>, is_error: bool) {
        self.status = text.into();
        self.status_is_error = is_error;
    }

    fn load(&mut self, ctx: &egui::Context) {
        let (Some(audio), Some(analysis)) = (self.audio_path.clone(), self.analysis_path.clone())
        else {
            return;
        };
        self.set_status(String::new(), false);
        match self.load_inner(ctx, &audio, &analysis) {
            Ok(()) => {}
            Err(error) => {
                self.loaded = None;
                self.set_status(error, true);
            }
        }
    }

    fn load_inner(
        &mut self,
        ctx: &egui::Context,
        audio: &Path,
        analysis: &Path,
    ) -> Result<(), String> {
        let probe = probe_audio(audio).map_err(|e| format!("cannot read audio: {e}"))?;
        let reader = AnalysisReader::open(analysis).map_err(|e| e.to_string())?;
        if reader.header.sample_rate != f64::from(probe.sample_rate) {
            return Err(format!(
                "sample rate mismatch: audio is {} Hz but analysis is {:.0} Hz \
                 (rebuild the analysis from this exact file)",
                probe.sample_rate, reader.header.sample_rate
            ));
        }
        let playback = match Playback::new(audio) {
            Ok(playback) => Some(playback),
            Err(error) => {
                self.set_status(
                    format!("no audio output available ({error}); view-only mode"),
                    true,
                );
                None
            }
        };
        let duration = reader.header.duration();
        let complete = reader.is_complete();
        self.view_start = 0.0;
        self.view_span = duration.clamp(MIN_SPAN, MAX_SPAN.min(duration.max(MIN_SPAN)));
        self.follow = true;
        self.loaded = Some(Loaded {
            spec: Spectrogram::new(ctx, &reader),
            reader,
            audio_path: audio.to_path_buf(),
            playback,
            duration,
        });
        if !complete && !self.status_is_error {
            self.set_status(
                format!(
                    "incomplete analysis (still processing or terminated early) — \
                     showing the first {}; press Load to refresh",
                    fmt_time(duration)
                ),
                false,
            );
        }
        Ok(())
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Analysis viewer");
        ui.add_space(4.0);

        let mut do_load = false;
        let prev_audio = self.audio_path.clone();
        ui.horizontal_wrapped(|ui| {
            file_row(ui, "Audio", &mut self.audio_path, false);
            file_row(ui, "Analysis (.gca)", &mut self.analysis_path, true);
            if ui
                .add_enabled(
                    self.audio_path.is_some() && self.analysis_path.is_some(),
                    egui::Button::new("Load"),
                )
                .clicked()
            {
                do_load = true;
            }
        });
        if self.audio_path != prev_audio {
            // New audio: point the analysis path at its sibling .gca.
            self.analysis_path = self.audio_path.as_ref().and_then(|audio| {
                let sibling = audio.with_extension("gca");
                sibling.exists().then_some(sibling)
            });
        }
        if do_load {
            self.load(ui.ctx());
        }
        if !self.status.is_empty() {
            let color = if self.status_is_error {
                egui::Color32::LIGHT_RED
            } else {
                ui.style().visuals.text_color()
            };
            ui.colored_label(color, &self.status);
        }

        let Self {
            loaded,
            follow,
            view_start,
            view_span,
            floor_db,
            ceil_db,
            status,
            ..
        } = self;
        let Some(loaded) = loaded else {
            ui.label("Load an audio file and its .gca analysis to begin.");
            return;
        };

        let header = &loaded.reader.header;
        let mut info = format!(
            "{:.0} Hz · {} ch · {:.0}–{:.0} Hz · {} · {} · {} · {} ({})",
            header.sample_rate,
            header.num_channels,
            header.f_range[0],
            header.f_range[1],
            mode_label(header.mode),
            control_mode_label(header.control_mode),
            value_kind_label(header.value_kind),
            fmt_time(header.duration()),
            crate::builder_ui::human_bytes(
                header.num_samples * header.values_per_sample() as u64 * 4
            ),
        );
        if header.value_kind == VALUE_SALIENCE {
            let scales = header
                .scales
                .iter()
                .map(|scale| format!("{scale:.1}"))
                .collect::<Vec<_>>()
                .join(", ");
            info = format!("{info} · scales [{scales}]");
        }
        if loaded.reader.is_binaural() {
            let tau_max_ms = header
                .tau_seconds
                .iter()
                .map(|tau| tau.abs())
                .fold(0.0_f64, f64::max)
                * 1000.0;
            let iid_max_db = header
                .iid_db
                .iter()
                .map(|iid| iid.abs())
                .fold(0.0_f64, f64::max);
            info = format!(
                "{info} · {} EI units ({} ITD × {} IID), ±{tau_max_ms:.1} ms, ±{iid_max_db:.0} dB",
                header.tau_seconds.len() * header.iid_db.len(),
                header.tau_seconds.len(),
                header.iid_db.len(),
            );
        }
        if !loaded.reader.is_complete() {
            info = format!("{info} · INCOMPLETE");
        }
        ui.label(info);
        ui.add_space(4.0);

        // Keyboard transport (when no text field has focus).
        if !ui.ctx().egui_wants_keyboard_input() {
            let (space, left, right, shift) = ui.ctx().input(|i| {
                (
                    i.key_pressed(egui::Key::Space),
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                    i.modifiers.shift,
                )
            });
            if space {
                toggle_play(loaded, status);
            }
            let step = if shift { 5.0 } else { 1.0 };
            if left {
                let pos = position(loaded);
                seek_to(loaded, status, view_start, *view_span, pos - step);
            }
            if right {
                let pos = position(loaded);
                seek_to(loaded, status, view_start, *view_span, pos + step);
            }
        }

        let playing = is_playing(loaded);
        let pos = position(loaded);

        // Transport bar.
        let mut seek_request: Option<f64> = None;
        ui.horizontal(|ui| {
            let has_playback = loaded.playback.is_some();
            if ui
                .add_enabled(
                    has_playback,
                    egui::Button::new(if playing { "Pause" } else { "Play" }),
                )
                .clicked()
            {
                toggle_play(loaded, status);
            }
            ui.label(format!("{} / {}", fmt_time(pos), fmt_time(loaded.duration)));

            let seek_width = (ui.available_width() - 260.0).max(80.0);
            let (response, painter) =
                ui.allocate_painter(egui::vec2(seek_width, 14.0), egui::Sense::click_and_drag());
            let rect = response.rect;
            painter.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
            let frac = if loaded.duration > 0.0 {
                (pos / loaded.duration) as f32
            } else {
                0.0
            };
            let mut played = rect;
            played.set_width(rect.width() * frac);
            painter.rect_filled(played, 3.0, ui.visuals().selection.bg_fill);
            if (response.clicked() || response.dragged())
                && let Some(pointer) = response.interact_pointer_pos()
            {
                let frac = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                seek_request = Some(f64::from(frac) * loaded.duration);
            }

            ui.checkbox(follow, "Follow");
            if ui.button("−").clicked() {
                zoom(loaded, view_start, view_span, 1.5, pos);
            }
            if ui.button("+").clicked() {
                zoom(loaded, view_start, view_span, 1.0 / 1.5, pos);
            }
        });
        if let Some(t) = seek_request {
            seek_to(loaded, status, view_start, *view_span, t);
        }

        // Follow the playhead during playback (continuous scroll).
        if playing && *follow {
            *view_start = (pos - *view_span * 0.9).max(0.0);
        }
        clamp_view(loaded, view_start, view_span);

        // Spectrogram.
        let remaining = ui.available_size() - egui::vec2(0.0, 28.0);
        let (response, painter) = ui.allocate_painter(
            remaining.max(egui::vec2(100.0, 60.0)),
            egui::Sense::click_and_drag(),
        );
        let rect = response.rect;

        let fs = loaded.reader.header.sample_rate;
        let spp = *view_span * fs / TEX_W as f64;
        let p_view_f = *view_start * TEX_W as f64 / *view_span;
        let p_view = p_view_f.floor().max(0.0) as u64;
        loaded
            .spec
            .ensure(&loaded.reader, spp, p_view, *floor_db, *ceil_db);

        // Paint the ring-buffered texture as two UV slices, offset by the
        // fractional column for smooth scrolling.
        let col_w = rect.width() / TEX_W as f32;
        let frac_col = (p_view_f - p_view as f64) as f32;
        let off = (p_view % TEX_W as u64) as usize;
        let left_cols = TEX_W - off;
        let tex_id = loaded.spec.texture.id();
        let x0 = rect.left() - frac_col * col_w;
        let left_w = left_cols as f32 * col_w;
        painter.image(
            tex_id,
            egui::Rect::from_min_size(
                egui::pos2(x0, rect.top()),
                egui::vec2(left_w, rect.height()),
            ),
            egui::Rect::from_min_max(
                egui::pos2(off as f32 / TEX_W as f32, 0.0),
                egui::pos2(1.0, 1.0),
            ),
            egui::Color32::WHITE,
        );
        painter.image(
            tex_id,
            egui::Rect::from_min_size(
                egui::pos2(x0 + left_w, rect.top()),
                egui::vec2(off as f32 * col_w, rect.height()),
            ),
            egui::Rect::from_min_max(
                egui::pos2(0.0, 0.0),
                egui::pos2(off as f32 / TEX_W as f32, 1.0),
            ),
            egui::Color32::WHITE,
        );
        painter.rect_stroke(
            rect,
            0.0,
            ui.visuals().widgets.noninteractive.bg_stroke,
            egui::StrokeKind::Outside,
        );

        // Time grid + labels.
        let step = tick_step(*view_span);
        let mut t = (*view_start / step).ceil() * step;
        while t < *view_start + *view_span {
            let x = rect.left() + ((t - *view_start) / *view_span) as f32 * rect.width();
            painter.line_segment(
                [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_white_alpha(24)),
            );
            painter.text(
                egui::pos2(x + 2.0, rect.top() + 2.0),
                egui::Align2::LEFT_TOP,
                fmt_time(t),
                egui::FontId::proportional(11.0),
                egui::Color32::from_white_alpha(180),
            );
            t += step;
        }

        // Frequency axis labels from the header's channel center frequencies.
        let num_ch = loaded.reader.header.num_channels as usize;
        for i in 0..5 {
            let ch = i * (num_ch - 1) / 4;
            let freq = loaded.reader.header.channel_freqs[ch];
            let row = (num_ch - 1 - ch) as f32 + 0.5;
            let y = rect.top() + row / num_ch as f32 * rect.height();
            let label = if freq >= 1000.0 {
                format!("{:.1}k", freq / 1000.0)
            } else {
                format!("{freq:.0}")
            };
            let galley = painter.layout_no_wrap(
                label,
                egui::FontId::proportional(11.0),
                egui::Color32::WHITE,
            );
            let label_rect = egui::Rect::from_min_size(
                egui::pos2(rect.left() + 3.0, y - galley.size().y / 2.0),
                galley.size(),
            );
            painter.rect_filled(
                label_rect.expand(1.5),
                2.0,
                egui::Color32::from_black_alpha(150),
            );
            painter.galley(label_rect.min, galley, egui::Color32::WHITE);
        }

        // Playhead.
        let p_now = pos * TEX_W as f64 / *view_span;
        let playhead_x = rect.left() + ((p_now - p_view_f) as f32) * col_w;
        if playhead_x >= rect.left() && playhead_x <= rect.right() {
            painter.line_segment(
                [
                    egui::pos2(playhead_x, rect.top()),
                    egui::pos2(playhead_x, rect.bottom()),
                ],
                egui::Stroke::new(1.5, egui::Color32::WHITE),
            );
        }

        // Interactions: click/drag scrubs, wheel pans, pinch/ctrl-wheel zooms.
        if (response.clicked() || response.dragged())
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let frac = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            seek_request = Some(*view_start + f64::from(frac) * *view_span);
        }
        if response.hovered() {
            let zoom_delta = ui.ctx().input(|i| i.zoom_delta());
            if zoom_delta != 1.0 {
                let anchor = response
                    .hover_pos()
                    .map(|p| {
                        *view_start + f64::from((p.x - rect.left()) / rect.width()) * *view_span
                    })
                    .unwrap_or(pos);
                zoom(
                    loaded,
                    view_start,
                    view_span,
                    1.0 / f64::from(zoom_delta),
                    anchor,
                );
            } else {
                let scroll = ui.ctx().input(|i| i.smooth_scroll_delta);
                if scroll.x != 0.0 || scroll.y != 0.0 {
                    *view_start -=
                        f64::from(scroll.x + scroll.y) / f64::from(rect.width()) * *view_span;
                    *follow = false;
                    clamp_view(loaded, view_start, view_span);
                }
            }
        }
        if let Some(t) = seek_request {
            seek_to(loaded, status, view_start, *view_span, t);
        }

        // Hover readout.
        if let Some(pointer) = response.hover_pos() {
            let t_frac = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
            let time = *view_start + f64::from(t_frac) * *view_span;
            let ch_frac = ((pointer.y - rect.top()) / rect.height()).clamp(0.0, 1.0);
            let ch = (num_ch as f32 * (1.0 - ch_frac)) as usize;
            let ch = ch.min(num_ch - 1);
            let sample = (time * fs).clamp(0.0, (loaded.reader.header.num_samples - 1) as f64);
            let value_label = match loaded.reader.header.value_kind {
                VALUE_SALIENCE => "salience dB",
                VALUE_AMPLITUDE => "dB",
                _ => "energy dB",
            };
            if loaded.reader.is_binaural() {
                let sample = sample as u64;
                let amp = loaded.reader.dcgc_row(sample)[ch].abs();
                let db = value_db(loaded.reader.header.value_kind, amp);
                let iid_db = loaded.reader.iid_row(sample)[ch];
                let itd_ms = loaded.reader.itd_row(sample)[ch] * 1e3;
                let ei_mu = loaded.reader.ei_row(sample)[ch];
                response.clone().on_hover_text(format!(
                    "{} · {:.0} Hz (ch {}) · {:.1} {value_label} · {iid_db:+.1} dB IID · {itd_ms:.2} ms ITD · {ei_mu:.3} MU EI",
                    fmt_time(time),
                    loaded.reader.header.channel_freqs[ch],
                    ch,
                    db,
                ));
            } else {
                let value = loaded.reader.row(sample as u64)[ch];
                let db = value_db(loaded.reader.header.value_kind, value);
                response.clone().on_hover_text(format!(
                    "{} · {:.0} Hz (ch {}) · {:.1} {value_label}",
                    fmt_time(time),
                    loaded.reader.header.channel_freqs[ch],
                    ch,
                    db
                ));
            }
        }

        // Bottom bar: dB range controls.
        ui.horizontal(|ui| {
            ui.label("dB range");
            let floor_changed = ui
                .add(egui::Slider::new(floor_db, -140.0..=-6.0).text("floor"))
                .changed();
            let ceil_changed = ui
                .add(egui::Slider::new(ceil_db, -120.0..=20.0).text("ceiling"))
                .changed();
            ui.checkbox(&mut self.auto_range, "Auto range");
            if self.auto_range {
                let new_ceil = auto_ceiling(&loaded.reader, *view_start, *view_span);
                if let Some(last_auto_range) = self.last_auto_range {
                    if last_auto_range.elapsed() > Duration::from_millis(50) {
                        self.last_auto_range = None;
                    } else {
                        ui.request_repaint();
                    }
                }
                if self.last_auto_range.is_none() {
                    if (new_ceil - *ceil_db) > 2.0 {
                        *ceil_db += 0.1;
                        *floor_db = (*ceil_db - 45.0).max(-140.0);
                        loaded.spec.invalidate();
                        ui.request_repaint();
                        self.last_auto_range = Some(Instant::now());
                    }
                    if (*ceil_db - new_ceil) > 2.0 {
                        *ceil_db -= 0.1;
                        *floor_db = (*ceil_db - 45.0).max(-140.0);
                        loaded.spec.invalidate();
                        ui.request_repaint();
                        self.last_auto_range = Some(Instant::now());
                    }
                }
            }
            if floor_changed || ceil_changed {
                if *floor_db >= *ceil_db {
                    *floor_db = *ceil_db - 6.0;
                }
                loaded.spec.invalidate();
                ui.request_repaint();
            }
            if loaded.reader.is_binaural() {
                ui.separator();
                let mut stereo_changed = false;
                egui::ComboBox::from_label("Stereo variable")
                    .selected_text(match loaded.spec.stereo_var {
                        StereoVar::Iid => "IID",
                        StereoVar::Itd => "ITD",
                    })
                    .show_ui(ui, |ui| {
                        stereo_changed |= ui
                            .selectable_value(&mut loaded.spec.stereo_var, StereoVar::Iid, "IID")
                            .changed();
                        stereo_changed |= ui
                            .selectable_value(&mut loaded.spec.stereo_var, StereoVar::Itd, "ITD")
                            .changed();
                    });
                if stereo_changed {
                    loaded.spec.invalidate();
                    ui.request_repaint();
                }
            }
            ui.separator();
            ui.label("Space: play/pause · ←/→: seek (Shift = 5 s) · drag: scrub · wheel: pan · pinch/ctrl-wheel: zoom");
        });

        if playing {
            ui.ctx().request_repaint();
        }
    }
}

fn seek_to(
    loaded: &mut Loaded,
    status: &mut String,
    view_start: &mut f64,
    view_span: f64,
    seconds: f64,
) {
    let seconds = seconds.clamp(0.0, loaded.duration);
    if let Some(playback) = &mut loaded.playback {
        playback.seek(&loaded.audio_path, seconds, status);
    }
    // Keep the playhead visible after a seek.
    if seconds < *view_start || seconds > *view_start + view_span {
        *view_start = (seconds - view_span * 0.3).max(0.0);
    }
}

fn toggle_play(loaded: &mut Loaded, status: &mut String) {
    let Some(playback) = &mut loaded.playback else {
        return;
    };
    if playback.player.is_paused() {
        if playback.player.empty() {
            playback.restart(&loaded.audio_path, status);
        }
        playback.player.play();
    } else {
        playback.player.pause();
    }
}

fn position(loaded: &Loaded) -> f64 {
    match &loaded.playback {
        Some(playback) if !playback.player.empty() => {
            playback.player.get_pos().as_secs_f64().min(loaded.duration)
        }
        Some(_) => loaded.duration, // finished
        None => 0.0,
    }
}

fn is_playing(loaded: &Loaded) -> bool {
    loaded
        .playback
        .as_ref()
        .is_some_and(|p| !p.player.is_paused() && !p.player.empty())
}

fn zoom(loaded: &Loaded, view_start: &mut f64, view_span: &mut f64, factor: f64, anchor: f64) {
    let max_span = loaded.duration.min(MAX_SPAN);
    let new_span = (*view_span * factor).clamp(MIN_SPAN, max_span.max(MIN_SPAN));
    let factor = new_span / *view_span;
    *view_start = anchor - (anchor - *view_start) * factor;
    *view_span = new_span;
    clamp_view(loaded, view_start, view_span);
}

fn clamp_view(loaded: &Loaded, view_start: &mut f64, view_span: &mut f64) {
    *view_span = (*view_span).clamp(MIN_SPAN, loaded.duration.clamp(MIN_SPAN, MAX_SPAN));
    *view_start = (*view_start).clamp(0.0, (loaded.duration - *view_span).max(0.0));
}

impl Playback {
    fn new(path: &Path) -> Result<Self, String> {
        let handle = DeviceSinkBuilder::open_default_sink().map_err(|e| e.to_string())?;
        let player = Player::connect_new(handle.mixer());
        let decoder = streaming_decoder(path).map_err(|e| e.to_string())?;
        player.append(decoder);
        player.pause();
        Ok(Self {
            _handle: handle,
            player,
        })
    }

    fn restart(&mut self, path: &Path, status: &mut String) {
        self.player.stop();
        match streaming_decoder(path) {
            Ok(decoder) => self.player.append(decoder),
            Err(error) => *status = format!("cannot re-open audio: {error}"),
        }
    }

    fn seek(&mut self, path: &Path, seconds: f64, status: &mut String) {
        if self.player.empty() {
            self.restart(path, status);
        }
        if let Err(error) = self.player.try_seek(Duration::from_secs_f64(seconds)) {
            *status = format!("seek failed: {error}");
        }
    }
}

/// Ring-buffered spectrogram texture over the analysis mmap. Column `p`
/// (covering `[p*spp, (p+1)*spp)` samples) lives at texel `p % TEX_W`; during
/// playback only newly visible columns are computed and appended.
struct Spectrogram {
    texture: egui::TextureHandle,
    image: egui::ColorImage,
    num_ch: usize,
    spp: f64,
    next_p: u64,
    filled: u64,
    dirty: bool,
    sums: Vec<f32>,
    lut: Vec<egui::Color32>,
    /// Binaural rendering state; `iid_sums`, `itd_sums`, and `lut2d` stay
    /// empty for mono analyses, and `sums` doubles as the dcGC sums buffer.
    binaural: bool,
    tau_max: f32,
    iid_max: f32,
    stereo_var: StereoVar,
    iid_sums: Vec<f32>,
    itd_sums: Vec<f32>,
    lut2d: Vec<egui::Color32>,
}

/// Which interaural variable drives the hue of a binaural spectrogram;
/// the downmixed amplitude always drives lightness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum StereoVar {
    /// Characteristic IID of the lowest-activity EI unit.
    #[default]
    Iid,
    /// Characteristic ITD of the lowest-activity EI unit.
    Itd,
}

impl Spectrogram {
    fn new(ctx: &egui::Context, reader: &AnalysisReader) -> Self {
        let num_ch = reader.header.num_channels as usize;
        let image =
            egui::ColorImage::new([TEX_W, num_ch], vec![egui::Color32::BLACK; TEX_W * num_ch]);
        let texture = ctx.load_texture("spectrogram", image.clone(), egui::TextureOptions::NEAREST);
        let binaural = reader.is_binaural();
        let tau_max = reader
            .header
            .tau_seconds
            .iter()
            .map(|tau| tau.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12) as f32;
        let iid_max = reader
            .header
            .iid_db
            .iter()
            .map(|iid| iid.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-12) as f32;
        Self {
            texture,
            image,
            num_ch,
            spp: 0.0,
            next_p: 0,
            filled: 0,
            dirty: false,
            sums: vec![0.0; num_ch],
            lut: magma_lut(),
            binaural,
            tau_max,
            iid_max,
            stereo_var: StereoVar::default(),
            iid_sums: if binaural {
                vec![0.0; num_ch]
            } else {
                Vec::new()
            },
            itd_sums: if binaural {
                vec![0.0; num_ch]
            } else {
                Vec::new()
            },
            lut2d: if binaural {
                crate::colormap::bivariate_lut()
            } else {
                Vec::new()
            },
        }
    }

    fn invalidate(&mut self) {
        self.filled = 0;
    }

    fn lo(&self) -> u64 {
        self.next_p - self.filled
    }

    fn ensure(
        &mut self,
        reader: &AnalysisReader,
        spp: f64,
        p_lo: u64,
        floor_db: f32,
        ceil_db: f32,
    ) {
        let want_hi = p_lo + TEX_W as u64;
        let need_reset = self.filled == 0
            || (spp / self.spp - 1.0).abs() > 1e-9
            || p_lo < self.lo()
            || p_lo > self.next_p;
        if need_reset {
            self.spp = spp;
            for p in p_lo..want_hi {
                self.render_column(reader, p, floor_db, ceil_db);
            }
            self.next_p = want_hi;
            self.filled = TEX_W as u64;
        } else if want_hi > self.next_p {
            for p in self.next_p..want_hi {
                self.render_column(reader, p, floor_db, ceil_db);
            }
            self.filled = (self.filled + (want_hi - self.next_p)).min(TEX_W as u64);
            self.next_p = want_hi;
        }
        if self.dirty {
            self.texture
                .set(self.image.clone(), egui::TextureOptions::NEAREST);
            self.dirty = false;
        }
    }

    fn render_column(&mut self, reader: &AnalysisReader, p: u64, floor_db: f32, ceil_db: f32) {
        let num_samples = reader.header.num_samples;
        let x = (p % TEX_W as u64) as usize;
        let a = (p as f64 * self.spp) as u64;
        if a >= num_samples {
            for y in 0..self.num_ch {
                self.image.pixels[y * TEX_W + x] = egui::Color32::BLACK;
            }
            self.dirty = true;
            return;
        }
        let b = (((p + 1) as f64 * self.spp) as u64)
            .max(a + 1)
            .min(num_samples);
        let span = (ceil_db - floor_db).max(1e-6);
        let value_kind = reader.header.value_kind;
        if self.binaural {
            reader.aggregate_binaural_column(
                a,
                b,
                &mut self.sums,
                &mut self.iid_sums,
                &mut self.itd_sums,
            );
            let count = (b - a) as f32;
            for ch in 0..self.num_ch {
                // Lightness is the column mean of the stored dcGC mean.
                let db = value_db(value_kind, self.sums[ch] / count);
                let t = ((db - floor_db) / span).clamp(0.0, 1.0);
                // The stored values are the characteristic IID/ITD of each
                // sample's lowest-activity EI unit; use the column mean.
                let s = match self.stereo_var {
                    StereoVar::Iid => {
                        let iid_db = self.iid_sums[ch] / count;
                        (iid_db / self.iid_max + 1.0) / 2.0
                    }
                    StereoVar::Itd => {
                        let tau = self.itd_sums[ch] / count;
                        (tau / self.tau_max + 1.0) / 2.0
                    }
                };
                let s = s.clamp(0.0, 1.0);
                let color = self.lut2d[(t * 255.0) as usize * 256 + (s * 255.0) as usize];
                // Row 0 is the top of the image; put low frequencies at the bottom.
                let y = self.num_ch - 1 - ch;
                self.image.pixels[y * TEX_W + x] = color;
            }
            self.dirty = true;
            return;
        }
        reader.column_means(a, b, &mut self.sums);
        for ch in 0..self.num_ch {
            let db = value_db(value_kind, self.sums[ch]);
            let t = ((db - floor_db) / span).clamp(0.0, 1.0);
            let color = self.lut[(t * 255.0) as usize];
            // Row 0 is the top of the image; put low frequencies at the bottom.
            let y = self.num_ch - 1 - ch;
            self.image.pixels[y * TEX_W + x] = color;
        }
        self.dirty = true;
    }
}

fn auto_ceiling(reader: &AnalysisReader, view_start: f64, view_span: f64) -> f32 {
    let fs = reader.header.sample_rate;
    let spp = view_span * fs / TEX_W as f64;
    let p0 = (view_start * TEX_W as f64 / view_span).max(0.0) as u64;
    let mut max = 0.0_f32;
    let num_ch = reader.header.num_channels as usize;
    let mut means = vec![0.0_f32; num_ch];
    if reader.is_binaural() {
        // Ceiling from the column means of the stored two-ear dcGC mean.
        let mut iid_sums = vec![0.0_f32; num_ch];
        let mut itd_sums = vec![0.0_f32; num_ch];
        for i in 0..64_u64 {
            let p = p0 + i * (TEX_W as u64 / 64);
            let a = (p as f64 * spp) as u64;
            let b = (((p + 1) as f64 * spp) as u64).max(a + 1);
            reader.aggregate_binaural_column(a, b, &mut means, &mut iid_sums, &mut itd_sums);
            let count = (b - a) as f32;
            for &v in &means {
                max = max.max(v / count);
            }
        }
    } else {
        for i in 0..64_u64 {
            let p = p0 + i * (TEX_W as u64 / 64);
            let a = (p as f64 * spp) as u64;
            let b = (((p + 1) as f64 * spp) as u64).max(a + 1);
            reader.column_means(a, b, &mut means);
            for &v in &means {
                max = max.max(v);
            }
        }
    }
    value_db(reader.header.value_kind, max).ceil()
}

/// Approximate matplotlib "magma" via anchor-point interpolation.
fn magma_lut() -> Vec<egui::Color32> {
    const ANCHORS: [(f32, (u8, u8, u8)); 9] = [
        (0.00, (0, 0, 4)),
        (0.13, (28, 16, 68)),
        (0.25, (79, 18, 123)),
        (0.38, (129, 37, 129)),
        (0.50, (181, 54, 122)),
        (0.63, (229, 80, 100)),
        (0.75, (251, 135, 97)),
        (0.88, (254, 194, 135)),
        (1.00, (252, 253, 245)),
    ];
    (0..256)
        .map(|i| {
            let t = i as f32 / 255.0;
            let upper = ANCHORS
                .iter()
                .position(|(at, _)| t <= *at)
                .unwrap_or(ANCHORS.len() - 1)
                .max(1);
            let (t0, c0) = ANCHORS[upper - 1];
            let (t1, c1) = ANCHORS[upper];
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * f) as u8;
            egui::Color32::from_rgb(mix(c0.0, c1.0), mix(c0.1, c1.1), mix(c0.2, c1.2))
        })
        .collect()
}

fn tick_step(span: f64) -> f64 {
    const STEPS: [f64; 12] = [
        0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0,
    ];
    for step in STEPS {
        if span / step <= 12.0 {
            return step;
        }
    }
    120.0
}

pub fn fmt_time(secs: f64) -> String {
    let secs = secs.max(0.0);
    let minutes = (secs / 60.0).floor();
    let rest = secs - minutes * 60.0;
    format!("{minutes:02.0}:{rest:06.3}")
}

fn file_row(ui: &mut egui::Ui, label: &str, path: &mut Option<PathBuf>, analysis: bool) {
    ui.label(label);
    if ui.button("Browse…").clicked() {
        let mut dialog = rfd::FileDialog::new();
        dialog = if analysis {
            dialog.add_filter("gammachirp analysis", &["gca"])
        } else {
            dialog.add_filter(
                "audio",
                &[
                    "wav", "flac", "mp3", "ogg", "opus", "m4a", "mp4", "mkv", "mka", "webm", "aac",
                    "aif", "aiff", "caf", "alac",
                ],
            )
        };
        if let Some(picked) = dialog.pick_file() {
            *path = Some(picked);
        }
    }
    let text = path
        .as_ref()
        .map(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string())
        })
        .unwrap_or_else(|| "(none)".into());
    ui.add(egui::Label::new(egui::RichText::new(text).monospace()).wrap())
        .on_hover_text(
            path.as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        );
}
