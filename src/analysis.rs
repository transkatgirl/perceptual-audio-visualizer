//! Offline per-sample analysis: mono GCFB v2.34 or binaural Breebaart-2001 /
//! GCFB hybrid, with a binary file format, builder core, and a memory-mapped
//! reader. This module contains no GUI code so it can be unit-tested headlessly.
//!
//! File format (`.gca`, little-endian):
//!
//! ```text
//! 0   "GCA1" magic (4 B)
//! 4   u32  version = 1
//! 8   u32  engine = 1  (1 = gcfb_v234 per-sample)
//! 12  f64  sample_rate
//! 20  u32  num_channels
//! 24  u64  num_samples           (patched in at end of write)
//! 32  f64  f_range_low
//! 40  f64  f_range_high
//! 48  u32  control_mode (0=Static, 1=Dynamic, 2=Level)
//! 52  u32  header_len = 68 + 8*(num_channels + num_tau + num_iid)
//! 56  u32  mode (0 = mono, 1 = binaural)
//! 60  u32  num_tau (ITD grid size; 0 for mono)
//! 64  u32  num_iid (IID grid size; 0 for mono)
//! 68  f64 × num_channels   channel center frequencies (gc_resp.fr1)
//! ..  f64 × num_tau        EI population ITD grid in seconds (binaural)
//! ..  f64 × num_iid        EI population IID grid in dB (binaural)
//! header_len..  sample-major f32 data:
//!               mono:     per sample [dcgc num_ch]
//!               binaural: per sample [dcgc_mean num_ch][iid num_ch]
//!                         [itd num_ch][ei num_ch] — the mean of the two ears'
//!                         dcGC amplitudes plus, per channel, the characteristic
//!                         IID (dB) and ITD (seconds) of the lowest-activity EI
//!                         unit plus that unit's activity
//! ```

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytemuck::cast_slice;
use gammachirp_rs::gcfb_v234::{ControlMode, DcgcEvent, DynHpaf, GcParam, GcfbStream};
use gammachirp_rs::{
    breebaart2001::{
        EiConfig, EiIntegrationBoundary, EiStreamSample, EiUnit, HybridBinauralConfig,
        HybridBinauralStream, HybridBinauralStreamStep, PeripheralConfig,
    },
    gcfb_v234::gcfb_v234::LvlEst,
};
use memmap2::Mmap;
use rodio::{Decoder, Source};

const MAGIC: &[u8; 4] = b"GCA1";
const VERSION: u32 = 1;
const ENGINE_GCFB_V234_SAMPLE: u32 = 1;
const FIXED_HEADER_LEN: usize = 68;
const NUM_SAMPLES_OFFSET: u64 = 24;

/// Numeric analysis-mode tags stored in the file header.
pub const MODE_MONO: u32 = 0;
pub const MODE_BINAURAL: u32 = 1;

/// Errors returned by [`probe_audio`] and [`run_analysis`].
#[derive(Debug)]
pub enum AnalysisError {
    /// The user cancelled the run; the partial output file was deleted.
    Cancelled,
    Message(String),
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisError::Cancelled => write!(f, "analysis cancelled"),
            AnalysisError::Message(message) => write!(f, "{message}"),
        }
    }
}

impl Error for AnalysisError {}

impl From<std::io::Error> for AnalysisError {
    fn from(error: std::io::Error) -> Self {
        AnalysisError::Message(error.to_string())
    }
}

fn message<E: fmt::Display>(error: E) -> AnalysisError {
    AnalysisError::Message(error.to_string())
}

/// Decoded audio metadata, read without decoding the whole stream.
pub struct AudioProbe {
    pub sample_rate: u32,
    pub channels: u32,
    pub total_duration: Option<Duration>,
}

/// Open a streaming decoder for `path` and read its stream metadata.
pub fn probe_audio(path: &Path) -> Result<AudioProbe, AnalysisError> {
    let decoder = streaming_decoder(path)?;
    Ok(AudioProbe {
        sample_rate: decoder.sample_rate().get(),
        channels: decoder.channels().get() as u32,
        total_duration: decoder.total_duration(),
    })
}

/// Open a streaming decoder for `path`. The decoder reads from disk as it is
/// iterated, so arbitrarily large files stay out of memory.
pub fn streaming_decoder(path: &Path) -> Result<Decoder<BufReader<File>>, AnalysisError> {
    let file = File::open(path).map_err(message)?;
    Decoder::try_from(BufReader::new(file)).map_err(message)
}

/// Analysis kind selected by the builder.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnalysisMode {
    /// Downmix all input channels to mono and run a single GCFB.
    #[default]
    Mono,
    /// Breebaart-2001 / GCFB hybrid of a stereo input: per-ear GCFB plus an
    /// excitation-inhibition population tuned to interaural time differences.
    Binaural,
}

impl AnalysisMode {
    /// Numeric tag stored in the file header.
    pub fn tag(self) -> u32 {
        match self {
            AnalysisMode::Mono => MODE_MONO,
            AnalysisMode::Binaural => MODE_BINAURAL,
        }
    }
}

/// Human-readable label for a stored analysis-mode tag.
pub fn mode_label(tag: u32) -> &'static str {
    match tag {
        MODE_MONO => "Mono",
        MODE_BINAURAL => "Binaural",
        _ => "Unknown",
    }
}

/// Binaural-specific parameters: the EI population shape plus the peripheral
/// and EI stages of the Breebaart hybrid.
///
/// The population is a two-dimensional grid over characteristic ITD and IID
/// (like the `breebaart2001_hybrid` example); per channel and sample only the
/// lowest-activity unit's characteristic IID, ITD, and activity are stored.
/// `ei.integration_boundary` is forced to causal zero-state and
/// `ei.max_abs_delay_seconds` to `tau_max_seconds` by the builder; every
/// other item is user-facing.
#[derive(Clone, Debug)]
pub struct BinauralParams {
    /// Largest characteristic ITD of the EI population, in seconds.
    pub tau_max_seconds: f64,
    /// Number of ITD grid points, spaced linearly over ±`tau_max_seconds`.
    pub num_tau: usize,
    /// Largest characteristic IID of the EI population, in dB.
    pub iid_max_db: f64,
    /// Number of IID grid points, spaced linearly over ±`iid_max_db`.
    pub num_iid: usize,
    /// Inner-hair-cell, level-calibration, and adaptation-loop settings.
    pub peripheral: PeripheralConfig,
    /// EI population settings, including post-EI internal noise.
    pub ei: EiConfig,
}

impl Default for BinauralParams {
    fn default() -> Self {
        Self {
            tau_max_seconds: 5e-3,
            num_tau: 9,
            iid_max_db: 5.0,
            num_iid: 19,
            peripheral: PeripheralConfig::default(),
            ei: EiConfig::streaming(),
        }
    }
}

/// `n` points linearly spaced over ±`max`; a single point sits at zero.
fn linspace_symmetric(max: f64, n: usize) -> Vec<f64> {
    let n = n.max(1);
    if n == 1 {
        return vec![0.0];
    }
    (0..n)
        .map(|index| max * (2.0 * index as f64 / (n - 1) as f64 - 1.0))
        .collect()
}

impl BinauralParams {
    /// The ITD grid: `num_tau` points linearly spaced over ±`tau_max_seconds`.
    pub fn tau_grid(&self) -> Vec<f64> {
        linspace_symmetric(self.tau_max_seconds, self.num_tau)
    }

    /// The IID grid: `num_iid` points linearly spaced over ±`iid_max_db`.
    pub fn iid_grid(&self) -> Vec<f64> {
        linspace_symmetric(self.iid_max_db, self.num_iid)
    }

    /// The EI population: the full ITD × IID grid, IID-major and ITD-minor
    /// (the same ordering as the `breebaart2001_hybrid` example).
    pub fn units(&self) -> Vec<EiUnit> {
        let tau_grid = self.tau_grid();
        self.iid_grid()
            .iter()
            .flat_map(|&iid_db| tau_grid.iter().map(move |&tau| EiUnit::new(tau, iid_db)))
            .collect()
    }
}

/// Parameters for one builder run: a complete [`GcParam`] template plus the
/// analysis mode and its binaural settings.
///
/// Every user-facing GcParam item is customizable. The exceptions are managed
/// by the builder itself: `fs` is forced to the input file's sample rate and
/// `dyn_hpaf.str_prc` is forced to `"sample-base"` (that is what makes this a
/// per-sample analysis), while `hloss`, `fr1`, and the derived `lvl_est`
/// fields are computed by the library's `set_param`.
#[derive(Clone, Debug)]
pub struct BuilderParams {
    pub gc: GcParam,
    pub mode: AnalysisMode,
    /// Used only when `mode` is [`AnalysisMode::Binaural`].
    pub binaural: BinauralParams,
}

impl Default for BuilderParams {
    fn default() -> Self {
        Self {
            gc: GcParam {
                num_ch: 100,
                f_range: [40.0, 16_000.0],
                out_mid_crct: "ELC".into(),
                ctrl: ControlMode::Dynamic,
                dyn_hpaf: DynHpaf {
                    str_prc: "sample-base".into(),
                    ..DynHpaf::default()
                },
                lvl_est: LvlEst {
                    rms2spldb: 100.0,
                    ..Default::default()
                },
                ..GcParam::default()
            },
            mode: AnalysisMode::Mono,
            binaural: BinauralParams::default(),
        }
    }
}

/// Numeric control-mode tag stored in the file header.
pub fn control_mode_tag(control: ControlMode) -> u32 {
    match control {
        ControlMode::Static => 0,
        ControlMode::Dynamic => 1,
        ControlMode::Level => 2,
    }
}

/// Human-readable label for a stored control-mode tag.
pub fn control_mode_label(tag: u32) -> &'static str {
    match tag {
        0 => "Static",
        1 => "Dynamic",
        2 => "Level",
        _ => "Unknown",
    }
}

/// Parsed `.gca` header.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisHeader {
    pub sample_rate: f64,
    pub num_channels: u32,
    pub num_samples: u64,
    pub f_range: [f64; 2],
    /// See [`control_mode_tag`].
    pub control_mode: u32,
    /// See [`MODE_MONO`] / [`MODE_BINAURAL`].
    pub mode: u32,
    /// ITD grid of the EI population in seconds (empty for mono).
    pub tau_seconds: Vec<f64>,
    /// IID grid of the EI population in dB (empty for mono).
    pub iid_db: Vec<f64>,
    /// Channel center frequencies in Hz (`num_channels` entries).
    pub channel_freqs: Vec<f64>,
}

impl AnalysisHeader {
    pub fn header_len(&self) -> usize {
        FIXED_HEADER_LEN
            + 8 * (self.channel_freqs.len() + self.tau_seconds.len() + self.iid_db.len())
    }

    pub fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }

    /// f32 values stored per sample: `num_ch` for mono, `4 * num_ch` for
    /// binaural (the two-ear dcGC mean plus the lowest EI unit's IID, ITD,
    /// and activity).
    pub fn values_per_sample(&self) -> usize {
        let num_ch = self.num_channels as usize;
        if self.mode == MODE_BINAURAL {
            4 * num_ch
        } else {
            num_ch
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![0_u8; self.header_len()];
        bytes[0..4].copy_from_slice(MAGIC);
        bytes[4..8].copy_from_slice(&VERSION.to_le_bytes());
        bytes[8..12].copy_from_slice(&ENGINE_GCFB_V234_SAMPLE.to_le_bytes());
        bytes[12..20].copy_from_slice(&self.sample_rate.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.num_channels.to_le_bytes());
        bytes[24..32].copy_from_slice(&self.num_samples.to_le_bytes());
        bytes[32..40].copy_from_slice(&self.f_range[0].to_le_bytes());
        bytes[40..48].copy_from_slice(&self.f_range[1].to_le_bytes());
        bytes[48..52].copy_from_slice(&self.control_mode.to_le_bytes());
        bytes[52..56].copy_from_slice(&(self.header_len() as u32).to_le_bytes());
        bytes[56..60].copy_from_slice(&self.mode.to_le_bytes());
        bytes[60..64].copy_from_slice(&(self.tau_seconds.len() as u32).to_le_bytes());
        bytes[64..68].copy_from_slice(&(self.iid_db.len() as u32).to_le_bytes());
        for (index, freq) in self.channel_freqs.iter().enumerate() {
            let start = FIXED_HEADER_LEN + 8 * index;
            bytes[start..start + 8].copy_from_slice(&freq.to_le_bytes());
        }
        for (index, tau) in self.tau_seconds.iter().enumerate() {
            let start = FIXED_HEADER_LEN + 8 * (self.channel_freqs.len() + index);
            bytes[start..start + 8].copy_from_slice(&tau.to_le_bytes());
        }
        for (index, iid) in self.iid_db.iter().enumerate() {
            let start =
                FIXED_HEADER_LEN + 8 * (self.channel_freqs.len() + self.tau_seconds.len() + index);
            bytes[start..start + 8].copy_from_slice(&iid.to_le_bytes());
        }
        bytes
    }

    fn decode(bytes: &[u8]) -> Result<Self, AnalysisError> {
        let invalid = |text: &str| AnalysisError::Message(format!("invalid analysis file: {text}"));
        if bytes.len() < FIXED_HEADER_LEN {
            return Err(invalid("file shorter than fixed header"));
        }
        if &bytes[0..4] != MAGIC {
            return Err(invalid("bad magic (not a .gca file)"));
        }
        let read_u32 =
            |offset: usize| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let read_u64 =
            |offset: usize| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        let read_f64 =
            |offset: usize| f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        let version = read_u32(4);
        if version != VERSION {
            return Err(invalid(&format!(
                "unsupported version {version} (rebuild the analysis)"
            )));
        }
        let engine = read_u32(8);
        if engine != ENGINE_GCFB_V234_SAMPLE {
            return Err(invalid(&format!("unsupported engine {engine}")));
        }
        let mode = read_u32(56);
        if mode != MODE_MONO && mode != MODE_BINAURAL {
            return Err(invalid(&format!("unknown analysis mode {mode}")));
        }
        let num_channels = read_u32(20);
        let num_tau = read_u32(60);
        let num_iid = read_u32(64);
        if num_channels == 0 || (mode == MODE_MONO) != (num_tau == 0 && num_iid == 0) {
            return Err(invalid("inconsistent mode / channel / EI unit counts"));
        }
        let header_len = read_u32(52) as usize;
        if header_len != FIXED_HEADER_LEN + 8 * (num_channels + num_tau + num_iid) as usize {
            return Err(invalid("inconsistent channel count / header length"));
        }
        if bytes.len() < header_len {
            return Err(invalid("file shorter than full header"));
        }
        let mut channel_freqs = Vec::with_capacity(num_channels as usize);
        for index in 0..num_channels as usize {
            channel_freqs.push(read_f64(FIXED_HEADER_LEN + 8 * index));
        }
        let mut tau_seconds = Vec::with_capacity(num_tau as usize);
        for index in 0..num_tau as usize {
            tau_seconds.push(read_f64(
                FIXED_HEADER_LEN + 8 * (num_channels as usize + index),
            ));
        }
        let mut iid_db = Vec::with_capacity(num_iid as usize);
        for index in 0..num_iid as usize {
            iid_db.push(read_f64(
                FIXED_HEADER_LEN + 8 * (num_channels as usize + num_tau as usize + index),
            ));
        }
        Ok(Self {
            sample_rate: read_f64(12),
            num_channels,
            num_samples: read_u64(24),
            f_range: [read_f64(32), read_f64(40)],
            control_mode: read_u32(48),
            mode,
            tau_seconds,
            iid_db,
            channel_freqs,
        })
    }
}

/// Validate the filterbank settings shared by both analysis modes.
fn validate_gc(gc: &GcParam, fs: f64) -> Result<(), AnalysisError> {
    if gc.num_ch < 2 {
        return Err(AnalysisError::Message(
            "channel count must be at least 2".into(),
        ));
    }
    if !(gc.f_range[0] > 0.0 && gc.f_range[0] < gc.f_range[1]) {
        return Err(AnalysisError::Message(format!(
            "invalid frequency range [{:.0}, {:.0}] Hz",
            gc.f_range[0], gc.f_range[1]
        )));
    }
    if gc.f_range[1] >= fs / 2.0 {
        return Err(AnalysisError::Message(format!(
            "max frequency {:.0} Hz must be below the Nyquist limit {:.0} Hz \
             (file sample rate {:.0} Hz)",
            gc.f_range[1],
            fs / 2.0,
            fs
        )));
    }
    Ok(())
}

/// The prepared filterbank template with the builder-forced fields applied.
fn prepared_gc(gc: &GcParam, fs: f64) -> GcParam {
    GcParam {
        fs,
        // Sample-base control is what makes this a per-sample analysis;
        // frame-base emits delayed frame-rate events instead.
        dyn_hpaf: DynHpaf {
            str_prc: "sample-base".into(),
            ..gc.dyn_hpaf.clone()
        },
        ..gc.clone()
    }
}

/// Run an offline per-sample analysis of `input` and stream the result to
/// `output`. In mono mode the input is downmixed and run through a single
/// GCFB v2.34; in binaural mode a stereo input runs through the
/// Breebaart-2001 / GCFB hybrid (per-ear GCFB plus an EI population).
///
/// Decoding, filtering, and writing are all chunked, so peak memory is
/// independent of the input length. `progress` is called with the number of
/// samples processed every chunk (~0.25 s of audio). Setting `cancel`
/// aborts the run; cancelled and failed runs delete the partial output file.
pub fn run_analysis(
    input: &Path,
    output: &Path,
    params: &BuilderParams,
    mut progress: impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<AnalysisHeader, AnalysisError> {
    let result = match params.mode {
        AnalysisMode::Mono => run_mono_inner(input, output, params, &mut progress, cancel),
        AnalysisMode::Binaural => run_binaural_inner(input, output, params, &mut progress, cancel),
    };
    if result.is_err() {
        // Never leave a partial or corrupt file behind.
        let _ = fs::remove_file(output);
    }
    result
}

fn run_mono_inner(
    input: &Path,
    output: &Path,
    params: &BuilderParams,
    progress: &mut impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<AnalysisHeader, AnalysisError> {
    let probe = probe_audio(input)?;
    let fs = probe.sample_rate as f64;
    validate_gc(&params.gc, fs)?;

    let mut stream = GcfbStream::new(prepared_gc(&params.gc, fs)).map_err(message)?;

    let header = AnalysisHeader {
        sample_rate: fs,
        num_channels: params.gc.num_ch as u32,
        num_samples: 0, // patched at the end
        f_range: params.gc.f_range,
        control_mode: control_mode_tag(params.gc.ctrl),
        mode: params.mode.tag(),
        tau_seconds: Vec::new(),
        iid_db: Vec::new(),
        channel_freqs: stream.gc_resp().fr1.to_vec(),
    };

    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header.encode())?;

    let decoder = streaming_decoder(input)?;
    let channels = probe.channels.max(1) as usize;
    let chunk_len = (probe.sample_rate as usize / 4).max(1); // ~0.25 s of mono audio
    let num_ch = params.gc.num_ch;

    let mut mono_chunk: Vec<f64> = Vec::with_capacity(chunk_len);
    let mut mix_acc = 0.0_f64;
    let mut mix_count = 0_usize;
    let mut write_buf: Vec<f32> = Vec::with_capacity(chunk_len * num_ch);
    let mut num_samples = 0_u64;

    for sample in decoder {
        mix_acc += f64::from(sample);
        mix_count += 1;
        if mix_count == channels {
            mono_chunk.push(mix_acc / channels as f64);
            mix_acc = 0.0;
            mix_count = 0;
        }
        if mono_chunk.len() == chunk_len {
            process_chunk(
                &mut stream,
                &mono_chunk,
                &mut write_buf,
                &mut writer,
                &mut num_samples,
                progress,
                cancel,
            )?;
            mono_chunk.clear();
        }
    }
    if !mono_chunk.is_empty() {
        process_chunk(
            &mut stream,
            &mono_chunk,
            &mut write_buf,
            &mut writer,
            &mut num_samples,
            progress,
            cancel,
        )?;
    }
    if num_samples == 0 {
        return Err(AnalysisError::Message(
            "no decodable audio samples in input file".into(),
        ));
    }
    // Sample-base modes emit no tail events, but the contract requires this.
    stream.finish().map_err(message)?;

    finalize(writer, num_samples)?;
    progress(num_samples);
    Ok(AnalysisHeader {
        num_samples,
        ..header
    })
}

#[allow(clippy::too_many_arguments)]
fn process_chunk(
    stream: &mut GcfbStream,
    mono: &[f64],
    write_buf: &mut Vec<f32>,
    writer: &mut BufWriter<File>,
    num_samples: &mut u64,
    progress: &mut impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<(), AnalysisError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(AnalysisError::Cancelled);
    }
    write_buf.clear();
    for &sample in mono {
        let step = stream.process_sample(sample).map_err(message)?;
        match step.event {
            Some(DcgcEvent::Sample { dcgc_out, .. }) => {
                write_buf.extend(dcgc_out.iter().map(|&v| v as f32));
            }
            // Static, Level, and Dynamic sample-base control always produce a
            // per-sample event; this fallback only keeps the one-row-per-
            // sample invariant if the library ever changes that.
            _ => write_buf.extend(step.scgc_smpl.iter().map(|&v| v as f32)),
        }
        *num_samples += 1;
    }
    writer.write_all(cast_slice(write_buf))?;
    progress(*num_samples);
    Ok(())
}

/// Binaural output assembly: the mean of the per-ear dcGC rows is buffered
/// for the short EI latency, then interleaved with the lowest-EI-unit data as
/// `[dcgc_mean | iid | itd | ei]` rows. EI events arrive in sample order,
/// exactly one per input sample, so the queue head always matches the next
/// event.
struct BinauralState {
    pending: VecDeque<Vec<f32>>,
    units: Vec<EiUnit>,
    iid_block: Vec<f32>,
    itd_block: Vec<f32>,
    ei_block: Vec<f32>,
    write_buf: Vec<f32>,
    num_samples: u64,
}

impl BinauralState {
    fn new(chunk_len: usize, num_ch: usize, units: Vec<EiUnit>) -> Self {
        Self {
            pending: VecDeque::new(),
            units,
            iid_block: vec![0.0; num_ch],
            itd_block: vec![0.0; num_ch],
            ei_block: vec![0.0; num_ch],
            write_buf: Vec::with_capacity(chunk_len * 4 * num_ch),
            num_samples: 0,
        }
    }

    fn push_step(&mut self, step: HybridBinauralStreamStep) -> Result<(), AnalysisError> {
        let dcgc_left = dcgc_f32(&step.left_filterbank.event, &step.left_filterbank.scgc_smpl);
        let dcgc_right = dcgc_f32(
            &step.right_filterbank.event,
            &step.right_filterbank.scgc_smpl,
        );
        let dcgc_mean = dcgc_left
            .iter()
            .zip(&dcgc_right)
            .map(|(&l, &r)| (l + r) / 2.0)
            .collect();
        self.pending.push_back(dcgc_mean);
        if let Some(event) = step.ei_event {
            self.emit(event)?;
        }
        Ok(())
    }

    fn emit(&mut self, event: EiStreamSample) -> Result<(), AnalysisError> {
        let dcgc_mean = self.pending.pop_front().ok_or_else(|| {
            AnalysisError::Message("EI event without a matching filterbank sample".into())
        })?;
        debug_assert_eq!(event.sample_index as u64, self.num_samples);
        self.write_buf.extend_from_slice(&dcgc_mean);
        // The EI stage integrates (L − R)², so the unit tuned to the
        // stimulus cancels it: per channel, the lowest-activity unit carries
        // the characteristic IID and ITD (as in the breebaart2001_hybrid
        // example). Only that unit's data is stored.
        let num_ch = dcgc_mean.len();
        let activity = &event.activity;
        for ch in 0..num_ch {
            let (lowest, &activity) = activity
                .column(ch)
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.total_cmp(b.1))
                .expect("EI population is never empty");
            let unit = self.units[lowest];
            self.iid_block[ch] = unit.iid_db as f32;
            self.itd_block[ch] = unit.delay_seconds as f32;
            self.ei_block[ch] = activity as f32;
        }
        self.write_buf.extend_from_slice(&self.iid_block);
        self.write_buf.extend_from_slice(&self.itd_block);
        self.write_buf.extend_from_slice(&self.ei_block);
        self.num_samples += 1;
        Ok(())
    }
}

fn dcgc_f32(event: &Option<DcgcEvent>, scgc_smpl: &ndarray::Array1<f64>) -> Vec<f32> {
    match event {
        Some(DcgcEvent::Sample { dcgc_out, .. }) => dcgc_out.iter().map(|&v| v as f32).collect(),
        // See the mono fallback in process_chunk.
        _ => scgc_smpl.iter().map(|&v| v as f32).collect(),
    }
}

fn run_binaural_inner(
    input: &Path,
    output: &Path,
    params: &BuilderParams,
    progress: &mut impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<AnalysisHeader, AnalysisError> {
    let probe = probe_audio(input)?;
    if probe.channels != 2 {
        return Err(AnalysisError::Message(format!(
            "binaural analysis requires a stereo input file (found {} channels)",
            probe.channels
        )));
    }
    let fs = probe.sample_rate as f64;
    validate_gc(&params.gc, fs)?;
    let bin = &params.binaural;
    if !(bin.tau_max_seconds.is_finite() && bin.tau_max_seconds > 0.0) {
        return Err(AnalysisError::Message(format!(
            "ITD range must be positive and finite (got {:.2e} s)",
            bin.tau_max_seconds
        )));
    }
    if bin.num_tau == 0 || bin.num_iid == 0 {
        return Err(AnalysisError::Message(
            "EI population must contain at least one unit per dimension".into(),
        ));
    }
    if !(bin.iid_max_db.is_finite() && bin.iid_max_db > 0.0) {
        return Err(AnalysisError::Message(format!(
            "IID range must be positive and finite (got {:.2e} dB)",
            bin.iid_max_db
        )));
    }
    let units = bin.units();

    let config = HybridBinauralConfig {
        filterbank: prepared_gc(&params.gc, fs),
        peripheral: bin.peripheral.clone(),
        ei: EiConfig {
            // The streaming EI stage requires causal integration; the
            // population limit follows the configured ITD range.
            integration_boundary: EiIntegrationBoundary::CausalZeroState,
            max_abs_delay_seconds: bin.tau_max_seconds,
            ..bin.ei.clone()
        },
    };
    let mut stream = HybridBinauralStream::new(&units, config).map_err(message)?;

    let num_ch = params.gc.num_ch;
    let header = AnalysisHeader {
        sample_rate: fs,
        num_channels: num_ch as u32,
        num_samples: 0, // patched at the end
        f_range: params.gc.f_range,
        control_mode: control_mode_tag(params.gc.ctrl),
        mode: params.mode.tag(),
        tau_seconds: bin.tau_grid(),
        iid_db: bin.iid_grid(),
        channel_freqs: stream.center_frequencies_hz().to_vec(),
    };

    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(&header.encode())?;

    let decoder = streaming_decoder(input)?;
    let chunk_len = (probe.sample_rate as usize / 4).max(1); // ~0.25 s of audio
    let mut state = BinauralState::new(chunk_len, num_ch, units);

    let mut left_chunk: Vec<f64> = Vec::with_capacity(chunk_len);
    let mut right_chunk: Vec<f64> = Vec::with_capacity(chunk_len);
    let mut is_left = true;
    for sample in decoder {
        if is_left {
            left_chunk.push(f64::from(sample));
        } else {
            right_chunk.push(f64::from(sample));
        }
        is_left = !is_left;
        if left_chunk.len() == chunk_len && right_chunk.len() == chunk_len {
            process_binaural_chunk(
                &mut stream,
                &mut state,
                &left_chunk,
                &right_chunk,
                &mut writer,
                progress,
                cancel,
            )?;
            left_chunk.clear();
            right_chunk.clear();
        }
    }
    // A trailing unpaired sample (odd-length stereo) is dropped.
    let tail_len = left_chunk.len().min(right_chunk.len());
    if tail_len > 0 {
        process_binaural_chunk(
            &mut stream,
            &mut state,
            &left_chunk[..tail_len],
            &right_chunk[..tail_len],
            &mut writer,
            progress,
            cancel,
        )?;
    }
    if stream.samples_processed() == 0 {
        return Err(AnalysisError::Message(
            "no decodable audio samples in input file".into(),
        ));
    }
    // Flush the EI latency tail (zero-extended past the last input sample).
    state.write_buf.clear();
    for event in stream.finish().map_err(message)? {
        state.emit(event)?;
    }
    writer.write_all(cast_slice(&state.write_buf))?;
    if !state.pending.is_empty() {
        return Err(AnalysisError::Message(format!(
            "EI stage emitted no event for {} samples",
            state.pending.len()
        )));
    }

    finalize(writer, state.num_samples)?;
    progress(state.num_samples);
    Ok(AnalysisHeader {
        num_samples: state.num_samples,
        ..header
    })
}

#[allow(clippy::too_many_arguments)]
fn process_binaural_chunk(
    stream: &mut HybridBinauralStream,
    state: &mut BinauralState,
    left: &[f64],
    right: &[f64],
    writer: &mut BufWriter<File>,
    progress: &mut impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<(), AnalysisError> {
    if cancel.load(Ordering::Relaxed) {
        return Err(AnalysisError::Cancelled);
    }
    state.write_buf.clear();
    for (&l, &r) in left.iter().zip(right) {
        let step = stream.process_sample(l, r).map_err(message)?;
        state.push_step(step)?;
    }
    writer.write_all(cast_slice(&state.write_buf))?;
    progress(state.num_samples);
    Ok(())
}

/// Patch the sample count into the header and sync the file to disk.
fn finalize(mut writer: BufWriter<File>, num_samples: u64) -> Result<(), AnalysisError> {
    writer.flush()?;
    writer.seek(SeekFrom::Start(NUM_SAMPLES_OFFSET))?;
    writer.write_all(&num_samples.to_le_bytes())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    Ok(())
}

/// A `.gca` file mapped into memory. The OS pages data in on demand, so the
/// resident set stays small regardless of file size.
pub struct AnalysisReader {
    pub header: AnalysisHeader,
    mmap: Mmap,
}

impl AnalysisReader {
    pub fn open(path: &Path) -> Result<Self, AnalysisError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        let mut fixed = [0_u8; FIXED_HEADER_LEN];
        {
            use std::io::Read;
            let mut reader = &file;
            reader.read_exact(&mut fixed).map_err(|_| {
                AnalysisError::Message("invalid analysis file: truncated header".into())
            })?;
        }
        // Decode needs the full header; read channel frequencies and the EI
        // population grids too. The magic must be checked first: a non-.gca
        // file would otherwise yield a garbage (possibly huge) channel count.
        if &fixed[0..4] != MAGIC {
            return Err(AnalysisError::Message(
                "invalid analysis file: bad magic (not a .gca file)".into(),
            ));
        }
        let num_channels = u32::from_le_bytes(fixed[20..24].try_into().unwrap_or([0; 4])) as usize;
        let num_tau = u32::from_le_bytes(fixed[60..64].try_into().unwrap_or([0; 4])) as usize;
        let num_iid = u32::from_le_bytes(fixed[64..68].try_into().unwrap_or([0; 4])) as usize;
        let header_len = FIXED_HEADER_LEN + 8 * (num_channels + num_tau + num_iid);
        if file_len < header_len {
            return Err(AnalysisError::Message(
                "invalid analysis file: truncated header".into(),
            ));
        }
        // Reads through &File share one file offset, which now sits just past
        // the fixed header, so only the variable header fields remain.
        let mut header_bytes = fixed.to_vec();
        header_bytes.resize(header_len, 0);
        {
            use std::io::Read;
            let mut reader = &file;
            reader
                .read_exact(&mut header_bytes[FIXED_HEADER_LEN..])
                .map_err(|_| {
                    AnalysisError::Message("invalid analysis file: truncated header".into())
                })?;
        }
        let header = AnalysisHeader::decode(&header_bytes)?;
        let expected = header_len + header.num_samples as usize * header.values_per_sample() * 4;
        if file_len < expected {
            return Err(AnalysisError::Message(format!(
                "invalid analysis file: expected at least {expected} bytes, found {file_len}"
            )));
        }
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { header, mmap })
    }

    /// True for a binaural (dcGC mean + EI population) analysis.
    pub fn is_binaural(&self) -> bool {
        self.header.mode == MODE_BINAURAL
    }

    /// The complete sample-major matrix as `num_samples * values_per_sample`
    /// f32. See the module docs for the per-sample layout.
    pub fn data(&self) -> &[f32] {
        let start = self.header.header_len();
        let len = self.header.num_samples as usize * self.header.values_per_sample();
        cast_slice(&self.mmap[start..start + len * 4])
    }

    /// Samples `[start, end)` of a mono analysis as contiguous rows of
    /// `num_channels` floats. Binaural files use the dedicated binaural
    /// accessors instead.
    pub fn rows(&self, start: u64, end: u64) -> &[f32] {
        let num_ch = self.header.num_channels as usize;
        let start = start.min(self.header.num_samples) as usize;
        let end = end.min(self.header.num_samples).max(start as u64) as usize;
        &self.data()[start * num_ch..end * num_ch]
    }

    /// One sample's channel vector of a mono analysis.
    pub fn row(&self, sample: u64) -> &[f32] {
        self.rows(sample, sample + 1)
    }

    /// Per-channel peak absolute amplitude over samples `[start, end)` of a
    /// mono analysis.
    pub fn column_peaks(&self, start: u64, end: u64, out: &mut [f32]) {
        out.fill(0.0);
        let num_ch = self.header.num_channels as usize;
        for row in self.rows(start, end).chunks_exact(num_ch) {
            for (peak, &value) in out.iter_mut().zip(row.iter()) {
                *peak = peak.max(value.abs());
            }
        }
    }

    /// Samples `[start, end)` of a binaural analysis as contiguous
    /// `[dcgc_mean | iid | itd | ei]` rows of `values_per_sample` floats.
    fn binaural_rows(&self, start: u64, end: u64) -> &[f32] {
        debug_assert!(self.is_binaural());
        let stride = self.header.values_per_sample();
        let start = start.min(self.header.num_samples) as usize;
        let end = end.min(self.header.num_samples).max(start as u64) as usize;
        &self.data()[start * stride..end * stride]
    }

    /// Mean of the two ears' dcGC amplitudes, per channel, for one sample of
    /// a binaural analysis.
    pub fn dcgc_row(&self, sample: u64) -> &[f32] {
        let num_ch = self.header.num_channels as usize;
        &self.binaural_rows(sample, sample + 1)[..num_ch]
    }

    /// Characteristic IID in dB of the lowest-activity EI unit, per channel,
    /// for one sample of a binaural analysis.
    pub fn iid_row(&self, sample: u64) -> &[f32] {
        let num_ch = self.header.num_channels as usize;
        &self.binaural_rows(sample, sample + 1)[num_ch..2 * num_ch]
    }

    /// Characteristic ITD in seconds of the lowest-activity EI unit, per
    /// channel, for one sample of a binaural analysis.
    pub fn itd_row(&self, sample: u64) -> &[f32] {
        let num_ch = self.header.num_channels as usize;
        &self.binaural_rows(sample, sample + 1)[2 * num_ch..3 * num_ch]
    }

    /// Activity of the lowest-activity EI unit, per channel, for one sample
    /// of a binaural analysis.
    pub fn ei_row(&self, sample: u64) -> &[f32] {
        let num_ch = self.header.num_channels as usize;
        &self.binaural_rows(sample, sample + 1)[3 * num_ch..]
    }

    /// One-pass aggregation of a binaural column over samples `[start, end)`:
    /// per-channel absolute peaks of the stored dcGC mean plus per-channel
    /// sums of the lowest-activity unit's characteristic IID and ITD
    /// (`iid_sums` and `itd_sums` are `num_channels` long). The renderer
    /// divides the sums by the sample count to get column means.
    pub fn aggregate_binaural_column(
        &self,
        start: u64,
        end: u64,
        peaks: &mut [f32],
        iid_sums: &mut [f32],
        itd_sums: &mut [f32],
    ) {
        peaks.fill(0.0);
        iid_sums.fill(0.0);
        itd_sums.fill(0.0);
        let num_ch = self.header.num_channels as usize;
        let stride = self.header.values_per_sample();
        for row in self.binaural_rows(start, end).chunks_exact(stride) {
            let (dcgc, rest) = row.split_at(num_ch);
            let (iid, rest) = rest.split_at(num_ch);
            let (itd, _) = rest.split_at(num_ch);
            for (peak, &value) in peaks.iter_mut().zip(dcgc.iter()) {
                *peak = peak.max(value.abs());
            }
            for (sum, &value) in iid_sums.iter_mut().zip(iid.iter()) {
                *sum += value;
            }
            for (sum, &value) in itd_sums.iter_mut().zip(itd.iter()) {
                *sum += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn header_roundtrip() {
        let mono = AnalysisHeader {
            sample_rate: 44_100.0,
            num_channels: 3,
            num_samples: 123_456,
            f_range: [40.0, 16_000.0],
            control_mode: 1,
            mode: MODE_MONO,
            tau_seconds: Vec::new(),
            iid_db: Vec::new(),
            channel_freqs: vec![40.0, 1000.0, 16_000.0],
        };
        assert_eq!(AnalysisHeader::decode(&mono.encode()).unwrap(), mono);

        let binaural = AnalysisHeader {
            mode: MODE_BINAURAL,
            tau_seconds: vec![-1e-3, -0.5e-3, 0.0, 0.5e-3, 1e-3],
            iid_db: vec![-10.0, 0.0, 10.0],
            ..mono.clone()
        };
        let decoded = AnalysisHeader::decode(&binaural.encode()).unwrap();
        assert_eq!(decoded, binaural);
        assert_eq!(decoded.values_per_sample(), 4 * 3);
    }

    #[test]
    fn header_rejects_garbage() {
        assert!(AnalysisHeader::decode(b"not a gca file").is_err());
        let mono = AnalysisHeader {
            sample_rate: 48_000.0,
            num_channels: 2,
            num_samples: 0,
            f_range: [40.0, 16_000.0],
            control_mode: 0,
            mode: MODE_MONO,
            tau_seconds: Vec::new(),
            iid_db: Vec::new(),
            channel_freqs: vec![100.0, 200.0],
        };
        let mut bytes = mono.encode();
        bytes[0] = b'X';
        assert!(AnalysisHeader::decode(&bytes).is_err());
        // A mono header with a nonzero EI population is inconsistent.
        let mut bytes = mono.encode();
        bytes[60..64].copy_from_slice(&1_u32.to_le_bytes());
        assert!(AnalysisHeader::decode(&bytes).is_err());
        let mut bytes = mono.encode();
        bytes[64..68].copy_from_slice(&1_u32.to_le_bytes());
        assert!(AnalysisHeader::decode(&bytes).is_err());
    }

    /// Write a 16-bit PCM WAV. `data` is interleaved little-endian i16.
    fn write_wav(path: &Path, sample_rate: u32, channels: u16, data: &[u8]) {
        let data_len = data.len() as u32;
        let block_align = channels * 2;
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
        wav.extend_from_slice(&block_align.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(data);
        fs::write(path, wav).unwrap();
    }

    fn tone_sample(t: f64, gain: f64) -> i16 {
        (gain * 0.5 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 32_767.0) as i16
    }

    /// Write a 0.25 s mono 44.1 kHz WAV containing a 1 kHz tone.
    fn write_mono_wav(path: &Path) -> u64 {
        let sample_rate = 44_100_u32;
        let num_samples = sample_rate as usize / 4;
        let mut data = Vec::with_capacity(num_samples * 2);
        for n in 0..num_samples {
            let t = n as f64 / f64::from(sample_rate);
            data.extend_from_slice(&tone_sample(t, 1.0).to_le_bytes());
        }
        write_wav(path, sample_rate, 1, &data);
        num_samples as u64
    }

    /// Write a 0.25 s stereo 44.1 kHz 16-bit PCM WAV containing a 1 kHz tone
    /// identical in both ears. Returns the number of per-channel samples.
    fn write_test_wav(path: &Path) -> u64 {
        let sample_rate = 44_100_u32;
        let num_samples = sample_rate as usize / 4;
        let mut data = Vec::with_capacity(num_samples * 4);
        for n in 0..num_samples {
            let t = n as f64 / f64::from(sample_rate);
            let value = tone_sample(t, 1.0);
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
        }
        write_wav(path, sample_rate, 2, &data);
        num_samples as u64
    }

    /// Write a 0.25 s stereo deterministic-noise WAV whose left ear is
    /// attenuated by `iid_db` relative to the full-scale right ear. Broadband
    /// noise at this level keeps the interaural level difference visible
    /// through the adaptation loops, as in the breebaart2001_hybrid example.
    fn write_binaural_noise_wav(path: &Path, sample_rate: u32, iid_db: f64) -> u64 {
        let num_samples = sample_rate as usize / 4;
        let left_gain = 10_f64.powf(-iid_db / 20.0);
        let mut state = 0x6a09_e667_f3bc_c909_u64;
        let mut data = Vec::with_capacity(num_samples * 4);
        for _ in 0..num_samples {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
            let uniform = 2.0 * bits as f64 / ((1_u64 << 53) - 1) as f64 - 1.0;
            let left = (left_gain * uniform * 32_767.0) as i16;
            let right = (uniform * 32_767.0) as i16;
            data.extend_from_slice(&left.to_le_bytes());
            data.extend_from_slice(&right.to_le_bytes());
        }
        write_wav(path, sample_rate, 2, &data);
        num_samples as u64
    }
    /// Write a 0.25 s stereo tone whose right ear is delayed by
    /// `right_delay_samples` and scaled by `right_gain` relative to the left.
    ///
    /// The tone is 500 Hz, comfortably below the model's 770 Hz inner-hair-
    /// cell lowpass, so the peripheral representation retains the temporal
    /// fine structure that carries ITD information.
    fn write_binaural_wav(path: &Path, right_delay_samples: usize, right_gain: f64) -> u64 {
        let sample_rate = 44_100_u32;
        let num_samples = sample_rate as usize / 4;
        let mut data = Vec::with_capacity(num_samples * 4);
        for n in 0..num_samples {
            let t = n as f64 / f64::from(sample_rate);
            let t_right = (n as f64 - right_delay_samples as f64) / f64::from(sample_rate);
            let left = (0.5 * (2.0 * std::f64::consts::PI * 500.0 * t).sin() * 32_767.0) as i16;
            let right = (right_gain
                * 0.5
                * (2.0 * std::f64::consts::PI * 500.0 * t_right).sin()
                * 32_767.0) as i16;
            data.extend_from_slice(&left.to_le_bytes());
            data.extend_from_slice(&right.to_le_bytes());
        }
        write_wav(path, sample_rate, 2, &data);
        num_samples as u64
    }

    fn temp_paths(name: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("pav_{name}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        (dir.join("tone.wav"), dir.join("tone.gca"))
    }

    /// Deterministic, noise-free binaural parameters with static control.
    /// The 50 dB gain reference keeps the peripheral near-linear (as in the
    /// breebaart2001_hybrid example), so interaural level differences survive
    /// the adaptation loops.
    fn binaural_test_params() -> BuilderParams {
        use gammachirp_rs::gcfb_v234::GainReference;
        BuilderParams {
            gc: GcParam {
                num_ch: 32,
                f_range: [40.0, 16_000.0],
                ctrl: ControlMode::Static,
                gain_ref: GainReference::Db(50.0),
                lvl_est: LvlEst {
                    rms2spldb: 30.0,
                    ..BuilderParams::default().gc.lvl_est
                },
                ..BuilderParams::default().gc
            },
            mode: AnalysisMode::Binaural,
            binaural: BinauralParams {
                tau_max_seconds: 1e-3,
                num_tau: 9,
                iid_max_db: 10.0,
                num_iid: 5,
                peripheral: PeripheralConfig {
                    absolute_threshold_noise_level_db_spl: None,
                    ..BinauralParams::default().peripheral
                },
                ei: EiConfig {
                    internal_noise_std_mu: 0.0,
                    ..EiConfig::streaming()
                },
            },
        }
    }

    /// Channel index whose center frequency is nearest `freq` Hz.
    fn channel_near(header: &AnalysisHeader, freq: f64) -> usize {
        header
            .channel_freqs
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - freq).abs().partial_cmp(&(b.1 - freq).abs()).unwrap())
            .unwrap()
            .0
    }

    #[test]
    fn end_to_end_tone_analysis() {
        let (wav, gca) = temp_paths("e2e");
        let expected_samples = write_test_wav(&wav);

        let params = BuilderParams {
            gc: GcParam {
                num_ch: 32,
                f_range: [40.0, 16_000.0],
                ..BuilderParams::default().gc
            },
            ..BuilderParams::default()
        };
        let cancel = AtomicBool::new(false);
        let header = run_analysis(&wav, &gca, &params, |_| {}, &cancel).unwrap();
        assert_eq!(header.num_samples, expected_samples);
        assert_eq!(header.sample_rate, 44_100.0);
        assert_eq!(header.num_channels, 32);
        assert_eq!(header.control_mode, control_mode_tag(ControlMode::Dynamic));
        assert_eq!(header.mode, MODE_MONO);
        assert_eq!(header.channel_freqs.len(), 32);

        let reader = AnalysisReader::open(&gca).unwrap();
        assert_eq!(reader.header, header);
        assert!(!reader.is_binaural());
        assert_eq!(reader.data().len(), expected_samples as usize * 32);
        assert_eq!(reader.row(0).len(), 32);
        assert_eq!(reader.rows(10, 20).len(), 10 * 32);

        // Mean |dcgc| per channel; the strongest channel must sit near 1 kHz.
        let mut energy = [0.0_f32; 32];
        for row in reader.data().chunks_exact(32) {
            for (sum, &value) in energy.iter_mut().zip(row.iter()) {
                *sum += value.abs();
            }
        }
        let (best, _) = energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let best_freq = reader.header.channel_freqs[best];
        assert!(
            (600.0..=1700.0).contains(&best_freq),
            "peak channel {best} at {best_freq:.0} Hz, expected near 1000 Hz"
        );

        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    #[test]
    fn cancel_deletes_partial_file() {
        let (wav, gca) = temp_paths("cancel");
        write_test_wav(&wav);
        let cancel = AtomicBool::new(true); // cancelled before the first chunk
        let result = run_analysis(&wav, &gca, &BuilderParams::default(), |_| {}, &cancel);
        assert!(matches!(result, Err(AnalysisError::Cancelled)));
        assert!(!gca.exists());
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    #[test]
    fn rejects_frequency_above_nyquist() {
        let (wav, gca) = temp_paths("nyquist");
        write_test_wav(&wav);
        let params = BuilderParams {
            gc: GcParam {
                f_range: [40.0, 23_000.0], // 44.1 kHz file → Nyquist 22.05 kHz
                ..BuilderParams::default().gc
            },
            ..BuilderParams::default()
        };
        let cancel = AtomicBool::new(false);
        let result = run_analysis(&wav, &gca, &params, |_| {}, &cancel);
        assert!(matches!(result, Err(AnalysisError::Message(_))));
        assert!(!gca.exists());
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    /// Customized `GcParam` items must flow through the builder unchanged
    /// (except `fs` and `dyn_hpaf.str_prc`, which the builder forces).
    #[test]
    fn end_to_end_customized_gc_param() {
        use gammachirp_rs::gcfb_v234::GainReference;

        let (wav, gca) = temp_paths("custom");
        let expected_samples = write_test_wav(&wav);

        let params = BuilderParams {
            gc: GcParam {
                num_ch: 16,
                out_mid_crct: "ELC".into(),
                n: 3.5,
                b1: [1.5, 0.1],
                ctrl: ControlMode::Static,
                level_db_scgcfb: 40.0,
                gain_ref: GainReference::Db(50.0),
                num_update_asym_cmp: 4,
                hloss_type: "HL3".into(),
                hloss_compression_health: Some(0.7),
                ..BuilderParams::default().gc
            },
            ..BuilderParams::default()
        };
        let cancel = AtomicBool::new(false);
        let header = run_analysis(&wav, &gca, &params, |_| {}, &cancel).unwrap();
        assert_eq!(header.num_samples, expected_samples);
        assert_eq!(header.num_channels, 16);
        assert_eq!(header.control_mode, control_mode_tag(ControlMode::Static));
        assert_eq!(header.channel_freqs.len(), 16);

        let reader = AnalysisReader::open(&gca).unwrap();
        assert_eq!(reader.header, header);
        assert_eq!(reader.data().len(), expected_samples as usize * 16);
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    #[test]
    fn binaural_rejects_mono_input() {
        let (wav, gca) = temp_paths("bin_mono");
        write_mono_wav(&wav);
        let cancel = AtomicBool::new(false);
        let result = run_analysis(&wav, &gca, &binaural_test_params(), |_| {}, &cancel);
        assert!(matches!(result, Err(AnalysisError::Message(_))));
        assert!(!gca.exists());
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    #[test]
    fn binaural_cancel_deletes_partial_file() {
        let (wav, gca) = temp_paths("bin_cancel");
        write_binaural_wav(&wav, 0, 1.0);
        let cancel = AtomicBool::new(true); // cancelled before the first chunk
        let result = run_analysis(&wav, &gca, &binaural_test_params(), |_| {}, &cancel);
        assert!(matches!(result, Err(AnalysisError::Cancelled)));
        assert!(!gca.exists());
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    /// A right-ear delay must show up as the characteristic ITD of the
    /// lowest-activity EI unit: the EI stage computes (L − R)², so the unit
    /// tuned to the stimulus ITD cancels.
    #[test]
    fn binaural_itd_tracks_right_ear_delay() {
        let (wav, gca) = temp_paths("bin_itd");
        let delay_samples = 22_usize; // ≈ 0.499 ms at 44.1 kHz
        let expected_samples = write_binaural_wav(&wav, delay_samples, 1.0);

        let params = binaural_test_params();
        let cancel = AtomicBool::new(false);
        let header = run_analysis(&wav, &gca, &params, |_| {}, &cancel).unwrap();
        assert_eq!(header.num_samples, expected_samples);
        assert_eq!(header.mode, MODE_BINAURAL);
        assert_eq!(header.tau_seconds.len(), 9);
        assert_eq!(header.iid_db.len(), 5);
        assert_eq!(header.values_per_sample(), 4 * 32);

        let reader = AnalysisReader::open(&gca).unwrap();
        assert!(reader.is_binaural());
        assert_eq!(reader.header, header);
        assert_eq!(reader.dcgc_row(0).len(), 32);
        assert_eq!(reader.iid_row(0).len(), 32);
        assert_eq!(reader.itd_row(0).len(), 32);
        assert_eq!(reader.ei_row(0).len(), 32);

        let num_ch = 32_usize;
        let mut peaks = vec![0.0_f32; num_ch];
        let mut iid_sums = vec![0.0_f32; num_ch];
        let mut itd_sums = vec![0.0_f32; num_ch];
        reader.aggregate_binaural_column(
            0,
            header.num_samples,
            &mut peaks,
            &mut iid_sums,
            &mut itd_sums,
        );
        assert!(peaks.iter().all(|&p| p > 0.0));

        let channel = channel_near(&header, 500.0);
        let itd = itd_sums[channel] as f64 / header.num_samples as f64;
        // Paper-symmetric convention: a right-ear delay of d cancels at
        // characteristic ITD τ = −d.
        let expected_itd = -(delay_samples as f64 / 44_100.0);
        assert!(
            (itd - expected_itd).abs() <= 0.26e-3,
            "dominant ITD {itd:.6} s, expected near {expected_itd:.6} s"
        );
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    /// A right ear louder by 3 dB must show up as the characteristic IID of
    /// the lowest-activity EI unit, snapping to the nearest IID grid point
    /// (+5 dB on the ±10 dB, 5-unit test grid). Like the breebaart2001_hybrid
    /// example, this uses a mid-range filterbank; judgment happens at the
    /// channel with the deepest cancellation (see below).
    #[test]
    fn binaural_iid_reflects_level_ratio() {
        let (wav, gca) = temp_paths("bin_iid");
        write_binaural_noise_wav(&wav, 16_000, 3.0);

        let params = BuilderParams {
            gc: GcParam {
                num_ch: 24,
                f_range: [100.0, 6_000.0],
                out_mid_crct: "No".into(),
                ..binaural_test_params().gc
            },
            ..binaural_test_params()
        };
        let cancel = AtomicBool::new(false);
        let header = run_analysis(&wav, &gca, &params, |_| {}, &cancel).unwrap();

        let reader = AnalysisReader::open(&gca).unwrap();
        let num_ch = 24_usize;
        let mut peaks = vec![0.0_f32; num_ch];
        let mut iid_sums = vec![0.0_f32; num_ch];
        let mut itd_sums = vec![0.0_f32; num_ch];
        reader.aggregate_binaural_column(
            0,
            header.num_samples,
            &mut peaks,
            &mut iid_sums,
            &mut itd_sums,
        );

        let count = header.num_samples as f64;
        // Per-channel mean characteristic IID: at the near-threshold
        // operating point the peripheral compresses the IID away in the
        // highest channels and expands it in the quietest ones, so judge the
        // median across channels (the breebaart2001_hybrid example likewise
        // aggregates across frequency).
        let mut iid_means: Vec<f64> = iid_sums.iter().map(|&v| v as f64 / count).collect();
        iid_means.sort_by(f64::total_cmp);
        let iid_db = iid_means[num_ch / 2];
        assert!(
            (iid_db - 5.0).abs() < 1.0,
            "characteristic IID {iid_db:.2} dB, expected near +5 dB (grid point next to +3 dB)"
        );

        // Zero-delay input: the cancellation trough sits at τ = 0.
        let mut itd_means: Vec<f64> = itd_sums.iter().map(|&v| v as f64 / count).collect();
        itd_means.sort_by(f64::total_cmp);
        let itd = itd_means[num_ch / 2];
        assert!(
            itd.abs() <= 0.26e-3,
            "dominant ITD {itd:.6} s, expected near 0 s"
        );
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }

    /// Binaural mode also works with the default dynamic control (the hybrid
    /// forces sample-base processing internally).
    #[test]
    fn binaural_dynamic_smoke() {
        let (wav, gca) = temp_paths("bin_dyn");
        let expected_samples = write_binaural_wav(&wav, 0, 1.0);

        let params = BuilderParams {
            mode: AnalysisMode::Binaural,
            binaural: BinauralParams {
                peripheral: PeripheralConfig {
                    absolute_threshold_noise_level_db_spl: None,
                    ..PeripheralConfig::default()
                },
                ei: EiConfig {
                    internal_noise_std_mu: 0.0,
                    ..EiConfig::streaming()
                },
                ..BinauralParams::default()
            },
            ..BuilderParams::default()
        };
        let cancel = AtomicBool::new(false);
        let header = run_analysis(&wav, &gca, &params, |_| {}, &cancel).unwrap();
        assert_eq!(header.num_samples, expected_samples);
        assert_eq!(header.control_mode, control_mode_tag(ControlMode::Dynamic));

        let reader = AnalysisReader::open(&gca).unwrap();
        assert_eq!(reader.data().len(), expected_samples as usize * 4 * 100);
        let _ = fs::remove_dir_all(wav.parent().unwrap());
    }
}
