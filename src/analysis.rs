//! Offline per-sample GCFB v2.34 analysis: binary file format, builder core,
//! and a memory-mapped reader. This module contains no GUI code so it can be
//! unit-tested headlessly.
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
//! 52  u32  header_len = 64 + 8*num_channels
//! 56  ..64 reserved (zeros)
//! 64  f64 × num_channels   channel center frequencies (gc_resp.fr1)
//! header_len..  f32 × num_samples × num_channels, SAMPLE-MAJOR
//!               (sample n, channel c at header_len + (n*num_ch + c)*4)
//! ```

use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use bytemuck::cast_slice;
use gammachirp_rs::gcfb_v234::{ControlMode, DcgcEvent, DynHpaf, GcParam, GcfbStream};
use memmap2::Mmap;
use rodio::{Decoder, Source};

const MAGIC: &[u8; 4] = b"GCA1";
const VERSION: u32 = 1;
const ENGINE_GCFB_V234_SAMPLE: u32 = 1;
const FIXED_HEADER_LEN: usize = 64;
const NUM_SAMPLES_OFFSET: u64 = 24;

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

/// Parameters for one builder run: a complete [`GcParam`] template.
///
/// Every user-facing GcParam item is customizable. The exceptions are managed
/// by the builder itself: `fs` is forced to the input file's sample rate and
/// `dyn_hpaf.str_prc` is forced to `"sample-base"` (that is what makes this a
/// per-sample analysis), while `hloss`, `fr1`, and the derived `lvl_est`
/// fields are computed by the library's `set_param`.
#[derive(Clone, Debug)]
pub struct BuilderParams {
    pub gc: GcParam,
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
                ..GcParam::default()
            },
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
    /// Channel center frequencies in Hz (`num_channels` entries).
    pub channel_freqs: Vec<f64>,
}

impl AnalysisHeader {
    pub fn header_len(&self) -> usize {
        FIXED_HEADER_LEN + 8 * self.channel_freqs.len()
    }

    pub fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
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
        for (index, freq) in self.channel_freqs.iter().enumerate() {
            let start = FIXED_HEADER_LEN + 8 * index;
            bytes[start..start + 8].copy_from_slice(&freq.to_le_bytes());
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
            return Err(invalid(&format!("unsupported version {version}")));
        }
        let engine = read_u32(8);
        if engine != ENGINE_GCFB_V234_SAMPLE {
            return Err(invalid(&format!("unsupported engine {engine}")));
        }
        let num_channels = read_u32(20);
        let header_len = read_u32(52) as usize;
        if num_channels == 0 || header_len != FIXED_HEADER_LEN + 8 * num_channels as usize {
            return Err(invalid("inconsistent channel count / header length"));
        }
        if bytes.len() < header_len {
            return Err(invalid("file shorter than full header"));
        }
        let mut channel_freqs = Vec::with_capacity(num_channels as usize);
        for index in 0..num_channels as usize {
            channel_freqs.push(read_f64(FIXED_HEADER_LEN + 8 * index));
        }
        Ok(Self {
            sample_rate: read_f64(12),
            num_channels,
            num_samples: read_u64(24),
            f_range: [read_f64(32), read_f64(40)],
            control_mode: read_u32(48),
            channel_freqs,
        })
    }
}

/// Run an offline per-sample GCFB v2.34 analysis of `input` and stream the
/// result to `output`.
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
    let result = run_analysis_inner(input, output, params, &mut progress, cancel);
    if result.is_err() {
        // Never leave a partial or corrupt file behind.
        let _ = fs::remove_file(output);
    }
    result
}

fn run_analysis_inner(
    input: &Path,
    output: &Path,
    params: &BuilderParams,
    progress: &mut impl FnMut(u64),
    cancel: &AtomicBool,
) -> Result<AnalysisHeader, AnalysisError> {
    let probe = probe_audio(input)?;
    let fs = probe.sample_rate as f64;
    if params.gc.num_ch < 2 {
        return Err(AnalysisError::Message(
            "channel count must be at least 2".into(),
        ));
    }
    if !(params.gc.f_range[0] > 0.0 && params.gc.f_range[0] < params.gc.f_range[1]) {
        return Err(AnalysisError::Message(format!(
            "invalid frequency range [{:.0}, {:.0}] Hz",
            params.gc.f_range[0], params.gc.f_range[1]
        )));
    }
    if params.gc.f_range[1] >= fs / 2.0 {
        return Err(AnalysisError::Message(format!(
            "max frequency {:.0} Hz must be below the Nyquist limit {:.0} Hz \
             (file sample rate {:.0} Hz)",
            params.gc.f_range[1],
            fs / 2.0,
            fs
        )));
    }

    let gc_param = GcParam {
        fs,
        // Sample-base control is what makes this a per-sample analysis;
        // frame-base emits delayed frame-rate events instead.
        dyn_hpaf: DynHpaf {
            str_prc: "sample-base".into(),
            ..params.gc.dyn_hpaf.clone()
        },
        ..params.gc.clone()
    };
    let mut stream = GcfbStream::new(gc_param).map_err(message)?;

    let header = AnalysisHeader {
        sample_rate: fs,
        num_channels: params.gc.num_ch as u32,
        num_samples: 0, // patched at the end
        f_range: params.gc.f_range,
        control_mode: control_mode_tag(params.gc.ctrl),
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

    writer.flush()?;
    writer.seek(SeekFrom::Start(NUM_SAMPLES_OFFSET))?;
    writer.write_all(&num_samples.to_le_bytes())?;
    writer.flush()?;
    writer.get_ref().sync_all()?;

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
        // Decode needs the full header; read channel frequencies too. The
        // magic must be checked first: a non-.gca file would otherwise yield
        // a garbage (possibly huge) channel count.
        if &fixed[0..4] != MAGIC {
            return Err(AnalysisError::Message(
                "invalid analysis file: bad magic (not a .gca file)".into(),
            ));
        }
        let num_channels = u32::from_le_bytes(fixed[20..24].try_into().unwrap_or([0; 4])) as usize;
        let header_len = FIXED_HEADER_LEN + 8 * num_channels;
        if file_len < header_len {
            return Err(AnalysisError::Message(
                "invalid analysis file: truncated header".into(),
            ));
        }
        // Reads through &File share one file offset, which now sits just past
        // the fixed header, so only the channel frequencies remain to be read.
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
        let expected = header_len + header.num_samples as usize * header.num_channels as usize * 4;
        if file_len < expected {
            return Err(AnalysisError::Message(format!(
                "invalid analysis file: expected at least {expected} bytes, found {file_len}"
            )));
        }
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self { header, mmap })
    }

    /// The complete sample-major matrix as `num_samples * num_channels` f32.
    pub fn data(&self) -> &[f32] {
        let start = self.header.header_len();
        let len = self.header.num_samples as usize * self.header.num_channels as usize;
        cast_slice(&self.mmap[start..start + len * 4])
    }

    /// Samples `[start, end)` as contiguous rows of `num_channels` floats.
    pub fn rows(&self, start: u64, end: u64) -> &[f32] {
        let num_ch = self.header.num_channels as usize;
        let start = start.min(self.header.num_samples) as usize;
        let end = end.min(self.header.num_samples).max(start as u64) as usize;
        &self.data()[start * num_ch..end * num_ch]
    }

    /// One sample's channel vector.
    pub fn row(&self, sample: u64) -> &[f32] {
        self.rows(sample, sample + 1)
    }

    /// Per-channel peak absolute amplitude over samples `[start, end)`.
    pub fn column_peaks(&self, start: u64, end: u64, out: &mut [f32]) {
        out.fill(0.0);
        let num_ch = self.header.num_channels as usize;
        for row in self.rows(start, end).chunks_exact(num_ch) {
            for (peak, &value) in out.iter_mut().zip(row.iter()) {
                *peak = peak.max(value.abs());
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
        let header = AnalysisHeader {
            sample_rate: 44_100.0,
            num_channels: 3,
            num_samples: 123_456,
            f_range: [40.0, 16_000.0],
            control_mode: 1,
            channel_freqs: vec![40.0, 1000.0, 16_000.0],
        };
        let decoded = AnalysisHeader::decode(&header.encode()).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn header_rejects_garbage() {
        assert!(AnalysisHeader::decode(b"not a gca file").is_err());
        let mut bytes = AnalysisHeader {
            sample_rate: 48_000.0,
            num_channels: 2,
            num_samples: 0,
            f_range: [40.0, 16_000.0],
            control_mode: 0,
            channel_freqs: vec![100.0, 200.0],
        }
        .encode();
        bytes[0] = b'X';
        assert!(AnalysisHeader::decode(&bytes).is_err());
    }

    /// Write a 0.25 s stereo 44.1 kHz 16-bit PCM WAV containing a 1 kHz tone.
    /// Returns the number of per-channel samples.
    fn write_test_wav(path: &Path) -> u64 {
        let sample_rate = 44_100_u32;
        let num_samples = sample_rate as usize / 4;
        let mut data = Vec::with_capacity(num_samples * 4);
        for n in 0..num_samples {
            let t = n as f64 / f64::from(sample_rate);
            let value = (0.5 * (2.0 * std::f64::consts::PI * 1000.0 * t).sin() * 32_767.0) as i16;
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
        }
        let data_len = data.len() as u32;
        let mut wav = Vec::with_capacity(44 + data.len());
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVE");
        wav.extend_from_slice(b"fmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes()); // PCM
        wav.extend_from_slice(&2_u16.to_le_bytes()); // stereo
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * 4).to_le_bytes()); // byte rate
        wav.extend_from_slice(&4_u16.to_le_bytes()); // block align
        wav.extend_from_slice(&16_u16.to_le_bytes()); // bits per sample
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        wav.extend_from_slice(&data);
        fs::write(path, wav).unwrap();
        num_samples as u64
    }

    fn temp_paths(name: &str) -> (PathBuf, PathBuf) {
        let dir = std::env::temp_dir().join(format!("pav_{name}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        (dir.join("tone.wav"), dir.join("tone.gca"))
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
        };
        let cancel = AtomicBool::new(false);
        let header = run_analysis(&wav, &gca, &params, |_| {}, &cancel).unwrap();
        assert_eq!(header.num_samples, expected_samples);
        assert_eq!(header.sample_rate, 44_100.0);
        assert_eq!(header.num_channels, 32);
        assert_eq!(header.control_mode, control_mode_tag(ControlMode::Dynamic));
        assert_eq!(header.channel_freqs.len(), 32);

        let reader = AnalysisReader::open(&gca).unwrap();
        assert_eq!(reader.header, header);
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
}
