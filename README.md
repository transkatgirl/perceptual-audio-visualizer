# perceptual-audio-visualizer

An [egui](https://github.com/emilk/egui) front-end for
[gammachirp-rs](gammachirp/) (GCFB v2.34) with two tools:

- **Analysis builder** — offline *per-sample* analysis of an audio file
  (wav/flac/mp3/ogg/m4a). The file is decoded, downmixed to mono, and streamed
  through a dynamic compressive gammachirp filterbank one sample at a time
  (default range 40 Hz – 16 kHz, 100 ERB-spaced channels, dynamic control).
  The resulting dcGC output is written straight to disk as a `.gca` file, so
  inputs larger than RAM are fine.
- **Analysis viewer** — plots a saved `.gca` as a scrolling
  spectrogram (time × auditory channel, dB color scale) in sync with playback
  of the source audio. Supports play/pause (Space), seeking (←/→, Shift for
  5 s steps, click/drag on the plot or the seek bar), panning (scroll wheel),
  and zooming (pinch / Ctrl-wheel / ± buttons). The analysis is memory-mapped
  and only the visible window is read, so analyses larger than RAM are fine.

## Run

```bash
cargo run --release
```

Typical flow: in **Analysis builder**, pick an audio file and press
*Run analysis* (writes `<name>.gca` next to it, with progress and
cancellation). Then switch to **Analysis viewer**, pick the same audio file —
the sibling `.gca` is picked up automatically — press *Load*, and hit *Play*.

The max frequency must be below half the audio file's sample rate, so the
default 16 kHz ceiling requires a ≥ 32.1 kHz file. Dynamic control is
several × slower than realtime; Static and Level are cheaper.

## `.gca` format

Little-endian, self-describing header (magic `GCA1`, version, sample rate,
channel count, sample count, frequency range, control mode, per-channel center
frequencies in Hz), followed by the analysis as sample-major `f32` rows
(`num_samples × num_channels` dcGC values). A time window is therefore one
contiguous byte range, which is what makes cheap memory-mapped viewing
possible.

## Development

```bash
cargo test                 # format roundtrip + end-to-end tone analysis
cargo build --release
```
