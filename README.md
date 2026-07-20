# perceptual-audio-visualizer

An [egui](https://github.com/emilk/egui) front-end for
[gammachirp-rs](gammachirp/) (GCFB v2.34) with two tools:

- **Analysis builder** — offline *per-sample* analysis of an audio file
  (wav/flac/mp3/ogg/m4a) in one of two modes:
  - **Mono downmix** — the file is decoded, downmixed to mono, and streamed
    through a dynamic compressive gammachirp filterbank one sample at a time
    (default range 40 Hz – 16 kHz, 100 ERB-spaced channels, dynamic control).
  - **Binaural (Breebaart 2001)** — a stereo file is streamed through the
    gammachirp/Breebaart hybrid: one GCFB per ear, inner-hair-cell lowpass and
    adaptation loops, and an excitation-inhibition (EI) population tuned to a
    configurable range of characteristic ITDs (default 9 units over ±1 ms).
    The per-ear dcGC outputs and the EI activity are all stored.

  Every `GcParam` item is customizable in collapsible sections: gammachirp
  filter coefficients, gain/level references, level estimation, outer/middle-ear
  correction, and hearing-loss characteristics. Binaural mode additionally
  exposes the EI population (ITD range and unit count), the EI stage
  (integration, compression, delay convention, internal noise), and the
  peripheral stage (IHC cutoff, adaptation time constants, level calibration,
  overshoot limit, absolute-threshold noise). The resulting output is written
  straight to disk as a `.gca` file, so inputs larger than RAM are fine.
- **Analysis viewer** — plots a saved `.gca` as a scrolling
  spectrogram (time × auditory channel) in sync with playback of the source
  audio. Supports play/pause (Space), seeking (←/→, Shift for 5 s steps,
  click/drag on the plot or the seek bar), panning (scroll wheel), and
  zooming (pinch / Ctrl-wheel / ± buttons). The analysis is memory-mapped
  and only the visible window is read, so analyses larger than RAM are fine.

  Mono files use a dB-scaled magma color map. Binaural files use a bivariate
  color map computed in the OkLCH color space (cached in a 256×256 lookup
  table): the mono-downmixed amplitude drives lightness, and a stereo
  variable drives hue as a blue↔orange diverging scale — toggleable between
  **IID** (per-ear level difference, with an adjustable ±dB range) and
  **ITD** (the characteristic delay of the best-cancelling EI unit — the EI
  stage computes (L − R)², so the stimulus ITD appears as a trough).

## Run

```bash
cargo run --release
```

Typical flow: in **Analysis builder**, pick an audio file and press
*Run analysis* (writes `<name>.gca` next to it, with progress and
cancellation). Then switch to **Analysis viewer**, pick the same audio file —
the sibling `.gca` is picked up automatically — press *Load*, and hit *Play*.

The max frequency must be below half the audio file's sample rate, so the
default 16 kHz ceiling requires a ≥ 32.1 kHz file. Binaural mode requires a
stereo file. Dynamic control is several × slower than realtime; Static and
Level are cheaper. Binaural files are `(2 + num_tau)` × larger than mono
ones (11× at the default 9 EI units).

## `.gca` format

Little-endian, self-describing header (magic `GCA1`, version 1, sample rate,
channel count, sample count, frequency range, control mode, analysis mode,
per-channel center frequencies in Hz, and — for binaural analyses — the EI
population's characteristic ITDs in seconds), followed by the analysis as
sample-major `f32` rows:

- **mono**: `num_samples × num_channels` dcGC values;
- **binaural**: per sample `[left dcGC | right dcGC | EI activity]`, i.e.
  `num_samples × (2 + num_tau) × num_channels` values with the EI activity
  stored unit-major.

A time window is therefore one contiguous byte range in either mode, which is
what makes cheap memory-mapped viewing possible.

## Development

```bash
cargo test                 # format roundtrip + end-to-end mono/binaural analyses + colormap
cargo build --release
```
