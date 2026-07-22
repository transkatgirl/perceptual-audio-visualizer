//! Bivariate color map for binaural spectrograms, computed in the OkLCH
//! color space and cached in a lookup table.
//!
//! The first variable (amplitude, `t`) drives lightness; the second
//! (interaural difference, `s`, with 0.5 = zero) drives hue as a blue↔orange
//! diverging scale around an achromatic center. Quiet or centered content is
//! dark gray; lateralized content is tinted blue (negative IID/ITD) or orange
//! (positive IID/ITD).

use eframe::egui;

/// Side length of the square lookup table.
pub const LUT_SIZE: usize = 256;

/// Convert OkLCH to clipped sRGB (each component in 0..=1).
///
/// OkLCH→OkLab→linear sRGB matrices are Björn Ottosson's
/// (<https://bottosson.github.io/posts/oklab/>); out-of-gamut results are
/// simply clamped, which is fine for a display LUT.
pub fn oklch_to_srgb(l: f32, c: f32, h_degrees: f32) -> [f32; 3] {
    let h = h_degrees.to_radians();
    let a = c * h.cos();
    let b = c * h.sin();

    let l_ = l + 0.396_337_8 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let l3 = l_ * l_ * l_;
    let m3 = m_ * m_ * m_;
    let s3 = s_ * s_ * s_;

    let red = 4.076_741_7 * l3 - 3.307_711_6 * m3 + 0.230_969_94 * s3;
    let green = -1.268_438 * l3 + 2.609_757_4 * m3 - 0.341_319_38 * s3;
    let blue = -0.004_196_086_3 * l3 - 0.703_418_6 * m3 + 1.707_614_7 * s3;

    [red, green, blue].map(|linear| {
        let srgb = if linear <= 0.003_130_8 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        };
        srgb.clamp(0.0, 1.0)
    })
}

/// The bivariate color at normalized amplitude `t` ∈ [0, 1] and normalized
/// interaural variable `s` ∈ [0, 1] (0.5 = no interaural difference).
pub fn bivariate_color(t: f32, s: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let s = s.clamp(0.0, 1.0);
    let lightness = 0.12 + 0.78 * t;
    // Chroma grows with both amplitude and distance from the neutral center.
    let chroma = 0.14 * t * (2.0 * (s - 0.5).abs()).min(1.0);
    let hue = if s < 0.5 { 195.0 } else { 328.0 };
    oklch_to_srgb(lightness, chroma, hue)
}

/// The cached 256×256 lookup table, indexed as `lut[t * 255][s * 255]` when
/// flattened row-major as `(t * 255) * 256 + (s * 255)`.
pub fn bivariate_lut() -> Vec<egui::Color32> {
    let mut lut = Vec::with_capacity(LUT_SIZE * LUT_SIZE);
    for t in 0..LUT_SIZE {
        for s in 0..LUT_SIZE {
            let [r, g, b] = bivariate_color(t as f32 / 255.0, s as f32 / 255.0);
            lut.push(egui::Color32::from_rgb(
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            ));
        }
    }
    lut
}
