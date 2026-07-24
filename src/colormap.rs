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

/// Convert sRGB (each component in 0..=1) to OkLCH `(lightness, chroma, hue)`
/// with the hue in degrees. The inverse of [`oklch_to_srgb`].
pub fn srgb_to_oklch(red: f32, green: f32, blue: f32) -> (f32, f32, f32) {
    let linear = |srgb: f32| {
        if srgb <= 0.040_45 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / 1.055).powf(2.4)
        }
    };
    let r = linear(red);
    let g = linear(green);
    let b = linear(blue);

    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_993 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    let lightness = 0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_;
    let a = 1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_;
    let b_ = 0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_;
    let chroma = a.hypot(b_);
    let hue = b_.atan2(a).to_degrees().rem_euclid(360.0);
    (lightness, chroma, hue)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `oklch_to_srgb` must invert `srgb_to_oklch` for in-gamut colors:
    /// every sRGB color survives the roundtrip.
    #[test]
    fn oklch_srgb_roundtrip() {
        for (r, g, b) in [
            (0_u8, 0, 4),
            (28, 16, 68),
            (79, 18, 123),
            (129, 37, 129),
            (181, 54, 122),
            (229, 80, 100),
            (251, 135, 97),
            (254, 194, 135),
            (252, 253, 245),
            (0, 0, 0),
            (255, 255, 255),
            (17, 220, 96),
        ] {
            let rgb = [r, g, b].map(|v| f32::from(v) / 255.0);
            let (l, c, h) = srgb_to_oklch(rgb[0], rgb[1], rgb[2]);
            let roundtrip = oklch_to_srgb(l, c, h);
            for (before, after) in rgb.iter().zip(roundtrip) {
                assert!(
                    (before - after).abs() < 2e-3,
                    "{rgb:?} -> ({l}, {c}, {h}) -> {roundtrip:?}"
                );
            }
        }
    }
}
