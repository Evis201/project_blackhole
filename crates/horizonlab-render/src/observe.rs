//! Observational post-processing: telescope beam blur and radial profiles.
//!
//! Image-plane lengths are in units of M (apparent impact parameter). For a
//! source of mass M at distance D, one M on the sky subtends the angular
//! gravitational radius `θ_g = GM / (c² D)`.

use clap::ValueEnum;

/// `GM_⊙/c²` in metres.
const GM_SUN_OVER_C2: f64 = 1476.625;
const PARSEC: f64 = 3.085_677_581e16;
const MICROARCSEC_PER_RAD: f64 = 180.0 / std::f64::consts::PI * 3600.0 * 1e6;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Source {
    /// M87*: M = 6.5e9 M_⊙, D = 16.8 Mpc (EHT 2019).
    M87,
    /// Sgr A*: M = 4.0e6 M_⊙, D = 8.15 kpc (EHT 2022).
    Sgra,
}

impl Source {
    /// `(mass in M_⊙, distance in pc)`
    fn parameters(self) -> (f64, f64) {
        match self {
            Source::M87 => (6.5e9, 16.8e6),
            Source::Sgra => (4.0e6, 8.15e3),
        }
    }

    /// Angular size of one gravitational radius, in μas.
    pub fn theta_g_uas(self) -> f64 {
        let (mass, distance_pc) = self.parameters();
        mass * GM_SUN_OVER_C2 / (distance_pc * PARSEC) * MICROARCSEC_PER_RAD
    }
}

/// In-place separable Gaussian blur of a linear-RGB image.
/// Edges are handled by renormalising the truncated kernel.
pub fn gaussian_blur(pixels: &mut [[f32; 3]], width: usize, height: usize, sigma_px: f64) {
    if sigma_px <= 0.0 {
        return;
    }
    let radius = (4.0 * sigma_px).ceil() as isize;
    let kernel: Vec<f64> = (-radius..=radius)
        .map(|i| (-0.5 * (i as f64 / sigma_px).powi(2)).exp())
        .collect();

    let pass = |src: &[[f32; 3]], dst: &mut [[f32; 3]], horizontal: bool| {
        for y in 0..height {
            for x in 0..width {
                let mut acc = [0.0f64; 3];
                let mut norm = 0.0;
                for (k, w) in kernel.iter().enumerate() {
                    let off = k as isize - radius;
                    let (sx, sy) = if horizontal {
                        (x as isize + off, y as isize)
                    } else {
                        (x as isize, y as isize + off)
                    };
                    if sx < 0 || sy < 0 || sx >= width as isize || sy >= height as isize {
                        continue;
                    }
                    let p = src[sy as usize * width + sx as usize];
                    for c in 0..3 {
                        acc[c] += w * p[c] as f64;
                    }
                    norm += w;
                }
                dst[y * width + x] = acc.map(|v| (v / norm) as f32);
            }
        }
    };

    let mut tmp = vec![[0.0f32; 3]; pixels.len()];
    pass(pixels, &mut tmp, true);
    pass(&tmp, pixels, false);
}

pub fn luminance(c: [f32; 3]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
}

/// Azimuthally averaged luminance about the image centre, in bins of width
/// `bin` (same units as `px_scale`, the size of one pixel). Returns
/// `(bin centre, mean)` for every non-empty bin out to the inscribed circle.
pub fn radial_profile(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    px_scale: f64,
    bin: f64,
) -> Vec<(f64, f64)> {
    let r_max = 0.5 * width.min(height) as f64 * px_scale;
    let n_bins = (r_max / bin).floor() as usize;
    let mut sums = vec![(0.0f64, 0usize); n_bins];
    let (cx, cy) = (0.5 * width as f64, 0.5 * height as f64);
    for y in 0..height {
        for x in 0..width {
            let dx = (x as f64 + 0.5 - cx) * px_scale;
            let dy = (y as f64 + 0.5 - cy) * px_scale;
            let i = ((dx * dx + dy * dy).sqrt() / bin) as usize;
            if i < n_bins {
                sums[i].0 += luminance(pixels[y * width + x]);
                sums[i].1 += 1;
            }
        }
    }
    sums.iter()
        .enumerate()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(i, (s, n))| ((i as f64 + 0.5) * bin, s / *n as f64))
        .collect()
}

/// Radians per microarcsecond.
pub const RAD_PER_UAS: f64 = 1.0 / MICROARCSEC_PER_RAD;

/// Normalised visibility amplitude `|V(u,v)| / V(0)` of an image as a
/// function of baseline length.
///
/// `V(u, v) = Σ I(x, y) exp(-2πi (u x + v y))`, with `(x, y)` in radians
/// on the sky and `(u, v)` in wavelengths. Computed by direct DFT of the
/// pixel luminance (no FFT, no padding artefacts) on a polar grid.
pub struct VisibilityCurve {
    /// Baseline length in Gλ.
    pub baseline: Vec<f64>,
    /// Mean over position angle.
    pub mean: Vec<f64>,
    /// East–west baselines (image x direction).
    pub east_west: Vec<f64>,
    /// North–south baselines (image y direction).
    pub north_south: Vec<f64>,
}

impl VisibilityCurve {
    /// Baseline (Gλ) of the first local minimum of the mean curve.
    pub fn first_minimum(&self) -> Option<f64> {
        (1..self.mean.len() - 1)
            .find(|&i| self.mean[i] < self.mean[i - 1] && self.mean[i] <= self.mean[i + 1])
            .map(|i| self.baseline[i])
    }
}

/// `px_rad`: angular size of one pixel. `n_angles` must be even so that
/// the east–west and north–south directions are sampled.
pub fn visibility_curve(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    px_rad: f64,
    max_glambda: f64,
    n_radii: usize,
    n_angles: usize,
) -> VisibilityCurve {
    use rayon::prelude::*;
    assert!(n_angles.is_multiple_of(2) && n_radii > 0);
    let (cx, cy) = (0.5 * width as f64, 0.5 * height as f64);
    // Bright pixels only: (x, y, I). y grows upwards on the sky.
    let sources: Vec<(f64, f64, f64)> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .filter_map(|(x, y)| {
            let i = luminance(pixels[y * width + x]);
            (i > 0.0).then_some({
                (
                    (x as f64 + 0.5 - cx) * px_rad,
                    (cy - y as f64 - 0.5) * px_rad,
                    i,
                )
            })
        })
        .collect();
    let total: f64 = sources.iter().map(|s| s.2).sum();

    let amplitude = |u: f64, v: f64| -> f64 {
        let (mut re, mut im) = (0.0, 0.0);
        for &(x, y, i) in &sources {
            let (sn, cs) = (-2.0 * std::f64::consts::PI * (u * x + v * y)).sin_cos();
            re += i * cs;
            im += i * sn;
        }
        re.hypot(im) / total
    };

    let rows: Vec<(f64, f64, f64, f64)> = (0..=n_radii)
        .into_par_iter()
        .map(|k| {
            let rho_gl = max_glambda * k as f64 / n_radii as f64;
            let rho = rho_gl * 1e9;
            let amps: Vec<f64> = (0..n_angles)
                .map(|j| {
                    let psi = std::f64::consts::PI * j as f64 / n_angles as f64;
                    amplitude(rho * psi.cos(), rho * psi.sin())
                })
                .collect();
            let mean = amps.iter().sum::<f64>() / n_angles as f64;
            (rho_gl, mean, amps[0], amps[n_angles / 2])
        })
        .collect();

    VisibilityCurve {
        baseline: rows.iter().map(|r| r.0).collect(),
        mean: rows.iter().map(|r| r.1).collect(),
        east_west: rows.iter().map(|r| r.2).collect(),
        north_south: rows.iter().map(|r| r.3).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn theta_g_matches_published_values() {
        // EHT: θ_g ≈ 3.8 μas (M87*), ≈ 4.8 μas (Sgr A*).
        assert!((Source::M87.theta_g_uas() - 3.82).abs() < 0.02);
        assert!((Source::Sgra.theta_g_uas() - 4.84).abs() < 0.02);
    }

    #[test]
    fn blur_of_point_source_is_gaussian_and_conserves_flux() {
        let (w, h) = (101, 101);
        let mut img = vec![[0.0f32; 3]; w * h];
        img[50 * w + 50] = [1.0; 3];
        let sigma = 3.0;
        gaussian_blur(&mut img, w, h, sigma);

        let total: f64 = img.iter().map(|p| p[0] as f64).sum();
        assert!((total - 1.0).abs() < 1e-5, "flux {total}");

        // Second moment along x equals σ².
        let var: f64 = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| (x as f64 - 50.0).powi(2) * img[y * w + x][0] as f64)
            .sum();
        // Truncating the kernel at 4σ drops the tails: relative bias ≈ 5e-4.
        assert!((var / (sigma * sigma) - 1.0).abs() < 1e-3, "variance {var}");
    }

    #[test]
    fn radial_profile_of_uniform_image_is_flat() {
        let (w, h) = (64, 48);
        let img = vec![[0.5f32; 3]; w * h];
        let prof = radial_profile(&img, w, h, 0.25, 1.0);
        assert_eq!(prof.len(), 6);
        for (_, v) in prof {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn gaussian_visibility_is_gaussian() {
        // I ∝ exp(-x²/2σ²) ⇒ |V(ρ)| = exp(-2π² σ² ρ²).
        let (w, h) = (128, 128);
        let sigma_px = 4.0;
        let px_rad = 1e-11;
        let img: Vec<[f32; 3]> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| {
                let dx = x as f64 + 0.5 - 64.0;
                let dy = y as f64 + 0.5 - 64.0;
                let v = (-(dx * dx + dy * dy) / (2.0 * sigma_px * sigma_px)).exp() as f32;
                [v; 3]
            })
            .collect();
        let curve = visibility_curve(&img, w, h, px_rad, 10.0, 10, 8);
        let sigma = sigma_px * px_rad;
        for (rho, v) in curve.baseline.iter().zip(&curve.mean) {
            let expected =
                (-2.0 * std::f64::consts::PI.powi(2) * (sigma * rho * 1e9).powi(2)).exp();
            assert!(
                (v - expected).abs() < 1e-4,
                "ρ = {rho} Gλ: {v} vs {expected}"
            );
        }
        assert_eq!(curve.mean[0], 1.0);
    }

    #[test]
    fn thin_ring_first_null_matches_bessel_zero() {
        // Thin ring of radius R: |V| = |J₀(2πRρ)|, first null at 2.4048/(2πR).
        let (w, h) = (256, 256);
        let r_px = 60.0;
        let px_rad = 1e-11;
        let img: Vec<[f32; 3]> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| {
                let d =
                    ((x as f64 + 0.5 - 128.0).powi(2) + (y as f64 + 0.5 - 128.0).powi(2)).sqrt();
                [if (d - r_px).abs() < 0.5 { 1.0 } else { 0.0 }; 3]
            })
            .collect();
        let curve = visibility_curve(&img, w, h, px_rad, 1.0, 200, 8);
        let expected = 2.4048 / (2.0 * std::f64::consts::PI * r_px * px_rad) / 1e9;
        let got = curve.first_minimum().unwrap();
        assert!(
            (got - expected).abs() / expected < 0.01,
            "null {got} vs {expected} Gλ"
        );
    }
}
