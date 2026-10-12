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
        assert!((var - sigma * sigma).abs() < 1e-3, "variance {var}");
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
}
