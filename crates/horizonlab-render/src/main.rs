//! HorizonLab CPU reference renderer.
//!
//! Double-precision, one geodesic per sample. Slow but exact enough to
//! validate a future GPU (f32) renderer against.

mod observe;
mod scene;
mod vec3;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use horizonlab_core::{Jmn1, Kerr, Schwarzschild, Tolerance, TraceConfig};
use rayon::prelude::*;

use observe::{gaussian_blur, luminance, radial_profile, Source};
use scene::{Camera, Disk, Gas, GasFlow, Outcome, Scene, Sky, Spacetime};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricKind {
    Schwarzschild,
    Jmn1,
    Kerr,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GasKind {
    None,
    Static,
    Infall,
}

/// Render a compact object (black hole or mimicker) by backward ray tracing.
/// Lengths are in units of the mass M.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(long, value_enum, default_value = "schwarzschild")]
    metric: MetricKind,
    /// Mass M (sets the length unit; leave at 1).
    #[arg(long, default_value_t = 1.0)]
    mass: f64,
    /// JMN-1 boundary radius R_b (> 2M). R_b < 3M ⇒ photon sphere exists.
    #[arg(long, default_value_t = 2.5)]
    rb: f64,
    /// Kerr spin a/M, |a| < 1. Negative = disk retrograde to the spin.
    #[arg(long, default_value_t = 0.9, allow_negative_numbers = true)]
    spin: f64,

    /// Also render this metric with identical settings and write a
    /// side-by-side comparison with a residual map: [main | vs | main − vs].
    #[arg(long, value_enum)]
    vs: Option<MetricKind>,
    /// R_b for the `--vs` metric when it is JMN-1.
    #[arg(long, default_value_t = 4.0)]
    vs_rb: f64,
    /// Spin for the `--vs` metric when it is Kerr.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    vs_spin: f64,

    #[arg(long, default_value_t = 800)]
    width: u32,
    #[arg(long, default_value_t = 450)]
    height: u32,
    /// Vertical field of view in degrees.
    #[arg(long, default_value_t = 30.0)]
    fov: f64,
    /// Observer radius.
    #[arg(long, default_value_t = 30.0)]
    distance: f64,
    /// Angle between line of sight and disk normal, degrees.
    #[arg(long, default_value_t = 80.0)]
    inclination: f64,
    /// Supersamples per pixel axis (N×N rays per pixel).
    #[arg(long, default_value_t = 1)]
    samples: u32,

    /// Inner disk radius (default: ISCO — 6M, or the Kerr ISCO).
    #[arg(long)]
    disk_in: Option<f64>,
    #[arg(long, default_value_t = 20.0)]
    disk_out: f64,
    #[arg(long)]
    no_disk: bool,
    /// Disk brightness multiplier.
    #[arg(long, default_value_t = 1.5)]
    exposure: f64,

    /// Optically thin spherical gas (j ∝ r⁻²).
    #[arg(long, value_enum, default_value = "none")]
    gas: GasKind,
    /// Outer radius of the gas.
    #[arg(long, default_value_t = 50.0)]
    gas_out: f64,
    /// Gas brightness multiplier.
    #[arg(long, default_value_t = 1.0)]
    gas_brightness: f64,

    /// Equirectangular background image (default: procedural grid).
    #[arg(long)]
    sky_image: Option<PathBuf>,
    /// No background light at all.
    #[arg(long, conflicts_with = "sky_image")]
    black_sky: bool,

    /// Relative tolerance of the geodesic integrator.
    #[arg(long, default_value_t = 1e-8)]
    rtol: f64,

    /// Convolve with a circular Gaussian beam of this FWHM in μas
    /// (EHT 230 GHz ≈ 20, ngEHT 345 GHz ≈ 15). Needs a distant camera
    /// (--distance ≥ 100) so pixels map linearly to impact parameter.
    #[arg(long)]
    blur_fwhm: Option<f64>,
    /// Source that sets the angular scale θ_g = GM/(c²D) for --blur-fwhm.
    #[arg(long, value_enum, default_value = "m87")]
    source: Source,
    /// Write azimuthally averaged luminance vs impact parameter (CSV).
    #[arg(long)]
    profile: Option<PathBuf>,
    /// Radial bin width for --profile, in M.
    #[arg(long, default_value_t = 0.25)]
    profile_bin: f64,

    #[arg(short, long, default_value = "horizonlab.png")]
    out: PathBuf,
}

impl Args {
    fn build_spacetime(&self, kind: MetricKind, rb: f64, spin: f64) -> Spacetime {
        match kind {
            MetricKind::Schwarzschild => {
                Spacetime::Spherical(Box::new(Schwarzschild::new(self.mass)))
            }
            MetricKind::Jmn1 => Spacetime::Spherical(Box::new(Jmn1::new(self.mass, rb))),
            MetricKind::Kerr => Spacetime::Kerr(Kerr::new(self.mass, spin * self.mass)),
        }
    }

    fn build_scene<'a>(
        &self,
        spacetime: &'a Spacetime,
    ) -> Result<Scene<'a>, Box<dyn std::error::Error>> {
        if matches!(spacetime, Spacetime::Kerr(_)) && !matches!(self.gas, GasKind::None) {
            return Err("--gas is only implemented for spherical metrics, not Kerr".into());
        }
        let disk_in = self
            .disk_in
            .unwrap_or_else(|| spacetime.default_disk_inner());
        let sky = match &self.sky_image {
            Some(path) => Sky::load(path)?,
            None if self.black_sky => Sky::Black,
            None => Sky::Checker,
        };
        let flow = match self.gas {
            GasKind::None => None,
            GasKind::Static => Some(GasFlow::Static),
            GasKind::Infall => Some(GasFlow::Infall),
        };
        Ok(Scene {
            spacetime,
            camera: Camera::new(self.distance, self.inclination, self.fov),
            disk: (!self.no_disk).then(|| Disk::new(disk_in, self.disk_out)),
            gas: flow.map(|flow| Gas {
                flow,
                r_out: self.gas_out,
                brightness: self.gas_brightness,
            }),
            sky,
            trace: TraceConfig {
                tol: Tolerance {
                    rtol: self.rtol,
                    atol: self.rtol * 1e-2,
                },
                r_escape: (10.0 * self.distance).max(1000.0),
                ..TraceConfig::default()
            },
            disk_brightness: self.exposure,
        })
    }
}

/// Linear-RGB frame plus per-outcome sample counts.
struct Frame {
    width: usize,
    height: usize,
    pixels: Vec<[f32; 3]>,
    stats: HashMap<Outcome, usize>,
}

fn render(scene: &Scene, width: usize, height: usize, samples: usize) -> Frame {
    let s = samples.max(1);
    let aspect = width as f64 / height as f64;
    let rows: Vec<_> = (0..height)
        .into_par_iter()
        .map(|y| {
            let mut stats = HashMap::new();
            let row: Vec<[f32; 3]> = (0..width)
                .map(|x| {
                    let mut acc = [0.0f32; 3];
                    for sy in 0..s {
                        for sx in 0..s {
                            let u = (x as f64 + (sx as f64 + 0.5) / s as f64) / width as f64;
                            let v = (y as f64 + (sy as f64 + 0.5) / s as f64) / height as f64;
                            let d = scene.camera.ray((2.0 * u - 1.0) * aspect, 1.0 - 2.0 * v);
                            let (c, outcome) = scene.shade(d);
                            *stats.entry(outcome).or_insert(0) += 1;
                            for i in 0..3 {
                                acc[i] += c[i];
                            }
                        }
                    }
                    acc.map(|c| c / (s * s) as f32)
                })
                .collect();
            (row, stats)
        })
        .collect();

    let mut pixels = Vec::with_capacity(width * height);
    let mut stats = HashMap::new();
    for (row, row_stats) in rows {
        pixels.extend(row);
        for (k, v) in row_stats {
            *stats.entry(k).or_insert(0) += v;
        }
    }
    Frame {
        width,
        height,
        pixels,
        stats,
    }
}

fn print_stats(label: &str, frame: &Frame) {
    let mut totals: Vec<_> = frame.stats.iter().collect();
    totals.sort_by_key(|(k, _)| format!("{k:?}"));
    let summary: Vec<String> = totals.iter().map(|(k, v)| format!("{k:?}={v}")).collect();
    println!("  {label}: {}", summary.join(", "));
}

/// Diverging colormap for `t ∈ [-1, 1]`: blue (negative) – black – orange (positive).
fn diverging(t: f64) -> [u8; 3] {
    let t = t.clamp(-1.0, 1.0);
    let m = t.abs().sqrt();
    let c = if t >= 0.0 {
        [1.0, 0.55, 0.1]
    } else {
        [0.2, 0.55, 1.0]
    };
    c.map(|v| (v * m * 255.0 + 0.5) as u8)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let (w, h, s) = (
        args.width as usize,
        args.height as usize,
        args.samples.max(1) as usize,
    );

    let metric = args.build_spacetime(args.metric, args.rb, args.spin);
    let scene = args.build_scene(&metric)?;
    println!(
        "Rendering {} at {w}x{h} ({s}x{s} spp), r_obs={}M, i={}°",
        metric.name(),
        args.distance,
        args.inclination
    );
    let start = Instant::now();
    let mut frame = render(&scene, w, h, s);
    print_stats(&metric.name(), &frame);

    let mut vs = match args.vs {
        Some(kind) => {
            let vs_metric = args.build_spacetime(kind, args.vs_rb, args.vs_spin);
            let vs_scene = args.build_scene(&vs_metric)?;
            println!("Rendering {} for comparison", vs_metric.name());
            let vs_frame = render(&vs_scene, w, h, s);
            print_stats(&vs_metric.name(), &vs_frame);
            Some((vs_metric.name(), vs_frame))
        }
        None => None,
    };

    // Image-plane scale: apparent impact parameter b = r sin α / √A(r).
    let px_scale = scene.camera.pixel_angle(h) * args.distance / metric.lapse(args.distance);
    if let Some(fwhm_uas) = args.blur_fwhm {
        let theta_g = args.source.theta_g_uas();
        let sigma_px = fwhm_uas / theta_g / px_scale / (8.0 * 2f64.ln()).sqrt();
        println!(
            "Beam: FWHM {fwhm_uas} μas = {:.2}M for {:?} (θ_g = {theta_g:.2} μas), σ = {sigma_px:.1} px",
            fwhm_uas / theta_g,
            args.source
        );
        gaussian_blur(&mut frame.pixels, w, h, sigma_px);
        if let Some((_, f)) = vs.as_mut() {
            gaussian_blur(&mut f.pixels, w, h, sigma_px);
        }
    }

    if let Some(path) = &args.profile {
        let main = radial_profile(&frame.pixels, w, h, px_scale, args.profile_bin);
        let other = vs
            .as_ref()
            .map(|(_, f)| radial_profile(&f.pixels, w, h, px_scale, args.profile_bin));
        let mut out = BufWriter::new(File::create(path)?);
        match &vs {
            Some((name, _)) => writeln!(out, "b_M,\"{}\",\"{}\"", metric.name(), name)?,
            None => writeln!(out, "b_M,\"{}\"", metric.name())?,
        }
        for (i, (b, v)) in main.iter().enumerate() {
            match &other {
                Some(o) => writeln!(out, "{b},{v},{}", o[i].1)?,
                None => writeln!(out, "{b},{v}")?,
            }
        }
        println!("Profile -> {}", path.display());
    }

    let Some((vs_name, vs_frame)) = vs else {
        let mut img = image::RgbImage::new(args.width, args.height);
        for (i, c) in frame.pixels.iter().enumerate() {
            img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(c.map(tonemap)));
        }
        img.save(&args.out)?;
        println!("Done in {:.2?} -> {}", start.elapsed(), args.out.display());
        return Ok(());
    };

    let residual: Vec<f64> = frame
        .pixels
        .iter()
        .zip(&vs_frame.pixels)
        .map(|(a, b)| luminance(*a) - luminance(*b))
        .collect();
    let max_lum = frame
        .pixels
        .iter()
        .chain(&vs_frame.pixels)
        .map(|c| luminance(*c))
        .fold(0.0, f64::max);
    let max_res = residual.iter().fold(0.0f64, |m, r| m.max(r.abs()));
    let threshold = 1e-3 * max_lum;
    let differing = residual.iter().filter(|r| r.abs() > threshold).count();

    let mut img = image::RgbImage::new(3 * args.width, args.height);
    for y in 0..frame.height {
        for x in 0..frame.width {
            let i = y * frame.width + x;
            let (xu, yu) = (x as u32, y as u32);
            img.put_pixel(xu, yu, image::Rgb(frame.pixels[i].map(tonemap)));
            img.put_pixel(
                xu + args.width,
                yu,
                image::Rgb(vs_frame.pixels[i].map(tonemap)),
            );
            let t = if max_res > 0.0 {
                residual[i] / max_res
            } else {
                0.0
            };
            img.put_pixel(xu + 2 * args.width, yu, image::Rgb(diverging(t)));
        }
    }
    img.save(&args.out)?;

    println!(
        "Residual (luminance, {} − {}): max |Δ| = {:.3e} ({:.2}% of peak), {} of {} pixels differ by > 0.1% of peak",
        metric.name(),
        vs_name,
        max_res,
        if max_lum > 0.0 { 100.0 * max_res / max_lum } else { 0.0 },
        differing,
        residual.len()
    );
    println!("Residual panel: orange = first brighter, blue = second brighter, scaled to max |Δ|");
    println!("Done in {:.2?} -> {}", start.elapsed(), args.out.display());
    Ok(())
}

/// Reinhard tone map + sRGB encode.
fn tonemap(c: f32) -> u8 {
    let x = c / (1.0 + c);
    let srgb = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (srgb.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
