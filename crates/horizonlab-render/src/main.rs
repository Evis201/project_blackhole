//! HorizonLab CPU reference renderer.
//!
//! Double-precision, one geodesic per sample. Slow but exact enough to
//! validate a future GPU (f32) renderer against.

mod scene;
mod vec3;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use horizonlab_core::{Jmn1, Schwarzschild, SphericalMetric, Tolerance, TraceConfig};
use rayon::prelude::*;

use scene::{Camera, Disk, Gas, GasFlow, Outcome, Scene, Sky};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricKind {
    Schwarzschild,
    Jmn1,
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

    /// Also render this metric with identical settings and write a
    /// side-by-side comparison with a residual map: [main | vs | main − vs].
    #[arg(long, value_enum)]
    vs: Option<MetricKind>,
    /// R_b for the `--vs` metric when it is JMN-1.
    #[arg(long, default_value_t = 4.0)]
    vs_rb: f64,

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

    #[arg(long, default_value_t = 6.0)]
    disk_in: f64,
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

    #[arg(short, long, default_value = "horizonlab.png")]
    out: PathBuf,
}

impl Args {
    fn build_metric(&self, kind: MetricKind, rb: f64) -> Box<dyn SphericalMetric> {
        match kind {
            MetricKind::Schwarzschild => Box::new(Schwarzschild::new(self.mass)),
            MetricKind::Jmn1 => Box::new(Jmn1::new(self.mass, rb)),
        }
    }

    fn build_scene<'a>(
        &self,
        metric: &'a dyn SphericalMetric,
    ) -> Result<Scene<'a>, image::ImageError> {
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
            metric,
            camera: Camera::new(self.distance, self.inclination, self.fov),
            disk: (!self.no_disk).then(|| Disk::new(self.disk_in, self.disk_out)),
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

fn luminance(c: [f32; 3]) -> f64 {
    0.2126 * c[0] as f64 + 0.7152 * c[1] as f64 + 0.0722 * c[2] as f64
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

    let metric = args.build_metric(args.metric, args.rb);
    let scene = args.build_scene(metric.as_ref())?;
    println!(
        "Rendering {} at {w}x{h} ({s}x{s} spp), r_obs={}M, i={}°",
        metric.name(),
        args.distance,
        args.inclination
    );
    let start = Instant::now();
    let frame = render(&scene, w, h, s);
    print_stats(&metric.name(), &frame);

    let Some(vs_kind) = args.vs else {
        let mut img = image::RgbImage::new(args.width, args.height);
        for (i, c) in frame.pixels.iter().enumerate() {
            img.put_pixel((i % w) as u32, (i / w) as u32, image::Rgb(c.map(tonemap)));
        }
        img.save(&args.out)?;
        println!("Done in {:.2?} -> {}", start.elapsed(), args.out.display());
        return Ok(());
    };

    let vs_metric = args.build_metric(vs_kind, args.vs_rb);
    let vs_scene = args.build_scene(vs_metric.as_ref())?;
    println!("Rendering {} for comparison", vs_metric.name());
    let vs_frame = render(&vs_scene, w, h, s);
    print_stats(&vs_metric.name(), &vs_frame);

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
        vs_metric.name(),
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
