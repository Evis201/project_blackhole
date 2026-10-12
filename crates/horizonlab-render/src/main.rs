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

use scene::{Camera, Disk, Outcome, Scene, Sky};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricKind {
    Schwarzschild,
    Jmn1,
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
    /// Equirectangular background image (default: procedural grid).
    #[arg(long)]
    sky_image: Option<PathBuf>,

    /// Relative tolerance of the geodesic integrator.
    #[arg(long, default_value_t = 1e-8)]
    rtol: f64,

    #[arg(short, long, default_value = "horizonlab.png")]
    out: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let metric: Box<dyn SphericalMetric> = match args.metric {
        MetricKind::Schwarzschild => Box::new(Schwarzschild::new(args.mass)),
        MetricKind::Jmn1 => Box::new(Jmn1::new(args.mass, args.rb)),
    };
    let sky = match &args.sky_image {
        Some(path) => Sky::load(path)?,
        None => Sky::Checker,
    };
    let scene = Scene {
        metric: metric.as_ref(),
        camera: Camera::new(args.distance, args.inclination, args.fov),
        disk: (!args.no_disk).then(|| Disk::new(args.disk_in, args.disk_out)),
        sky,
        trace: TraceConfig {
            tol: Tolerance {
                rtol: args.rtol,
                atol: args.rtol * 1e-2,
            },
            r_escape: (10.0 * args.distance).max(1000.0),
            ..TraceConfig::default()
        },
        disk_brightness: args.exposure,
    };

    let (w, h, s) = (
        args.width as usize,
        args.height as usize,
        args.samples.max(1) as usize,
    );
    let aspect = w as f64 / h as f64;
    println!(
        "Rendering {} at {w}x{h} ({s}x{s} spp), r_obs={}M, i={}°",
        metric.name(),
        args.distance,
        args.inclination
    );
    let start = Instant::now();

    let rows: Vec<_> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut stats = HashMap::new();
            let row: Vec<[f32; 3]> = (0..w)
                .map(|x| {
                    let mut acc = [0.0f32; 3];
                    for sy in 0..s {
                        for sx in 0..s {
                            let u = (x as f64 + (sx as f64 + 0.5) / s as f64) / w as f64;
                            let v = (y as f64 + (sy as f64 + 0.5) / s as f64) / h as f64;
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

    let mut totals: HashMap<Outcome, usize> = HashMap::new();
    let mut img = image::RgbImage::new(args.width, args.height);
    for (y, (row, stats)) in rows.into_iter().enumerate() {
        for (k, v) in stats {
            *totals.entry(k).or_insert(0) += v;
        }
        for (x, c) in row.into_iter().enumerate() {
            img.put_pixel(x as u32, y as u32, image::Rgb(c.map(tonemap)));
        }
    }
    img.save(&args.out)?;

    let mut totals: Vec<_> = totals.into_iter().collect();
    totals.sort_by_key(|(k, _)| format!("{k:?}"));
    println!("Done in {:.2?} -> {}", start.elapsed(), args.out.display());
    for (k, v) in totals {
        println!("  {k:?}: {v}");
    }
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
