//! Radial free fall: proper time, tidal forces and curvature along the way.
//!
//! Follows an observer released from rest at `--r0` down to `--r-end`
//! (through the horizon for Schwarzschild, to the naked singularity for
//! JMN-1) and reports the tidal acceleration across a body of length
//! `--body` for an object of `--mass-solar` solar masses.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use horizonlab_core::tidal::{radial_infall, InfallSample};
use horizonlab_core::{Jmn1, Schwarzschild, SphericalMetric};

/// `GM_⊙/c²` in metres and `GM_⊙/c³` in seconds.
const GM_SUN_M: f64 = 1476.625;
const GM_SUN_S: f64 = 4.925_491e-6;
const C2: f64 = 8.987_551_787e16;
const G_EARTH: f64 = 9.806_65;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum MetricKind {
    Schwarzschild,
    Jmn1,
}

/// Tidal forces on a body falling radially into a black hole or JMN-1
/// naked singularity. Lengths in units of M unless stated.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(long, value_enum, default_value = "schwarzschild")]
    metric: MetricKind,
    /// JMN-1 boundary radius R_b (> 2M).
    #[arg(long, default_value_t = 2.5)]
    rb: f64,
    /// Release radius (from rest).
    #[arg(long, default_value_t = 10.0)]
    r0: f64,
    /// Stop radius.
    #[arg(long, default_value_t = 1e-3)]
    r_end: f64,
    /// Number of samples.
    #[arg(long, default_value_t = 4000)]
    samples: usize,
    /// Mass of the object in solar masses (sets physical units).
    #[arg(long, default_value_t = 10.0)]
    mass_solar: f64,
    /// Size of the falling body in metres.
    #[arg(long, default_value_t = 2.0)]
    body: f64,
    /// Tidal acceleration considered lethal, in units of g.
    #[arg(long, default_value_t = 10.0)]
    lethal_g: f64,
    /// Write all samples as CSV.
    #[arg(long)]
    out: Option<PathBuf>,
}

/// Log-log slope of `|f|` over the last decade of samples (`r` shrinking).
fn tail_slope(samples: &[InfallSample], f: impl Fn(&InfallSample) -> f64) -> f64 {
    let last = samples.last().expect("samples");
    let ref_r = last.r * 10.0;
    let first = samples
        .iter()
        .rev()
        .find(|s| s.r >= ref_r)
        .unwrap_or(&samples[0]);
    (f(last).abs().ln() - f(first).abs().ln()) / (last.r.ln() - first.r.ln())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let metric: Box<dyn SphericalMetric> = match args.metric {
        MetricKind::Schwarzschild => Box::new(Schwarzschild::new(1.0)),
        MetricKind::Jmn1 => Box::new(Jmn1::new(1.0, args.rb)),
    };
    let samples = radial_infall(metric.as_ref(), args.r0, args.r_end, args.samples);

    let len_m = GM_SUN_M * args.mass_solar;
    let time_s = GM_SUN_S * args.mass_solar;
    // Stretch is in M⁻²; acceleration across the body = stretch · c² · body / len².
    let accel_g = |stretch: f64| stretch * C2 * args.body / (len_m * len_m) / G_EARTH;

    println!(
        "{} — release from rest at r0 = {}M, M = {} M_⊙ (M = {:.4e} m, {:.4e} s), body {} m",
        metric.name(),
        args.r0,
        args.mass_solar,
        len_m,
        time_s,
        args.body
    );

    // Proper time to notable radii.
    let tau_at = |r_mark: f64| samples.iter().find(|s| s.r <= r_mark).map(|s| s.tau);
    let mut marks = vec![];
    match args.metric {
        MetricKind::Schwarzschild => marks.push(("horizon r = 2M", 2.0)),
        MetricKind::Jmn1 => marks.push(("surface r = R_b", args.rb)),
    }
    for (label, r_mark) in marks {
        if let Some(tau) = tau_at(r_mark) {
            println!("  τ to {label}: {tau:.4} M = {:.4e} s", tau * time_s);
        }
    }
    let last = samples.last().expect("samples");
    println!(
        "  τ to r = {:.3e}M: {:.4} M = {:.4e} s",
        last.r,
        last.tau,
        last.tau * time_s
    );

    // Where tidal forces become lethal.
    let lethal = samples.iter().find(|s| {
        accel_g(s.radial_stretch).abs() > args.lethal_g
            || accel_g(s.transverse_stretch).abs() > args.lethal_g
    });
    match lethal {
        Some(s) => println!(
            "  Tidal acceleration exceeds {} g at r = {:.4}M ({:.4e} m), τ = {:.4} M{}",
            args.lethal_g,
            s.r,
            s.r * len_m,
            s.tau,
            match args.metric {
                MetricKind::Schwarzschild if s.r > 2.0 => " — outside the horizon",
                MetricKind::Schwarzschild => " — inside the horizon",
                MetricKind::Jmn1 if s.r > args.rb => " — outside the object",
                MetricKind::Jmn1 => " — inside the object",
            }
        ),
        None => println!("  Tidal acceleration stays below {} g", args.lethal_g),
    }

    println!(
        "  At r = {:.3e}M: radial {:+.3e} g, transverse {:+.3e} g (+ stretch, − squeeze), Kretschmann {:.3e} M⁻⁴",
        last.r,
        accel_g(last.radial_stretch),
        accel_g(last.transverse_stretch),
        last.kretschmann
    );
    println!(
        "  Scaling over the last decade: radial ∝ r^{:.2}, transverse ∝ r^{:.2}, Kretschmann ∝ r^{:.2}",
        tail_slope(&samples, |s| s.radial_stretch),
        tail_slope(&samples, |s| s.transverse_stretch),
        tail_slope(&samples, |s| s.kretschmann)
    );

    if let Some(path) = &args.out {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(
            w,
            "r_M,tau_M,radial_stretch_M-2,transverse_stretch_M-2,kretschmann_M-4,radial_g,transverse_g"
        )?;
        for s in &samples {
            writeln!(
                w,
                "{},{},{},{},{},{},{}",
                s.r,
                s.tau,
                s.radial_stretch,
                s.transverse_stretch,
                s.kretschmann,
                accel_g(s.radial_stretch),
                accel_g(s.transverse_stretch)
            )?;
        }
        println!("  Samples -> {}", path.display());
    }
    Ok(())
}
