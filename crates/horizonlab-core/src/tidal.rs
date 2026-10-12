//! Curvature and tidal forces for radial infall in static spherical metrics.
//!
//! Riemann components in the static orthonormal frame of
//! `ds² = -A dt² + B dr² + r² dΩ²`:
//!
//! ```text
//! R_t̂r̂t̂r̂ = (1/B) (A''/(2A) - A'²/(4A²) - A'B'/(4AB))
//! R_t̂θ̂t̂θ̂ = A' / (2 r A B)
//! R_r̂θ̂r̂θ̂ = B' / (2 r B²)
//! R_θ̂φ̂θ̂φ̂ = (1 - 1/B) / r²
//! ```
//!
//! A radially moving observer with conserved energy `E = A dt/dτ` is related
//! to the static frame by a radial boost with `γ² = E²/A`. The radial
//! component is boost-invariant; the transverse one becomes
//!
//! ```text
//! K_⊥ = R_t̂θ̂t̂θ̂ + (E² - A)/A · (A'B + AB') / (2 r A B²)
//! ```
//!
//! which equals `R_t̂θ̂t̂θ̂` whenever `AB = const` (Schwarzschild). Written this
//! way the expressions stay finite inside a horizon (`A < 0`) as long as
//! `AB` is regular, so Schwarzschild infall can be followed to `r → 0`.
//!
//! Tidal acceleration between two free particles separated by `ξ` along a
//! frame axis is `δa = -K ξ`. Below, `stretch = -K`: positive means the
//! particles are pulled apart, negative means they are squeezed together.

use crate::metric::SphericalMetric;

/// Independent Riemann components in the static orthonormal frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaticRiemann {
    pub trtr: f64,
    pub tqtq: f64,
    pub rqrq: f64,
    pub qpqp: f64,
}

impl StaticRiemann {
    /// Kretschmann invariant `R_abcd R^abcd`.
    pub fn kretschmann(&self) -> f64 {
        4.0 * self.trtr.powi(2)
            + 8.0 * self.tqtq.powi(2)
            + 8.0 * self.rqrq.powi(2)
            + 4.0 * self.qpqp.powi(2)
    }
}

pub fn static_riemann<M: SphericalMetric + ?Sized>(metric: &M, r: f64) -> StaticRiemann {
    let (a, da, d2a) = (metric.a(r), metric.da(r), metric.d2a(r));
    let (b, db) = (metric.b(r), metric.db(r));
    StaticRiemann {
        trtr: (d2a / (2.0 * a) - da * da / (4.0 * a * a) - da * db / (4.0 * a * b)) / b,
        tqtq: da / (2.0 * r * a * b),
        rqrq: db / (2.0 * r * b * b),
        qpqp: (1.0 - 1.0 / b) / (r * r),
    }
}

/// Tidal stretch `(radial, transverse)` felt by a radial geodesic observer
/// with energy `e` at radius `r` (units of `M⁻²`; multiply by separation).
pub fn infall_stretch<M: SphericalMetric + ?Sized>(metric: &M, e: f64, r: f64) -> (f64, f64) {
    let riem = static_riemann(metric, r);
    let (a, da, b, db) = (metric.a(r), metric.da(r), metric.b(r), metric.db(r));
    let k_perp = riem.tqtq + (e * e - a) / a * (da * b + a * db) / (2.0 * r * a * b * b);
    (-riem.trtr, -k_perp)
}

/// One sample along a radial infall.
#[derive(Clone, Copy, Debug)]
pub struct InfallSample {
    pub r: f64,
    /// Proper time since release.
    pub tau: f64,
    pub radial_stretch: f64,
    pub transverse_stretch: f64,
    pub kretschmann: f64,
}

/// `(dr/dτ)² = (E² - A)/(AB)`.
fn radial_speed_sq<M: SphericalMetric + ?Sized>(metric: &M, e: f64, r: f64) -> f64 {
    let a = metric.a(r);
    let v = (e * e - a) / (a * metric.b(r));
    if v.is_finite() {
        v
    } else {
        // AB = 0·∞ exactly on a horizon: step off it.
        radial_speed_sq(metric, e, r * (1.0 + 1e-9))
    }
}

/// Free fall from rest at `r0` down to `r_end`, sampled `n + 1` times
/// (uniform in `u = √(r0 - r)`, which concentrates samples near release and
/// removes the `1/√(r0 - r)` singularity of `dτ/dr` there).
///
/// Proper time uses the trapezoid rule on `dτ/du = 2u / √((E²-A)/(AB))`,
/// whose limit at `u = 0` is `2 / √(A'/(AB))|_{r0}`.
pub fn radial_infall<M: SphericalMetric + ?Sized>(
    metric: &M,
    r0: f64,
    r_end: f64,
    n: usize,
) -> Vec<InfallSample> {
    assert!(r0 > r_end && r_end > 0.0 && n > 0);
    assert!(
        metric.a(r0) > 0.0,
        "release point must be outside any horizon"
    );
    let e = metric.a(r0).sqrt();
    let u_max = (r0 - r_end).sqrt();
    let integrand = |u: f64| -> f64 {
        if u == 0.0 {
            2.0 / (metric.da(r0) / (metric.a(r0) * metric.b(r0))).sqrt()
        } else {
            2.0 * u / radial_speed_sq(metric, e, r0 - u * u).sqrt()
        }
    };

    let mut out = Vec::with_capacity(n + 1);
    let mut tau = 0.0;
    let mut prev = integrand(0.0);
    for i in 0..=n {
        let u = u_max * i as f64 / n as f64;
        let r = r0 - u * u;
        if i > 0 {
            let cur = integrand(u);
            tau += 0.5 * (u_max / n as f64) * (prev + cur);
            prev = cur;
        }
        let (radial_stretch, transverse_stretch) = infall_stretch(metric, e, r);
        out.push(InfallSample {
            r,
            tau,
            radial_stretch,
            transverse_stretch,
            kretschmann: static_riemann(metric, r).kretschmann(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::{Jmn1, Schwarzschild};

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    #[test]
    fn schwarzschild_curvature_outside_and_inside_horizon() {
        let s = Schwarzschild::new(1.0);
        for &r in &[1.0, 1.5, 3.0, 10.0] {
            let rm = static_riemann(&s, r);
            let r3 = r * r * r;
            assert!(close(rm.trtr, -2.0 / r3, 1e-12), "trtr at r={r}");
            assert!(close(rm.tqtq, 1.0 / r3, 1e-12), "tqtq at r={r}");
            assert!(
                close(rm.kretschmann(), 48.0 / (r3 * r3), 1e-12),
                "K at r={r}"
            );
        }
    }

    #[test]
    fn schwarzschild_tidal_forces_are_boost_invariant() {
        // Radial infall at any energy: stretch 2M/r³, squeeze M/r³.
        let s = Schwarzschild::new(1.0);
        for &e in &[0.5, 0.9, 1.0, 2.0] {
            for &r in &[0.5, 1.9, 4.0] {
                let (rad, trans) = infall_stretch(&s, e, r);
                assert!(close(rad, 2.0 / r.powi(3), 1e-10), "E={e} r={r}");
                assert!(close(trans, -1.0 / r.powi(3), 1e-9), "E={e} r={r}: {trans}");
            }
        }
    }

    #[test]
    fn jmn1_interior_curvature_is_analytic() {
        // Inside: A = (1-M₀)(r/R_b)^α, B = 1/(1-M₀):
        // R_trtr = (1-M₀) α(α-2)/(4r²), R_tθtθ = M₀/(2r²), R_rθrθ = 0,
        // R_θφθφ = M₀/r².
        let j = Jmn1::new(1.0, 2.5);
        let (m0, al) = (j.compactness(), j.alpha());
        for &r in &[0.01, 0.5, 2.0] {
            let rm = static_riemann(&j, r);
            let r2 = r * r;
            assert!(close(
                rm.trtr,
                (1.0 - m0) * al * (al - 2.0) / (4.0 * r2),
                1e-12
            ));
            assert!(close(rm.tqtq, m0 / (2.0 * r2), 1e-12));
            assert_eq!(rm.rqrq, 0.0);
            assert!(close(rm.qpqp, m0 / r2, 1e-12));
        }
    }

    #[test]
    fn schwarzschild_infall_proper_time_matches_cycloid() {
        // From rest at r0: r = (r0/2)(1 + cos η), τ = √(r0³/8M)(η + sin η).
        let s = Schwarzschild::new(1.0);
        let r0 = 10.0;
        let tau_at = |r: f64| {
            let eta = (2.0 * r / r0 - 1.0).acos();
            (r0.powi(3) / 8.0).sqrt() * (eta + eta.sin())
        };
        let samples = radial_infall(&s, r0, 1e-3, 20000);
        for smp in samples.iter().step_by(1000) {
            assert!(
                (smp.tau - tau_at(smp.r)).abs() < 1e-6 * tau_at(1e-3),
                "r={}: τ={} vs {}",
                smp.r,
                smp.tau,
                tau_at(smp.r)
            );
        }
        // Total to r → 0 approaches π √(r0³/8M) ≈ 35.12M.
        let last = samples.last().unwrap();
        assert!((last.tau - std::f64::consts::PI * (r0.powi(3) / 8.0).sqrt()).abs() < 1e-3);
    }

    #[test]
    fn jmn1_tidal_power_laws_near_naked_singularity() {
        // Radial: R_trtr ∝ r⁻² (static, boost-invariant). Transverse: the
        // static term ∝ r⁻² is multiplied by γ² = E²/A ∝ r⁻α ⇒ r^-(α+2).
        for &rb in &[2.5, 4.0] {
            let j = Jmn1::new(1.0, rb);
            let e = j.a(10.0).sqrt();
            let slope = |f: &dyn Fn(f64) -> f64| {
                let (r1, r2) = (1e-4, 1e-3);
                (f(r2).abs().ln() - f(r1).abs().ln()) / (r2.ln() - r1.ln())
            };
            let radial = slope(&|r| infall_stretch(&j, e, r).0);
            let trans = slope(&|r| infall_stretch(&j, e, r).1);
            assert!(
                (radial + 2.0).abs() < 1e-9,
                "R_b={rb}: radial slope {radial}"
            );
            assert!(
                (trans + j.alpha() + 2.0).abs() < 1e-3,
                "R_b={rb}: transverse slope {trans}"
            );
        }
    }
}
