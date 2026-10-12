//! Novikov–Thorne thin-disk flux for any stationary, axisymmetric spacetime
//! with equatorial circular geodesics.
//!
//! Page & Thorne (1974): with accretion rate `Ṁ = 1`, the flux emitted from
//! each face of the disk, measured in the comoving frame, is
//!
//! ```text
//! F(r) = - Ω' / (4π √(-g) (E - ΩL)²) · ∫_{r_in}^{r} (E - ΩL) L' dr
//! ```
//!
//! where `E`, `L`, `Ω` are the specific energy, angular momentum and angular
//! velocity of circular orbits and `√(-g)` is the determinant of the
//! equatorial `(t, r, φ)` metric. The inner edge has zero torque (`F = 0`).
//!
//! Energy conservation gives the radiative efficiency identity
//! `∫ 4π √(-g) E F dr = 1 - E(r_in)` (both faces, Ṁ = 1), used as a test.

use crate::kerr::Kerr;
use crate::metric::SphericalMetric;

/// Equatorial circular orbits, counter-clockwise about `+z`.
pub trait EquatorialOrbits {
    /// `(E, L, Ω)` of the circular geodesic at `r`, or `None` if none exists.
    fn orbit(&self, r: f64) -> Option<(f64, f64, f64)>;
    /// `√(-g)` of the equatorial `(t, r, φ)` metric.
    fn sqrt_g(&self, r: f64) -> f64;
}

/// Adapter for static spherical metrics: `E = A u^t`, `L = r² Ω u^t`,
/// `√(-g) = r √(AB)`.
pub struct SphericalOrbits<'a>(pub &'a dyn SphericalMetric);

impl EquatorialOrbits for SphericalOrbits<'_> {
    fn orbit(&self, r: f64) -> Option<(f64, f64, f64)> {
        let (omega, ut) = self.0.circular_orbit(r)?;
        Some((self.0.a(r) * ut, r * r * omega * ut, omega))
    }
    fn sqrt_g(&self, r: f64) -> f64 {
        r * (self.0.a(r) * self.0.b(r)).sqrt()
    }
}

impl EquatorialOrbits for Kerr {
    /// Bardeen, Press & Teukolsky (1972), with signed `a` (`a < 0`: the
    /// counter-clockwise orbit is retrograde with respect to the spin).
    fn orbit(&self, r: f64) -> Option<(f64, f64, f64)> {
        let (m, a) = (self.m, self.a);
        let (sm, sr) = (m.sqrt(), r.sqrt());
        let r32 = r * sr;
        let radicand = r32 - 3.0 * m * sr + 2.0 * a * sm;
        if radicand <= 0.0 {
            return None;
        }
        let denom = r.powf(0.75) * radicand.sqrt();
        let e = (r32 - 2.0 * m * sr + a * sm) / denom;
        let l = sm * (r * r - 2.0 * a * sm * sr + a * a) / denom;
        let omega = sm / (r32 + a * sm);
        Some((e, l, omega))
    }
    fn sqrt_g(&self, r: f64) -> f64 {
        r
    }
}

/// Tabulated Novikov–Thorne flux (Ṁ = 1, per face).
#[derive(Clone, Debug)]
pub struct NovikovThorne {
    pub r_in: f64,
    pub r_out: f64,
    /// Specific energy at the inner edge; efficiency is `1 - e_in`.
    pub e_in: f64,
    ln_r: Vec<f64>,
    flux: Vec<f64>,
}

impl NovikovThorne {
    const SAMPLES: usize = 4000;

    /// Build the table on a logarithmic grid. Fails if there is no circular
    /// orbit at `r_in`. Where `L' < 0` (unstable orbits, inside the ISCO) the
    /// model is unphysical and the flux is clamped at zero.
    pub fn new(orbits: &dyn EquatorialOrbits, r_in: f64, r_out: f64) -> Result<Self, String> {
        if !(r_in > 0.0 && r_out > r_in) {
            return Err(format!("need 0 < r_in < r_out (got {r_in}, {r_out})"));
        }
        let (e_in, _, _) = orbits
            .orbit(r_in)
            .ok_or_else(|| format!("no circular orbit at r_in = {r_in}"))?;

        let n = Self::SAMPLES;
        let ln_r: Vec<f64> = (0..=n)
            .map(|i| r_in.ln() + (r_out / r_in).ln() * i as f64 / n as f64)
            .collect();
        // Derivatives by central differences; the one-sided step at r_in
        // keeps every sample on an existing orbit.
        let deriv = |r: f64| -> Option<(f64, f64, f64, f64, f64)> {
            let (e, l, om) = orbits.orbit(r)?;
            let h = 1e-6 * r;
            let (lo, hi) = (orbits.orbit(r - h), orbits.orbit(r + h)?);
            let (dl, dom) = match lo {
                Some((_, l0, om0)) => ((hi.1 - l0) / (2.0 * h), (hi.2 - om0) / (2.0 * h)),
                None => ((hi.1 - l) / h, (hi.2 - om) / h),
            };
            Some((e, l, om, dl, dom))
        };

        let mut flux = Vec::with_capacity(n + 1);
        let mut integral = 0.0;
        let mut prev: Option<(f64, f64)> = None; // (r, integrand)
        for &lr in &ln_r {
            let r = lr.exp();
            let (e, l, om, dl, dom) =
                deriv(r).ok_or_else(|| format!("no circular orbit at r = {r}"))?;
            let integrand = (e - om * l) * dl;
            if let Some((r0, f0)) = prev {
                integral += 0.5 * (r - r0) * (f0 + integrand);
            }
            prev = Some((r, integrand));
            let f = -dom / (4.0 * std::f64::consts::PI * orbits.sqrt_g(r) * (e - om * l).powi(2))
                * integral;
            flux.push(f.max(0.0));
        }
        Ok(Self {
            r_in,
            r_out,
            e_in,
            ln_r,
            flux,
        })
    }

    /// Flux at `r` (linear interpolation in `ln r`); zero outside the disk.
    pub fn flux(&self, r: f64) -> f64 {
        if r < self.r_in || r > self.r_out {
            return 0.0;
        }
        let x = (r.ln() - self.ln_r[0]) / (self.ln_r[1] - self.ln_r[0]);
        let i = (x.floor() as usize).min(self.flux.len() - 2);
        let t = x - i as f64;
        self.flux[i] + t * (self.flux[i + 1] - self.flux[i])
    }

    pub fn peak(&self) -> f64 {
        self.flux.iter().cloned().fold(0.0, f64::max)
    }

    /// Radiative efficiency `1 - E(r_in)` (fraction of rest-mass energy
    /// radiated to infinity).
    pub fn efficiency(&self) -> f64 {
        1.0 - self.e_in
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::Schwarzschild;

    /// `∫ 4π √(-g) E F dr` over the tabulated disk (trapezoid in r).
    fn luminosity(nt: &NovikovThorne, orbits: &dyn EquatorialOrbits) -> f64 {
        let mut sum = 0.0;
        for w in nt.ln_r.windows(2).zip(nt.flux.windows(2)) {
            let ((l0, l1), (f0, f1)) = ((w.0[0], w.0[1]), (w.1[0], w.1[1]));
            let (r0, r1) = (l0.exp(), l1.exp());
            let g = |r: f64, f: f64| {
                4.0 * std::f64::consts::PI * orbits.sqrt_g(r) * orbits.orbit(r).unwrap().0 * f
            };
            sum += 0.5 * (r1 - r0) * (g(r0, f0) + g(r1, f1));
        }
        sum
    }

    #[test]
    fn kerr_without_spin_matches_schwarzschild_path() {
        let s = Schwarzschild::new(1.0);
        let a = NovikovThorne::new(&SphericalOrbits(&s), 6.0, 100.0).unwrap();
        let b = NovikovThorne::new(&Kerr::new(1.0, 0.0), 6.0, 100.0).unwrap();
        for &r in &[6.5, 8.0, 12.0, 50.0] {
            let (fa, fb) = (a.flux(r), b.flux(r));
            assert!((fa - fb).abs() < 1e-6 * fa, "r={r}: {fa} vs {fb}");
        }
    }

    #[test]
    fn radiated_energy_equals_binding_energy_at_isco() {
        // Schwarzschild: η = 1 - √(8/9) ≈ 0.0572; Kerr a = 0.9: η ≈ 0.1558.
        let s = Schwarzschild::new(1.0);
        let k = Kerr::new(1.0, 0.9);
        let cases: [(&dyn EquatorialOrbits, f64); 2] =
            [(&SphericalOrbits(&s), 6.0), (&k, k.isco(true))];
        for (orbits, r_in) in cases {
            let nt = NovikovThorne::new(orbits, r_in, 1e6).unwrap();
            let lum = luminosity(&nt, orbits);
            // The tail beyond r_out carries ≈ 3/(2 r_out) of luminosity.
            let err = (lum - nt.efficiency()).abs() / nt.efficiency();
            eprintln!(
                "r_in={r_in:.4}: L = {lum:.6}, η = {:.6}, rel err {err:.1e}",
                nt.efficiency()
            );
            assert!(
                err < 1e-4,
                "r_in={r_in}: L = {lum}, η = {}",
                nt.efficiency()
            );
        }
        assert!(
            (NovikovThorne::new(&SphericalOrbits(&s), 6.0, 10.0)
                .unwrap()
                .efficiency()
                - (1.0 - (8.0f64 / 9.0).sqrt()))
            .abs()
                < 1e-12
        );
        let eta_k = NovikovThorne::new(&k, k.isco(true), 10.0)
            .unwrap()
            .efficiency();
        assert!((eta_k - 0.1558).abs() < 1e-3, "η(a=0.9) = {eta_k}");
    }

    #[test]
    fn flux_vanishes_at_inner_edge_and_approaches_newtonian_far_out() {
        let s = Schwarzschild::new(1.0);
        let nt = NovikovThorne::new(&SphericalOrbits(&s), 6.0, 1e6).unwrap();
        assert_eq!(nt.flux(6.0), 0.0);
        assert!(nt.flux(6.1) > 0.0);
        // Far out: F → 3/(8π r³) (Newtonian, Ṁ = M = 1).
        let r = 1e5;
        let ratio = nt.flux(r) * 8.0 * std::f64::consts::PI * r.powi(3) / 3.0;
        eprintln!("Newtonian ratio at r = 1e5: {ratio:.5}");
        assert!((ratio - 1.0).abs() < 2e-2, "ratio {ratio}");
    }
}
