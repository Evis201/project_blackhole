//! Static, spherically symmetric metrics `ds² = -A dt² + B dr² + r² dΩ²`.

/// What absorbs a ray that falls below [`SphericalMetric::capture_radius`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Capture {
    /// Event horizon (ray never returns to infinity).
    Horizon,
    /// Central curvature singularity (naked, for JMN-1).
    Singularity,
}

/// A static, spherically symmetric spacetime described by `A(r)` and `B(r)`.
///
/// Implementors must provide analytic radial derivatives; the geodesic
/// equations use them directly.
pub trait SphericalMetric: Sync {
    /// `-g_tt`
    fn a(&self, r: f64) -> f64;
    /// `dA/dr`
    fn da(&self, r: f64) -> f64;
    /// `g_rr`
    fn b(&self, r: f64) -> f64;
    /// `dB/dr`
    fn db(&self, r: f64) -> f64;
    /// `d²A/dr²` (curvature needs it). Default: central difference of `A'`.
    fn d2a(&self, r: f64) -> f64 {
        let h = 1e-5 * r;
        (self.da(r + h) - self.da(r - h)) / (2.0 * h)
    }

    /// Rays with `r` below this radius are terminated as captured.
    fn capture_radius(&self) -> f64;
    fn capture_kind(&self) -> Capture;
    fn name(&self) -> String;

    /// Angular velocity `Ω = dφ/dt` and `u^t` of a circular timelike geodesic
    /// at radius `r`, or `None` if no such orbit exists there.
    ///
    /// For a static spherical metric, `Ω² = A'/(2r)` and
    /// `u^t = 1/sqrt(A - r²Ω²)`. Stability is not checked.
    fn circular_orbit(&self, r: f64) -> Option<(f64, f64)> {
        let da = self.da(r);
        if da <= 0.0 {
            return None;
        }
        let omega = (da / (2.0 * r)).sqrt();
        let denom = self.a(r) - r * r * omega * omega;
        if denom <= 0.0 {
            return None;
        }
        Some((omega, 1.0 / denom.sqrt()))
    }
}

/// Schwarzschild black hole of mass `m`.
#[derive(Clone, Copy, Debug)]
pub struct Schwarzschild {
    pub m: f64,
}

impl Schwarzschild {
    /// Rays are stopped this fraction outside the horizon. Any inward-moving
    /// ray inside the photon sphere (`r < 3m`) is captured anyway, so this
    /// only avoids integrating into the coordinate singularity at `r = 2m`.
    const HORIZON_MARGIN: f64 = 1e-2;

    pub fn new(m: f64) -> Self {
        assert!(m > 0.0, "mass must be positive");
        Self { m }
    }

    pub fn horizon(&self) -> f64 {
        2.0 * self.m
    }

    pub fn photon_sphere(&self) -> f64 {
        3.0 * self.m
    }

    /// Critical impact parameter `b_c = 3√3 m` (shadow radius seen from infinity).
    pub fn critical_impact_parameter(&self) -> f64 {
        3.0 * 3f64.sqrt() * self.m
    }
}

impl SphericalMetric for Schwarzschild {
    fn a(&self, r: f64) -> f64 {
        1.0 - 2.0 * self.m / r
    }
    fn da(&self, r: f64) -> f64 {
        2.0 * self.m / (r * r)
    }
    fn b(&self, r: f64) -> f64 {
        1.0 / self.a(r)
    }
    fn db(&self, r: f64) -> f64 {
        let a = self.a(r);
        -self.da(r) / (a * a)
    }
    fn d2a(&self, r: f64) -> f64 {
        -4.0 * self.m / (r * r * r)
    }
    fn capture_radius(&self) -> f64 {
        self.horizon() * (1.0 + Self::HORIZON_MARGIN)
    }
    fn capture_kind(&self) -> Capture {
        Capture::Horizon
    }
    fn name(&self) -> String {
        format!("Schwarzschild(M={})", self.m)
    }
}

/// Joshi–Malafarina–Narayan (JMN-1) naked singularity spacetime.
///
/// Interior (`r ≤ R_b`), with compactness `M₀ = 2m/R_b`:
///
/// ```text
/// A(r) = (1 - M₀) (r/R_b)^{M₀/(1-M₀)},   B(r) = 1/(1 - M₀)
/// ```
///
/// matched to exterior Schwarzschild of mass `m` at `R_b`. `A`, `A'` and `B`
/// are continuous across `R_b`; `B'` jumps (anisotropic-pressure shell-free
/// matching), which the adaptive integrator absorbs by step rejection.
///
/// Photon sphere: exists (at `3m`, in the exterior) only for `R_b < 3m`,
/// i.e. `M₀ > 2/3`.
#[derive(Clone, Copy, Debug)]
pub struct Jmn1 {
    pub m: f64,
    pub rb: f64,
    m0: f64,
    alpha: f64,
    exterior: Schwarzschild,
}

impl Jmn1 {
    /// Rays reaching this radius are treated as absorbed by the naked
    /// singularity. Must stay far below any turning point we care about.
    pub const SINGULARITY_CUTOFF: f64 = 1e-4;

    pub fn new(m: f64, rb: f64) -> Self {
        assert!(m > 0.0, "mass must be positive");
        assert!(rb > 2.0 * m, "JMN-1 requires R_b > 2M (no horizon)");
        let m0 = 2.0 * m / rb;
        Self {
            m,
            rb,
            m0,
            alpha: m0 / (1.0 - m0),
            exterior: Schwarzschild::new(m),
        }
    }

    /// Compactness `M₀ = 2M/R_b ∈ (0, 1)`.
    pub fn compactness(&self) -> f64 {
        self.m0
    }

    /// Exponent `α = M₀/(1-M₀)` of the interior `A(r) ∝ r^α`.
    ///
    /// `α > 2` ⇔ `M₀ > 2/3` ⇔ photon sphere exists. In that regime the
    /// interior has no radial turning points for null rays: anything that
    /// crosses `R_b` reaches the singularity.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn has_photon_sphere(&self) -> bool {
        self.rb < 3.0 * self.m
    }
}

impl SphericalMetric for Jmn1 {
    fn a(&self, r: f64) -> f64 {
        if r >= self.rb {
            self.exterior.a(r)
        } else {
            (1.0 - self.m0) * (r / self.rb).powf(self.alpha)
        }
    }
    fn da(&self, r: f64) -> f64 {
        if r >= self.rb {
            self.exterior.da(r)
        } else {
            self.alpha * self.a(r) / r
        }
    }
    fn b(&self, r: f64) -> f64 {
        if r >= self.rb {
            self.exterior.b(r)
        } else {
            1.0 / (1.0 - self.m0)
        }
    }
    fn db(&self, r: f64) -> f64 {
        if r >= self.rb {
            self.exterior.db(r)
        } else {
            0.0
        }
    }
    fn d2a(&self, r: f64) -> f64 {
        if r >= self.rb {
            self.exterior.d2a(r)
        } else {
            self.alpha * (self.alpha - 1.0) * self.a(r) / (r * r)
        }
    }
    fn capture_radius(&self) -> f64 {
        Self::SINGULARITY_CUTOFF * self.m
    }
    fn capture_kind(&self) -> Capture {
        Capture::Singularity
    }
    fn name(&self) -> String {
        format!("JMN-1(M={}, R_b={})", self.m, self.rb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    #[test]
    fn schwarzschild_derivatives_match_finite_differences() {
        let s = Schwarzschild::new(1.0);
        for &r in &[2.5, 3.0, 6.0, 40.0] {
            let h = 1e-6 * r;
            let fd_a = (s.a(r + h) - s.a(r - h)) / (2.0 * h);
            let fd_b = (s.b(r + h) - s.b(r - h)) / (2.0 * h);
            assert!(rel(s.da(r), fd_a) < 1e-7, "dA at r={r}");
            assert!(rel(s.db(r), fd_b) < 1e-7, "dB at r={r}");
        }
    }

    #[test]
    fn jmn1_interior_derivatives_match_finite_differences() {
        let j = Jmn1::new(1.0, 2.5);
        for &r in &[0.1, 1.0, 2.0, 2.4] {
            let h = 1e-6 * r;
            let fd_a = (j.a(r + h) - j.a(r - h)) / (2.0 * h);
            assert!(rel(j.da(r), fd_a) < 1e-6, "dA at r={r}");
        }
    }

    #[test]
    fn jmn1_matching_is_c1_in_a_and_c0_in_b() {
        for &rb in &[2.2, 2.5, 2.9, 4.0, 10.0] {
            let j = Jmn1::new(1.0, rb);
            let s = Schwarzschild::new(1.0);
            let inside = rb * (1.0 - 1e-12);
            assert!(rel(j.a(inside), s.a(rb)) < 1e-9, "A jump at R_b={rb}");
            assert!(rel(j.da(inside), s.da(rb)) < 1e-9, "A' jump at R_b={rb}");
            assert!(rel(j.b(inside), s.b(rb)) < 1e-9, "B jump at R_b={rb}");
        }
    }

    #[test]
    fn jmn1_photon_sphere_condition() {
        assert!(Jmn1::new(1.0, 2.5).has_photon_sphere());
        assert!(Jmn1::new(1.0, 2.5).alpha() > 2.0);
        assert!(!Jmn1::new(1.0, 4.0).has_photon_sphere());
        assert!(Jmn1::new(1.0, 4.0).alpha() < 2.0);
    }

    #[test]
    fn schwarzschild_circular_orbit_is_keplerian() {
        let s = Schwarzschild::new(1.0);
        let (omega, ut) = s.circular_orbit(6.0).unwrap();
        assert!(rel(omega, 6f64.powf(-1.5)) < 1e-12);
        // u^t = 1/sqrt(1 - 3M/r)
        assert!(rel(ut, 1.0 / 0.5f64.sqrt()) < 1e-12);
        // No timelike circular orbits inside the photon sphere.
        assert!(s.circular_orbit(2.9).is_none());
    }

    #[test]
    fn jmn1_interior_circular_orbits_exist_only_without_photon_sphere() {
        assert!(Jmn1::new(1.0, 4.0).circular_orbit(1.0).is_some());
        assert!(Jmn1::new(1.0, 2.5).circular_orbit(1.0).is_none());
    }
}
