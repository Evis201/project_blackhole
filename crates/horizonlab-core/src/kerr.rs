//! Kerr spacetime in Boyer–Lindquist coordinates.
//!
//! Null geodesics are integrated in Mino time `σ` (`dλ = Σ dσ`), using the
//! separable "super-Hamiltonian" with `E = -p_t = 1` and `L = p_φ`:
//!
//! ```text
//! 𝓗 = Σ H = ½ ( Δ p_r² + p_θ² - P²/Δ + K² ),   P = r² + a² - aL,
//!                                              K = L/sinθ - a sinθ
//! ```
//!
//! Hamilton's equations for `𝓗` in `σ` (valid on the shell `𝓗 = 0`):
//!
//! ```text
//! dr/dσ   = Δ p_r
//! dθ/dσ   = p_θ
//! dφ/dσ   = aP/Δ + L/sin²θ - a
//! dp_r/dσ = -½ Δ' p_r² + 2rP/Δ - P² Δ'/(2Δ²)
//! dp_θ/dσ = K cosθ (L/sin²θ + a)
//! ```
//!
//! Mino time removes the `Σ` factors, so the system is regular everywhere
//! outside the horizon, including the ring-singularity-free equatorial plane.
//!
//! Kerr is not time-reversal symmetric (only under `t → -t, φ → -φ`), so a
//! camera ray must be the *received* photon integrated backward in time,
//! not a photon emitted by the camera. [`trace`] takes a time direction.

use crate::geodesic::{Termination, TraceConfig, TraceResult};
use crate::integrator::Dopri5;
use crate::metric::Capture;

/// `[r, θ, φ, p_r, p_θ]`
pub type KerrState = [f64; 5];

const R: usize = 0;
const TH: usize = 1;
const PR: usize = 3;
const PTH: usize = 4;

/// Kerr black hole of mass `m` and spin parameter `a = J/m` (`|a| < m`).
/// Spin along `+z` for `a > 0`.
#[derive(Clone, Copy, Debug)]
pub struct Kerr {
    pub m: f64,
    pub a: f64,
}

/// Zero-angular-momentum observer (ZAMO) frame quantities at `(r, θ)`.
#[derive(Clone, Copy, Debug)]
pub struct Zamo {
    /// Lapse `α = √(ΣΔ/𝒜)`.
    pub alpha: f64,
    /// Frame-dragging angular velocity `ω = 2mar/𝒜`.
    pub omega: f64,
    /// Cylindrical radius `ϖ = sinθ √(𝒜/Σ)`.
    pub varpi: f64,
    pub sigma: f64,
    pub delta: f64,
}

impl Kerr {
    /// Rays are stopped this fraction outside the outer horizon.
    const HORIZON_MARGIN: f64 = 1e-2;

    pub fn new(m: f64, a: f64) -> Self {
        assert!(m > 0.0, "mass must be positive");
        assert!(a.abs() < m, "need |a| < M (extremal Kerr not supported)");
        Self { m, a }
    }

    pub fn delta(&self, r: f64) -> f64 {
        r * r - 2.0 * self.m * r + self.a * self.a
    }

    pub fn sigma(&self, r: f64, theta: f64) -> f64 {
        let c = theta.cos();
        r * r + self.a * self.a * c * c
    }

    /// Outer horizon `r₊ = m + √(m² - a²)`.
    pub fn horizon(&self) -> f64 {
        self.m + (self.m * self.m - self.a * self.a).sqrt()
    }

    pub fn capture_radius(&self) -> f64 {
        self.horizon() * (1.0 + Self::HORIZON_MARGIN)
    }

    /// Equatorial ISCO (Bardeen, Press & Teukolsky 1972).
    pub fn isco(&self, prograde: bool) -> f64 {
        let m = self.m;
        let chi = self.a / m;
        let z1 = 1.0 + (1.0 - chi * chi).cbrt() * ((1.0 + chi).cbrt() + (1.0 - chi).cbrt());
        let z2 = (3.0 * chi * chi + z1 * z1).sqrt();
        let root = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt();
        m * if prograde {
            3.0 + z2 - root
        } else {
            3.0 + z2 + root
        }
    }

    /// `|L/E|` of the equatorial circular photon orbit.
    ///
    /// Prograde: `b = -a + 6m cos(⅓ arccos(-a/m))`;
    /// retrograde: `b = a + 6m cos(⅓ arccos(a/m))`.
    pub fn equatorial_critical_impact(&self, prograde: bool) -> f64 {
        let (m, a) = (self.m, self.a);
        if prograde {
            -a + 6.0 * m * ((-a / m).acos() / 3.0).cos()
        } else {
            a + 6.0 * m * ((a / m).acos() / 3.0).cos()
        }
    }

    /// Prograde equatorial circular geodesic at `r`: `(Ω, u^t)`, or `None`
    /// inside the prograde photon orbit. Stability is not checked.
    pub fn circular_orbit(&self, r: f64) -> Option<(f64, f64)> {
        let (m, a) = (self.m, self.a);
        let sm = m.sqrt();
        let r32 = r * r.sqrt();
        let radicand = r32 - 3.0 * m * r.sqrt() + 2.0 * a * sm;
        if radicand <= 0.0 {
            return None;
        }
        let omega = sm / (r32 + a * sm);
        let ut = (r32 + a * sm) / (r.powf(0.75) * radicand.sqrt());
        Some((omega, ut))
    }

    pub fn zamo(&self, r: f64, theta: f64) -> Zamo {
        let (m, a) = (self.m, self.a);
        let s = theta.sin();
        let sigma = self.sigma(r, theta);
        let delta = self.delta(r);
        let r2a2 = r * r + a * a;
        let big_a = r2a2 * r2a2 - a * a * delta * s * s;
        Zamo {
            alpha: (sigma * delta / big_a).sqrt(),
            omega: 2.0 * m * a * r / big_a,
            varpi: s.abs() * (big_a / sigma).sqrt(),
            sigma,
            delta,
        }
    }

    pub fn name(&self) -> String {
        format!("Kerr(M={}, a={})", self.m, self.a)
    }
}

#[inline]
fn rhs(k: &Kerr, l: f64, y: &KerrState) -> KerrState {
    let (m, a) = (k.m, k.a);
    let r = y[R];
    let (s, c) = y[TH].sin_cos();
    let pr = y[PR];
    let delta = k.delta(r);
    let ddelta = 2.0 * r - 2.0 * m;
    let p = r * r + a * a - a * l;
    let kk = l / s - a * s;
    let l_s2 = l / (s * s);
    [
        delta * pr,
        y[PTH],
        a * p / delta + l_s2 - a,
        -0.5 * ddelta * pr * pr + 2.0 * r * p / delta - p * p * ddelta / (2.0 * delta * delta),
        kk * c * (l_s2 + a),
    ]
}

/// Hamiltonian `H = 𝓗/Σ` (same normalisation as the spherical code); zero on
/// a null geodesic with `E = 1`.
pub fn hamiltonian(k: &Kerr, l: f64, y: &KerrState) -> f64 {
    let r = y[R];
    let s = y[TH].sin();
    let delta = k.delta(r);
    let p = r * r + k.a * k.a - k.a * l;
    let kk = l / s - k.a * s;
    0.5 * (delta * y[PR] * y[PR] + y[PTH] * y[PTH] - p * p / delta + kk * kk) / k.sigma(r, y[TH])
}

/// Photon with `E = 1` at `(r, θ, φ)` moving in the ZAMO-frame direction
/// `n = (n_r, n_θ, n_φ)` (unit; components along `∂_r, ∂_θ, ∂_φ`).
/// Returns `(state, L)`.
pub fn initial_state(k: &Kerr, r: f64, theta: f64, phi: f64, n: [f64; 3]) -> (KerrState, f64) {
    let z = k.zamo(r, theta);
    let e_loc = 1.0 / (z.alpha + z.omega * z.varpi * n[2]);
    let pr = e_loc * n[0] * (z.sigma / z.delta).sqrt();
    let pth = e_loc * n[1] * z.sigma.sqrt();
    let l = e_loc * n[2] * z.varpi;
    ([r, theta, phi, pr, pth], l)
}

/// Unit propagation direction `(n_r, n_θ, n_φ)` in the local ZAMO frame.
pub fn local_direction(k: &Kerr, l: f64, y: &KerrState) -> [f64; 3] {
    let z = k.zamo(y[R], y[TH]);
    let e_loc = (1.0 - z.omega * l) / z.alpha;
    let n = [
        y[PR] / (e_loc * (z.sigma / z.delta).sqrt()),
        y[PTH] / (e_loc * z.sigma.sqrt()),
        l / (e_loc * z.varpi),
    ];
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|v| v / norm)
}

/// Photon energy measured by the ZAMO at the photon's position (`E = 1`).
pub fn zamo_energy(k: &Kerr, l: f64, y: &KerrState) -> f64 {
    let z = k.zamo(y[R], y[TH]);
    (1.0 - z.omega * l) / z.alpha
}

/// Integrate a null geodesic. `time_sign = +1` follows the photon forward;
/// `-1` integrates backward in time (use this for camera rays, starting
/// from the *received* photon's momentum). Escape is declared once `r`
/// exceeds `cfg.r_escape` while still increasing along the integration.
///
/// `on_step(prev, cur, |Δσ|)` is called after every accepted step.
pub fn trace<T, F>(
    k: &Kerr,
    state: KerrState,
    l: f64,
    time_sign: f64,
    cfg: &TraceConfig,
    mut on_step: F,
) -> TraceResult<T, KerrState>
where
    F: FnMut(&KerrState, &KerrState, f64) -> Option<T>,
{
    let solver = Dopri5::new(cfg.tol);
    let f = |y: &KerrState| rhs(k, l, y).map(|v| time_sign * v);
    let r_capture = k.capture_radius();
    // dλ ≈ (r² + a²) dσ: cap σ-steps so the affine step stays ≤ frac·r.
    let h_cap = |r: f64| cfg.max_step_frac * r / (r * r + k.a * k.a);

    let mut y = state;
    let mut h = 1e-2 * h_cap(y[R]);
    let mut max_constraint = hamiltonian(k, l, &y).abs();
    let result = |termination, state, event, steps, max_constraint| TraceResult {
        termination,
        state,
        event,
        steps,
        max_constraint,
    };

    for steps in 0..cfg.max_steps {
        let (y_new, taken, h_next) = match solver.adaptive_step(&f, &y, h, h_cap(y[R])) {
            Ok(s) => s,
            Err(e) => return result(Termination::Failed(e), y, None, steps, max_constraint),
        };
        let prev = y;
        y = y_new;
        h = h_next;

        if y[R] <= r_capture || !y[R].is_finite() {
            return result(
                Termination::Captured(Capture::Horizon),
                y,
                None,
                steps + 1,
                max_constraint,
            );
        }
        max_constraint = max_constraint.max(hamiltonian(k, l, &y).abs());

        if let Some(event) = on_step(&prev, &y, taken) {
            return result(
                Termination::Event,
                y,
                Some(event),
                steps + 1,
                max_constraint,
            );
        }
        if y[R] >= cfg.r_escape && time_sign * y[PR] > 0.0 {
            return result(Termination::Escaped, y, None, steps + 1, max_constraint);
        }
    }
    result(
        Termination::MaxSteps,
        y,
        None,
        cfg.max_steps,
        max_constraint,
    )
}

/// State where `g` changes sign inside the accepted step `prev → |Δσ|`
/// (as reported to `on_step`), integrating with the same `time_sign`.
pub fn refine_crossing<G>(
    k: &Kerr,
    l: f64,
    time_sign: f64,
    prev: &KerrState,
    dsigma: f64,
    g: G,
) -> KerrState
where
    G: Fn(&KerrState) -> f64,
{
    let f = |y: &KerrState| rhs(k, l, y).map(|v| time_sign * v);
    Dopri5::default().bisect_event(&f, prev, dsigma, g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isco_matches_known_values() {
        assert!((Kerr::new(1.0, 0.0).isco(true) - 6.0).abs() < 1e-12);
        assert!((Kerr::new(1.0, 0.0).isco(false) - 6.0).abs() < 1e-12);
        // a = 0.9: prograde 2.3209, retrograde 8.7174.
        let k = Kerr::new(1.0, 0.9);
        assert!((k.isco(true) - 2.3209).abs() < 1e-4, "{}", k.isco(true));
        assert!((k.isco(false) - 8.7174).abs() < 1e-4, "{}", k.isco(false));
    }

    #[test]
    fn photon_orbit_impact_limits() {
        let s = 3.0 * 3f64.sqrt();
        let k0 = Kerr::new(1.0, 0.0);
        assert!((k0.equatorial_critical_impact(true) - s).abs() < 1e-12);
        assert!((k0.equatorial_critical_impact(false) - s).abs() < 1e-12);
        // Near-extremal: prograde → 2M, retrograde → 7M.
        let k = Kerr::new(1.0, 1.0 - 1e-12);
        assert!((k.equatorial_critical_impact(true) - 2.0).abs() < 1e-3);
        assert!((k.equatorial_critical_impact(false) - 7.0).abs() < 1e-3);
    }

    #[test]
    fn circular_orbit_reduces_to_schwarzschild() {
        let (omega, ut) = Kerr::new(1.0, 0.0).circular_orbit(6.0).unwrap();
        assert!((omega - 6f64.powf(-1.5)).abs() < 1e-12);
        assert!((ut - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn initial_state_is_null() {
        let k = Kerr::new(1.0, 0.7);
        for &(r, th) in &[(50.0, 1.0), (4.0, 0.3), (2.5, 1.5)] {
            let n = [0.3f64, -0.5, 0.6];
            let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            let n = n.map(|v| v / norm);
            let (y, l) = initial_state(&k, r, th, 0.0, n);
            assert!(hamiltonian(&k, l, &y).abs() < 1e-10, "r={r} θ={th}");
            let back = local_direction(&k, l, &y);
            for i in 0..3 {
                assert!((back[i] - n[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn zamo_frame_dragging_vanishes_without_spin() {
        let z = Kerr::new(1.0, 0.0).zamo(10.0, 1.0);
        assert_eq!(z.omega, 0.0);
        assert!((z.alpha - (1.0f64 - 0.2).sqrt()).abs() < 1e-12);
    }
}
