//! Planar null geodesics in static, spherically symmetric spacetimes.
//!
//! With `E = -p_t = 1` and `L = p_φ ≥ 0` (orbital plane chosen so the ray
//! moves in `+φ`), the Hamiltonian
//!
//! ```text
//! H = ½ ( -E²/A + p_r²/B + L²/r² ) = 0
//! ```
//!
//! gives, in affine parameter λ,
//!
//! ```text
//! dr/dλ   =  p_r / B
//! dp_r/dλ = -½ ( E² A'/A² - p_r² B'/B² - 2 L²/r³ )
//! dφ/dλ   =  L / r²
//! ```
//!
//! Integrating `p_r` instead of `(dr/dλ)²` handles radial turning points
//! without sign bookkeeping. `|H|` is a free accuracy monitor.

use crate::integrator::{Dopri5, StepError, Tolerance};
use crate::metric::{Capture, SphericalMetric};

/// `[r, p_r, φ]`
pub type State = [f64; 3];

const R: usize = 0;
const PR: usize = 1;
const PHI: usize = 2;

#[inline]
fn rhs<M: SphericalMetric + ?Sized>(metric: &M, l: f64, y: &State) -> State {
    let r = y[R];
    let pr = y[PR];
    let a = metric.a(r);
    let b = metric.b(r);
    let da = metric.da(r);
    let db = metric.db(r);
    let l2 = l * l;
    [
        pr / b,
        -0.5 * (da / (a * a) - pr * pr * db / (b * b) - 2.0 * l2 / (r * r * r)),
        l / (r * r),
    ]
}

/// Null constraint `H`; zero for an exact null geodesic with `E = 1`.
pub fn hamiltonian<M: SphericalMetric + ?Sized>(metric: &M, l: f64, y: &State) -> f64 {
    let r = y[R];
    0.5 * (-1.0 / metric.a(r) + y[PR] * y[PR] / metric.b(r) + l * l / (r * r))
}

/// Build the initial state for a ray leaving a static observer at radius `r`
/// in the local orthonormal direction with radial cosine `n_r` and
/// tangential component `n_t ≥ 0` (`n_r² + n_t² = 1`). Returns `(state, L)`
/// with `φ = 0`.
///
/// Static-frame components: `n_r = √(AB) dr/dλ`, `n_t = √A L / r` (for E=1).
pub fn initial_state<M: SphericalMetric + ?Sized>(
    metric: &M,
    r: f64,
    n_r: f64,
    n_t: f64,
) -> (State, f64) {
    let a = metric.a(r);
    let b = metric.b(r);
    let l = r * n_t / a.sqrt();
    let pr = (b / a).sqrt() * n_r;
    ([r, pr, 0.0], l)
}

/// Direction cosines `(n_r, n_t)` of the ray in the static observer frame at
/// the ray's current position.
pub fn local_direction<M: SphericalMetric + ?Sized>(metric: &M, l: f64, y: &State) -> (f64, f64) {
    let r = y[R];
    let a = metric.a(r);
    let b = metric.b(r);
    let n_r = (a / b).sqrt() * y[PR];
    let n_t = a.sqrt() * l / r;
    let norm = (n_r * n_r + n_t * n_t).sqrt();
    (n_r / norm, n_t / norm)
}

/// In-plane angle of the ray's propagation direction (measured from the
/// plane's `φ = 0` axis). For an escaped ray this is the asymptotic
/// direction on the celestial sphere, up to `O(M/r)` residual bending.
pub fn escape_angle<M: SphericalMetric + ?Sized>(metric: &M, l: f64, y: &State) -> f64 {
    let (n_r, n_t) = local_direction(metric, l, y);
    y[PHI] + n_t.atan2(n_r)
}

#[derive(Clone, Copy, Debug)]
pub struct TraceConfig {
    pub tol: Tolerance,
    /// Outgoing rays beyond this radius count as escaped.
    pub r_escape: f64,
    pub max_steps: usize,
    /// Cap on the affine step, as a fraction of the current `r`. Keeps the
    /// per-step `Δφ` small so callers can interpolate events between steps.
    pub max_step_frac: f64,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            tol: Tolerance::default(),
            r_escape: 1000.0,
            max_steps: 200_000,
            max_step_frac: 0.05,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Termination {
    Escaped,
    Captured(Capture),
    /// Stopped by the caller's step callback.
    Event,
    MaxSteps,
    Failed(StepError),
}

#[derive(Clone, Copy, Debug)]
pub struct TraceResult<T> {
    pub termination: Termination,
    pub state: State,
    pub event: Option<T>,
    pub steps: usize,
    /// Largest `|H|` seen at accepted steps.
    pub max_constraint: f64,
}

/// Integrate a null ray until it escapes, is captured, or `on_step` returns
/// `Some(event)`. `on_step(prev, cur)` is called after every accepted step.
pub fn trace<M, T, F>(
    metric: &M,
    state: State,
    l: f64,
    cfg: &TraceConfig,
    mut on_step: F,
) -> TraceResult<T>
where
    M: SphericalMetric + ?Sized,
    F: FnMut(&State, &State) -> Option<T>,
{
    let solver = Dopri5::new(cfg.tol);
    let f = |y: &State| rhs(metric, l, y);
    let r_capture = metric.capture_radius();

    let mut y = state;
    let mut h = 1e-2 * y[R];
    let mut max_constraint = hamiltonian(metric, l, &y).abs();
    let finish = |termination, y, event, steps, max_constraint| TraceResult {
        termination,
        state: y,
        event,
        steps,
        max_constraint,
    };

    for steps in 0..cfg.max_steps {
        let h_max = cfg.max_step_frac * y[R].max(r_capture);
        let (y_new, _taken, h_next) = match solver.adaptive_step(&f, &y, h, h_max) {
            Ok(s) => s,
            Err(e) => {
                // Steps collapse when a ray dives into the singular core;
                // that is a capture, not a numerical failure.
                if y[PR] < 0.0 && y[R] < 2.0 * r_capture.max(1e-3) {
                    return finish(
                        Termination::Captured(metric.capture_kind()),
                        y,
                        None,
                        steps,
                        max_constraint,
                    );
                }
                return finish(Termination::Failed(e), y, None, steps, max_constraint);
            }
        };
        let prev = y;
        y = y_new;
        h = h_next;

        if y[R] <= r_capture || !y[R].is_finite() || y[R] <= 0.0 {
            return finish(
                Termination::Captured(metric.capture_kind()),
                y,
                None,
                steps + 1,
                max_constraint,
            );
        }
        max_constraint = max_constraint.max(hamiltonian(metric, l, &y).abs());

        if let Some(event) = on_step(&prev, &y) {
            return finish(
                Termination::Event,
                y,
                Some(event),
                steps + 1,
                max_constraint,
            );
        }
        if y[R] >= cfg.r_escape && y[PR] > 0.0 {
            return finish(Termination::Escaped, y, None, steps + 1, max_constraint);
        }
    }
    finish(
        Termination::MaxSteps,
        y,
        None,
        cfg.max_steps,
        max_constraint,
    )
}
