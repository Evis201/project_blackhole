//! HorizonLab core: spacetime metrics and null-geodesic integration.
//!
//! Units: geometric (G = c = 1), with all lengths expressed in units of the
//! ADM mass `M` of the object (so `M = 1` is the usual choice).
//!
//! The MVP handles static, spherically symmetric spacetimes
//!
//! ```text
//! ds² = -A(r) dt² + B(r) dr² + r² dΩ²
//! ```
//!
//! For these, every null geodesic lies in a plane through the origin, so rays
//! are integrated in that plane as `(r, p_r, φ)`. This avoids the coordinate
//! singularity of spherical coordinates at the poles.
//!
//! Kerr ([`kerr`]) has no such reduction: rays are integrated in full
//! Boyer–Lindquist coordinates `(r, θ, φ, p_r, p_θ)` in Mino time.

pub mod geodesic;
pub mod integrator;
pub mod kerr;
pub mod metric;

pub use geodesic::{
    escape_angle, hamiltonian, initial_state, local_direction, refine_crossing, trace, State,
    Termination, TraceConfig, TraceResult,
};
pub use integrator::{Dopri5, StepError, Tolerance};
pub use kerr::{Kerr, KerrState};
pub use metric::{Capture, Jmn1, Schwarzschild, SphericalMetric};
