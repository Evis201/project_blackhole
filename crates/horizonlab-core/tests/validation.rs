//! Physics validation against analytic results.

use horizonlab_core::{
    escape_angle, initial_state, trace, Capture, Jmn1, Schwarzschild, SphericalMetric, Termination,
    Tolerance, TraceConfig, TraceResult,
};

/// Shoot a ray inward from radius `r0` with impact parameter `b` (E = 1,
/// so `L = b`). Returns the trace and the in-plane deflection angle.
fn shoot<M: SphericalMetric>(
    metric: &M,
    r0: f64,
    b: f64,
    cfg: &TraceConfig,
) -> (TraceResult<()>, f64) {
    let n_t = b * metric.a(r0).sqrt() / r0;
    let n_r = -(1.0 - n_t * n_t).sqrt();
    let (state, l) = initial_state(metric, r0, n_r, n_t);
    assert!((l - b).abs() < 1e-9 * b.max(1.0));
    let psi0 = escape_angle(metric, l, &state);
    let res = trace(metric, state, l, cfg, |_, _| None::<()>);
    let deflection = escape_angle(metric, l, &res.state) - psi0;
    (res, deflection)
}

fn precise(r_escape: f64) -> TraceConfig {
    TraceConfig {
        tol: Tolerance {
            rtol: 1e-12,
            atol: 1e-14,
        },
        r_escape,
        ..TraceConfig::default()
    }
}

#[test]
fn schwarzschild_weak_deflection_matches_post_newtonian_series() {
    let s = Schwarzschild::new(1.0);
    let r0 = 1e7;
    for &b in &[200.0, 1000.0] {
        let (res, alpha) = shoot(&s, r0, b, &precise(r0));
        assert_eq!(res.termination, Termination::Escaped);
        let pi = std::f64::consts::PI;
        let expected = 4.0 / b
            + 15.0 * pi / (4.0 * b * b)
            + 128.0 / (3.0 * b * b * b)
            + 3465.0 * pi / (64.0 * b.powi(4));
        let err = (alpha - expected).abs() / expected;
        eprintln!("b={b}: deflection {alpha:.12e}, series {expected:.12e}, rel err {err:.2e}");
        // Residual is dominated by the truncated O(b⁻⁵) term.
        assert!(
            err < 1e-6,
            "b={b}: got {alpha}, expected {expected}, rel err {err}"
        );
    }
}

#[test]
fn schwarzschild_critical_impact_parameter() {
    let s = Schwarzschild::new(1.0);
    let bc = s.critical_impact_parameter();
    let r0 = 1e4;
    let cfg = precise(r0);
    let (inside, _) = shoot(&s, r0, bc * (1.0 - 1e-4), &cfg);
    let (outside, _) = shoot(&s, r0, bc * (1.0 + 1e-4), &cfg);
    assert_eq!(inside.termination, Termination::Captured(Capture::Horizon));
    assert_eq!(outside.termination, Termination::Escaped);
}

#[test]
fn null_constraint_is_preserved() {
    let s = Schwarzschild::new(1.0);
    let (res, _) = shoot(&s, 100.0, 6.0, &TraceConfig::default());
    assert_eq!(res.termination, Termination::Escaped);
    assert!(
        res.max_constraint < 1e-8,
        "|H| max = {}",
        res.max_constraint
    );
}

#[test]
fn jmn1_with_photon_sphere_is_schwarzschild_outside_rb() {
    let s = Schwarzschild::new(1.0);
    let j = Jmn1::new(1.0, 2.5);
    let cfg = TraceConfig::default();
    // b = 6 > b_c ⇒ periapsis ≈ 4.1M > R_b: ray never sees the interior.
    let (rs, ds) = shoot(&s, 50.0, 6.0, &cfg);
    let (rj, dj) = shoot(&j, 50.0, 6.0, &cfg);
    assert_eq!(rs.termination, Termination::Escaped);
    assert_eq!(rj.termination, Termination::Escaped);
    assert!((ds - dj).abs() < 1e-12, "deflections differ: {ds} vs {dj}");
}

#[test]
fn jmn1_with_photon_sphere_has_same_critical_curve() {
    let j = Jmn1::new(1.0, 2.5);
    let bc = Schwarzschild::new(1.0).critical_impact_parameter();
    let r0 = 1e4;
    let cfg = precise(r0);
    let (inside, _) = shoot(&j, r0, bc * (1.0 - 1e-4), &cfg);
    let (outside, _) = shoot(&j, r0, bc * (1.0 + 1e-4), &cfg);
    // Inside b_c the ray crosses R_b and, since α > 2, has no interior
    // turning point: it ends on the naked singularity.
    assert_eq!(
        inside.termination,
        Termination::Captured(Capture::Singularity)
    );
    assert_eq!(outside.termination, Termination::Escaped);
}

#[test]
fn jmn1_without_photon_sphere_is_transparent() {
    // R_b = 4M ⇒ M₀ = 1/2, α = 1 < 2: every ray with L > 0 has an interior
    // turning point, passes through the object and escapes.
    let j = Jmn1::new(1.0, 4.0);
    let cfg = TraceConfig::default();
    for &b in &[0.3, 1.0, 3.0, 5.0] {
        let (res, _) = shoot(&j, 50.0, b, &cfg);
        assert_eq!(res.termination, Termination::Escaped, "b={b}");
        assert!(
            res.max_constraint < 1e-6,
            "b={b}: |H| max = {}",
            res.max_constraint
        );
    }
}
