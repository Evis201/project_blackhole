//! Kerr geodesic validation.

use horizonlab_core::kerr::{self, Kerr, KerrState};
use horizonlab_core::{
    escape_angle, initial_state as spherical_initial_state, trace as spherical_trace, Capture,
    Schwarzschild, SphericalMetric, Termination, Tolerance, TraceConfig,
};

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

/// Cartesian direction of a ZAMO-frame vector at `(θ, φ)`.
fn to_cartesian(theta: f64, phi: f64, n: [f64; 3]) -> [f64; 3] {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    let er = [st * cp, st * sp, ct];
    let et = [ct * cp, ct * sp, -st];
    let ep = [-sp, cp, 0.0];
    std::array::from_fn(|i| n[0] * er[i] + n[1] * et[i] + n[2] * ep[i])
}

/// Photon moving inward from `r0` on the equator with impact parameter `b`,
/// its orbital plane tilted by `tilt` from the equator. Integrated forward.
fn shoot(k: &Kerr, r0: f64, b: f64, tilt: f64, cfg: &TraceConfig) -> (Termination, f64, f64) {
    let z = k.zamo(r0, std::f64::consts::FRAC_PI_2);
    // Far away the ZAMO frame is nearly static, so n_t ≈ b α / r0.
    let n_t = b * z.alpha / r0;
    let n = [
        -(1.0 - n_t * n_t).sqrt(),
        -n_t * tilt.sin(),
        n_t * tilt.cos(),
    ];
    let (y0, l) = kerr::initial_state(k, r0, std::f64::consts::FRAC_PI_2, 0.0, n);
    let d0 = to_cartesian(y0[1], y0[2], kerr::local_direction(k, l, &y0));
    let res = kerr::trace(k, y0, l, 1.0, cfg, |_: &KerrState, _: &KerrState, _| {
        None::<()>
    });
    let y = res.state;
    let d1 = to_cartesian(y[1], y[2], kerr::local_direction(k, l, &y));
    let dot: f64 = (0..3).map(|i| d0[i] * d1[i]).sum();
    (
        res.termination,
        dot.clamp(-1.0, 1.0).acos(),
        res.max_constraint,
    )
}

#[test]
fn zero_spin_matches_schwarzschild_deflection_in_tilted_plane() {
    // The tilted plane exercises θ dynamics; at a = 0 the answer must not
    // depend on the tilt and must equal the planar Schwarzschild result.
    let k = Kerr::new(1.0, 0.0);
    let s = Schwarzschild::new(1.0);
    let r0 = 1e4;
    let cfg = precise(r0);
    for &b in &[6.0, 10.0, 50.0] {
        let n_t = b * s.a(r0).sqrt() / r0;
        let (st, l) = spherical_initial_state(&s, r0, -(1.0 - n_t * n_t).sqrt(), n_t);
        let psi0 = escape_angle(&s, l, &st);
        let res = spherical_trace(&s, st, l, &cfg, |_, _, _| None::<()>);
        let expected = escape_angle(&s, l, &res.state) - psi0;

        for &tilt in &[0.0, 0.7, 1.4] {
            let (term, defl, h) = shoot(&k, r0, b, tilt, &cfg);
            assert_eq!(term, Termination::Escaped);
            // Deflection > π possible for b near b_c; compare mod 2π via cos.
            let err = (defl.cos() - expected.cos()).abs();
            assert!(err < 1e-8, "b={b} tilt={tilt}: {defl} vs {expected}");
            assert!(h < 1e-8, "constraint {h}");
        }
    }
}

#[test]
fn equatorial_critical_impact_parameters_with_spin() {
    let k = Kerr::new(1.0, 0.9);
    let r0 = 1e4;
    let cfg = precise(r0);
    for prograde in [true, false] {
        let bc = k.equatorial_critical_impact(prograde);
        // tilt 0: L > 0 (prograde), tilt π: L < 0 (retrograde).
        let tilt = if prograde { 0.0 } else { std::f64::consts::PI };
        let (inside, _, _) = shoot(&k, r0, bc * (1.0 - 1e-4), tilt, &cfg);
        let (outside, _, _) = shoot(&k, r0, bc * (1.0 + 1e-4), tilt, &cfg);
        assert_eq!(
            inside,
            Termination::Captured(Capture::Horizon),
            "prograde={prograde}"
        );
        assert_eq!(outside, Termination::Escaped, "prograde={prograde}");
    }
}

#[test]
fn backward_integration_retraces_forward_path() {
    // Integrate forward to some event, then backward from there: must
    // return to the start (checks the time_sign plumbing).
    let k = Kerr::new(1.0, 0.8);
    let n = [-0.9f64, 0.3, 0.3];
    let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    let (y0, l) = kerr::initial_state(&k, 30.0, 1.2, 0.0, n.map(|v| v / norm));
    let cfg = precise(1e3);
    // Forward: stop while still inbound, at r < 20.
    let fwd = kerr::trace(&k, y0, l, 1.0, &cfg, |_, cur: &KerrState, _| {
        (cur[0] < 20.0).then_some(())
    });
    assert_eq!(fwd.termination, Termination::Event);
    // Backward: stop at the upward crossing of r = 30, interpolated.
    let back = kerr::trace(
        &k,
        fwd.state,
        l,
        -1.0,
        &cfg,
        |prev: &KerrState, cur: &KerrState, _| {
            (prev[0] < 30.0 && cur[0] >= 30.0).then(|| {
                let t = (30.0 - prev[0]) / (cur[0] - prev[0]);
                [
                    prev[1] + t * (cur[1] - prev[1]),
                    prev[2] + t * (cur[2] - prev[2]),
                ]
            })
        },
    );
    let [theta, phi] = back.event.expect("returned to r = 30");
    assert!((theta - y0[1]).abs() < 1e-4, "θ {theta} vs {}", y0[1]);
    assert!((phi - y0[2]).abs() < 1e-4, "φ {phi} vs {}", y0[2]);
}
