//! Dormand–Prince 5(4) embedded Runge–Kutta with adaptive step control.

/// Mixed absolute/relative error tolerance.
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub rtol: f64,
    pub atol: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            rtol: 1e-9,
            atol: 1e-11,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StepError {
    /// Step size shrank below `h_min` without meeting the tolerance.
    StepTooSmall { h: f64 },
}

/// Adaptive Dormand–Prince 5(4) integrator (5th-order solution, 4th-order
/// embedded error estimate, local extrapolation).
#[derive(Clone, Copy, Debug)]
pub struct Dopri5 {
    pub tol: Tolerance,
    pub h_min: f64,
    pub safety: f64,
    pub min_factor: f64,
    pub max_factor: f64,
}

impl Default for Dopri5 {
    fn default() -> Self {
        Self::new(Tolerance::default())
    }
}

// Butcher tableau. The nodes c_i are omitted: the geodesic equations are
// autonomous (no explicit λ dependence).

const A21: f64 = 1.0 / 5.0;
const A31: f64 = 3.0 / 40.0;
const A32: f64 = 9.0 / 40.0;
const A41: f64 = 44.0 / 45.0;
const A42: f64 = -56.0 / 15.0;
const A43: f64 = 32.0 / 9.0;
const A51: f64 = 19372.0 / 6561.0;
const A52: f64 = -25360.0 / 2187.0;
const A53: f64 = 64448.0 / 6561.0;
const A54: f64 = -212.0 / 729.0;
const A61: f64 = 9017.0 / 3168.0;
const A62: f64 = -355.0 / 33.0;
const A63: f64 = 46732.0 / 5247.0;
const A64: f64 = 49.0 / 176.0;
const A65: f64 = -5103.0 / 18656.0;
// 5th-order weights (also row 7 of A; FSAL).
const B1: f64 = 35.0 / 384.0;
const B3: f64 = 500.0 / 1113.0;
const B4: f64 = 125.0 / 192.0;
const B5: f64 = -2187.0 / 6784.0;
const B6: f64 = 11.0 / 84.0;
// Error weights: b(5th) - b(4th).
const E1: f64 = 71.0 / 57600.0;
const E3: f64 = -71.0 / 16695.0;
const E4: f64 = 71.0 / 1920.0;
const E5: f64 = -17253.0 / 339200.0;
const E6: f64 = 22.0 / 525.0;
const E7: f64 = -1.0 / 40.0;

#[inline]
fn axpy<const N: usize>(y: &[f64; N], terms: &[(f64, &[f64; N])], h: f64) -> [f64; N] {
    let mut out = *y;
    for i in 0..N {
        let mut acc = 0.0;
        for (c, k) in terms {
            acc += c * k[i];
        }
        out[i] += h * acc;
    }
    out
}

impl Dopri5 {
    pub fn new(tol: Tolerance) -> Self {
        Self {
            tol,
            h_min: 1e-14,
            safety: 0.9,
            min_factor: 0.2,
            max_factor: 5.0,
        }
    }

    /// One fixed step of size `h`. Returns the 5th-order solution and the
    /// scaled RMS error norm (accept iff `≤ 1`).
    pub fn step<const N: usize, F>(&self, f: &F, y: &[f64; N], h: f64) -> ([f64; N], f64)
    where
        F: Fn(&[f64; N]) -> [f64; N],
    {
        let k1 = f(y);
        let k2 = f(&axpy(y, &[(A21, &k1)], h));
        let k3 = f(&axpy(y, &[(A31, &k1), (A32, &k2)], h));
        let k4 = f(&axpy(y, &[(A41, &k1), (A42, &k2), (A43, &k3)], h));
        let k5 = f(&axpy(
            y,
            &[(A51, &k1), (A52, &k2), (A53, &k3), (A54, &k4)],
            h,
        ));
        let k6 = f(&axpy(
            y,
            &[(A61, &k1), (A62, &k2), (A63, &k3), (A64, &k4), (A65, &k5)],
            h,
        ));
        let y_new = axpy(
            y,
            &[(B1, &k1), (B3, &k3), (B4, &k4), (B5, &k5), (B6, &k6)],
            h,
        );
        let k7 = f(&y_new);

        let mut sum = 0.0;
        for i in 0..N {
            let err =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
            let scale = self.tol.atol + self.tol.rtol * y[i].abs().max(y_new[i].abs());
            sum += (err / scale).powi(2);
        }
        let norm = (sum / N as f64).sqrt();
        (
            y_new,
            if norm.is_finite() {
                norm
            } else {
                f64::INFINITY
            },
        )
    }

    /// Take one accepted adaptive step, starting from trial size `h` and
    /// never exceeding `h_max`. Returns `(y_new, h_taken, h_next)`.
    pub fn adaptive_step<const N: usize, F>(
        &self,
        f: &F,
        y: &[f64; N],
        mut h: f64,
        h_max: f64,
    ) -> Result<([f64; N], f64, f64), StepError>
    where
        F: Fn(&[f64; N]) -> [f64; N],
    {
        h = h.min(h_max);
        loop {
            let (y_new, err) = self.step(f, y, h);
            let finite = y_new.iter().all(|v| v.is_finite());
            if finite && err <= 1.0 {
                let factor = if err == 0.0 {
                    self.max_factor
                } else {
                    (self.safety * err.powf(-0.2)).clamp(self.min_factor, self.max_factor)
                };
                return Ok((y_new, h, (h * factor).min(h_max)));
            }
            let factor = if finite && err.is_finite() {
                (self.safety * err.powf(-0.2)).clamp(self.min_factor, 1.0)
            } else {
                self.min_factor
            };
            h *= factor;
            if h < self.h_min {
                return Err(StepError::StepTooSmall { h });
            }
        }
    }

    /// Locate a sign change of `g` inside an accepted step of size `h` from
    /// `y0`, by bisecting on the sub-step length (each trial is one RK step
    /// from `y0`, so the located state is 5th-order accurate). Returns the
    /// state at the crossing.
    pub fn bisect_event<const N: usize, F, G>(&self, f: &F, y0: &[f64; N], h: f64, g: G) -> [f64; N]
    where
        F: Fn(&[f64; N]) -> [f64; N],
        G: Fn(&[f64; N]) -> f64,
    {
        let g0 = g(y0);
        let (mut lo, mut hi) = (0.0, h);
        let mut y_hi = self.step(f, y0, h).0;
        for _ in 0..48 {
            let mid = 0.5 * (lo + hi);
            let y_mid = self.step(f, y0, mid).0;
            if (g(&y_mid) > 0.0) == (g0 > 0.0) {
                lo = mid;
            } else {
                hi = mid;
                y_hi = y_mid;
            }
        }
        y_hi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_step_is_fifth_order() {
        // y' = y, y(0) = 1 → y(1) = e. Halving h should cut the error ~32×.
        let solver = Dopri5::default();
        let f = |y: &[f64; 1]| [y[0]];
        let run = |n: usize| {
            let h = 1.0 / n as f64;
            let mut y = [1.0];
            for _ in 0..n {
                y = solver.step(&f, &y, h).0;
            }
            (y[0] - std::f64::consts::E).abs()
        };
        let ratio = run(10) / run(20);
        assert!(ratio > 25.0 && ratio < 40.0, "convergence ratio {ratio}");
    }

    #[test]
    fn bisect_event_finds_zero_crossing() {
        // Oscillator x = cos t, stepping h = 0.05 from t = 1.55 (a step size
        // the controller accepts at default tolerance): crossing at t = π/2.
        let solver = Dopri5::default();
        let f = |y: &[f64; 2]| [y[1], -y[0]];
        let y0 = [1.55f64.cos(), -1.55f64.sin()];
        let y = solver.bisect_event(&f, &y0, 0.05, |y| y[0]);
        assert!(y[0].abs() < 1e-12, "x = {}", y[0]);
        assert!((y[1] + 1.0).abs() < 1e-12, "v = {}", y[1]);
    }

    #[test]
    fn adaptive_harmonic_oscillator_conserves_energy() {
        let solver = Dopri5::new(Tolerance {
            rtol: 1e-10,
            atol: 1e-12,
        });
        let f = |y: &[f64; 2]| [y[1], -y[0]];
        let mut y = [1.0, 0.0];
        let mut t = 0.0;
        let mut h = 1e-3;
        let t_end = 20.0 * std::f64::consts::PI;
        while t < t_end {
            let (y_new, taken, next) = solver.adaptive_step(&f, &y, h, t_end - t).unwrap();
            y = y_new;
            t += taken;
            h = next;
        }
        assert!((y[0] - 1.0).abs() < 1e-7, "x = {}", y[0]);
        assert!(y[1].abs() < 1e-7, "v = {}", y[1]);
    }
}
