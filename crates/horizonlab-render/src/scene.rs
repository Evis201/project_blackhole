//! Camera, thin accretion disk, celestial sphere and per-pixel shading.

use std::f64::consts::PI;
use std::path::Path;

use horizonlab_core::{
    escape_angle, initial_state, trace, Capture, SphericalMetric, State, Termination, TraceConfig,
};

use crate::vec3::Vec3;

/// Static pinhole camera at distance `r_obs`, looking at the origin.
///
/// The world frame has the disk in the `z = 0` plane. `inclination` is the
/// angle between the line of sight and the disk normal `+z`.
pub struct Camera {
    pub r_obs: f64,
    pos_hat: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    tan_half_fov: f64,
}

impl Camera {
    pub fn new(r_obs: f64, inclination_deg: f64, fov_deg: f64) -> Self {
        let inc = inclination_deg.clamp(0.01, 179.99).to_radians();
        let pos_hat = Vec3::new(inc.sin(), 0.0, inc.cos());
        let forward = -pos_hat;
        let up = (Vec3::Z - forward * Vec3::Z.dot(forward)).normalized();
        let right = forward.cross(up);
        Self {
            r_obs,
            pos_hat,
            forward,
            right,
            up,
            tan_half_fov: (fov_deg.to_radians() * 0.5).tan(),
        }
    }

    /// Ray direction (unit, static-observer frame) for screen coordinates
    /// `sx ∈ [-aspect, aspect]`, `sy ∈ [-1, 1]` (y up).
    pub fn ray(&self, sx: f64, sy: f64) -> Vec3 {
        (self.forward + self.right * (sx * self.tan_half_fov) + self.up * (sy * self.tan_half_fov))
            .normalized()
    }
}

/// Geometrically thin, optically thick disk in the `z = 0` plane on
/// circular geodesics (prograde about `+z`).
///
/// Emission uses the Newtonian thin-disk flux profile
/// `F(r) ∝ r⁻³ (1 - √(r_in/r))` — a placeholder for Novikov–Thorne, which
/// needs metric-specific orbital quantities.
pub struct Disk {
    pub r_in: f64,
    pub r_out: f64,
    f_max: f64,
}

impl Disk {
    pub fn new(r_in: f64, r_out: f64) -> Self {
        assert!(r_in > 0.0 && r_out > r_in, "need 0 < r_in < r_out");
        let profile = |r: f64| r.powi(-3) * (1.0 - (r_in / r).sqrt());
        // Peak of the profile is at r = (49/36) r_in.
        let f_max = profile((49.0 / 36.0) * r_in);
        Self { r_in, r_out, f_max }
    }

    fn flux(&self, r: f64) -> f64 {
        (r.powi(-3) * (1.0 - (self.r_in / r).sqrt())).max(0.0) / self.f_max
    }
}

pub enum Sky {
    /// No background light: isolates emission from disk and gas.
    Black,
    /// Latitude/longitude grid, to make lensing visible.
    Checker,
    /// Equirectangular panorama, linear RGB.
    Image {
        width: usize,
        height: usize,
        data: Vec<[f32; 3]>,
    },
}

impl Sky {
    pub fn load(path: &Path) -> Result<Self, image::ImageError> {
        let img = image::open(path)?.to_rgb32f();
        let (width, height) = (img.width() as usize, img.height() as usize);
        let data = img
            .pixels()
            .map(|p| {
                [
                    srgb_to_linear(p[0]),
                    srgb_to_linear(p[1]),
                    srgb_to_linear(p[2]),
                ]
            })
            .collect();
        Ok(Sky::Image {
            width,
            height,
            data,
        })
    }

    fn sample(&self, dir: Vec3) -> [f32; 3] {
        let theta = dir.z.clamp(-1.0, 1.0).acos();
        let phi = dir.y.atan2(dir.x);
        match self {
            Sky::Black => [0.0; 3],
            Sky::Checker => {
                let i = (theta / PI * 18.0).floor() as i64;
                let j = ((phi + PI) / (2.0 * PI) * 36.0).floor() as i64;
                if (i + j).rem_euclid(2) == 0 {
                    [0.030, 0.032, 0.045]
                } else {
                    [0.006, 0.006, 0.010]
                }
            }
            Sky::Image {
                width,
                height,
                data,
            } => {
                let u = (phi + PI) / (2.0 * PI);
                let v = theta / PI;
                let x = ((u * *width as f64) as usize).min(width - 1);
                let y = ((v * *height as f64) as usize).min(height - 1);
                data[y * width + x]
            }
        }
    }
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Velocity field of the optically thin gas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GasFlow {
    /// At rest with respect to static observers.
    Static,
    /// Radial free fall from rest at infinity: `u_t = -1`,
    /// `u^r = -√((1-A)/(AB))`.
    Infall,
}

/// Optically thin, spherically symmetric emitting gas with emissivity
/// `j ∝ r⁻²` for `r < r_out` (the model of Narayan, Johnson & Gammie 2019).
///
/// Bolometric transfer without absorption: `dI = g⁴ j dl`, where `dl` is
/// proper length in the gas frame, `dl = E_emit dλ`. With `g = E_obs/E_emit`
/// this is `dI = E_obs⁴ j / E_emit³ dλ`.
pub struct Gas {
    pub flow: GasFlow,
    pub r_out: f64,
    pub brightness: f64,
}

impl Gas {
    /// Photon energy in the gas frame, for a ray traced with `E = 1` and
    /// radial momentum `p_r` (traced direction; the physical photon has
    /// `k_r = -p_r`, so `E_emit = -k·u = u^t + p_r u^r`).
    fn emitted_energy(&self, metric: &dyn SphericalMetric, r: f64, p_r: f64) -> f64 {
        let a = metric.a(r);
        match self.flow {
            GasFlow::Static => 1.0 / a.sqrt(),
            GasFlow::Infall => {
                let ur = -((1.0 - a).max(0.0) / (a * metric.b(r))).sqrt();
                1.0 / a + p_r * ur
            }
        }
    }

    fn integrand(&self, metric: &dyn SphericalMetric, e_obs: f64, y: &State) -> f64 {
        let r = y[0];
        let e_emit = self.emitted_energy(metric, r, y[1]);
        e_obs.powi(4) / (r * r * e_emit.powi(3))
    }

    /// Emission accumulated over one integration step (trapezoid rule,
    /// clipped at `r_out` by linear interpolation in `r`).
    fn segment(
        &self,
        metric: &dyn SphericalMetric,
        e_obs: f64,
        prev: &State,
        cur: &State,
        dl: f64,
    ) -> f64 {
        let (r0, r1) = (prev[0], cur[0]);
        match (r0 <= self.r_out, r1 <= self.r_out) {
            (false, false) => 0.0,
            (true, true) => {
                0.5 * dl
                    * (self.integrand(metric, e_obs, prev) + self.integrand(metric, e_obs, cur))
            }
            (true, false) => {
                let frac = (self.r_out - r0) / (r1 - r0);
                frac * dl * self.integrand(metric, e_obs, prev)
            }
            (false, true) => {
                let frac = (self.r_out - r1) / (r0 - r1);
                frac * dl * self.integrand(metric, e_obs, cur)
            }
        }
    }
}

/// Color of the gas emission (linear RGB, scaled by intensity).
const GAS_COLOR: [f64; 3] = [1.0, 0.55, 0.25];

/// How a pixel's ray ended, for render statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Outcome {
    Sky,
    Disk,
    Horizon,
    Singularity,
    Failed,
}

pub struct Scene<'a> {
    pub metric: &'a dyn SphericalMetric,
    pub camera: Camera,
    pub disk: Option<Disk>,
    pub gas: Option<Gas>,
    pub sky: Sky,
    pub trace: TraceConfig,
    pub disk_brightness: f64,
}

struct DiskHit {
    r: f64,
}

impl Scene<'_> {
    /// Trace one camera ray and return linear RGB radiance.
    pub fn shade(&self, d: Vec3) -> ([f32; 3], Outcome) {
        let cam = &self.camera;
        let metric = self.metric;

        // Orbital plane spanned by the radial direction e1 and the ray's
        // tangential part e2. φ is measured from e1 towards e2.
        let e1 = cam.pos_hat;
        let n_r = d.dot(e1);
        let tangential = d - e1 * n_r;
        let n_t = tangential.norm();
        let e2 = if n_t > 1e-12 {
            tangential * (1.0 / n_t)
        } else {
            cam.up
        };
        let (state, l) = initial_state(metric, cam.r_obs, n_r, n_t);

        // z(φ) ∝ cos φ e1.z + sin φ e2.z = R cos(φ - φ0); zeros at φ0 + π/2 + kπ.
        let amp = e1.z.hypot(e2.z);
        let phi0 = e2.z.atan2(e1.z);
        let e_obs = 1.0 / metric.a(cam.r_obs).sqrt();
        let mut gas_intensity = 0.0;
        let on_step = |prev: &State, cur: &State, dl: f64| -> Option<DiskHit> {
            if let Some(gas) = &self.gas {
                gas_intensity += gas.segment(metric, e_obs, prev, cur, dl);
            }
            let disk = self.disk.as_ref()?;
            if amp < 1e-12 {
                return None;
            }
            let (p0, p1) = (prev[2], cur[2]);
            let k = ((p0 - phi0 - 0.5 * PI) / PI).ceil();
            let phi_z = phi0 + 0.5 * PI + k * PI;
            if !(phi_z > p0 && phi_z <= p1) {
                return None;
            }
            let t = (phi_z - p0) / (p1 - p0);
            let r = prev[0] + t * (cur[0] - prev[0]);
            if r < disk.r_in || r > disk.r_out || metric.circular_orbit(r).is_none() {
                return None;
            }
            Some(DiskHit { r })
        };

        let res = trace(metric, state, l, &self.trace, on_step);
        let (background, outcome) = match res.termination {
            Termination::Event => {
                let hit = res.event.expect("event termination carries a hit");
                (self.disk_radiance(hit.r, l, e1.cross(e2)), Outcome::Disk)
            }
            Termination::Escaped => {
                let psi = escape_angle(metric, l, &res.state);
                let dir = e1 * psi.cos() + e2 * psi.sin();
                (self.sky.sample(dir), Outcome::Sky)
            }
            Termination::Captured(Capture::Horizon) => ([0.0; 3], Outcome::Horizon),
            Termination::Captured(Capture::Singularity) => ([0.0; 3], Outcome::Singularity),
            Termination::MaxSteps | Termination::Failed(_) => ([0.0; 3], Outcome::Failed),
        };
        let gas = self
            .gas
            .as_ref()
            .map_or(0.0, |g| g.brightness * gas_intensity);
        let radiance = std::array::from_fn(|i| background[i] + (GAS_COLOR[i] * gas) as f32);
        (radiance, outcome)
    }

    /// Observed radiance from the disk at radius `r`, including gravitational
    /// and Doppler shift `g = E_obs / E_emit` and bolometric `I ∝ g⁴`.
    fn disk_radiance(&self, r: f64, l: f64, plane_normal: Vec3) -> [f32; 3] {
        let disk = self.disk.as_ref().expect("disk radiance without disk");
        let (omega, ut) = self
            .metric
            .circular_orbit(r)
            .expect("disk hit only where circular orbits exist");
        // The traced ray runs backwards; the physical photon has
        // L_z = -L n̂_z, so E_emit = u^t (1 - Ω L_z) = u^t (1 + Ω L n̂_z).
        let e_emit = ut * (1.0 + omega * l * plane_normal.z);
        let e_obs = 1.0 / self.metric.a(self.camera.r_obs).sqrt();
        let g = e_obs / e_emit;
        let flux = disk.flux(r);
        let temperature = g * flux.powf(0.25);
        let intensity = self.disk_brightness * g.powi(4) * flux;
        let c = thermal_ramp(temperature);
        [
            (c[0] * intensity) as f32,
            (c[1] * intensity) as f32,
            (c[2] * intensity) as f32,
        ]
    }
}

/// Rough blackbody-like color ramp; `t = 1` ≈ hottest disk region at rest.
fn thermal_ramp(t: f64) -> [f64; 3] {
    const STOPS: [(f64, [f64; 3]); 6] = [
        (0.0, [0.35, 0.02, 0.0]),
        (0.3, [0.75, 0.12, 0.01]),
        (0.55, [1.0, 0.42, 0.08]),
        (0.8, [1.0, 0.78, 0.45]),
        (1.0, [1.0, 0.95, 0.88]),
        (1.6, [0.72, 0.82, 1.0]),
    ];
    if t <= STOPS[0].0 {
        return STOPS[0].1;
    }
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let s = (t - t0) / (t1 - t0);
            return [
                c0[0] + s * (c1[0] - c0[0]),
                c0[1] + s * (c1[1] - c0[1]),
                c0[2] + s * (c1[2] - c0[2]),
            ];
        }
    }
    STOPS[STOPS.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;
    use horizonlab_core::Schwarzschild;

    fn gas_scene(metric: &dyn SphericalMetric, flow: GasFlow) -> Scene<'_> {
        Scene {
            metric,
            camera: Camera::new(100.0, 90.0, 30.0),
            disk: None,
            gas: Some(Gas {
                flow,
                r_out: 50.0,
                brightness: 1.0,
            }),
            sky: Sky::Black,
            trace: TraceConfig::default(),
            disk_brightness: 1.0,
        }
    }

    /// Unit ray from the camera with flat-space impact parameter `b`.
    fn ray_with_impact(cam: &Camera, b: f64) -> Vec3 {
        let s = b / cam.r_obs;
        (cam.forward * (1.0 - s * s).sqrt() + cam.up * s).normalized()
    }

    #[test]
    fn gas_column_matches_flat_space_integral() {
        // M → 0: straight rays, g = 1. ∫ dl / r² along a chord of the sphere
        // r < R at impact b is (2/b) atan(√(R² - b²)/b).
        let metric = Schwarzschild::new(1e-9);
        for flow in [GasFlow::Static, GasFlow::Infall] {
            let scene = gas_scene(&metric, flow);
            for &b in &[2.0, 5.0, 20.0] {
                let (c, outcome) = scene.shade(ray_with_impact(&scene.camera, b));
                assert_eq!(outcome, Outcome::Sky);
                let got = c[0] as f64 / GAS_COLOR[0];
                let r = 50.0f64;
                let expected = 2.0 / b * ((r * r - b * b).sqrt() / b).atan();
                let err = (got - expected).abs() / expected;
                assert!(err < 1e-3, "{flow:?} b={b}: got {got}, expected {expected}");
            }
        }
    }

    #[test]
    fn infall_dims_captured_rays() {
        // A captured ray has p_r < 0 throughout: it only samples gas whose
        // photons travel against the inflow, so E_emit = 1/A + |p_r u^r|
        // exceeds the static value 1/√A at every point. (Escaping rays also
        // cross blueshifted far-side gas, so no such ordering holds there.)
        let metric = Schwarzschild::new(1.0);
        let d = ray_with_impact(&gas_scene(&metric, GasFlow::Static).camera, 3.0);
        let (s, out) = gas_scene(&metric, GasFlow::Static).shade(d);
        let (i, _) = gas_scene(&metric, GasFlow::Infall).shade(d);
        assert_eq!(out, Outcome::Horizon);
        assert!(i[0] < s[0], "infall {} vs static {}", i[0], s[0]);
    }
}
