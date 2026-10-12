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
        let on_step = |prev: &State, cur: &State| -> Option<DiskHit> {
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
        match res.termination {
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
        }
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
