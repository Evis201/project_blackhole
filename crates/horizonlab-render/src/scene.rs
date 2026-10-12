//! Camera, thin accretion disk, celestial sphere and per-pixel shading.

use std::f64::consts::PI;
use std::path::Path;
use std::sync::OnceLock;

use horizonlab_core::kerr::{self, Kerr, KerrState};
use horizonlab_core::{
    escape_angle, initial_state, refine_crossing, trace, Capture, SphericalMetric, State,
    Termination, TraceConfig,
};
use horizonlab_core::{NovikovThorne, Schwarzschild, SphericalOrbits};

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

    /// Angle subtended by one pixel at the image centre, in radians
    /// (small-angle: exact to O(fov²) across the frame).
    pub fn pixel_angle(&self, height: usize) -> f64 {
        2.0 * self.tan_half_fov / height as f64
    }

    /// Ray direction (unit, static-observer frame) for screen coordinates
    /// `sx ∈ [-aspect, aspect]`, `sy ∈ [-1, 1]` (y up).
    pub fn ray(&self, sx: f64, sy: f64) -> Vec3 {
        (self.forward + self.right * (sx * self.tan_half_fov) + self.up * (sy * self.tan_half_fov))
            .normalized()
    }
}

/// Geometrically thin, optically thick disk in the `z = 0` plane on
/// circular geodesics (counter-clockwise about `+z`).
///
/// The flux normalisation is absolute (same accretion rate for every disk):
/// `F = 1` at the peak of a Schwarzschild disk with `r_in = 6M` under the
/// same flux model. Disks reaching closer in (Kerr ISCO, JMN-1 interior) are
/// correspondingly brighter and hotter, so images of different spacetimes
/// are directly comparable.
pub struct Disk {
    pub r_in: f64,
    pub r_out: f64,
    model: FluxModel,
}

enum FluxModel {
    /// `F ∝ r⁻³ (1 - √(r_in/r))`: metric-independent placeholder.
    Newtonian,
    /// Relativistic Novikov–Thorne flux for the actual spacetime.
    NovikovThorne(NovikovThorne),
}

/// Peak of `r⁻³(1 - √(6/r))`, reached at `r = (49/36)·6`.
fn newtonian_reference_peak() -> f64 {
    let r: f64 = 49.0 / 36.0 * 6.0;
    r.powi(-3) * (1.0 - (6.0 / r).sqrt())
}

/// Peak Novikov–Thorne flux of a Schwarzschild disk from `r_in = 6M`.
fn novikov_thorne_reference_peak() -> f64 {
    static PEAK: OnceLock<f64> = OnceLock::new();
    *PEAK.get_or_init(|| {
        let s = Schwarzschild::new(1.0);
        NovikovThorne::new(&SphericalOrbits(&s), 6.0, 100.0)
            .expect("Schwarzschild ISCO has a circular orbit")
            .peak()
    })
}

impl Disk {
    /// Newtonian flux profile (metric-independent).
    pub fn newtonian(r_in: f64, r_out: f64) -> Self {
        assert!(r_in > 0.0 && r_out > r_in, "need 0 < r_in < r_out");
        Self {
            r_in,
            r_out,
            model: FluxModel::Newtonian,
        }
    }

    /// Novikov–Thorne flux computed for `spacetime`. Fails if there is no
    /// circular orbit at `r_in` (e.g. inside the photon orbit).
    pub fn novikov_thorne(spacetime: &Spacetime, r_in: f64, r_out: f64) -> Result<Self, String> {
        let nt = match spacetime {
            Spacetime::Spherical(m) => {
                NovikovThorne::new(&SphericalOrbits(m.as_ref()), r_in, r_out)
            }
            Spacetime::Kerr(k) => NovikovThorne::new(k, r_in, r_out),
        }?;
        Ok(Self {
            r_in,
            r_out,
            model: FluxModel::NovikovThorne(nt),
        })
    }

    /// Radiative efficiency `1 - E(r_in)` (Novikov–Thorne only).
    pub fn efficiency(&self) -> Option<f64> {
        match &self.model {
            FluxModel::Newtonian => None,
            FluxModel::NovikovThorne(nt) => Some(nt.efficiency()),
        }
    }

    fn flux(&self, r: f64) -> f64 {
        match &self.model {
            FluxModel::Newtonian => {
                (r.powi(-3) * (1.0 - (self.r_in / r).sqrt())).max(0.0) / newtonian_reference_peak()
            }
            FluxModel::NovikovThorne(nt) => nt.flux(r) / novikov_thorne_reference_peak(),
        }
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
        self.clipped_trapezoid(prev[0], cur[0], dl, |first| {
            self.integrand(metric, e_obs, if first { prev } else { cur })
        })
    }

    /// Trapezoid rule over a step from `r0` to `r1` of parameter length `dl`,
    /// clipped at `r_out` by linear interpolation in `r`. `q(true)` is the
    /// integrand at the start, `q(false)` at the end.
    fn clipped_trapezoid(&self, r0: f64, r1: f64, dl: f64, q: impl Fn(bool) -> f64) -> f64 {
        match (r0 <= self.r_out, r1 <= self.r_out) {
            (false, false) => 0.0,
            (true, true) => 0.5 * dl * (q(true) + q(false)),
            (true, false) => (self.r_out - r0) / (r1 - r0) * dl * q(true),
            (false, true) => (self.r_out - r1) / (r0 - r1) * dl * q(false),
        }
    }

    /// Kerr: gas-frame photon energy for the received photon (`E = 1`,
    /// `p_φ = L`, state momenta are the physical ones). `Static` means
    /// ZAMO gas (static observers do not exist in the ergoregion; the two
    /// coincide at `a = 0`). `Infall` is free fall from rest at infinity
    /// with `L = Q = 0`, for which `u^θ = 0` and
    /// `Σu^t = (r²+a²)²/Δ - a² sin²θ`, `Σu^r = -√(2Mr(r²+a²))`,
    /// `Σu^φ = 2Mar/Δ`.
    fn kerr_emitted_energy(&self, k: &Kerr, l: f64, y: &KerrState) -> f64 {
        let (r, theta) = (y[0], y[1]);
        match self.flow {
            GasFlow::Static => kerr::zamo_energy(k, l, y),
            GasFlow::Infall => {
                let (m, a) = (k.m, k.a);
                let (sigma, delta) = (k.sigma(r, theta), k.delta(r));
                let r2a2 = r * r + a * a;
                let s2 = theta.sin().powi(2);
                let ut = (r2a2 * r2a2 / delta - a * a * s2) / sigma;
                let ur = -(2.0 * m * r * r2a2).sqrt() / sigma;
                let uphi = 2.0 * m * a * r / (delta * sigma);
                // -p·u with p_t = -1, p_r = y[3], p_φ = L (u^θ = 0).
                ut - y[3] * ur - l * uphi
            }
        }
    }

    /// Kerr emission over one Mino-time step: `dI = E_obs⁴ j Σ / E_emit³ dσ`.
    fn kerr_segment(
        &self,
        k: &Kerr,
        l: f64,
        e_obs: f64,
        prev: &KerrState,
        cur: &KerrState,
        dsigma: f64,
    ) -> f64 {
        let q = |y: &KerrState| {
            let e_emit = self.kerr_emitted_energy(k, l, y);
            e_obs.powi(4) * k.sigma(y[0], y[1]) / (y[0] * y[0] * e_emit.powi(3))
        };
        self.clipped_trapezoid(prev[0], cur[0], dsigma, |first| {
            q(if first { prev } else { cur })
        })
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

/// The spacetimes the renderer can trace through.
pub enum Spacetime {
    /// Static, spherically symmetric: planar rays (Schwarzschild, JMN-1).
    Spherical(Box<dyn SphericalMetric>),
    /// Rotating black hole, spin along `+z`. The disk always orbits
    /// counter-clockwise about `+z`, so `a < 0` gives a retrograde disk.
    Kerr(Kerr),
}

impl Spacetime {
    pub fn name(&self) -> String {
        match self {
            Spacetime::Spherical(m) => m.name(),
            Spacetime::Kerr(k) => k.name(),
        }
    }

    /// Lapse at the observer, used to convert pixel angles to apparent
    /// impact parameters (Kerr: monopole part only, `O(a²/r²)` error).
    pub fn lapse(&self, r: f64) -> f64 {
        match self {
            Spacetime::Spherical(m) => m.a(r).sqrt(),
            Spacetime::Kerr(k) => (1.0 - 2.0 * k.m / r).sqrt(),
        }
    }

    /// Natural inner disk edge: Schwarzschild ISCO for spherical metrics,
    /// the ISCO of the disk's orbital sense for Kerr.
    pub fn default_disk_inner(&self) -> f64 {
        match self {
            Spacetime::Spherical(_) => 6.0,
            Spacetime::Kerr(k) => k.isco(k.a >= 0.0),
        }
    }
}

pub struct Scene<'a> {
    pub spacetime: &'a Spacetime,
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
        match self.spacetime {
            Spacetime::Spherical(m) => self.shade_spherical(m.as_ref(), d),
            Spacetime::Kerr(k) => self.shade_kerr(k, d),
        }
    }

    /// Kerr: the received photon (travelling along `-d` at the camera) is
    /// integrated backward in time in Boyer–Lindquist coordinates. The
    /// camera is a ZAMO.
    fn shade_kerr(&self, k: &Kerr, d: Vec3) -> ([f32; 3], Outcome) {
        let cam = &self.camera;
        let theta0 = cam.pos_hat.z.clamp(-1.0, 1.0).acos();
        let phi0 = cam.pos_hat.y.atan2(cam.pos_hat.x);
        let [er, et, ep] = spherical_basis(theta0, phi0);
        let n = [-d.dot(er), -d.dot(et), -d.dot(ep)];
        let (y0, l) = kerr::initial_state(k, cam.r_obs, theta0, phi0, n);
        let e_obs = kerr::zamo_energy(k, l, &y0);

        let mut gas_intensity = 0.0;
        let on_step = |prev: &KerrState, cur: &KerrState, ds: f64| -> Option<f64> {
            if let Some(gas) = &self.gas {
                gas_intensity += gas.kerr_segment(k, l, e_obs, prev, cur, ds);
            }
            let disk = self.disk.as_ref()?;
            let equator = |y: &KerrState| y[1] - 0.5 * PI;
            if (equator(prev) > 0.0) == (equator(cur) > 0.0) {
                return None;
            }
            let r = kerr::refine_crossing(k, l, -1.0, prev, ds, equator)[0];
            (r >= disk.r_in && r <= disk.r_out && k.circular_orbit(r).is_some()).then_some(r)
        };

        let res = kerr::trace(k, y0, l, -1.0, &self.trace, on_step);
        let (background, outcome) = match res.termination {
            Termination::Event => {
                let r = res.event.expect("event termination carries a hit");
                let (omega, ut) = k.circular_orbit(r).expect("checked at hit");
                // Real (received) photon: E = 1, p_φ = L.
                let g = e_obs / (ut * (1.0 - omega * l));
                (self.disk_color(r, g), Outcome::Disk)
            }
            Termination::Escaped => {
                let y = res.state;
                let n = kerr::local_direction(k, l, &y);
                let [er, et, ep] = spherical_basis(y[1], y[2]);
                // The photon arrived from the opposite of its propagation direction.
                let dir = -(er * n[0] + et * n[1] + ep * n[2]);
                (self.sky.sample(dir), Outcome::Sky)
            }
            Termination::Captured(_) => ([0.0; 3], Outcome::Horizon),
            Termination::MaxSteps | Termination::Failed(_) => ([0.0; 3], Outcome::Failed),
        };
        (self.add_gas(background, gas_intensity), outcome)
    }

    /// Add accumulated gas emission to a background radiance.
    fn add_gas(&self, background: [f32; 3], intensity: f64) -> [f32; 3] {
        let gas = self.gas.as_ref().map_or(0.0, |g| g.brightness * intensity);
        std::array::from_fn(|i| background[i] + (GAS_COLOR[i] * gas) as f32)
    }

    /// Static spherical metrics: planar ray, traced forward from the camera
    /// (equivalent to the received photon by time-reversal symmetry).
    fn shade_spherical(&self, metric: &dyn SphericalMetric, d: Vec3) -> ([f32; 3], Outcome) {
        let cam = &self.camera;

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
            let plane_z = |y: &State| y[2].cos() * e1.z + y[2].sin() * e2.z;
            let r = refine_crossing(metric, l, prev, dl, plane_z)[0];
            if r < disk.r_in || r > disk.r_out || metric.circular_orbit(r).is_none() {
                return None;
            }
            Some(DiskHit { r })
        };

        let res = trace(metric, state, l, &self.trace, on_step);
        let (background, outcome) = match res.termination {
            Termination::Event => {
                let hit = res.event.expect("event termination carries a hit");
                (
                    self.disk_radiance(metric, hit.r, l, e1.cross(e2)),
                    Outcome::Disk,
                )
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
        (self.add_gas(background, gas_intensity), outcome)
    }

    /// Observed radiance from the disk at radius `r`, including gravitational
    /// and Doppler shift `g = E_obs / E_emit` and bolometric `I ∝ g⁴`.
    fn disk_radiance(
        &self,
        metric: &dyn SphericalMetric,
        r: f64,
        l: f64,
        plane_normal: Vec3,
    ) -> [f32; 3] {
        let (omega, ut) = metric
            .circular_orbit(r)
            .expect("disk hit only where circular orbits exist");
        // The traced ray runs backwards; the physical photon has
        // L_z = -L n̂_z, so E_emit = u^t (1 - Ω L_z) = u^t (1 + Ω L n̂_z).
        let e_emit = ut * (1.0 + omega * l * plane_normal.z);
        let e_obs = 1.0 / metric.a(self.camera.r_obs).sqrt();
        self.disk_color(r, e_obs / e_emit)
    }

    /// Disk radiance at radius `r` seen with shift factor `g`.
    fn disk_color(&self, r: f64, g: f64) -> [f32; 3] {
        let disk = self.disk.as_ref().expect("disk radiance without disk");
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

/// Orthonormal `(ê_r, ê_θ, ê_φ)` at polar angle `θ`, azimuth `φ` (world frame,
/// polar axis `+z`).
fn spherical_basis(theta: f64, phi: f64) -> [Vec3; 3] {
    let (st, ct) = theta.sin_cos();
    let (sp, cp) = phi.sin_cos();
    [
        Vec3::new(st * cp, st * sp, ct),
        Vec3::new(ct * cp, ct * sp, -st),
        Vec3::new(-sp, cp, 0.0),
    ]
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

    fn gas_scene(spacetime: &Spacetime, flow: GasFlow) -> Scene<'_> {
        Scene {
            spacetime,
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
        let metric = Spacetime::Spherical(Box::new(Schwarzschild::new(1e-9)));
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
        let metric = Spacetime::Spherical(Box::new(Schwarzschild::new(1.0)));
        let d = ray_with_impact(&gas_scene(&metric, GasFlow::Static).camera, 3.0);
        let (s, out) = gas_scene(&metric, GasFlow::Static).shade(d);
        let (i, _) = gas_scene(&metric, GasFlow::Infall).shade(d);
        assert_eq!(out, Outcome::Horizon);
        assert!(i[0] < s[0], "infall {} vs static {}", i[0], s[0]);
    }

    fn disk_scene(spacetime: &Spacetime, r_obs: f64, inclination: f64) -> Scene<'_> {
        Scene {
            spacetime,
            camera: Camera::new(r_obs, inclination, 30.0),
            disk: Some(Disk::newtonian(6.0, 20.0)),
            gas: None,
            sky: Sky::Checker,
            trace: TraceConfig::default(),
            disk_brightness: 1.0,
        }
    }

    #[test]
    fn kerr_without_spin_renders_like_schwarzschild() {
        // End-to-end check of the backward-in-time Kerr path (camera frame,
        // disk crossing, redshift sign, sky direction) against the planar
        // forward path.
        let kerr = Spacetime::Kerr(Kerr::new(1.0, 0.0));
        let schw = Spacetime::Spherical(Box::new(Schwarzschild::new(1.0)));
        let (mut sk, mut ss) = (disk_scene(&kerr, 40.0, 75.0), disk_scene(&schw, 40.0, 75.0));
        for novikov_thorne in [false, true] {
            if novikov_thorne {
                // Independent orbit formulas: Kerr (BPT 1972) vs spherical adapter.
                sk.disk = Some(Disk::novikov_thorne(&kerr, 6.0, 20.0).unwrap());
                ss.disk = Some(Disk::novikov_thorne(&schw, 6.0, 20.0).unwrap());
            }
            compare_renders(&sk, &ss);
        }
    }

    fn compare_renders(sk: &Scene, ss: &Scene) {
        let mut disk_hits = 0;
        for i in 0..15 {
            for j in 0..9 {
                let d = sk.camera.ray(-0.9 + 0.13 * i as f64, -0.8 + 0.2 * j as f64);
                let (ck, ok) = sk.shade(d);
                let (cs, os) = ss.shade(d);
                assert_eq!(ok, os, "ray ({i},{j})");
                disk_hits += (ok == Outcome::Disk) as usize;
                for c in 0..3 {
                    let tol = 1e-4 * cs[c].abs().max(1e-3);
                    assert!(
                        (ck[c] - cs[c]).abs() < tol,
                        "ray ({i},{j}) ch {c}: {ck:?} vs {cs:?}"
                    );
                }
            }
        }
        assert!(
            disk_hits > 10,
            "test grid should see the disk ({disk_hits})"
        );
    }

    #[test]
    fn kerr_shadow_edges_match_equatorial_photon_orbits() {
        // Edge-on observer far away: the shadow's horizontal extent along
        // the equator is set by the prograde and retrograde photon orbits.
        let k = Kerr::new(1.0, 0.9);
        let st = Spacetime::Kerr(k);
        let r_obs = 2000.0;
        let mut scene = disk_scene(&st, r_obs, 90.0);
        scene.disk = None;
        let cam = &scene.camera;
        let captured = |alpha: f64| {
            let d = (cam.forward * alpha.cos() + cam.right * alpha.sin()).normalized();
            scene.shade(d).1 == Outcome::Horizon
        };
        let edge = |sign: f64| {
            let (mut lo, mut hi) = (0.0, sign * 0.01);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if captured(mid) {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            r_obs * lo.abs().sin() / (1.0 - 2.0 / r_obs).sqrt()
        };
        let (left, right) = (edge(-1.0), edge(1.0));
        let (pro, retro) = (
            k.equatorial_critical_impact(true),
            k.equatorial_critical_impact(false),
        );
        let (small, large) = (left.min(right), left.max(right));
        assert!(
            (small - pro).abs() / pro < 3e-3,
            "prograde edge {small} vs {pro}"
        );
        assert!(
            (large - retro).abs() / retro < 3e-3,
            "retrograde edge {large} vs {retro}"
        );
    }

    #[test]
    fn kerr_without_spin_gas_matches_spherical_gas() {
        let kerr = Spacetime::Kerr(Kerr::new(1.0, 0.0));
        let schw = Spacetime::Spherical(Box::new(Schwarzschild::new(1.0)));
        for flow in [GasFlow::Static, GasFlow::Infall] {
            let (sk, ss) = (gas_scene(&kerr, flow), gas_scene(&schw, flow));
            for &b in &[1.0, 4.0, 6.0, 15.0] {
                let d = ray_with_impact(&sk.camera, b);
                let (ck, ok) = sk.shade(d);
                let (cs, os) = ss.shade(d);
                assert_eq!(ok, os, "{flow:?} b={b}");
                let err = (ck[0] - cs[0]).abs() / cs[0];
                assert!(err < 1e-3, "{flow:?} b={b}: {} vs {}", ck[0], cs[0]);
            }
        }
    }

    #[test]
    fn kerr_infall_velocity_is_normalised_with_zero_angular_momentum() {
        for &(a, r, theta) in &[(0.9, 3.0, 0.4), (0.5, 1.9, 1.2), (-0.7, 8.0, 2.5)] {
            let k = Kerr::new(1.0, a);
            let (sigma, delta) = (k.sigma(r, theta), k.delta(r));
            let r2a2 = r * r + a * a;
            let s2 = f64::sin(theta).powi(2);
            let ut = (r2a2 * r2a2 / delta - a * a * s2) / sigma;
            let ur = -(2.0 * r * r2a2).sqrt() / sigma;
            let uphi = 2.0 * a * r / (delta * sigma);
            let g_tt = -(1.0 - 2.0 * r / sigma);
            let g_tp = -2.0 * a * r * s2 / sigma;
            let g_pp = (r2a2 + 2.0 * a * a * r * s2 / sigma) * s2;
            let g_rr = sigma / delta;
            let norm =
                g_tt * ut * ut + 2.0 * g_tp * ut * uphi + g_pp * uphi * uphi + g_rr * ur * ur;
            assert!((norm + 1.0).abs() < 1e-12, "u·u = {norm}");
            // E = -u_t = 1, L = u_φ = 0.
            assert!((-(g_tt * ut + g_tp * uphi) - 1.0).abs() < 1e-12);
            assert!((g_tp * ut + g_pp * uphi).abs() < 1e-12);
        }
    }
}
