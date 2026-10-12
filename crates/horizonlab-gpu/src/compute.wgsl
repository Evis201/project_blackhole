// HorizonLab GPU preview: planar null geodesics in static spherical
// spacetimes (Schwarzschild, JMN-1), single precision, fixed-fraction RK4.
// Mirrors crates/horizonlab-render/src/scene.rs (shade_spherical); see
// that file for the physics. Units: M = 1.

struct Params {
    pos_hat: vec4<f32>,
    forward: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x: r_obs, y: tan(fov/2), z: aspect, w: exposure
    view: vec4<f32>,
    // x: kind (0 Schwarzschild, 1 JMN-1), y: M0, z: alpha, w: R_b
    metric: vec4<f32>,
    // x: r_in, y: r_out, z: disk on, w: flux model (0 Newtonian, 1 NT)
    disk: vec4<f32>,
    // x: width, y: height, z: r_escape, w: max steps
    frame: vec4<f32>,
    // x: ln r of first NT sample, y: Δ ln r, z: sample count, w: Newtonian reference peak
    table: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> nt_flux: array<f32>;

const PI: f32 = 3.14159265;

fn is_jmn_interior(r: f32) -> bool {
    return P.metric.x > 0.5 && r < P.metric.w;
}

fn metric_a(r: f32) -> f32 {
    if (is_jmn_interior(r)) {
        return (1.0 - P.metric.y) * pow(r / P.metric.w, P.metric.z);
    }
    return 1.0 - 2.0 / r;
}

fn metric_da(r: f32) -> f32 {
    if (is_jmn_interior(r)) {
        return P.metric.z * metric_a(r) / r;
    }
    return 2.0 / (r * r);
}

fn metric_b(r: f32) -> f32 {
    if (is_jmn_interior(r)) {
        return 1.0 / (1.0 - P.metric.y);
    }
    return 1.0 / (1.0 - 2.0 / r);
}

fn metric_db(r: f32) -> f32 {
    if (is_jmn_interior(r)) {
        return 0.0;
    }
    let a = 1.0 - 2.0 / r;
    return -(2.0 / (r * r)) / (a * a);
}

fn capture_radius() -> f32 {
    if (P.metric.x > 0.5) {
        return 1e-3;
    }
    return 2.02;
}

// d/dλ (r, p_r, φ) with E = 1.
fn rhs(y: vec3<f32>, l: f32) -> vec3<f32> {
    let r = y.x;
    let pr = y.y;
    let a = metric_a(r);
    let b = metric_b(r);
    let da = metric_da(r);
    let db = metric_db(r);
    return vec3<f32>(
        pr / b,
        -0.5 * (da / (a * a) - pr * pr * db / (b * b) - 2.0 * l * l / (r * r * r)),
        l / (r * r),
    );
}

fn rk4(y: vec3<f32>, l: f32, h: f32) -> vec3<f32> {
    let k1 = rhs(y, l);
    let k2 = rhs(y + 0.5 * h * k1, l);
    let k3 = rhs(y + 0.5 * h * k2, l);
    let k4 = rhs(y + h * k3, l);
    return y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

fn flux(r: f32) -> f32 {
    if (P.disk.w > 0.5) {
        let x = (log(r) - P.table.x) / P.table.y;
        let n = i32(P.table.z);
        let i = clamp(i32(floor(x)), 0, n - 2);
        let t = clamp(x - f32(i), 0.0, 1.0);
        return mix(nt_flux[i], nt_flux[i + 1], t);
    }
    return max(pow(r, -3.0) * (1.0 - sqrt(P.disk.x / r)), 0.0) / P.table.w;
}

fn thermal_ramp(t: f32) -> vec3<f32> {
    var ts = array<f32, 6>(0.0, 0.3, 0.55, 0.8, 1.0, 1.6);
    var cs = array<vec3<f32>, 6>(
        vec3<f32>(0.35, 0.02, 0.0),
        vec3<f32>(0.75, 0.12, 0.01),
        vec3<f32>(1.0, 0.42, 0.08),
        vec3<f32>(1.0, 0.78, 0.45),
        vec3<f32>(1.0, 0.95, 0.88),
        vec3<f32>(0.72, 0.82, 1.0),
    );
    if (t <= ts[0]) {
        return cs[0];
    }
    for (var i = 0; i < 5; i++) {
        if (t <= ts[i + 1]) {
            let s = (t - ts[i]) / (ts[i + 1] - ts[i]);
            return mix(cs[i], cs[i + 1], s);
        }
    }
    return cs[5];
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    let theta = acos(clamp(dir.z, -1.0, 1.0));
    let phi = atan2(dir.y, dir.x);
    let i = i32(floor(theta / PI * 18.0));
    let j = i32(floor((phi + PI) / (2.0 * PI) * 36.0));
    if (((i + j) % 2 + 2) % 2 == 0) {
        return vec3<f32>(0.030, 0.032, 0.045);
    }
    return vec3<f32>(0.006, 0.006, 0.010);
}

fn encode(c: f32) -> f32 {
    let x = c / (1.0 + c);
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn shade(d: vec3<f32>) -> vec3<f32> {
    let r_obs = P.view.x;
    let e1 = P.pos_hat.xyz;
    let n_r = dot(d, e1);
    let tangential = d - e1 * n_r;
    let n_t = length(tangential);
    var e2 = P.up.xyz;
    if (n_t > 1e-7) {
        e2 = tangential / n_t;
    }
    let a0 = metric_a(r_obs);
    let l = r_obs * n_t / sqrt(a0);
    var y = vec3<f32>(r_obs, sqrt(metric_b(r_obs) / a0) * n_r, 0.0);
    let e_obs = 1.0 / sqrt(a0);
    let r_cap = capture_radius();
    let max_steps = i32(P.frame.w);

    for (var i = 0; i < max_steps; i++) {
        // Limit the coordinate displacement per step to 2% of r: near the
        // JMN-1 singularity dr/dλ diverges and a fixed affine step overshoots.
        let v = rhs(y, l);
        let speed = max(1.0, length(vec2<f32>(v.x, y.x * v.z)));
        let y_new = rk4(y, l, 0.02 * y.x / speed);
        // Written so that NaN also counts as captured (comparisons with NaN
        // are false; GPU compilers may fold an explicit x == x test).
        if (!(y_new.x > r_cap)) {
            return vec3<f32>(0.0);
        }
        if (P.disk.z > 0.5) {
            let z0 = cos(y.z) * e1.z + sin(y.z) * e2.z;
            let z1 = cos(y_new.z) * e1.z + sin(y_new.z) * e2.z;
            if ((z0 > 0.0) != (z1 > 0.0)) {
                let t = z0 / (z0 - z1);
                let r = mix(y.x, y_new.x, t);
                let da = metric_da(r);
                let omega2 = da / (2.0 * r);
                let denom = metric_a(r) - r * r * omega2;
                if (r >= P.disk.x && r <= P.disk.y && da > 0.0 && denom > 0.0) {
                    let omega = sqrt(omega2);
                    let ut = 1.0 / sqrt(denom);
                    let nz = cross(e1, e2).z;
                    let g = e_obs / (ut * (1.0 + omega * l * nz));
                    let f = flux(r);
                    let intensity = P.view.w * pow(g, 4.0) * f;
                    return thermal_ramp(g * pow(f, 0.25)) * intensity;
                }
            }
        }
        if (y_new.x >= P.frame.z && y_new.y > 0.0) {
            let a = metric_a(y_new.x);
            let nr = sqrt(a / metric_b(y_new.x)) * y_new.y;
            let nt = sqrt(a) * l / y_new.x;
            let psi = y_new.z + atan2(nt, nr);
            return sky(e1 * cos(psi) + e2 * sin(psi));
        }
        y = y_new;
    }
    return vec3<f32>(1.0, 0.0, 1.0); // step budget exhausted: magenta
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(P.frame.x);
    let h = u32(P.frame.y);
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let u = (f32(gid.x) + 0.5) / f32(w);
    let v = (f32(gid.y) + 0.5) / f32(h);
    let sx = (2.0 * u - 1.0) * P.view.z * P.view.y;
    let sy = (1.0 - 2.0 * v) * P.view.y;
    let d = normalize(P.forward.xyz + P.right.xyz * sx + P.up.xyz * sy);
    let c = shade(d);
    textureStore(out_tex, vec2<i32>(gid.xy), vec4<f32>(encode(c.x), encode(c.y), encode(c.z), 1.0));
}
