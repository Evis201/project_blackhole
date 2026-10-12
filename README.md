# HorizonLab

Ray tracer for comparing black holes with horizonless black-hole mimickers.

Current state (MVP): double-precision CPU reference renderer for
**Schwarzschild** and **JMN-1** (Joshi–Malafarina–Narayan naked singularity),
with a thin accretion disk and a lensed background sky.

## Layout

| Crate | Purpose |
|---|---|
| `crates/horizonlab-core` | Metrics, Dormand–Prince 5(4) integrator, planar null geodesics. No dependencies. |
| `crates/horizonlab-render` | `horizonlab` CLI: camera, disk, sky, PNG output (rayon-parallel). |

## Quick start

```sh
cargo test --workspace
cargo run --release -- --metric schwarzschild --distance 100 --fov 22 --inclination 82 -o schw.png
cargo run --release -- --metric jmn1 --rb 4   --distance 100 --fov 22 --inclination 82 -o jmn1.png
cargo run --release -- --help

# Side-by-side comparison with residual map: [main | vs | main − vs]
cargo run --release -- --metric schwarzschild --vs jmn1 --vs-rb 2.5 \
    --no-disk --black-sky --gas static --gas-brightness 3 --distance 100 --fov 22 -o diff.png
```

In comparison mode, the program prints the largest luminance residual and the
number of pixels that differ by more than 0.1% of peak.

```sh
# Simulate an observation: camera far away (pixels linear in b), field ±25M,
# blur to a 20 μas beam at the M87* scale, plus a radial profile CSV
cargo run --release -- --distance 1000 --fov 2.864 --width 300 --height 300 \
    --metric schwarzschild --vs jmn1 --vs-rb 2.95 --no-disk --black-sky --gas static \
    --blur-fwhm 20 --source m87 --profile profile.csv -o obs.png
```

All lengths are in units of the mass `M` (G = c = 1).

## Physics

### Metrics

Both are static and spherically symmetric, `ds² = -A(r) dt² + B(r) dr² + r² dΩ²`.

| Metric | A(r) | B(r) | Absorber |
|---|---|---|---|
| Schwarzschild | `1 - 2M/r` | `1/A` | horizon `r = 2M` |
| JMN-1, `r < R_b` | `(1-M₀)(r/R_b)^{M₀/(1-M₀)}` | `1/(1-M₀)` | naked singularity `r = 0` |
| JMN-1, `r ≥ R_b` | Schwarzschild | Schwarzschild | |

`M₀ = 2M/R_b` is the compactness. At `R_b`, `A`, `A'` and `B` are continuous
and `B'` jumps (checked in `metric.rs` tests).

### Geodesics

Every null geodesic in a spherically symmetric spacetime lies in a plane
through the origin. Each ray is integrated in its own plane as `(r, p_r, φ)`
from the Hamiltonian `H = ½(-E²/A + p_r²/B + L²/r²) = 0`. This avoids the
polar coordinate singularity and makes the system 3-dimensional. `|H|` is
tracked as an accuracy check.

Integrator: adaptive Dormand–Prince 5(4), default `rtol = 1e-8` in the
renderer. Steps are capped at `0.05 r` so that disk crossings can be
interpolated between steps.

### Disk

The disk is thin and opaque, in the `z = 0` plane, on prograde circular
geodesics with `Ω² = A'/(2r)`. The shift factor
`g = E_obs/E_emit` includes the gravitational and Doppler terms, and observed
intensity scales as `g⁴ F(r)`. The flux profile `F(r)` is the Newtonian
thin-disk form, used as a placeholder.

### Optically thin gas

The gas is spherical, has emissivity `j ∝ r⁻²` for `r < r_out`, and does not
absorb (after Narayan, Johnson & Gammie 2019). Bolometric transfer along the
ray is `dI = g⁴ j dl`, with proper length in the gas frame
`dl = E_emit dλ`, which gives `dI = E_obs⁴ j / E_emit³ dλ`. Integration uses
the trapezoid rule over integrator steps, clipped at `r_out`. Two velocity
fields are available:

| `--gas` | Gas 4-velocity | `E_emit` (traced ray, `E = 1`) |
|---|---|---|
| `static` | static observer | `1/√A` |
| `infall` | free fall from rest at ∞: `u_t = -1`, `u^r = -√((1-A)/(AB))` | `1/A + p_r u^r` |

In JMN-1 the same gas model also fills the interior `r < R_b`. This is a
kinematic assumption: it ignores the gas's interaction with the matter that
makes up the JMN-1 object.

## Results so far

1. **JMN-1 with a photon sphere (`R_b < 3M`) gives the same image as
   Schwarzschild for a thin disk.** The exterior is exactly Schwarzschild,
   so the critical curve is the same. Inside, `A ∝ r^α` with `α > 2`, so null
   rays have no turning points: every ray that crosses `R_b` reaches the
   singularity, just as every such ray reaches the horizon in Schwarzschild.
   No timelike circular orbits exist for `r < 3M` (`A - rA'/2 < 0`), so the
   disk cannot extend inside either. The renderer gives identical
   per-outcome pixel counts for both metrics.
   **Consequence:** in this regime, telling the two apart needs an emission
   model with sources inside `R_b` (e.g. optically thin spherical inflow).
   Spacetime geometry plus a thin disk is not enough.
2. **JMN-1 without a photon sphere (`R_b > 3M`) has no shadow.** Every ray with
   `L > 0` has an interior turning point and comes back out. The center shows
   lensed sky and extra disk images.
3. **With optically thin gas, JMN-1 (`R_b < 3M`) is brighter only inside the
   shadow, and only by a little unless `R_b` is close to `3M`.** The extra
   light comes from gas at `r < R_b`, seen along rays inside `b_c` before they
   reach the singularity. Schwarzschild has no such region: those rays hit
   the horizon. Residual vs Schwarzschild (r_obs = 100M, 400×225 px, maximum
   luminance difference as a fraction of peak):

   | R_b | static gas | infall gas |
   |---|---|---|
   | 2.05M | < 0.01% | < 0.01% |
   | 2.5M | 0.38% | < 0.01% |
   | 2.95M | 10.1% | 0.85% |

   Infall suppresses the effect strongly: along captured rays the gas
   recedes from the photon (extra redshift). These numbers change by < 1.5%
   between `rtol = 1e-6` and `1e-10`. The differing pixels are exactly the
   2792 shadow pixels (or a subset), never the photon ring.
4. **At EHT resolution, the gap shrinks to a few percent.** Setup: M87* scale
   (θ_g = 3.82 μas), camera at 1000M, inclination 17°, maximum residual as a
   fraction of the blurred image's peak:

   | Gas, R_b | no blur | ngEHT 15 μas (3.9M) | EHT 20 μas (5.2M) |
   |---|---|---|---|
   | static, 2.5M | 0.40% | 0.47% | 0.49% |
   | static, 2.95M | 11.3% | 2.9% | 2.9% |
   | infall, 2.5M | 0.01% | < 0.01% | < 0.01% |
   | infall, 2.95M | 0.93% | 0.07% | 0.06% |

   Radial profile, static gas, R_b = 2.95M, 20 μas beam: JMN-1 is 4.5%
   brighter at the center, and the ratio falls to 1.00 by b ≈ 11M. The
   brightness depression (center / ring peak) is 0.67 for JMN-1 vs 0.65 for
   Schwarzschild. **Reading:** a 2-point contrast difference at the center is
   far smaller than the uncertainty in the emission model (emissivity
   profile, flow velocity). In this framework, JMN-1 with `R_b < 3M` cannot be
   distinguished from Schwarzschild by its image alone, unless the plasma
   model is known independently.

## Validation (`cargo test`)

| Test | Reference |
|---|---|
| Weak-field deflection, b = 200M, 1000M | Series to O(b⁻⁴); relative error 1e-7 and 3e-9 |
| Critical impact parameter | `b_c = 3√3 M` ± 1e-4 → captured / escapes |
| Null constraint | `|H| < 1e-8` along an escaping ray |
| JMN-1 exterior | Deflection identical to Schwarzschild for b > b_c |
| JMN-1 `R_b < 3M` | Same critical curve; interior rays end on the singularity |
| JMN-1 `R_b > 3M` | Rays pass through the interior and escape |
| DP5 | 5th-order convergence; oscillator energy over 10 periods |
| Metric | Analytic derivatives vs finite differences; C¹ matching at R_b |
| Gas transfer, M → 0 | `∫dl/r²` along a chord = `(2/b) atan(√(R²-b²)/b)`, error < 1e-3 |
| Gas, captured ray | Infall strictly dimmer than static (`E_emit > 1/√A` pointwise) |
| Comparison mode | Schwarzschild vs JMN-1 (R_b=2.5) with a thin disk: residual exactly 0 |
| θ_g | M87* 3.82 μas, Sgr A* 4.84 μas (published EHT values) |
| Gaussian beam | Flux conserved; point-source variance = σ² (4σ cutoff bias 5e-4) |
| Radial profile | Uniform image → flat profile |

The renderer has also been checked by hand: at r_obs = 100M, the shadow area
matches the analytic angular radius `sin α = b_c √(1-2M/r)/r` to within 1%.

## Known limitations

- No Kerr yet: it needs a full 4D Hamiltonian path, because orbits are not planar.
- Rays reaching `r < 10⁻⁴ M` in JMN-1 count as absorbed by the singularity.
  The physics of the singularity is not modeled.
- Disk flux profile is Newtonian, and stability of circular orbits is not
  checked (only that they exist).
- Escape direction is read at `r_escape = max(10 r_obs, 1000M)`, leaving
  `O(M/r_escape)` residual bending.
- The beam is an image-plane Gaussian, not real interferometric sampling
  (sparse (u,v) coverage, noise, calibration). Pixel scale is small-angle;
  use `--distance ≥ 1000` for quantitative profiles.

## Roadmap

1. ~~Difference-map mode~~ (`--vs`) ✓
2. ~~Optically thin emission~~ (`--gas static|infall`) ✓
3. ~~EHT/ngEHT blur and radial profiles~~ (`--blur-fwhm`, `--profile`) ✓
4. Kerr (4D Hamiltonian, Boyer–Lindquist).
5. GPU preview renderer (wgpu, f32), validated against this CPU reference.
6. Interior views (tidal tensor along infalling geodesics; Kasner regime).
7. Interferometric observables (visibility amplitudes on real EHT baselines)
   in place of a simple image-plane blur.
