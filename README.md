# HorizonLab

Ray tracer for comparing black holes with horizonless black-hole mimickers.

Features:

- **Spacetimes:** **Schwarzschild**, **Kerr** and **JMN-1**
  (Joshi–Malafarina–Narayan naked singularity).
- **CPU reference renderer** (double precision): Novikov–Thorne thin disk,
  optically thin gas, lensed sky, comparison maps.
- **Observables:** telescope-resolution blurring, radial profiles and
  interferometric visibility amplitudes.
- **Interior:** tidal forces along radial infall.
- **Interactive GPU viewer** (wgpu) for the static spacetimes.

The physics, validation and results are written up in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Layout

| Crate | Purpose |
|---|---|
| `crates/horizonlab-core` | Metrics, Dormand–Prince 5(4) integrator with event location, planar null geodesics (spherical), Boyer–Lindquist geodesics (Kerr). No dependencies. |
| `crates/horizonlab-render` | `horizonlab` CLI: camera, disk, gas, sky, comparison, blur, profiles, visibilities (rayon-parallel). `horizonlab-infall` CLI: tidal forces along radial free fall. |
| `crates/horizonlab-gpu` | `horizonlab-view`: interactive wgpu viewer (Schwarzschild, JMN-1), single precision, checked against the CPU renderer with `--compare`. |

## Quick start

```sh
cargo test --workspace
cargo run --release -- --metric schwarzschild --distance 100 --fov 22 --inclination 82 -o schw.png
cargo run --release -- --metric jmn1 --rb 4   --distance 100 --fov 22 --inclination 82 -o jmn1.png
cargo run --release -- --metric kerr --spin 0.99 --distance 100 --fov 22 --inclination 82 -o kerr.png
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
cargo run --release -- --distance 1000 --fov 2.864 --width 300 --height 300 --metric schwarzschild --vs jmn1 --vs-rb 2.95 --no-disk --black-sky --gas static     --blur-fwhm 20 --source m87 --profile profile.csv -o obs.png
```

```sh
# Interferometric visibility amplitudes |V|/V0 vs baseline (unblurred image)
cargo run --release -- --distance 1000 --fov 2.864 --width 200 --height 200 --no-disk     --black-sky --gas infall --gas-out 20 --metric kerr --spin 0.9 --vs schwarzschild     --visibilities vis.csv -o vis.png

# Interactive GPU viewer; headless check against a CPU render with the same flags
cargo run --release --bin horizonlab-view -- --metric jmn1 --rb 2.5
cargo run --release -- --metric jmn1 --rb 2.5 --width 640 --height 360 -o cpu.png
cargo run --release --bin horizonlab-view -- --metric jmn1 --rb 2.5 --width 640 --height 360     --headless gpu.png --compare cpu.png
```

Viewer controls: left-drag to orbit, wheel to zoom, `M` switches metric,
`↑`/`↓` change R_b, `D` toggles the disk, `N` switches between
Novikov–Thorne and Newtonian flux, `+`/`-` set exposure, `[` `]` set field
of view, `Esc` quits. A frame is re-traced only when something changes.

```sh
# Radial free fall: proper time, tidal forces, curvature (CSV optional)
cargo run --release --bin horizonlab-infall -- --metric schwarzschild --mass-solar 4e6
cargo run --release --bin horizonlab-infall -- --metric jmn1 --rb 2.5 --out infall.csv
```

All lengths are in units of the mass `M` (G = c = 1).

## Physics

### Metrics

Schwarzschild and JMN-1 are static and spherically symmetric,
`ds² = -A(r) dt² + B(r) dr² + r² dΩ²`. Kerr is described below.

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
renderer. Steps are capped at `0.05 r` (affine length). Disk crossings are
located exactly: bisection on the sub-step length inside the accepted step,
each trial being one RK step (`Dopri5::bisect_event`). Linear interpolation
between step endpoints gave errors of order 1e-3 in disk brightness.

### Kerr

Boyer–Lindquist coordinates, spin along `+z`. Rays are integrated in Mino
time `σ` (`dλ = Σ dσ`) with the separable super-Hamiltonian, `E = 1`, `L = p_φ`:

```text
𝓗 = ½ ( Δ p_r² + p_θ² - P²/Δ + (L/sinθ - a sinθ)² ),   P = r² + a² - aL
```

The state is `(r, θ, φ, p_r, p_θ)`. Mino time removes the `Σ` factors.

- **Camera:** a zero-angular-momentum observer (ZAMO). A pixel's direction
  `d` maps to a *received* photon moving along `-d` in the ZAMO frame.
- **Backward tracing:** Kerr is only symmetric under `t → -t, φ → -φ`
  together, so the static-metric trick of tracing forward from the camera is
  invalid here. The photon is integrated **backward in time**
  (`kerr::trace(.., time_sign = -1, ..)`).
- **Disk:** prograde equatorial circular geodesics,
  `Ω = √M / (r^{3/2} + a√M)` (Bardeen et al. 1972). The shift factor is
  `g = E_ZAMO,cam / (u^t (1 - Ω L))`. The disk always orbits
  counter-clockwise about `+z`, so `--spin` < 0 gives a retrograde disk. The
  default inner edge is the matching ISCO.
- **Gas in Kerr:** `--gas static` means ZAMO gas, since static observers do
  not exist inside the ergoregion; it is identical to static gas at a = 0.
  `--gas infall` is geodesic free fall from rest at infinity with
  `L = Q = 0`: `Σu^t = (r²+a²)²/Δ - a²sin²θ`, `Σu^r = -√(2Mr(r²+a²))`,
  `Σu^φ = 2Mar/Δ`. Emission is integrated per Mino step,
  `dI = E_obs⁴ j Σ / E_emit³ dσ`.

### Disk

The disk is thin and opaque, in the `z = 0` plane, on prograde circular
geodesics with `Ω² = A'/(2r)`. The shift factor
`g = E_obs/E_emit` includes the gravitational and Doppler terms, and observed
intensity scales as `g⁴ F(r)`.

Flux `F(r)` (`--disk-model`):

- **`novikov-thorne`** (default): relativistic thin disk, computed for the
  actual spacetime (Page & Thorne 1974):

  ```text
  F(r) = - Ω' / (4π √(-g) (E - ΩL)²) · ∫_{r_in}^{r} (E - ΩL) L' dr
  ```

  `E, L, Ω` come from the circular orbits of the metric. For static
  spherical metrics, `E = A u^t` and `L = r²Ω u^t`; for Kerr they are the
  Bardeen–Press–Teukolsky formulas. The torque vanishes at `r_in`, so the
  flux does too. The flux is tabulated on a log grid (4000 points) and the
  radiative efficiency `1 - E(r_in)` is printed. Where `L' < 0` (unstable
  orbits) the flux is clamped at 0.
- **`newtonian`**: `F ∝ r⁻³(1 - √(r_in/r))`, independent of the metric.

The normalization is absolute: `F = 1` at the peak of a Schwarzschild disk
with `r_in = 6M` under the same model, i.e. the same accretion rate for
every disk. A disk reaching further in (Kerr ISCO, JMN-1 interior) is
therefore brighter and hotter, and images of different spacetimes can be
compared directly. Lower `--exposure` for high-spin images, which
saturate.

### Tidal forces and interior (`tidal.rs`, `horizonlab-infall`)

Riemann components in the static orthonormal frame of the spherical metric:

```text
R_t̂r̂t̂r̂ = (1/B)(A''/2A - A'²/4A² - A'B'/4AB)    R_t̂θ̂t̂θ̂ = A'/(2rAB)
R_r̂θ̂r̂θ̂ = B'/(2rB²)                              R_θ̂φ̂θ̂φ̂ = (1 - 1/B)/r²
```

For an observer falling radially with energy `E`, the radial component is
boost-invariant. The transverse component becomes
`K_⊥ = R_t̂θ̂t̂θ̂ + (E²-A)/A · (AB)'/(2rAB²)`; the extra term vanishes for
Schwarzschild (`AB = 1`). These expressions stay finite inside a horizon,
so Schwarzschild infall is followed to `r → 0`. The tidal acceleration
across a body of length `ℓ` is `-K ℓ` (positive = stretch, negative =
squeeze), converted to physical units with `GM/c²`. Proper time comes from
`(dr/dτ)² = (E²-A)/(AB)`, using the substitution `r = r₀ - u²` to remove the
singularity at the release point.

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

5. **Kerr: spin shapes the image far more than the mimicker does.** With the
   same disk and accretion rate (r_obs = 100M, i = 82°):
   - a = 0.99 prograde: the disk reaches the ISCO at 1.45M and dominates
     (luminance residual up to ~157, against an r_in = 6M disk peaking near
     1 at the same exposure). The shadow is smaller,
     flattened on the approaching side, and shifted.
   - a = −0.99 (retrograde disk): the ISCO is at 8.97M, leaving a wide gap
     between the disk and a larger shadow.
   - Edge-on, the horizontal extent of the shadow matches the analytic
     equatorial photon orbits (a = 0.9: b = 2.85M prograde, 6.83M retrograde)
     to < 0.3%.

6. **Radiative efficiency `1 - E(r_in)`** of the Novikov–Thorne disk:

   | Spacetime | r_in | η |
   |---|---|---|
   | Schwarzschild | 6M | 5.72% |
   | Kerr a = 0.5 | 4.233M | 8.21% |
   | Kerr a = 0.9 | 2.321M | 15.58% |
   | Kerr a = 0.99 | 1.454M | 26.40% |
   | Kerr a = −0.99 (retrograde) | 8.972M | 3.79% |
   | JMN-1, R_b = 4M | 0.5M | 64.6% |

   JMN-1 without a photon sphere has circular orbits deep in its interior,
   so a disk there can exceed the efficiency of any Kerr black hole. Joshi,
   Malafarina & Narayan (2014) report these orbits to be stable for
   `M₀ < 2/3`; that is not checked here beyond `F ≥ 0` along this disk. This
   is an observable different in kind from shadow imaging.

7. **Falling into a naked singularity is not gentler.** Release from rest at
   10M, body 2 m (`horizonlab-infall`):

   | | Schwarzschild | JMN-1, R_b = 2.5M (α = 4) | JMN-1, R_b = 4M (α = 1) |
   |---|---|---|---|
   | Kretschmann near r → 0 | ∝ r⁻⁶ | ∝ r⁻⁴ | ∝ r⁻⁴ |
   | Radial tidal | stretch ∝ r⁻³ | **squeeze** ∝ r⁻² | stretch ∝ r⁻² |
   | Transverse tidal | squeeze ∝ r⁻³ | squeeze ∝ **r⁻⁶** | squeeze ∝ r⁻³ |
   | τ from 10M to 10⁻³M | 35.12M | 34.09M | 34.64M |

   In JMN-1 the curvature invariant diverges more slowly than in
   Schwarzschild, yet the transverse force *felt* diverges as `r^-(α+2)`.
   The infaller approaches the speed of light relative to the static frame
   (`γ² = E²/A ∝ r^-α`), which amplifies the static curvature. The radial
   direction switches from stretching to squeezing when `α > 2`, i.e.
   exactly when the photon sphere exists.

   For Sgr A* (4·10⁶ M☉), a 2 m body crosses the horizon unharmed and
   passes 10 g of tidal acceleration only at r = 0.046M, matching the
   estimate `r = (4c²/(10g·ℓ_M²))^{1/3}` ≈ 0.047M. For a 10 M☉ black hole,
   10 g is already exceeded at the release point, 10M.

8. **In visibility space, spin is visible and JMN-1 is not.** Infalling or
   static gas with `j ∝ r⁻²` out to 20M, M87* scale, camera at 1000M.
   Maximum difference in normalized visibility amplitude `|V|/V₀` out to
   13 Gλ:

   | Comparison | i | max Δ(|V|/V₀) |
   |---|---|---|
   | JMN-1 (R_b = 2.95M) vs Schwarzschild, static gas | 17° | 0.0051 |
   | JMN-1 (R_b = 2.95M) vs Schwarzschild, infalling gas | 17° | 0.0001 |
   | Kerr a = 0.9 vs Schwarzschild, infalling gas | 17° | 0.027 |
   | Kerr a = 0.9 vs Schwarzschild, infalling gas | 80° | 0.071 |

   Kerr also makes the visibilities **anisotropic**. At i = 80° and 2 Gλ,
   |V|/V₀ = 0.112 on east-west baselines and 0.069 on north-south ones;
   Schwarzschild gives 0.033 for both. In the image plane, the 20 μas
   blurred residual for Kerr vs Schwarzschild is 24% (i = 17°) to 55%
   (i = 80°) of peak. It comes from the spin-induced displacement of the
   shadow (crescent-shaped residual), compared with ~3% for the best JMN-1
   case.

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
| Event location | Oscillator zero crossing located to 1e-12 within one step |
| Kerr ISCO | a = 0: 6M; a = 0.9: 2.3209M / 8.7174M (Bardeen et al. 1972) |
| Kerr photon orbits | a = 0: 3√3 M; a → 1: 2M prograde, 7M retrograde |
| Kerr initial state | `H = 0` and direction round-trip for arbitrary ZAMO directions |
| Kerr a = 0, tilted planes | Deflection = planar Schwarzschild for b = 6, 10, 50M; plane tilts 0, 0.7, 1.4 rad (< 1e-8) |
| Kerr a = 0.9 capture | Equatorial b_c(1 ∓ 1e-4) → captured / escapes, prograde and retrograde |
| Kerr backward tracing | Forward then backward integration returns to start (< 1e-4) |
| Kerr a = 0 render | Same outcome and disk/sky color as the Schwarzschild path, 135 rays, < 1e-4 relative |
| Kerr a = 0.9 shadow | Edge-on shadow extent = analytic photon orbits (< 0.3%) |
| Novikov–Thorne energy | `∫4π√(-g) E F dr = 1 - E(r_in)`: Schwarzschild error 2.8e-5 (matches truncation at 10⁶M), Kerr a = 0.9 error 1.3e-5 |
| Novikov–Thorne, two paths | Kerr a = 0 vs spherical adapter: flux agrees < 1e-6; rendered disk < 1e-4 |
| Novikov–Thorne limits | F(r_in) = 0; F → 3/(8πr³) far out |
| Riemann, Schwarzschild | `R_trtr = -2M/r³`, `R_tθtθ = M/r³`, `K = 48M²/r⁶`, outside and inside the horizon |
| Tidal boost invariance | Schwarzschild: (2M/r³, -M/r³) for E = 0.5…2, including r < 2M |
| Riemann, JMN-1 interior | Analytic `(1-M₀)α(α-2)/4r²`, `M₀/2r²`, `0`, `M₀/r²` |
| Proper time | Schwarzschild infall from 10M matches the cycloid `τ = √(r₀³/8M)(η + sin η)` through the horizon to 10⁻³M |
| JMN-1 power laws | Radial `r⁻²`, transverse `r^-(α+2)` for R_b = 2.5, 4 |
| Kerr gas, a = 0 | Static (ZAMO) and infalling gas match the spherical path (< 1e-3), b = 1–15M |
| Kerr infall velocity | `u·u = -1`, `E = 1`, `L = 0` at arbitrary (a, r, θ), to 1e-12 |
| Visibilities | Gaussian image → Gaussian |V| (< 1e-4); thin ring's first null at the J₀ zero 2.405/(2πR) (< 1%) |
| GPU vs CPU (`--compare`, manual) | ≤ 0.04% of pixels differ by > 16/255 for Schwarzschild (r = 30, 100M), JMN-1 R_b = 2.5 and R_b = 4 |

The renderer has also been checked by hand: at r_obs = 100M, the shadow area
matches the analytic angular radius `sin α = b_c √(1-2M/r)/r` to within 1%.

## Known limitations

- The Kerr camera is a ZAMO and the spherical camera is a static observer;
  they agree as `r_obs → ∞`.
- The GPU viewer handles static spherical metrics only (no Kerr, no gas).
  It uses f32 fixed-fraction RK4 and linear disk-crossing interpolation; it
  is a preview, and the CPU renderer is the reference.
- Rays reaching `r < 10⁻⁴ M` in JMN-1 count as absorbed by the singularity.
  The physics of the singularity is not modeled.
- Stability of circular orbits is not checked directly. Novikov–Thorne
  clamps the flux to 0 where `L' < 0`. The disk is opaque and emits as a
  colored ramp with intensity `g⁴F`; there is no blackbody spectrum or limb
  darkening.
- Escape direction is read at `r_escape = max(10 r_obs, 1000M)`, leaving
  `O(M/r_escape)` residual bending.
- The beam is an image-plane Gaussian. Visibilities are computed on a
  dense polar (u,v) grid as a function of baseline *length*, not on the
  sparse tracks of real EHT stations; there is no thermal noise, gain
  calibration or closure-quantity analysis. Pixel scale is small-angle; use
  `--distance ≥ 1000` for quantitative profiles and visibilities, and keep
  emission inside the field of view, because a square crop adds spurious
  long-baseline structure.

## Roadmap

1. ~~Difference-map mode~~ (`--vs`) ✓
2. ~~Optically thin emission~~ (`--gas static|infall`) ✓
3. ~~EHT/ngEHT blur and radial profiles~~ (`--blur-fwhm`, `--profile`) ✓
4. ~~Kerr~~ (`--metric kerr --spin a`) ✓
4b. ~~Novikov–Thorne disk flux~~ (`--disk-model`) ✓, ~~gas for Kerr~~ ✓
5. ~~GPU preview renderer~~ (`horizonlab-view`) ✓. Next: Kerr on the GPU.
6. ~~Tidal tensor along radial infall~~ (`horizonlab-infall`) ✓. Not done:
   Kasner regime of *scalarized* black-hole interiors, which needs that
   model's interior field equations (an ODE system, not an analytic
   metric); non-radial infall; Kerr interior (Cauchy horizon, which is
   unstable to mass inflation).
7. ~~Visibility amplitudes vs baseline length~~ (`--visibilities`) ✓. Next:
   real station (u,v) tracks, noise, closure phases.
