# Can images tell a black hole from a naked singularity? Results from HorizonLab

## Summary

We compare the Schwarzschild and Kerr black holes with the horizonless
JMN-1 spacetime (Joshi, Malafarina & Narayan 2011). All three are traced
with the same validated geodesic code, under the same accretion models and
the same observational post-processing.

- **Thin disks: no difference.** When JMN-1 has a photon sphere
  (`R_b < 3M`), thin-disk images are *pixel-for-pixel identical* to
  Schwarzschild.
- **Optically thin gas: small differences.** The images differ only inside
  the shadow: up to 11% of peak brightness at full resolution, ~3% after a
  20 μas beam, and ≤ 0.5% in interferometric visibility amplitude. All of
  these are below the uncertainty in the emission model.
- **Spin: large differences.** Black-hole spin changes the image by 24–55%
  of peak at the same resolution, and makes the visibilities anisotropic.
- **Different observables.** JMN-1 differs instead in quantities that do not
  require imaging the shadow:
  - accretion efficiency: up to 65% when the disk enters the interior,
    against 5.7% for Schwarzschild;
  - tidal forces near the center: the curvature invariant diverges more
    slowly than at a Schwarzschild singularity, while the transverse tidal
    force felt by an infalling body diverges *faster*.

## 1. Spacetimes

All quantities are in units with `G = c = M = 1`.

| Spacetime | Line element | Horizon | Photon sphere |
|---|---|---|---|
| Schwarzschild | `-A dt² + dr²/A + r²dΩ²`, `A = 1 - 2/r` | `r = 2` | `r = 3` |
| JMN-1, `r < R_b` | `A = (1-M₀)(r/R_b)^α`, `B = 1/(1-M₀)`, `M₀ = 2/R_b`, `α = M₀/(1-M₀)` | none (naked singularity at 0) | `r = 3` if `R_b < 3` |
| JMN-1, `r ≥ R_b` | Schwarzschild | | |
| Kerr | Boyer–Lindquist, spin `a` along `+z` | `r₊ = 1 + √(1-a²)` | equatorial `b = ∓a + 6cos(⅓ arccos(∓a))` |

At `R_b`, `A`, `A'` and `B` are continuous and `B'` jumps; the code tests
this.

## 2. Methods

**Geodesics.**
- **Spherical metrics:** each null ray stays in its own orbital plane,
  integrated as `(r, p_r, φ)` from `H = ½(-1/A + p_r²/B + L²/r²) = 0`.
- **Kerr:** rays are integrated in Mino time with the separable
  super-Hamiltonian, as `(r, θ, φ, p_r, p_θ)`. Because Kerr is not
  time-reversal symmetric, the received photon is traced backward in time
  from a ZAMO camera.
- **Integrator:** adaptive Dormand–Prince 5(4), `rtol = 1e-8`. Disk
  crossings are located exactly by bisection inside each accepted step.

**Emission.**
- **Thin disk:** opaque, on prograde equatorial circular geodesics, with
  bolometric intensity `g⁴F`. The flux `F` is the Novikov–Thorne (Page &
  Thorne 1974) profile, computed for each spacetime from its own `E(r)`,
  `L(r)`, `Ω(r)`. All disks share one accretion rate, normalized to the peak
  of the Schwarzschild `r_in = 6M` disk.
- **Optically thin gas:** `j ∝ r⁻²` (Narayan, Johnson & Gammie 2019), either
  static (ZAMO in Kerr) or in free fall from rest at infinity. It is
  integrated as `dI = E_obs⁴ j / E_emit³ dλ`.

**Observables.**
- **Beam:** image-plane Gaussian with FWHM in μas, at the M87* scale
  (`θ_g = 3.82 μas`) or the Sgr A* scale (`4.84 μas`).
- **Radial profiles.**
- **Visibility amplitudes:** `|V(u,v)|/V₀` by direct DFT, as a function of
  baseline length up to 13 Gλ.

**Interior.**
- Static-frame Riemann components and the Kretschmann invariant.
- Tidal tensor of a radially infalling observer, using the boosted
  transverse component, valid through the Schwarzschild horizon.
- Proper time to each radius.

**GPU viewer.** The same spherical-metric physics in a WGSL compute shader
(f32 RK4), checked against the CPU renderer.

## 3. Validation

Every number below comes from code covered by 42 automated tests
(`cargo test --workspace`), in addition to manual checks. Highlights:

| Quantity | Check | Agreement |
|---|---|---|
| Light bending | weak-field series to O(b⁻⁴), b = 200M, 1000M | 1e-7, 3e-9 rel. |
| Schwarzschild shadow | `b_c = 3√3`; rendered shadow area at 100M | ±1e-4 in b; 1% in area |
| Kerr | ISCO (BPT 1972), equatorial photon orbits, a = 0 limit of geodesics and full renders | 1e-4, < 0.3%, < 1e-8 / 1e-4 |
| Novikov–Thorne | `∫4π√(-g) E F dr = 1 - E(r_in)` | 2.8e-5 (= truncation tail), 1.3e-5 |
| Gas transfer | flat-space `∫dl/r²` chord; Kerr a = 0 vs spherical | 1e-3 |
| Curvature | Schwarzschild `K = 48/r⁶` inside and outside the horizon; JMN-1 interior analytic | 1e-12 |
| Proper time | cycloid `τ = √(r₀³/8)(η + sin η)` through the horizon | 1e-6 rel. |
| Visibilities | Gaussian → Gaussian; ring null at the J₀ zero | 1e-4; 1% |
| GPU | vs CPU, four scenes | ≤ 0.04% of pixels off by > 16/255 |

## 4. Results

Commands reproduce each result with
`cargo run --release [--bin …] -- <flags>`.

### 4.1 JMN-1 with a photon sphere is invisible to thin-disk imaging

`--metric schwarzschild --vs jmn1 --vs-rb 2.5 --distance 100 --fov 22 --inclination 82`
gives a residual that is **exactly zero**, with identical per-outcome pixel
counts.

The reasons:
- Outside `R_b` the metric *is* Schwarzschild, so the critical curve is the
  same.
- Inside, `A ∝ r^α` with `α > 2` leaves null rays no turning point, so every
  ray crossing `R_b` ends on the singularity, just as it would end on the
  horizon in Schwarzschild.
- Timelike circular orbits do not exist inside `r = 3M`, so the disk cannot
  enter either.

### 4.2 Optically thin gas: small, interior-only differences

JMN-1 is brighter only at impact parameters below `b_c`, where rays cross
gas inside `R_b` before reaching the singularity. Maximum luminance residual
vs Schwarzschild, as a fraction of peak:

| Gas \ R_b | 2.05M | 2.5M | 2.95M |
|---|---|---|---|
| static, full resolution | < 0.01% | 0.38% | 10.1% |
| infall, full resolution | < 0.01% | < 0.01% | 0.85% |
| static, 20 μas beam (M87*) | — | 0.49% | 2.9% |
| infall, 20 μas beam (M87*) | — | < 0.01% | 0.06% |

With the 20 μas beam and static gas at `R_b = 2.95M`, the central brightness
is 4.5% higher, and the depression contrast is 0.67 vs 0.65. That is a
two-point change, far below the spread between plausible emissivity and
velocity models.

### 4.3 Spin dominates

With the same disk and accretion rate:
- a = 0.99 moves the ISCO to 1.45M; the disk dominates the image and the
  shadow shrinks and shifts.
- a = −0.99 opens a gap out to 8.97M.

With infalling gas, the 20 μas residual between Kerr a = 0.9 and
Schwarzschild is 24% of peak at i = 17° and 55% at i = 80°. The residual is
crescent-shaped: the shadow is displaced.

### 4.4 Visibility amplitudes

Gas inside 20M, M87* scale, `--visibilities`:

| Comparison | i | max Δ(|V|/V₀) up to 13 Gλ |
|---|---|---|
| JMN-1 2.95M vs Schw., static gas | 17° | 0.0051 |
| JMN-1 2.95M vs Schw., infalling gas | 17° | 0.0001 |
| Kerr 0.9 vs Schw., infalling gas | 17° | 0.027 |
| Kerr 0.9 vs Schw., infalling gas | 80° | 0.071 |

Kerr also breaks the isotropy of the visibilities: at i = 80° and 2 Gλ,
east-west |V|/V₀ = 0.112 against north-south 0.069, while Schwarzschild
gives 0.033 in both directions.

### 4.5 Accretion efficiency (Novikov–Thorne)

| Spacetime | r_in | η = 1 − E(r_in) |
|---|---|---|
| Schwarzschild | 6M | 5.72% |
| Kerr a = 0.5 / 0.9 / 0.99 | 4.233 / 2.321 / 1.454M | 8.21 / 15.58 / 26.40% |
| Kerr a = −0.99 | 8.972M | 3.79% |
| JMN-1, R_b = 4M | 0.5M | 64.6% |

When JMN-1 has no photon sphere, circular orbits continue deep into the
interior, and a disk reaching them radiates far more than around any Kerr
black hole. Joshi, Malafarina & Narayan (2014) report these orbits to be
stable for `M₀ < 2/3`; the code checks only that the flux stays
non-negative (`L' ≥ 0`). This difference shows up in luminosity and
spectrum, not in the shadow.

### 4.6 Tidal forces on the way in

Release from rest at 10M (`horizonlab-infall`):

| Near r → 0 | Schwarzschild | JMN-1, R_b = 2.5M (α = 4) | JMN-1, R_b = 4M (α = 1) |
|---|---|---|---|
| Kretschmann | ∝ r⁻⁶ | ∝ r⁻⁴ | ∝ r⁻⁴ |
| Radial tide | stretch ∝ r⁻³ | squeeze ∝ r⁻² | stretch ∝ r⁻² |
| Transverse tide | squeeze ∝ r⁻³ | squeeze ∝ r⁻⁶ | squeeze ∝ r⁻³ |
| τ from 10M to 10⁻³M | 35.12M | 34.09M | 34.64M |

In JMN-1 the transverse tide felt by the infaller scales as `r^-(α+2)`. The
static curvature (∝ r⁻²) is amplified by the boost factor `γ² = E²/A ∝ r^-α`,
because the infaller approaches the speed of light relative to static
observers. The radial tide turns from stretching into squeezing exactly when
`α > 2`, the condition for a photon sphere.

For scale: a 2 m body falling into Sgr A* first feels 10 g at r = 0.046M,
well inside the horizon, matching the estimate `(4c²/(10g·ℓ_M²))^{1/3}`. For
a 10 M☉ black hole, 10 g is already exceeded at 10M.

## 5. Discussion

The original question, *can we distinguish black holes from mimickers?*,
has a sharper answer than "look at the photon ring":

1. For JMN-1 with a photon sphere, geometry alone gives **no** signature in
   thin-disk images, and only a **model-dependent** few-percent signature
   with optically thin gas. Any claimed detection would need the plasma
   model to be known independently, to better than the effect itself.
2. Spin and inclination produce effects an order of magnitude larger, with a
   distinct morphology: displacement and anisotropy. Those must be fitted
   first, and any residual JMN-1 signal would sit beneath them.
3. More promising discriminants are **energetic**: radiative efficiency,
   and spectral hardness from a hot inner disk. Where a disk can enter the
   object, JMN-1 radiates up to ~11× more than Schwarzschild.

## 6. Limitations

- Emission models are simple: a thin disk with a color ramp rather than a
  spectrum; gas with `j ∝ r⁻²` and no absorption. There is no GRMHD.
- JMN-1 interior gas is kinematic; its coupling to the matter that sources
  the metric is ignored.
- Visibilities are computed against baseline *length* on a dense grid. Real
  sparse (u,v) tracks, noise, calibration and closure quantities are not
  modeled.
- Stability of circular orbits is not checked directly.
- The interior study covers radial infall only. The Kasner regime of
  scalarized black holes would need that model's interior field equations
  and is not included. The Kerr interior (Cauchy horizon) is not modeled.
- The GPU viewer covers spherical metrics in f32 and is a preview; the CPU
  renderer is the reference.

## References

- Bardeen, Press & Teukolsky 1972, ApJ 178, 347 — Kerr circular orbits, ISCO.
- Page & Thorne 1974, ApJ 191, 499 — thin-disk flux.
- Joshi, Malafarina & Narayan 2011, CQG 28, 235018 — JMN-1 spacetime.
- Joshi, Malafarina & Narayan 2014, CQG 31, 015002 — JMN-1 accretion disks.
- Narayan, Johnson & Gammie 2019, ApJL 885, L33 — shadows with spherical
  accretion.
- Event Horizon Telescope Collaboration 2019 (M87*), 2022 (Sgr A*).
