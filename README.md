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

## Roadmap

1. Difference-map mode (render two metrics with the same camera, write the residual).
2. Optically thin emission (spherical accretion), needed for result 1 above.
3. Kerr (4D Hamiltonian, Boyer–Lindquist).
4. GPU preview renderer (wgpu, f32), validated against this CPU reference.
5. Interior views (tidal tensor along infalling geodesics; Kasner regime).
