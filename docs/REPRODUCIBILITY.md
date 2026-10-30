# Reproducibility notes

How the shipped results were produced, and what was verified for this release.

## 1. Protocol (as reported)

| Item | Value |
|---|---|
| Seam routing | cyclic DP over `{x-1, x, x+1} mod W` (Eq. 5) |
| Width | `W ≤ 512` for all still-image carves |
| Seams | `k = 24` unless stated (`k = 48` reported for GT head disks) |
| Energy | `E = λ_g Ẽ_g + λ_p Ẽ_p + λ_s Ẽ_s`, `λ_g = λ_p = 1`, each term min–max normalised |
| Spherical term | `E_p = E_g cos φ`, `φ(y) = (y/(H-1) - 1/2)π`, circular padding in `x` |
| Gate | `λ_s = 4` iff `h_pred ≥ τ = 0.20` |
| GT head disk | latitude-aware radius `r(φ) = r_eq / max(\|cos φ\|, cos 75°)`, `r_eq = W·2°/360` |
| GT body column | 12° of latitude below each head, half-width `2.5·r(φ)` |
| Metrics | polar occupancy / routing share on the top+bottom 15% of rows; hit rates on the full ERP |
| CI | paired normal approximation, `δ̄ ± 1.96·std(δ)/√n` (Eq. 7), n = 171 |

Datasets: CVRG-Pano official val+test (**n=120**); PanoCount official val+test (**n=313**; dataset
density bins sparse 0–49 / medium 50–199 / dense 200+). Temporal control: ERP2 (source frame 1626,
`W=512`), with ERP4/ERP5 as consistency checks.

## 2. Environment

The paper's runs used Python ≥ 3.10 with NumPy, OpenCV, PyTorch, `transformers` (SegFormer-B0
ADE20K), Pillow and pytest — see `requirements.txt`. Nothing in the repo pins a CUDA build, and no
architecture-specific kernels are used. This release was re-verified on Python 3.14,
`torch 2.14.0+cu130`, `opencv 4.13.0`, `numpy 2.4.4`, `pytest 9.1.1`.

SegFormer-B0 is downloaded from the Hugging Face hub on first use (`transformers`); the practical
gate is the only experiment that needs it.

## 3. Shipped result dumps

`results/metrics/*.json` are the exact `metrics.json` files written by each runner, prefixed with
the run name:

| File | Runner | Contents |
|---|---|---|
| `spherical_set_metrics.json` | `scripts/run_b_cvrg_set.py` | 120 rows: plain vs. spherical routing/occupancy/step |
| `spherical_set_polar_texture.json` | `scripts/run_b_cvrg_set.py` | 120 rows: polar/equator Sobel ratio + Spearman summary |
| `panocrowd_metrics.json` | `scripts/run_c_panocrowd.py` | 313 rows: GT head-hit for (b)/(c) at k = 8/24/48, per bin |
| `panocrowd_body_energy_metrics.json` | `scripts/run_c_body_energy.py` | 313 rows: GT body-hit + `high_body` flag (oracle split) |
| `panocrowd_practical_gate_metrics.json` | `scripts/run_c_practical_gate.py` | 171 medium rows: gate decision, GT + predicted hits, oracle agreement |
| `c3_control_metrics.json` | `scripts/run_e_c3_control.py` | temporal control: synthetic + real (stride 1 / sparse) pan and photometric residual |

Each file has a `rows` array (per image) and a `summary` block with the run configuration.

## 4. Verification performed for this release

- `python scripts/reproduce_tables.py` recomputes Tables I, II and III(A)/(B) from the dumps. Every
  value matches the manuscript exactly, including the paired CIs:
  routing `+0.42 pp` (89/120 up); head-hit `−4.39 / −4.18 / −3.64 pp`; oracle `−35.2 / −3.4 pp`;
  practical `12.59% → 11.04% (−1.55, CI [−2.75, −0.35])`; ON subset `23.64% → 18.90% (−4.73,
  CI [−8.28, −1.19])`, 34 down / 5 tie / 17 up; OFF subset unchanged at 7.21%; predicted-person
  `18.68% → 5.16%`; gate/oracle agreement 30 caught, 19 missed, 26 extra.
- `python scripts/make_figures.py` regenerates every quantitative figure from the same dumps.
- `python -m pytest tests -q` passes without any dataset or network access (68 tests; one of them
  only needs a writable temporary directory for its `tmp_path` fixture).

## 5. Re-running from scratch

Requires the datasets (`PANOCOUNT_ROOT`, `CVRG_PANO_ROOT`) and, for the temporal control, `ERP2_VIDEO`.
The commands are in the README's *Reproducing the paper* section. Each runner writes its own
`metrics.json`; copy it into `results/metrics/` with the prefix from the table above and both scripts
will pick it up.

Runtime on a single mid-range GPU: the practical gate (171 images, SegFormer-B0 inference plus two
carves per image) is the slowest run, on the order of minutes; the CVRG-Pano set is seconds per image
on CPU because SegFormer is not involved.
