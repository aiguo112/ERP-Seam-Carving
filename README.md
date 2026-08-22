# Content-Conditioned Cyclic Seam Carving for Equirectangular Panoramas

Code and supplementary PDF for ERP (360°) seam carving with:

- **cyclic** wrap-around vertical seam routing
- **spherical** energy \(E_p = E_g \cos\phi\)
- **content-conditioned** semantic energy (SegFormer-B0 ADE20K person map), gated by predicted baseline person-hit risk

The main conference manuscript is **not** in this repository. Supplementary material:

- [`paper/supplementary_preview.pdf`](paper/supplementary_preview.pdf)
- [`paper/supplementary.pdf`](paper/supplementary.pdf)

## Setup

```bash
python -m pip install -r requirements.txt
```

Datasets and ERP videos are **not** included. Point scripts at a local copy:

| Env var | Purpose |
|---------|---------|
| `PANOCOUNT_ROOT` | PanoCount stills + head-point JSON (request-based; see [PanoCount360](https://github.com/aiguo112/PanoCount360)) |
| `CVRG_PANO_ROOT` | CVRG-Pano RGB + masks |
| `ERP2_VIDEO` | Path to ERP2.mp4 (temporal control) |
| `ERP_VIDEO_ROOT` / `PAVS10K_ROOT` | Folder of ERP video clips |
| `ERP_VIDEOS` | Extra video paths (`os.pathsep`-separated) |

You can also pass `--root`, `--cvrg-root`, `--video`, or `--videos` on the CLI.

## Quick demo (no dataset)

```bash
python scripts/run_a_classical.py path/to/erp.jpg --seams 24 --out output --cyclic
python scripts/run_b_spherical.py path/to/erp.jpg --seams 24 --out output/spherical
```

## Paper experiments (local data required)

```bash
# RQ1: spherical routing on CVRG-Pano (n=120, k=24, W<=512)
python scripts/run_b_cvrg_set.py --cvrg-root /path/to/CVRG-Pano --out output/spherical_set

# Always-on head disks / body columns / practical gate on PanoCount
python scripts/run_c_panocrowd.py --root /path/to/PanoCount --out output/panocrowd
python scripts/run_c_body_energy.py --root /path/to/PanoCount --out output/panocrowd_body_energy
python scripts/run_c_practical_gate.py --root /path/to/PanoCount --out output/panocrowd_practical_gate

# Temporal sampling control (ERP2)
python scripts/run_e_c3_control.py --video /path/to/ERP2.mp4 --out output/c3_control
```

Primary still-image settings in the paper: cyclic DP, \(W\le 512\), \(k=24\), gate \(\tau=0.20\).

## Layout

| Path | Role |
|------|------|
| `src/seam/` | Cyclic / non-cyclic DP and carving |
| `src/energy/` | Gradient, spherical, semantic, head/body hit metrics |
| `src/segmentation/` | SegFormer ADE20K / DeepLab / dataset taxonomies |
| `src/data/` | Loaders (paths via env; no datasets shipped) |
| `src/temporal/` | Phase-correlation pan, jitter, temporal controls |
| `scripts/` | Experiment runners |
| `tests/` | Unit tests (no external datasets required) |
| `paper/supplementary_preview.pdf` | Supplementary PDF |

## Tests

```bash
python -m pytest tests -q
```

## License / data

Code is provided for research reproducibility. PanoCount stills follow the request process of [PanoCount360](https://github.com/aiguo112/PanoCount360). CVRG-Pano and any ERP videos must be obtained separately.
