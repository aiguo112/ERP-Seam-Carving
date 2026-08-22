# Spherical seam carving — Stage A (classical baseline)

360° ERP images: `x` is longitude (periodic), `y` is latitude
`φ = (y/(H-1) - 0.5) π`. Stage A uses **Sobel gradient energy** and
**standard (non-cyclic) DP** so you can verify the pipeline before
spherical / semantic terms are added.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python scripts/run_a_classical.py data/demo_erp.png --seams 24 --out output
python scripts/run_a_classical.py path/to/erp.jpg --seams 40 --out output
```

Writes:

- `output/original.png`
- `output/carved.png` — width reduced by `N` vertical seams
- `output/seam_overlay.png` — first seam drawn on the ERP
- `output/comparison.png` — before / after

Optional: `--cyclic` turns on wrap-around 8-connected DP (off by default).

## Stage B — spherical weighting

Same image, two energy maps: plain Sobel vs Sobel × `cos(φ)`.

```bash
python scripts/run_b_spherical.py --seams 32 --out output/spherical
python scripts/run_b_spherical.py path/to/erp.jpg --seams 40 --out output/spherical
```

Prints polar-band pixel fraction (structurally ~30% for any vertical seam) and
polar **routing share** (`|Δx|` in the top/bottom 15% of rows). The spherical
run should route more lateral motion through the poles — that is intended.

## Stage C — semantic protection (real ERP ablation)

`E = λ_g·gradient + λ_p·spherical(gradient) + λ_s·semantic`, cyclic DP.
Four configs on the **same image / same seam count**:

| config | terms |
|--------|--------|
| (a) | gradient |
| (b) | gradient + spherical |
| (c) | gradient + spherical + semantic |
| (d) | semantic only (sanity) |

Metrics: seam–object intersection **% per protected class, void pixels excluded**,
plus Stage B polar **routing** share (not the ~30% occupancy).

**Indoor (Stanford 2D-3D-S GT).** Polar black caps and `#0D0D0D` missing labels
are void (semantic energy 0). Protected: door / window / board / bookcase.
Ceiling / floor / wall are cheap.

```bash
python scripts/run_c_semantic.py --dataset cvrg --cvrg-root D:\OutDoorSegmenation\CVRG-Pano --index 0 --seams 24 --out output/semantic
python scripts/run_c_cvrg_batch.py --cvrg-root D:\OutDoorSegmenation\CVRG-Pano --out output/cvrg_batch
```

**Outdoor 360 (no GT).** ADE20K SegFormer (`nvidia/segformer-b0-finetuned-ade-512-512`)
uses the **ADE20K** importance map, matched on `model.config.id2label` names
(person + vehicles the model actually predicts; ADE's generic `animal` is
left unlisted / low). DeepLabV3 uses the **VOC** map. Hits are prediction-based.

```bash
python scripts/run_c_semantic.py --dataset outdoor --image path/to/erp.jpg --segmenter segformer --seams 24 --out output/semantic
python scripts/run_c_semantic.py --dataset outdoor --image path/to/erp.jpg --segmenter deeplabv3 --seams 24 --out output/semantic
```

**VPN:** Hugging Face (SegFormer weights) and the Stanford dataset download
are often blocked. Enable a VPN if the first model load hangs or 403s. The
2D-3D-S dump is not fetched automatically — point `--stanford-root` at a
local copy (`area_*/pano/{rgb,semantic}` + `assets/semantic_labels.json`).

`--dataset demo` still paints VOC blobs (not an honest real-ERP table).

## Stage D0 — seam jitter (no temporal model)

Independent per-frame Stage B carving. Metric ``J = mean |S_t - S_{t-1}|`` (cyclic).

```bash
python scripts/run_d_jitter.py --root PATH/TO/PAVS10K --out output/jitter
python scripts/run_d_jitter.py --videos clip_slow.mp4 clip_fast.mp4 --out output/jitter
```

Writes side-by-side overlay|carved mp4s. No Mamba/Transformer.

Writes `output/semantic/comparison_abc.png` (original + a–c) and
`ablation_quad.png` (a–d).

`PYTHONPATH` is set by the script (`src/` is added automatically).

## Layout

| Path | Role |
|------|------|
| `src/io_erp.py` | load/save, `circular_pad` (wrap `x` only) |
| `src/energy/gradient.py` | Sobel magnitude energy |
| `src/energy/spherical.py` | `cos(φ)` area weight (Stage B) |
| `src/energy/semantic.py` | class-importance energy (VOC / ADE20K / Stanford) |
| `src/data/stanford2d3d.py` | Stanford 2D-3D-S ERP + 13-class GT + void mask |
| `src/segmentation/segmenter.py` | DeepLabV3-ResNet50 (VOC) |
| `src/segmentation/taxonomy.py` | VOC / ADE20K / Stanford-13 importance maps |
| `src/seam/dp.py` | `find_vertical_seam(energy, cyclic)` |
| `src/seam/carve.py` | remove seam, reduce width |
| `src/viz.py` | overlay + comparison figure |
