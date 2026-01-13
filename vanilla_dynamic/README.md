# Dynamic NeRF for Dynamic Scenes

This directory contains implementations of two approaches for modeling dynamic scenes with Neural Radiance Fields (NeRF).

## Overview

### Approach 1: Straightforward 6D Extension

The first approach directly extends NeRF by adding time as an additional input dimension:

$$\Psi(x, y, z, t, \theta, \phi) \rightarrow (c, \sigma)$$

The MLP takes a 6D input (3D position + time + 2D viewing direction) and outputs color and density.

### Approach 2: Deformation Network

The second approach uses two neural network modules:

1. **Canonical Network** $\Psi_x(x, d) \rightarrow (c, \sigma)$: Encodes the scene in a canonical configuration
2. **Deformation Network** $\Psi_t(x, t) \rightarrow \Delta x$: Predicts the deformation from time $t$ to canonical space

The pipeline:
- For time $t \neq 0$: Compute $\Delta x = \Psi_t(x, t)$, then query $\Psi_x(x + \Delta x, d)$
- For canonical time $t = 0$: Directly query $\Psi_x(x, d)$

## Dataset

This implementation is designed for the D-NeRF dataset:

```
D_NeRF_Dataset/
├── data/
│   ├── bouncingballs/
│   │   ├── train/
│   │   ├── val/
│   │   ├── test/
│   │   ├── transforms_train.json
│   │   ├── transforms_val.json
│   │   └── transforms_test.json
│   ├── hellwarrior/
│   ├── hook/
│   ├── jumpingjacks/
│   ├── lego/
│   ├── mutant/
│   ├── standup/
│   └── trex/
```

## Usage

### Training

**Train Straightforward (6D) Model:**
```bash
python train.py --datadir ../../D_NeRF_Dataset/data/bouncingballs --expname dnerf_straightforward_bouncingballs --network_type straightforward --N_iters 200000  --half_res --use_viewdirs
```

**Train Deformation Network Model:**
```bash
python -m NeRF.vanilla_dynamic.train \
    --datadir ../../D_NeRF_Dataset/data/bouncingballs \
    --expname dnerf_deformation_bouncingballs \
    --network_type deformation \
    --N_iters 200000 \
    --half_res \
    --use_viewdirs
```

**Using Config File:**
```bash
python -m NeRF.vanilla_dynamic.train --config configs/bouncingballs_deform.txt
```

### Rendering 360 Video

**Basic 360 Video:**
```bash
python -m NeRF.vanilla_dynamic.render_video \
    --ckpt logs/dnerf_deformation_bouncingballs/200000.tar \
    --datadir ../../D_NeRF_Dataset/data/bouncingballs \
    --output_dir ./videos \
    --network_type deformation \
    --n_frames 120 \
    --time_mode cycle
```

**Time Mode Options:**
- `cycle`: Time oscillates 0→1→0 as camera rotates (shows animation)
- `linear`: Time goes 0→1 linearly
- `fixed`: Static time (use `--fixed_time 0.5`)

### Evaluation & Comparison

**Compare Both Methods:**
```bash
python -m NeRF.vanilla_dynamic.evaluate \
    --datadir ../../D_NeRF_Dataset/data/bouncingballs \
    --ckpt_straightforward logs/dnerf_straightforward/200000.tar \
    --ckpt_deformation logs/dnerf_deformation/200000.tar \
    --output_dir ./evaluation_results
```

**Evaluate Single Model:**
```bash
python -m NeRF.vanilla_dynamic.evaluate \
    --datadir ../../D_NeRF_Dataset/data/bouncingballs \
    --ckpt_deformation logs/dnerf_deformation/200000.tar \
    --output_dir ./evaluation_results
```

## File Structure

```
vanilla_dynamic/
├── network.py          # Network architectures (both approaches)
├── model.py            # Model creation and checkpoint management
├── render.py           # Dynamic scene rendering
├── load_dnerf.py       # D-NeRF dataset loader
├── config.py           # Configuration parser
├── train.py            # Training script
├── utils.py            # Utilities and metrics
├── evaluate.py         # Quantitative evaluation
├── render_video.py     # 360 video generation
├── positional_encoding.py  # Positional encoding
├── configs/            # Config files
│   ├── bouncingballs_straightforward.txt
│   └── bouncingballs_deform.txt
└── README.md           # This file
```

## Configuration Options

### Network Architecture
| Argument | Default | Description |
|----------|---------|-------------|
| `--network_type` | deformation | 'straightforward' or 'deformation' |
| `--netdepth` | 8 | Layers in main network |
| `--netwidth` | 256 | Channels per layer |
| `--netdepth_deform` | 6 | Layers in deformation network |
| `--netwidth_deform` | 128 | Channels in deformation network |

### Positional Encoding
| Argument | Default | Description |
|----------|---------|-------------|
| `--multires` | 10 | Frequencies for position encoding |
| `--multires_views` | 4 | Frequencies for view direction |
| `--multires_time` | 10 | Frequencies for time encoding |

### Training
| Argument | Default | Description |
|----------|---------|-------------|
| `--N_iters` | 200000 | Training iterations |
| `--lrate` | 5e-4 | Learning rate |
| `--N_rand` | 1024 | Rays per batch |
| `--N_samples` | 64 | Coarse samples per ray |
| `--N_importance` | 128 | Fine samples per ray |

## Metrics

The evaluation produces the following metrics:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity): Higher is better
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better

## Expected Results

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| Straightforward (6D) | ~25-30 | ~0.90-0.95 | ~0.05-0.10 |
| Deformation Network | ~27-32 | ~0.92-0.97 | ~0.03-0.08 |

*Note: Results vary by scene and training duration.*