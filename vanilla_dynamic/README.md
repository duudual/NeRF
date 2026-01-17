# Dynamic NeRF for Dynamic Scenes

This directory contains implementations of two approaches for modeling dynamic scenes with Neural Radiance Fields (NeRF).

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

**Train Straightforward  Model:**
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

### Rendering 360 Video

```bash
python render_video.py \
    --ckpt "path/to/target.tar" \
    --data_basedir "path/to/D_NeRF_Dataset/data" \
    --scene bouncingballs \
    --network_type deformation \
    --n_frames 120 \
    --time_mode cycle \
    --fps 30
```

**Time Mode Options:**
- `cycle`: Time oscillates 0→1→0 as camera rotates (shows animation)
- `linear`: Time goes 0→1 linearly
- `fixed`: Static time (use `--fixed_time 0.5`)

### Evaluation 

**Evaluate Single Model:**
```bash
python -m NeRF.vanilla_dynamic.evaluate \
    --datadir ../../D_NeRF_Dataset/data/bouncingballs \
    --ckpt_deformation logs/dnerf_deformation/200000.tar \
    --output_dir ./evaluation_results
```

## Metrics

The evaluation produces the following metrics:
- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity): Higher is better
- **LPIPS** (Learned Perceptual Image Patch Similarity): Lower is better