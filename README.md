# NeRF
## File Structure
```
NeRF/
├── configs/           # Stores configs for different scenes
├── data/raw_nerf/     # Stores NeRF dataset
├── logs/              # Stores training results/evaluation
├── task1/             # Stores the result of task1
├── config.py          # Argument parsing for model and training parameters
├── train.py           # Main NeRF training script
├── evaluate.py        # NeRF evaluation and PSNR computation
├── train_2d.py        # task1: 2D image fitting training script
├── model.py           # NeRF model initialization and checkpoint loading
├── network.py         # NeRF MLP architecture definition
├── positional_encoding.py  # Positional encoding implementation
├── rays.py            # Camera ray generation logic
├── render.py          # Volume rendering and ray batch processing
├── sampling.py        # Coarse and fine point sampling
├── load_llff.py       # LLFF dataset loading (images, poses, intrinsics)
├── load_blender.py    # Blender dataset loading
├── utils.py           # Utility functions (MSE2PSNR, image normalization)
├── visualize_results.py  # Qualitative result visualization (comparison grids, PSNR plots)
├── run_nerf.py        # Entry point for NeRF training
└── README_RUNNING.md  # Detailed tutorial for running
```

## Dataset
You can get the dataset, pretrained weights and result of task1 from [supplementary_material](https://disk.pku.edu.cn/link/AAB3860F8642BC4A1196E5DC1DDA675912) via PKU netdisk.
```
data/
├── raw_nerf/
│   ├── nerf_synthetic/
|   |   ├── lego/
|   │   │   ├── train/
|   │   │   ├── val/
|   │   │   ├── test/
|   │   │   ├── transforms_train.json
|   │   │   ├── transforms_val.json
|   │   │   └── transforms_test.json
│   └── nerf_llff_data/
```

## Task 1: 2D Image Fitting with Positional Encoding
| Model Configuration | PSNR (dB) |
|:---------------------:|:-----------:|
| Positional Encoding (multires=12, D=8, W=256) | ~35 |
| No Positional Encoding (D=8, W=256) | ~24 |
| Positional Encoding (multires=12, D=4, W=256) | ~32.5 |
| Positional Encoding (multires=6, D=8, W=256) | ~28 |
| Positional Encoding (multires=12, D=8, W=128) | ~31 |


## Task 2: Implement NeRF and Fit on Multi-View Images

| Dataset | Average PSNR (dB) | Std Dev (dB) | Max (dB) | Min (dB) | Sample Number |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| lego (blender) | 31.55 | 1.26 | 33.63 | 28.20 | 25 |
| fern (llff) | 26.91 | 0.63 | 27.55 | 26.06 | 3 |
| flower (llff) | 27.57 | 1.14 | 29.60 | 26.15 | 5 |
| fortress (llff) | 31.68 | 0.99 | 33.39 | 30.19 | 6 |
| trex (llff) | 26.84 | 0.90 | 28.05 | 25.25 | 7 |
| horns (llff) | 27.78 | 1.70 | 29.78 | 25.00 | 8 |


## Quick Start
Run the following command line to perform task1:
```
python train_2d.py --img_path xxx.png --D 8 --W 128 --multires 12 --epochs 100 
--i_print 10 --i_save 50 --out_dir xxx
```

| Arg | Meaning |
| :-: | :-: |
| `--img_path` | input 2D image |
| `--D` and `--W` | depth and width of MLP |
| `--multires` | frequency of positional encoding |
| `--epochs` | training epochs |
| `--i_print` | how often the log is printed |
| `--i_save` | how often the image is stored |
| `--out_dir` | directory where the result is saved |
| `--no_posenc` | ignore positional encoding |

Run the following command line to render a video with pretrained weights from `logs/xxx_test/xxx.tar`. The results will be saved in `logs/xxx_test/renderonly_path_xxx/`:
```
python run_nerf.py --config configs/xxx.txt --render_only
```

Run the following command line to evaluate the quality of result on test set. The results will be saved in `logs/xxx_test/evaluation/`:
```
python evaluate.py --config configs/xxx.txt --render_test 
```
# Dynamic NeRF

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

# Novel NeRF
* [VGGT-NeRF](https://github.com/DongYu2005/vggt-NeRF): An enhanced version incorporating Visual Geometry Grounded Transformers for better generalizability. (Maintained by Group Member Dongge Yu, @DongYu2005)

# Novel Dynamic NeRF

A dynamic NeRF implementation based on VGGT that performs semantically consistent temporal interpolation in feature space to generate stable representations at intermediate time steps.

## Setup

1. **Download VGGT Pretrained Model**
   Download `model_weights/model.pt` from the online storage
2. **Extract Model to vggt Directory**

   Extract the downloaded `model.pt` file to the `vggt/model_weights/` directory:
   ```bash
   # Ensure the directory structure as follows
   vggt/
   ├── model_weights/
   │   └── model.pt
   ├── demo_gradio.py
   └── ...
   ```

## Quick Start

Run the following command to launch the Gradio interactive interface:

```bash
cd vggt
python demo_gradio.py
```
![alt text](image.png)

Open the displayed URL in your browser. The interface shows the original VGGT demo. Click the **Dynamic Scene Interpolation** button at the bottom and follow these steps:

- **Input**: Upload two multi-view images (or single images) at different time steps (t0, t1). The viewing angles for images at different time steps should correspond to each other.
- **Select Encoding Mode**: Choose the patch matching encoding mode. (Linear mode directly interpolates the aggregator output, which may cause the intermediate prediction to show mixed states from both time steps)
- **Encode**: Click the encode button. A confirmation message will appear after completion.
- **Temporal Control**: Use alpha to control the time parameter and generate predicted point clouds for intermediate time steps.

Alternatively, you can load our pre-configured history records, wait for loading to complete, and then directly control alpha to generate point clouds for the corresponding intermediate time steps.
[1] @misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch?tab=readme-ov-file}},
  year={2020}
}

