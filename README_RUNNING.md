# How to Run NeRF

This guide explains how to run the NeRF training and rendering pipeline with different command-line options.

## Basic Usage

### 1. Using a Config File (Recommended)

The easiest way is to use one of the provided config files:

```bash
# Train on Blender dataset (e.g., lego)
python run_nerf.py --config configs/lego.txt

# Train on LLFF dataset (e.g., fern)
python run_nerf.py --config configs/fern.txt
```

### 2. Override Config File with Command Line Arguments

You can use a config file and override specific parameters:

```bash
# Use config file but change experiment name and batch size
python run_nerf.py --config configs/lego.txt --expname my_lego_experiment --N_rand 2048

# Use config file but change learning rate
python run_nerf.py --config configs/fern.txt --lrate 1e-3
```

### 3. Command Line Only (No Config File)

You can specify all parameters via command line:

```bash
# Blender dataset example
python run_nerf.py \
    --expname blender_lego \
    --basedir ./logs \
    --datadir ./data/raw_nerf/nerf_synthetic/lego \
    --dataset_type blender \
    --use_viewdirs \
    --white_bkgd \
    --N_samples 64 \
    --N_importance 128 \
    --N_rand 1024 \
    --lrate 5e-4 \
    --lrate_decay 500 \
    --netdepth 8 \
    --netwidth 256 \
    --netdepth_fine 8 \
    --netwidth_fine 256 \
    --half_res

# LLFF dataset example
python run_nerf.py \
    --expname llff_fern \
    --basedir ./logs \
    --datadir ./data/raw_nerf/nerf_llff_data/fern \
    --dataset_type llff \
    --factor 8 \
    --llffhold 8 \
    --use_viewdirs \
    --N_samples 64 \
    --N_importance 64 \
    --N_rand 1024 \
    --raw_noise_std 1e0
```

## Common Use Cases

### Training

```bash
# Train on Blender lego scene
python run_nerf.py --config configs/lego.txt

# Train on LLFF fern scene
python run_nerf.py --config configs/fern.txt

# Train with custom experiment name
python run_nerf.py --config configs/lego.txt --expname my_custom_name
```

### Rendering Only (No Training)

After training, render from a checkpoint:

```bash
# Render the test set
python run_nerf.py --config configs/lego.txt --render_only --render_test

# Render the spiral path
python run_nerf.py --config configs/lego.txt --render_only

# Render with downsampling for faster preview
python run_nerf.py --config configs/lego.txt --render_only --render_factor 4
```

### Custom Training Parameters

```bash
# Smaller batch size (for limited GPU memory)
python run_nerf.py --config configs/lego.txt --N_rand 512

# More samples per ray (better quality, slower)
python run_nerf.py --config configs/lego.txt --N_samples 128 --N_importance 256

# Higher learning rate
python run_nerf.py --config configs/lego.txt --lrate 1e-3

# Disable batching (use single image at a time)
python run_nerf.py --config configs/lego.txt --no_batching

# Continue from checkpoint (auto-detects latest)
python run_nerf.py --config configs/lego.txt  # Automatically loads latest checkpoint

# Start fresh (don't load checkpoint)
python run_nerf.py --config configs/lego.txt --no_reload

# Load specific checkpoint
python run_nerf.py --config configs/lego.txt --ft_path ./logs/blender_paper_lego/100000.tar
```

## Available Config Files

The `configs/` directory contains pre-configured settings for different scenes:

**Blender Synthetic Scenes:**
- `lego.txt` - Lego scene
- `chair.txt` - Chair scene
- `drums.txt` - Drums scene
- `ficus.txt` - Ficus scene
- `hotdog.txt` - Hotdog scene
- `materials.txt` - Materials scene
- `mic.txt` - Microphone scene
- `ship.txt` - Ship scene

**LLFF Real Scenes:**
- `fern.txt` - Fern scene
- `flower.txt` - Flower scene
- `fortress.txt` - Fortress scene
- `horns.txt` - Horns scene
- `leaves.txt` - Leaves scene
- `orchids.txt` - Orchids scene
- `room.txt` - Room scene
- `trex.txt` - T-Rex scene

## Key Parameters

### Dataset Parameters
- `--dataset_type`: `llff` or `blender`
- `--datadir`: Path to dataset directory
- `--factor`: Downsample factor for LLFF (default: 8)
- `--half_res`: Use half resolution for Blender (400x400 instead of 800x800)
- `--white_bkgd`: Use white background for Blender scenes

### Network Parameters
- `--netdepth`: Number of layers in coarse network (default: 8)
- `--netwidth`: Channels per layer in coarse network (default: 256)
- `--netdepth_fine`: Number of layers in fine network (default: 8)
- `--netwidth_fine`: Channels per layer in fine network (default: 256)
- `--use_viewdirs`: Use viewing directions (5D input instead of 3D)
- `--multires`: Log2 of max freq for positional encoding (3D location, default: 10)
- `--multires_views`: Log2 of max freq for positional encoding (2D direction, default: 4)

### Training Parameters
- `--N_rand`: Batch size - number of random rays per gradient step (default: 4096)
- `--lrate`: Learning rate (default: 5e-4)
- `--lrate_decay`: Exponential learning rate decay (in 1000 steps, default: 250)
- `--N_samples`: Number of coarse samples per ray (default: 64)
- `--N_importance`: Number of additional fine samples per ray (default: 0)
- `--perturb`: Set to 0.0 for no jitter, 1.0 for jitter (default: 1.0)
- `--raw_noise_std`: Std dev of noise added to density (default: 0.0)
- `--precrop_iters`: Number of steps to train on central crops (default: 0)
- `--precrop_frac`: Fraction of image taken for central crops (default: 0.5)

### Rendering Parameters
- `--render_only`: Don't optimize, just render from checkpoint
- `--render_test`: Render the test set instead of spiral path
- `--render_factor`: Downsampling factor for rendering (0 = full res, 4 or 8 for preview)

### Logging Parameters
- `--i_print`: Frequency of console printout (default: 100)
- `--i_weights`: Frequency of weight checkpoint saving (default: 10000)
- `--i_testset`: Frequency of testset saving (default: 50000)
- `--i_video`: Frequency of render_poses video saving (default: 50000)

### Memory Management
- `--chunk`: Number of rays processed in parallel (default: 32768)
- `--netchunk`: Number of points sent through network in parallel (default: 65536)
- Decrease these if running out of memory

## Examples

### Example 1: Quick Test Run
```bash
# Train on lego with smaller batch size and fewer iterations
python run_nerf.py --config configs/lego.txt --N_rand 512 --i_weights 5000
```

### Example 2: High Quality Training
```bash
# Train with more samples and larger network
python run_nerf.py --config configs/lego.txt \
    --N_samples 128 \
    --N_importance 256 \
    --netwidth 512 \
    --netwidth_fine 512
```

### Example 3: Low Memory Training
```bash
# Reduce memory usage
python run_nerf.py --config configs/lego.txt \
    --N_rand 256 \
    --chunk 8192 \
    --netchunk 16384 \
    --half_res
```

### Example 4: Render Only
```bash
# Render test set from trained model
python run_nerf.py --config configs/lego.txt --render_only --render_test

# Render spiral path
python run_nerf.py --config configs/lego.txt --render_only
```

## Tips

1. **First Run**: Start with a config file and default settings
2. **GPU Memory**: If you get OOM errors, reduce `--N_rand`, `--chunk`, or `--netchunk`
3. **Speed vs Quality**: Increase `--N_samples` and `--N_importance` for better quality but slower training
4. **Checkpoints**: Checkpoints are saved in `{basedir}/{expname}/` directory
5. **Resume Training**: The program automatically loads the latest checkpoint unless `--no_reload` is specified

