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

## Citations
[1] @misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}