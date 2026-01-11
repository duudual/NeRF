# T-NeRF 渲染Demo使用说明

使用VGGT的NLP头预测NeRF MLP参数，然后渲染相机运动视频。

## 两种模式

### 1. 静态模式 (Static)
- **输入**: 一组多视角图像
- **行为**: 预测一次MLP参数，所有帧使用相同的NeRF
- **适用**: 单时刻场景重建

### 2. 动态模式 (Dynamic)  
- **输入**: 两组多视角图像（时刻A和时刻B）
- **行为**: 每一帧根据时间插值重新预测MLP参数（alpha从0→1）
- **适用**: 时序场景插值，展示场景变化过程

## 使用方法

### 静态模式

```bash
cd NeRF/tnerf

python demo_render.py \
    --mode static \
    --model_path checkpoints_tnerf/vggt_model.pth \
    --images data/scene/views/ \
    --output videos/static_render.mp4 \
    --n_poses 120
```

**说明**:
- 从`data/scene/views/`加载多视角图像
- 预测一次MLP参数
- 渲染120帧视频，相机绕场景旋转
- 所有帧使用相同的NeRF模型

### 动态模式

```bash
python demo_render.py \
    --mode dynamic \
    --model_path checkpoints_tnerf/vggt_model.pth \
    --images_a data/scene/time0/ \
    --images_b data/scene/time1/ \
    --output videos/dynamic_render.mp4 \
    --n_poses 120
```

**说明**:
- 从`time0/`和`time1/`加载两组多视角图像
- **每一帧**都根据`alpha = frame_idx / (n_poses - 1)`重新预测MLP
- 第0帧: alpha=0.0 (完全是time0)
- 第60帧: alpha=0.5 (time0和time1的中间)
- 第119帧: alpha=1.0 (完全是time1)
- 展示场景从time0到time1的平滑过渡

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--mode` | 模式选择: `static` 或 `dynamic` |
| `--model_path` | VGGT模型权重文件路径 |
| `--images` | [静态模式] 多视角图像目录 |
| `--images_a` | [动态模式] 时刻A的图像目录 |
| `--images_b` | [动态模式] 时刻B的图像目录 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | `output_video.mp4` | 输出视频路径 |
| `--H` | 400 | 渲染图像高度 |
| `--W` | 400 | 渲染图像宽度 |
| `--focal` | 555.0 | 相机焦距 |
| `--n_poses` | 120 | 视频帧数 |
| `--n_samples` | 64 | 每条光线的采样点数 |
| `--chunk` | 8192 | 批处理大小 |
| `--white_bkgd` | False | 使用白色背景 |
| `--device` | `cuda` | 计算设备 |
| `--multires` | 10 | 位置编码频率 |
| `--multires_views` | 4 | 视角编码频率 |

## 输入数据格式

多视角图像目录结构：
```
data/
├── scene/
│   ├── views/          # 静态模式
│   │   ├── view_00.png
│   │   ├── view_01.png
│   │   └── ...
│   ├── time0/          # 动态模式
│   │   ├── view_00.png
│   │   ├── view_01.png
│   │   └── ...
│   └── time1/          # 动态模式
│       ├── view_00.png
│       ├── view_01.png
│       └── ...
```

支持的图像格式: `.png`, `.jpg`, `.jpeg`

## 输出

1. **视频文件**: MP4格式，30fps
2. **关键帧**: `{output}_frames/`目录下的5张PNG图像

示例：
```
videos/
├── dynamic_render.mp4
└── dynamic_render_frames/
    ├── frame_000.png  # 第0帧 (alpha=0.0)
    ├── frame_030.png  # 第30帧 (alpha=0.25)
    ├── frame_060.png  # 第60帧 (alpha=0.5)
    ├── frame_090.png  # 第90帧 (alpha=0.75)
    └── frame_119.png  # 第119帧 (alpha=1.0)
```

## 工作流程

### 静态模式
```
多视角图像 → VGGT NLP头 → MLP参数 (一次)
                                ↓
生成相机轨迹 → 对每个位置: 使用相同MLP渲染 → 合成视频
```

### 动态模式
```
多视角图像A + 多视角图像B
         ↓
生成相机轨迹 (120帧)
         ↓
对每一帧:
  1. 计算 alpha = frame_idx / 119
  2. VGGT插值: interpolate(images_a, images_b, alpha)
  3. 预测当前时刻的MLP参数
  4. 使用MLP渲染该帧
         ↓
合成视频 (展示从A到B的平滑过渡)
```

## 性能优化

### 减少内存占用
```bash
--chunk 4096        # 减小批处理
--n_samples 32      # 减少采样点
--H 200 --W 200     # 降低分辨率
```

### 加快渲染速度
```bash
--n_poses 60        # 减少帧数
--n_samples 32      # 减少采样
--mode static       # 静态模式更快（只预测一次MLP）
```

### 提升质量
```bash
--n_samples 128     # 增加采样点
--H 800 --W 800     # 提高分辨率
--chunk 16384       # 增大批处理（需要更多内存）
```

## 示例场景

### Blender合成数据
```bash
python demo_render.py \
    --mode dynamic \
    --model_path checkpoints_tnerf/blender.pth \
    --images_a data/lego/time0/ \
    --images_b data/lego/time1/ \
    --output videos/lego_transition.mp4 \
    --white_bkgd \
    --H 400 --W 400
```

### LLFF真实场景
```bash
python demo_render.py \
    --mode static \
    --model_path checkpoints_tnerf/llff.pth \
    --images data/fern/views/ \
    --output videos/fern_orbit.mp4 \
    --H 378 --W 504 --focal 407.0
```

## 注意事项

1. **MLP参数映射**: 当前`apply_mlp_parameters_to_model()`是占位符，需要实现从NLP头输出到NeRF权重的具体映射

2. **动态模式计算量**: 每一帧都要运行VGGT推理，时间消耗约为静态模式的`n_poses`倍

3. **内存需求**: 取决于图像分辨率、采样点数和批处理大小

4. **相机轨迹**: 默认为螺旋轨迹，可在`generate_spiral_poses()`函数中自定义

## 故障排除

### 错误: CUDA out of memory
**解决**: 减小`--chunk`、`--H`、`--W`或`--n_samples`

### 错误: No images found
**解决**: 检查图像目录路径和文件扩展名

### 渲染结果异常
**解决**: 检查MLP参数映射实现，调整near/far平面（默认2.0/6.0）

### 动态模式太慢
**解决**: 减少`--n_poses`，或使用静态模式快速预览

## 相关文件

- `demo_render.py`: 主脚本
- `model/dynamic_tnerf.py`: DynamicVGGT实现
- `../network.py`: NeRF网络
- `../render.py`: 渲染工具
- `../rays.py`: 光线生成
