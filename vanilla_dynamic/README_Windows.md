# Dynamic NeRF Windows PowerShell Pipeline

Windows版本的PowerShell自动执行脚本，功能与bash版本完全一致。

## 使用方法

### 基本使用
```powershell
# 完整运行所有场景和方法
.\run_all_dnerf.ps1

# 只运行某个场景
.\run_all_dnerf.ps1 -Scenes bouncingballs

# 只运行straightforward方法
.\run_all_dnerf.ps1 -Straightforward

# 同时运行多个场景
.\run_all_dnerf.ps1 -Scenes bouncingballs,lego
```

### 阶段控制
```powershell
# 只训练，不渲染和评估
.\run_all_dnerf.ps1 -TrainOnly

# 只渲染，跳过训练和评估 
.\run_all_dnerf.ps1 -RenderOnly

# 只评估，跳过训练和渲染
.\run_all_dnerf.ps1 -EvalOnly

# 跳过训练，只渲染和评估
.\run_all_dnerf.ps1 -SkipTrain

# 跳过渲染
.\run_all_dnerf.ps1 -SkipRender

# 跳过评估
.\run_all_dnerf.ps1 -SkipEval
```

### DryRun模式
```powershell
# 只显示将要执行的命令，不实际运行
.\run_all_dnerf.ps1 -DryRun -Scenes bouncingballs -Straightforward
```

## 参数配置

默认参数与bash版本完全一致：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `N_iters` | 50000 | 训练迭代次数 |
| `N_rand` | 1024 | 批大小（每步光线数） |
| `N_samples` | 64 | 粗采样点数 |
| `N_importance` | 128 | 细采样点数 |
| `Lrate` | "5e-4" | 学习率 |
| `LrateDecay` | 250 | 学习率衰减 |
| `I_print` | 500 | 打印频率 |
| `I_weights` | 10000 | 保存权重频率 |
| `VideoFrames` | 120 | 视频帧数 |
| `VideoFps` | 30 | 视频FPS |
| `HalfRes` | 开启 | 半分辨率（400x400） |

## 路径配置

默认路径：
- 数据：`D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data`
- 模型：`D:/lecture/2.0_xk/CV/finalproject/NeRF/vanilla_dynamic`

可通过参数覆盖：
```powershell
.\run_all_dnerf.ps1 -DataBaseDir "您的数据路径" -ModelBaseDir "您的模型路径"
```

## 组合使用示例

```powershell
# 只训练bouncingballs场景的straightforward方法
.\run_all_dnerf.ps1 -TrainOnly -Scenes bouncingballs -Straightforward

# 跳过训练，渲染lego和mutant场景
.\run_all_dnerf.ps1 -SkipTrain -Scenes lego,mutant

# 完整流程，但只使用deformation方法
.\run_all_dnerf.ps1 -Deformation

# 先看看要执行什么命令，不实际运行
.\run_all_dnerf.ps1 -DryRun -Scenes bouncingballs,lego -Straightforward
```

## 输出结构

执行后会在模型目录下生成：
```
ModelBaseDir/
├── logs/                           # 所有日志文件
│   ├── dnerf_straightforward_*_train.log
│   ├── dnerf_straightforward_*_render_*.log  
│   └── dnerf_straightforward_*_eval.log
├── results/                        # 评估结果
│   └── summary.txt                # 汇总报告
└── dnerf_*_*/                     # 各个实验目录
    ├── best.tar                   # 最佳模型
    ├── videos/                    # 渲染视频
    └── evaluation/               # 评估指标
```

## 注意事项

1. 确保PowerShell执行策略允许脚本运行
2. 脚本会自动创建必要的目录
3. DryRun模式可以预览将要执行的命令
4. 所有输出都会保存到日志文件中
5. 与bash版本功能完全一致，参数设置相同