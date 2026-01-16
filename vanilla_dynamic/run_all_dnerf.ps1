<#
.SYNOPSIS
Run All Dynamic NeRF Pipeline (PowerShell for Windows)

.DESCRIPTION
Three-stage pipeline:
1) Train models on scenes (Straightforward + Deformation)  
2) Render videos from trained models
3) Evaluate and summarize metrics

.EXAMPLE
.\run_all_dnerf.ps1 -TrainOnly -Scenes bouncingballs -Straightforward
.\run_all_dnerf.ps1 -SkipTrain -RenderOnly -Scenes lego  
.\run_all_dnerf.ps1 -DryRun -Scenes bouncingballs,lego
#>

param(
    [string[]]$Scenes = @("bouncingballs","hellwarrior","hook","jumpingjacks","lego","mutant","standup","trex"),
    [switch]$TrainOnly,
    [switch]$RenderOnly, 
    [switch]$EvalOnly,
    [switch]$SkipTrain,
    [switch]$SkipRender,
    [switch]$SkipEval,
    [switch]$Straightforward,
    [switch]$Deformation,
    [string]$DataBaseDir = "D:/lecture/2.0_xk/CV/finalproject/D_NeRF_Dataset/data",
    [string]$ModelBaseDir = "D:/lecture/2.0_xk/CV/finalproject/NeRF/vanilla_dynamic",
    [int]$N_iters = 10000,
    [int]$N_rand = 1024,
    [int]$N_samples = 64,
    [int]$N_importance = 128,
    [string]$Lrate = "5e-4", 
    [int]$LrateDecay = 250,
    [int]$I_print = 500,
    [int]$I_weights = 10000,
    [switch]$HalfRes,
    [int]$VideoFrames = 120,
    [int]$VideoFps = 30,
    [string[]]$TimeModes = @("cycle"),
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Header($title) {
    Write-Host ""
    Write-Host "============================================="
    Write-Host $title
    Write-Host "============================================="
}

# Resolve pipeline stage flags
$runTrain = $true; $runRender = $true; $runEval = $true
if ($TrainOnly) { $runRender = $false; $runEval = $false }
if ($RenderOnly) { $runTrain = $false; $runEval = $false }  
if ($EvalOnly) { $runTrain = $false; $runRender = $false }
if ($SkipTrain) { $runTrain = $false }
if ($SkipRender) { $runRender = $false }
if ($SkipEval) { $runEval = $false }

# Resolve methods
$runStraightforward = $true; $runDeformation = $true
if ($Straightforward) { $runDeformation = $false }
if ($Deformation) { $runStraightforward = $false }

# Ensure directories
New-Item -ItemType Directory -Force -Path $ModelBaseDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ModelBaseDir 'logs') | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ModelBaseDir 'results') | Out-Null

Write-Header "Dynamic NeRF Three-Stage Pipeline"
Write-Host "Scenes: $($Scenes -join ', ')"
Write-Host "Methods:"
if ($runStraightforward) { Write-Host "  - Straightforward" }
if ($runDeformation) { Write-Host "  - Deformation" }
Write-Host ""
Write-Host "Stages:"
if ($runTrain) { Write-Host "  ✓ Training" }
if ($runRender) { Write-Host "  ✓ Rendering" }
if ($runEval) { Write-Host "  ✓ Evaluation" }
Write-Host ""
Write-Host "Paths:"
Write-Host "  Data: $DataBaseDir"
Write-Host "  Models: $ModelBaseDir"
Write-Host ""

# Helper: Build common train args
function Get-TrainArgs {
    param($DataDir, $ExpName, $NetworkType)
    $args = @(
        "--datadir", $DataDir,
        "--basedir", $ModelBaseDir, 
        "--expname", $ExpName,
        "--network_type", $NetworkType,
        "--N_iters", $N_iters,
        "--N_rand", $N_rand,
        "--N_samples", $N_samples,
        "--N_importance", $N_importance,
        "--lrate", $Lrate,
        "--lrate_decay", $LrateDecay,
        "--use_viewdirs",
        "--i_print", $I_print,
        "--i_weights", $I_weights,
        "--no_reload"
    )
    if ($HalfRes) { $args += "--half_res" }
    return $args
}

# Helper: Build render args
function Get-RenderArgs {
    param($Scene, $NetworkType, $TimeMode)
    $args = @(
        "--data_basedir", $DataBaseDir,
        "--model_basedir", $ModelBaseDir,
        "--scene", $Scene,
        "--network_type", $NetworkType,
        "--time_mode", $TimeMode,
        "--n_frames", $VideoFrames,
        "--fps", $VideoFps
    )
    if ($HalfRes) { $args += "--half_res" }
    return $args
}

# Helper: Build eval args  
function Get-EvalArgs {
    param($Scene, $NetworkType)
    $args = @(
        "--data_basedir", $DataBaseDir,
        "--model_basedir", $ModelBaseDir,
        "--scene", $Scene,
        "--network_type", $NetworkType
    )
    if ($HalfRes) { $args += "--half_res" }
    return $args
}

# Stage 1: Training
if ($runTrain) {
    Write-Header "STAGE 1: TRAINING"
    Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
    
    $totalExp = $Scenes.Count * ($runStraightforward.ToInt32($null) + $runDeformation.ToInt32($null))
    $currentExp = 0
    
    foreach ($scene in $Scenes) {
        $dataDir = Join-Path $DataBaseDir $scene
        if (-not (Test-Path $dataDir)) { 
            Write-Warning "Scene not found, skipping: $dataDir"
            continue 
        }
        
        Write-Host ""
        Write-Host "---------------------------------------------"
        Write-Host "Training scene: $scene"
        Write-Host "---------------------------------------------"

        if ($runStraightforward) {
            $currentExp++
            $exp = "dnerf_straightforward_$scene"
            $trainArgs = Get-TrainArgs -DataDir $dataDir -ExpName $exp -NetworkType "straightforward"
            $log = Join-Path $ModelBaseDir "logs\${exp}_train.log"
            
            Write-Host ""
            Write-Host "[$currentExp/$totalExp] Training: $scene (straightforward)"
            Write-Host "Experiment: $exp"
            Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            Write-Host "Command: python train.py $($trainArgs -join ' ')"
            
            if (-not $DryRun) { 
                $process = Start-Process -FilePath "python" -ArgumentList (@("train.py") + $trainArgs) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
                if ($process.ExitCode -ne 0) { Write-Warning "Training failed with exit code $($process.ExitCode)" }
            }
            Write-Host "Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        }
        
        if ($runDeformation) {
            $currentExp++
            $exp = "dnerf_deformation_$scene" 
            $trainArgs = Get-TrainArgs -DataDir $dataDir -ExpName $exp -NetworkType "deformation"
            $log = Join-Path $ModelBaseDir "logs\${exp}_train.log"
            
            Write-Host ""
            Write-Host "[$currentExp/$totalExp] Training: $scene (deformation)"
            Write-Host "Experiment: $exp"
            Write-Host "Start: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
            Write-Host "Command: python train.py $($trainArgs -join ' ')"
            
            if (-not $DryRun) {
                $process = Start-Process -FilePath "python" -ArgumentList (@("train.py") + $trainArgs) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
                if ($process.ExitCode -ne 0) { Write-Warning "Training failed with exit code: $($process.ExitCode)" }
            }
            Write-Host "Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        }
    }
    
    Write-Host ""
    Write-Header "STAGE 1 COMPLETED: All models trained"
    Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# Stage 2: Rendering
if ($runRender) {
    Write-Header "STAGE 2: RENDERING VIDEOS"
    Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
    
    foreach ($scene in $Scenes) {
        $dataDir = Join-Path $DataBaseDir $scene
        if (-not (Test-Path $dataDir)) {
            Write-Warning "Scene not found, skipping: $dataDir"  
            continue
        }
        
        Write-Host ""
        Write-Host "---------------------------------------------"
        Write-Host "Rendering scene: $scene"
        Write-Host "---------------------------------------------"
        
        if ($runStraightforward) {
            $exp = "dnerf_straightforward_$scene"
            $modelDir = Join-Path $ModelBaseDir $exp
            $ckpt = Join-Path $modelDir "best.tar"
            
            if (-not (Test-Path $ckpt)) {
                Write-Warning "Best checkpoint not found for $exp, skipping"
                continue
            }
            
            Write-Host ""
            Write-Host "Rendering: $scene (straightforward)"
            
            foreach ($timeMode in $TimeModes) {
                Write-Host "  - Time mode: $timeMode"
                $renderArgs = Get-RenderArgs -Scene $scene -NetworkType "straightforward" -TimeMode $timeMode
                $log = Join-Path $ModelBaseDir "logs\${exp}_render_${timeMode}.log"
                Write-Host "Command: python render_video.py $($renderArgs -join ' ')"
                
                if (-not $DryRun) {
                    $process = Start-Process -FilePath "python" -ArgumentList (@("render_video.py") + $renderArgs) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
                    if ($process.ExitCode -ne 0) { Write-Warning "Rendering failed with exit code: $($process.ExitCode)" }
                }
            }
            Write-Host "Completed rendering: $scene (straightforward)"
        }
        
        if ($runDeformation) {
            $exp = "dnerf_deformation_$scene"
            $modelDir = Join-Path $ModelBaseDir $exp
            $ckpt = Join-Path $modelDir "best.tar"
            
            if (-not (Test-Path $ckpt)) {
                Write-Warning "Best checkpoint not found for $exp, skipping"
                continue
            }
            
            Write-Host ""
            Write-Host "Rendering: $scene (deformation)"
            
            foreach ($timeMode in $TimeModes) {
                Write-Host "  - Time mode: $timeMode"
                $renderArgs = Get-RenderArgs -Scene $scene -NetworkType "deformation" -TimeMode $timeMode
                $log = Join-Path $ModelBaseDir "logs\${exp}_render_${timeMode}.log"
                Write-Host "Command: python render_video.py $($renderArgs -join ' ')"
                
                if (-not $DryRun) {
                    $process = Start-Process -FilePath "python" -ArgumentList (@("render_video.py") + $renderArgs) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
                    if ($process.ExitCode -ne 0) { Write-Warning "Rendering failed with exit code: $($process.ExitCode)" }
                }
            }
            Write-Host "Completed rendering: $scene (deformation)"
        }
    }
    
    Write-Host ""
    Write-Header "STAGE 2 COMPLETED: All videos rendered"
    Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
}

# Stage 3: Evaluation  
if ($runEval) {
    Write-Header "STAGE 3: EVALUATION"
    Write-Host "Start time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host ""
    
    $resultsDir = Join-Path $ModelBaseDir 'results'
    New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null
    
    foreach ($scene in $Scenes) {
        $dataDir = Join-Path $DataBaseDir $scene
        if (-not (Test-Path $dataDir)) {
            Write-Warning "Scene not found, skipping: $dataDir"
            continue
        }
        
        Write-Host ""
        Write-Host "---------------------------------------------"
        Write-Host "Evaluating scene: $scene"
        Write-Host "---------------------------------------------"
        
        if ($runStraightforward) {
            $exp = "dnerf_straightforward_$scene"
            $ckpt = Join-Path $ModelBaseDir "$exp\best.tar"
            
            if (Test-Path $ckpt) {
                Write-Host ""
                Write-Host "Evaluating: $scene (straightforward)"
                $evalArgs = Get-EvalArgs -Scene $scene -NetworkType "straightforward"
                $log = Join-Path $ModelBaseDir "logs\${exp}_eval.log"
                Write-Host "Command: python evaluate.py $($evalArgs -join ' ')"
                
                if (-not $DryRun) {
                    $process = Start-Process -FilePath "python" -ArgumentList (@("evaluate.py") + $evalArgs) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
                    if ($process.ExitCode -ne 0) { Write-Warning "Evaluation failed with exit code: $($process.ExitCode)" }
                }
                Write-Host "Completed evaluation: $scene (straightforward)"
            } else {
                Write-Warning "Checkpoint not found: $ckpt"
            }
        }
        
        if ($runDeformation) {
            $exp = "dnerf_deformation_$scene"
            $ckpt = Join-Path $ModelBaseDir "$exp\best.tar"
            
            if (Test-Path $ckpt) {
                Write-Host ""
                Write-Host "Evaluating: $scene (deformation)"
                $evalArgs = Get-EvalArgs -Scene $scene -NetworkType "deformation"  
                $log = Join-Path $ModelBaseDir "logs\${exp}_eval.log"
                Write-Host "Command: python evaluate.py $($evalArgs -join ' ')"
                
                if (-not $DryRun) {
                    $process = Start-Process -FilePath "python" -ArgumentList (@("evaluate.py") + $evalArgs) -NoNewWindow -Wait -PassThru -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
                    if ($process.ExitCode -ne 0) { Write-Warning "Evaluation failed with exit code: $($process.ExitCode)" }
                }
                Write-Host "Completed evaluation: $scene (deformation)"
            } else {
                Write-Warning "Checkpoint not found: $ckpt"  
            }
        }
    }
    
    Write-Host ""
    Write-Header "STAGE 3 COMPLETED: All evaluations done"
    Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    
    # Generate summary report
    Write-Host ""
    Write-Host "---------------------------------------------"
    Write-Host "Generating Summary Report"
    Write-Host "---------------------------------------------"
    
    $summary = Join-Path $resultsDir 'summary.txt'
    "Dynamic NeRF Evaluation Summary" | Out-File -FilePath $summary -Encoding UTF8
    "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $summary -Append -Encoding UTF8
    "========================================" | Out-File -FilePath $summary -Append -Encoding UTF8
    "" | Out-File -FilePath $summary -Append -Encoding UTF8
    
    foreach ($scene in $Scenes) {
        "Scene: $scene" | Out-File -FilePath $summary -Append -Encoding UTF8
        "----------------------------------------" | Out-File -FilePath $summary -Append -Encoding UTF8
        
        # Straightforward results
        if ($runStraightforward) {
            $metrics = Join-Path $ModelBaseDir "dnerf_straightforward_$scene\evaluation\metrics.json"
            if (Test-Path $metrics) {
                try {
                    $obj = Get-Content $metrics | ConvertFrom-Json
                    $m = $obj.metrics
                    "  Straightforward:" | Out-File -FilePath $summary -Append -Encoding UTF8
                    "    PSNR: $($m.psnr.ToString('F2')) ± $($m.psnr_std.ToString('F2')) dB" | Out-File -FilePath $summary -Append -Encoding UTF8
                    "    SSIM: $($m.ssim.ToString('F4')) ± $($m.ssim_std.ToString('F4'))" | Out-File -FilePath $summary -Append -Encoding UTF8
                    if ($m.PSObject.Properties.Name -contains 'lpips') {
                        "    LPIPS: $($m.lpips.ToString('F4')) ± $($m.lpips_std.ToString('F4'))" | Out-File -FilePath $summary -Append -Encoding UTF8
                    }
                } catch {
                    "    (metrics parsing failed)" | Out-File -FilePath $summary -Append -Encoding UTF8
                }
            }
        }
        
        # Deformation results  
        if ($runDeformation) {
            $metrics = Join-Path $ModelBaseDir "dnerf_deformation_$scene\evaluation\metrics.json"
            if (Test-Path $metrics) {
                try {
                    $obj = Get-Content $metrics | ConvertFrom-Json
                    $m = $obj.metrics
                    "  Deformation:" | Out-File -FilePath $summary -Append -Encoding UTF8
                    "    PSNR: $($m.psnr.ToString('F2')) ± $($m.psnr_std.ToString('F2')) dB" | Out-File -FilePath $summary -Append -Encoding UTF8
                    "    SSIM: $($m.ssim.ToString('F4')) ± $($m.ssim_std.ToString('F4'))" | Out-File -FilePath $summary -Append -Encoding UTF8
                    if ($m.PSObject.Properties.Name -contains 'lpips') {
                        "    LPIPS: $($m.lpips.ToString('F4')) ± $($m.lpips_std.ToString('F4'))" | Out-File -FilePath $summary -Append -Encoding UTF8  
                    }
                } catch {
                    "    (metrics parsing failed)" | Out-File -FilePath $summary -Append -Encoding UTF8
                }
            }
        }
        
        "" | Out-File -FilePath $summary -Append -Encoding UTF8
    }
    
    Write-Host "Summary report saved to: $summary"
    Write-Host ""
    Get-Content $summary | Write-Host
}

Write-Header "PIPELINE COMPLETED!"
Write-Host "End time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host ""

Write-Host "Summary of Outputs:"
Write-Host ""

foreach ($scene in $Scenes) {
    Write-Host "Scene: $scene"
    Write-Host "  Methods:"
    
    if ($runStraightforward) {
        $exp = "dnerf_straightforward_$scene"
        $modelDir = Join-Path $ModelBaseDir $exp
        
        if (Test-Path $modelDir) {
            Write-Host "    ✓ Straightforward"
            if (Test-Path (Join-Path $modelDir "best.tar")) { Write-Host "      - Model: $modelDir\best.tar" }
            if (Test-Path (Join-Path $modelDir "videos")) { Write-Host "      - Videos: $modelDir\videos\" }
            if (Test-Path (Join-Path $modelDir "evaluation")) { Write-Host "      - Evaluation: $modelDir\evaluation\" }
        }
    }
    
    if ($runDeformation) {
        $exp = "dnerf_deformation_$scene"
        $modelDir = Join-Path $ModelBaseDir $exp
        
        if (Test-Path $modelDir) {
            Write-Host "    ✓ Deformation"
            if (Test-Path (Join-Path $modelDir "best.tar")) { Write-Host "      - Model: $modelDir\best.tar" }
            if (Test-Path (Join-Path $modelDir "videos")) { Write-Host "      - Videos: $modelDir\videos\" }
            if (Test-Path (Join-Path $modelDir "evaluation")) { Write-Host "      - Evaluation: $modelDir\evaluation\" }
        }
    }
    
    Write-Host ""
}

Write-Host "============================================="
Write-Host "All logs saved to: $(Join-Path $ModelBaseDir 'logs')"
Write-Host "All results saved to: $(Join-Path $ModelBaseDir 'results')"
Write-Host "============================================="
