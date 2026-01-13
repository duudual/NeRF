# =============================================================================
# Run All Dynamic NeRF Training (PowerShell version for Windows)
# This script trains both Straightforward and Deformation networks 
# on all D-NeRF dataset scenes
# =============================================================================

param(
    [switch]$Straightforward,
    [switch]$Deform,
    [string[]]$Scene = @(),
    [switch]$Help
)

$AllScenes = @("bouncingballs", "hellwarrior", "hook", "jumpingjacks", "lego", "mutant", "standup", "trex")

function Print-Usage {
    Write-Host "Usage: .\run_all_dnerf.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -Straightforward    Run only straightforward method"
    Write-Host "  -Deform             Run only deformation method"
    Write-Host "  -Scene SCENE        Run only specified scene(s)"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "Available scenes: $($AllScenes -join ', ')"
}

if ($Help) {
    Print-Usage
    exit 0
}

# Determine which methods to run
$RunStraightforward = $true
$RunDeform = $true

if ($Straightforward -and -not $Deform) {
    $RunDeform = $false
}
if ($Deform -and -not $Straightforward) {
    $RunStraightforward = $false
}

# Determine which scenes to run
if ($Scene.Count -gt 0) {
    $Scenes = $Scene
} else {
    $Scenes = $AllScenes
}

# Print configuration
Write-Host "============================================="
Write-Host "Dynamic NeRF Training Configuration"
Write-Host "============================================="
Write-Host "Scenes: $($Scenes -join ', ')"
Write-Host "Run Straightforward: $RunStraightforward"
Write-Host "Run Deformation: $RunDeform"
Write-Host "============================================="
Write-Host ""

# Create logs directory if not exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

# Training loop
foreach ($scene in $Scenes) {
    Write-Host ""
    Write-Host "============================================="
    Write-Host "Processing scene: $scene"
    Write-Host "============================================="
    
    # Straightforward method
    if ($RunStraightforward) {
        $configFile = "configs\${scene}_straightforward.txt"
        if (Test-Path $configFile) {
            Write-Host ""
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training Straightforward on $scene..."
            python train.py --config $configFile
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Training failed for $scene (straightforward)" -ForegroundColor Red
            } else {
                Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Completed Straightforward on $scene" -ForegroundColor Green
            }
        } else {
            Write-Host "Warning: Config file not found: $configFile" -ForegroundColor Yellow
        }
    }
    
    # Deformation method
    if ($RunDeform) {
        $configFile = "configs\${scene}_deform.txt"
        if (Test-Path $configFile) {
            Write-Host ""
            Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Training Deformation on $scene..."
            python train.py --config $configFile
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error: Training failed for $scene (deformation)" -ForegroundColor Red
            } else {
                Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Completed Deformation on $scene" -ForegroundColor Green
            }
        } else {
            Write-Host "Warning: Config file not found: $configFile" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "============================================="
Write-Host "All training completed!"
Write-Host "============================================="
Write-Host ""

# Summary
Write-Host "Training Summary:"
foreach ($scene in $Scenes) {
    Write-Host "  ${scene}:"
    if ($RunStraightforward) {
        $logDir = "logs\dnerf_straightforward_$scene"
        if (Test-Path $logDir) {
            Write-Host "    - Straightforward: $logDir"
        }
    }
    if ($RunDeform) {
        $logDir = "logs\dnerf_deformation_$scene"
        if (Test-Path $logDir) {
            Write-Host "    - Deformation: $logDir"
        }
    }
}
