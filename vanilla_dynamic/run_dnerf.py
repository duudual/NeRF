"""
Dynamic NeRF Main Entry Point

Usage:
    python run_dnerf.py train --config configs/bouncingballs_deform.txt
    python run_dnerf.py render --ckpt logs/model/200000.tar --datadir data/bouncingballs
    python run_dnerf.py evaluate --ckpt logs/model/200000.tar --datadir data/bouncingballs
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        print("Dynamic NeRF - Neural Radiance Fields for Dynamic Scenes")
        print()
        print("Usage:")
        print("  python run_dnerf.py <command> [options]")
        print()
        print("Commands:")
        print("  train     Train a Dynamic NeRF model")
        print("  render    Render 360 video from trained model")
        print("  evaluate  Evaluate and compare models")
        print()
        print("Examples:")
        print("  python run_dnerf.py train --config configs/bouncingballs_deform.txt")
        print("  python run_dnerf.py render --ckpt logs/model/200000.tar --datadir data/bouncingballs")
        print("  python run_dnerf.py evaluate --ckpt_deformation logs/model/200000.tar --datadir data/bouncingballs")
        return
    
    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove the command from argv
    
    if command == 'train':
        from train import train
        train()
    elif command == 'render':
        from render_video import main as render_main
        render_main()
    elif command == 'evaluate':
        from evaluate import main as evaluate_main
        evaluate_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, render, evaluate")


if __name__ == '__main__':
    main()
