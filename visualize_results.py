import os
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from utils import device, img2mse, mse2psnr, to8b

def load_image(path):
    """Load and normalize an image."""
    img = imageio.imread(path).astype(np.float32) / 255.0
    if img.shape[-1] == 4:  # RGBA
        img = img[..., :3]
    return img

def compute_psnr(rendered_dir, gt_dir):
    """Compute PSNR between rendered images and ground truth."""
    # Get sorted list of image paths
    rendered_paths = sorted(glob(os.path.join(rendered_dir, '*.png')))
    gt_paths = sorted(glob(os.path.join(gt_dir, '*.png')))
    
    if len(rendered_paths) != len(gt_paths):
        print(f"Warning: Number of rendered images ({len(rendered_paths)}) doesn't match ground truth ({len(gt_paths)})")
        min_len = min(len(rendered_paths), len(gt_paths))
        rendered_paths = rendered_paths[:min_len]
        gt_paths = gt_paths[:min_len]
    
    psnr_values = []
    for i, (render_path, gt_path) in enumerate(tqdm(zip(rendered_paths, gt_paths), total=len(rendered_paths))):
        render_img = load_image(render_path)
        gt_img = load_image(gt_path)
        
        # Ensure images are the same size
        if render_img.shape != gt_img.shape:
            # Resize rendered image to match ground truth
            from PIL import Image
            render_img = np.array(Image.fromarray((render_img * 255).astype(np.uint8)).resize(
                (gt_img.shape[1], gt_img.shape[0]), resample=Image.LANCZOS)) / 255.0
        
        # Compute PSNR
        mse = img2mse(torch.Tensor(render_img).to(device), torch.Tensor(gt_img).to(device)).item()
        psnr = mse2psnr(torch.Tensor([mse]).to(device)).item()
        psnr_values.append(psnr)
    
    return psnr_values, rendered_paths, gt_paths

def create_visualizations(psnr_values, rendered_paths, gt_paths, output_dir):
    """Create visualizations of the results."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute statistics
    avg_psnr = np.mean(psnr_values)
    std_psnr = np.std(psnr_values)
    max_psnr = np.max(psnr_values)
    min_psnr = np.min(psnr_values)
    
    print(f"PSNR Statistics:")
    print(f"  Average: {avg_psnr:.2f} dB")
    print(f"  Standard Deviation: {std_psnr:.2f} dB")
    print(f"  Maximum: {max_psnr:.2f} dB")
    print(f"  Minimum: {min_psnr:.2f} dB")
    
    # Save PSNR values to file
    psnr_file = os.path.join(output_dir, 'psnr_values.txt')
    with open(psnr_file, 'w') as f:
        f.write(f'PSNR Statistics:\n')
        f.write(f'  Average: {avg_psnr:.2f} dB\n')
        f.write(f'  Standard Deviation: {std_psnr:.2f} dB\n')
        f.write(f'  Maximum: {max_psnr:.2f} dB\n')
        f.write(f'  Minimum: {min_psnr:.2f} dB\n\n')
        f.write('PSNR per image:\n')
        for i, psnr in enumerate(psnr_values):
            f.write(f'Image {i+1}: {psnr:.2f} dB\n')
    
    # 1. PSNR Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(psnr_values, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(avg_psnr, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_psnr:.2f} dB')
    plt.xlabel('PSNR (dB)')
    plt.ylabel('Frequency')
    plt.title(f'PSNR Distribution (Avg: {avg_psnr:.2f} dB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'psnr_histogram.png'), dpi=300, bbox_inches='tight')
    
    # 2. Comparison Grid for First Few Images
    num_comparisons = min(6, len(psnr_values))
    fig, axes = plt.subplots(num_comparisons, 2, figsize=(12, 3*num_comparisons))
    
    for i in range(num_comparisons):
        render_img = load_image(rendered_paths[i])
        gt_img = load_image(gt_paths[i])
        
        # Rendered image
        axes[i, 0].imshow(render_img)
        axes[i, 0].set_title(f'Rendered (PSNR: {psnr_values[i]:.2f} dB)')
        axes[i, 0].axis('off')
        
        # Ground truth image
        axes[i, 1].imshow(gt_img)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_grid.png'), dpi=300, bbox_inches='tight')
    
    # 3. PSNR Line Plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, marker='o', linestyle='-', markersize=4, alpha=0.7)
    plt.axhline(avg_psnr, color='red', linestyle='--', linewidth=2, label=f'Average: {avg_psnr:.2f} dB')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR per Test Image')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'psnr_plot.png'), dpi=300, bbox_inches='tight')
    
    # 4. Error Maps for First Few Images
    num_error_maps = min(4, len(psnr_values))
    fig, axes = plt.subplots(num_error_maps, 3, figsize=(15, 4*num_error_maps))
    
    for i in range(num_error_maps):
        render_img = load_image(rendered_paths[i])
        gt_img = load_image(gt_paths[i])
        
        # Ensure images are the same size
        if render_img.shape != gt_img.shape:
            from PIL import Image
            render_img = np.array(Image.fromarray((render_img * 255).astype(np.uint8)).resize(
                (gt_img.shape[1], gt_img.shape[0]), resample=Image.LANCZOS)) / 255.0
        
        # Compute error map (MSE per pixel)
        error_map = (render_img - gt_img) ** 2
        error_map_color = np.clip(error_map * 10, 0, 1)  # Amplify errors for visibility
        
        # Rendered image
        axes[i, 0].imshow(render_img)
        axes[i, 0].set_title(f'Rendered (PSNR: {psnr_values[i]:.2f} dB)')
        axes[i, 0].axis('off')
        
        # Ground truth image
        axes[i, 1].imshow(gt_img)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Error map
        im = axes[i, 2].imshow(error_map_color, cmap='hot')
        axes[i, 2].set_title('Error Map (Amplified)')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_maps.png'), dpi=300, bbox_inches='tight')
    
    # 5. Save all rendered images as a video
    if len(rendered_paths) > 1:
        rendered_images = [load_image(path) for path in rendered_paths]
        video_path = os.path.join(output_dir, 'rendered_video.mp4')
        imageio.mimwrite(video_path, to8b(rendered_images), fps=10, quality=8)
        print(f'Rendered video saved to {video_path}')
    
    plt.close('all')
    print(f'All visualizations saved to {output_dir}')

def visualize_results(rendered_dir, gt_dir, output_dir):
    """Main function to visualize results."""
    print(f'Loading rendered images from: {rendered_dir}')
    print(f'Loading ground truth from: {gt_dir}')
    
    # Compute PSNR
    psnr_values, rendered_paths, gt_paths = compute_psnr(rendered_dir, gt_dir)
    
    # Create visualizations
    create_visualizations(psnr_values, rendered_paths, gt_paths, output_dir)
    
    return psnr_values

if __name__ == '__main__':
    # Example usage: python visualize_results.py --rendered_dir logs/fern_test/renderonly_path_200000 --gt_dir data/raw_nerf/nerf_llff_data/fern/images --output_dir logs/fern_test/visualization
    import argparse
    parser = argparse.ArgumentParser(description='Visualize NeRF results and compute PSNR')
    parser.add_argument('--rendered_dir', type=str, required=True, help='Directory containing rendered images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing ground truth images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save visualizations')
    args = parser.parse_args()
    
    visualize_results(args.rendered_dir, args.gt_dir, args.output_dir)
