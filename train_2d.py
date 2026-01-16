import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from utils import device, img2mse, mse2psnr, to8b
from positional_encoding import Embedder


class MLP2D(nn.Module):
    """Simple MLP for 2D image fitting."""
    def __init__(self, D=4, W=256, input_ch=2, output_ch=3, skips=[2]):
        """Initialize 2D MLP model.
        
        Args:
            D: number of layers
            W: number of channels per layer
            input_ch: input channel dimension (2 for 2D coordinates, more with positional encoding)
            output_ch: output channel dimension (3 for RGB)
            skips: layers to add skip connections
        """
        super(MLP2D, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.skips = skips
        
        self.linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        self.output_linear = nn.Linear(W, output_ch)
        self.activation = F.relu

    def forward(self, x):
        h = x
        for i, l in enumerate(self.linears):
            h = self.linears[i](h)
            h = self.activation(h)
            if i in self.skips:
                h = torch.cat([x, h], -1)
        
        outputs = self.output_linear(h)
        return outputs


def load_image(img_path):
    """Load image and normalize to [0, 1]."""
    img = imageio.imread(img_path)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def create_composite_figure(pred_img, target_img, psnr_history, epoch, out_dir):
    """Create composite figure with predicted image, target image, and PSNR curve.
    
    Args:
        pred_img: Predicted image (H, W, 3)
        target_img: Target image (H, W, 3)
        psnr_history: List of PSNR values over epochs
        epoch: Current epoch number
        out_dir: Output directory to save the figure
    """
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 1.5])
    
    # Predicted image
    ax1 = plt.subplot(gs[0])
    ax1.imshow(pred_img)
    ax1.set_title(f'Iteration {epoch}')
    ax1.axis('off')
    
    # Target image
    ax2 = plt.subplot(gs[1])
    ax2.imshow(target_img)
    ax2.set_title('Target image')
    ax2.axis('off')
    
    # PSNR curve
    ax3 = plt.subplot(gs[2])
    ax3.plot(psnr_history, linewidth=2, color='blue')
    ax3.set_title('PSNR')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('PSNR (dB)')
    ax3.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(out_dir, f'composite_epoch_{epoch}.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path


def get_2d_coords(h, w):
    """Generate 2D coordinates in range [-1, 1]."""
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    coords = np.stack([xx, yy], axis=-1)
    return coords


def get_embedder_2d(multires, i=0):
    """Get positional encoding embedder for 2D inputs.
    
    Args:
        multires: log2 of max freq for positional encoding
        i: set 0 for default positional encoding, -1 for none
    """
    if i == -1:
        return nn.Identity(), 2
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


def train_2d():
    """Main training function for 2D image fitting."""
    parser = argparse.ArgumentParser(description='2D Image Fitting with MLP')
    
    # Model parameters
    parser.add_argument('--img_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--D', type=int, default=4, help='Number of layers in MLP')
    parser.add_argument('--W', type=int, default=256, help='Number of channels per layer')
    parser.add_argument('--skips', type=int, nargs='+', default=[2], help='Layers to add skip connections')
    
    # Positional encoding parameters
    parser.add_argument('--multires', type=int, default=10, help='Log2 of max freq for positional encoding')
    parser.add_argument('--no_posenc', action='store_true', help='Disable positional encoding')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--chunk', type=int, default=4096, help='Chunk size for evaluation')
    
    # Output parameters
    parser.add_argument('--out_dir', type=str, default='output_2d', help='Output directory')
    parser.add_argument('--i_print', type=int, default=100, help='Frequency of printing progress')
    parser.add_argument('--i_save', type=int, default=500, help='Frequency of saving images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Load image
    img = load_image(args.img_path)
    h, w, _ = img.shape
    print(f'Loaded image: {args.img_path}, shape: {h}x{w}x{img.shape[2]}')
    
    # Generate 2D coordinates
    coords = get_2d_coords(h, w)
    coords = coords.reshape(-1, 2)
    img_flat = img.reshape(-1, 3)
    
    # Convert to tensors
    coords = torch.Tensor(coords).to(device)
    img_flat = torch.Tensor(img_flat).to(device)
    
    # Get positional encoding
    if args.no_posenc:
        embed_fn, input_ch = get_embedder_2d(0, i=-1)
    else:
        embed_fn, input_ch = get_embedder_2d(args.multires)
    
    # Create model
    model = MLP2D(D=args.D, W=args.W, input_ch=input_ch, output_ch=3, skips=args.skips).to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    start_time = time.time()
    psnr_history = []
    
    for epoch in tqdm(range(args.epochs)):
        # Shuffle data
        perm = torch.randperm(coords.shape[0])
        coords_shuffled = coords[perm]
        img_shuffled = img_flat[perm]
        
        # Training batch loop
        total_loss = 0.0
        num_batches = coords.shape[0] // args.batch_size
        
        for i in range(num_batches):
            # Get batch
            start = i * args.batch_size
            end = start + args.batch_size
            coords_batch = coords_shuffled[start:end]
            img_batch = img_shuffled[start:end]
            
            # Embed coordinates
            coords_embedded = embed_fn(coords_batch)
            
            # Forward pass
            pred = model(coords_embedded)
            
            # Compute loss
            loss = img2mse(pred, img_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Compute average loss and PSNR
        avg_loss = total_loss / num_batches
        psnr = mse2psnr(torch.tensor(avg_loss)).item()
        psnr_history.append(psnr)
        
        # Print progress
        if (epoch + 1) % args.i_print == 0:
            tqdm.write(f'Epoch: {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, PSNR: {psnr:.2f} dB, Time: {time.time()-start_time:.2f}s')
        
        # Save image and create composite figure
        if (epoch + 1) % args.i_save == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                # Predict image in chunks
                pred_img = []
                for i in range(0, coords.shape[0], args.chunk):
                    end = min(i + args.chunk, coords.shape[0])
                    coords_chunk = coords[i:end]
                    coords_embedded = embed_fn(coords_chunk)
                    pred_chunk = model(coords_embedded)
                    pred_img.append(pred_chunk)
                pred_img = torch.cat(pred_img, dim=0)
                pred_img = pred_img.reshape(h, w, 3)
                pred_img_np = pred_img.cpu().numpy()
                pred_img_np = np.clip(pred_img_np, 0.0, 1.0)
            
            # Save individual prediction image
            img_path = os.path.join(args.out_dir, f'pred_epoch_{epoch+1}.png')
            imageio.imsave(img_path, to8b(pred_img_np))
            tqdm.write(f'Saved prediction to {img_path}')
            
            # Create and save composite figure
            composite_path = create_composite_figure(pred_img_np, img, psnr_history, epoch+1, args.out_dir)
            tqdm.write(f'Saved composite figure to {composite_path}')
            
            model.train()
    
    # Save final model
    model_path = os.path.join(args.out_dir, 'model_final.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path}')
    
    # Save PSNR history
    psnr_path = os.path.join(args.out_dir, 'psnr_history.npy')
    np.save(psnr_path, np.array(psnr_history))
    print(f'Saved PSNR history to {psnr_path}')
    
    # Save configuration
    config_path = os.path.join(args.out_dir, 'config.txt')
    with open(config_path, 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


if __name__ == '__main__':
    train_2d()
