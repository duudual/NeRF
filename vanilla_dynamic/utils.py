"""
Utility Functions for Dynamic NeRF

- Device configuration
- Image metrics (MSE, PSNR, SSIM, LPIPS)
- Image processing utilities
"""

import torch
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

# Basic utility functions
img2mse = lambda x, y: torch.mean((x - y) ** 2)
log10 = torch.log(torch.tensor([10.0], device=device))
mse2psnr = lambda x: -10.0 * torch.log(x) / log10
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def compute_psnr(img1, img2):
    """Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: [H, W, 3] or [N, H, W, 3] image(s) in range [0, 1]
        img2: [H, W, 3] or [N, H, W, 3] image(s) in range [0, 1]
    
    Returns:
        psnr: scalar or array of PSNR values
    """
    if isinstance(img1, np.ndarray):
        mse = np.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            return 100.0
        return 20.0 * np.log10(1.0 / np.sqrt(mse))
    else:
        mse = torch.mean((img1 - img2) ** 2)
        if mse < 1e-10:
            return torch.tensor(100.0)
        return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(img1, img2, window_size=11, size_average=True, data_range=1.0):
    """Compute Structural Similarity Index between two images.
    
    This is a simple implementation. For more accurate results,
    consider using skimage.metrics.structural_similarity.
    
    Args:
        img1: [H, W, 3] image in range [0, 1]
        img2: [H, W, 3] image in range [0, 1]
        window_size: size of the sliding window
        size_average: if True, return average SSIM
        data_range: data range of the images
    
    Returns:
        ssim: scalar SSIM value
    """
    try:
        from skimage.metrics import structural_similarity
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        # Handle different image shapes
        if img1.ndim == 3 and img1.shape[-1] == 3:
            return structural_similarity(img1, img2, channel_axis=-1, data_range=data_range)
        else:
            return structural_similarity(img1, img2, data_range=data_range)
    except ImportError:
        # Fallback to simple SSIM approximation
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim


def compute_lpips(img1, img2, lpips_model=None):
    """Compute Learned Perceptual Image Patch Similarity.
    
    Requires lpips package: pip install lpips
    
    Args:
        img1: [H, W, 3] image in range [0, 1]
        img2: [H, W, 3] image in range [0, 1]
        lpips_model: pre-loaded LPIPS model (optional)
    
    Returns:
        lpips: scalar LPIPS value (lower is better)
    """
    try:
        import lpips
        
        if lpips_model is None:
            lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # Convert to torch tensors if needed
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float()
        
        # Reshape to [B, C, H, W] and scale to [-1, 1]
        img1 = img1.permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        img2 = img2.permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1
        
        with torch.no_grad():
            lpips_val = lpips_model(img1, img2)
        
        return lpips_val.item()
    except ImportError:
        print("Warning: lpips package not installed. LPIPS metric unavailable.")
        return 0.0


class LPIPSMetric:
    """LPIPS metric wrapper for efficient batch computation."""
    
    def __init__(self):
        self.model = None
    
    def _load_model(self):
        if self.model is None:
            try:
                import lpips
                self.model = lpips.LPIPS(net='alex').to(device)
                self.model.eval()
            except ImportError:
                print("Warning: lpips package not installed.")
                self.model = None
    
    def compute(self, img1, img2):
        """Compute LPIPS between two images."""
        self._load_model()
        if self.model is None:
            return 0.0
        return compute_lpips(img1, img2, self.model)


def evaluate_images(pred_imgs, gt_imgs, compute_lpips_flag=True):
    """Evaluate a set of predicted images against ground truth.
    
    Args:
        pred_imgs: [N, H, W, 3] predicted images
        gt_imgs: [N, H, W, 3] ground truth images
        compute_lpips_flag: whether to compute LPIPS (slower)
    
    Returns:
        metrics: dict with 'psnr', 'ssim', 'lpips' (average over all images)
    """
    if isinstance(pred_imgs, torch.Tensor):
        pred_imgs = pred_imgs.cpu().numpy()
    if isinstance(gt_imgs, torch.Tensor):
        gt_imgs = gt_imgs.cpu().numpy()
    
    n_imgs = pred_imgs.shape[0]
    psnrs = []
    ssims = []
    lpips_vals = []
    
    lpips_metric = LPIPSMetric() if compute_lpips_flag else None
    
    for i in range(n_imgs):
        pred = pred_imgs[i]
        gt = gt_imgs[i]
        
        psnrs.append(compute_psnr(pred, gt))
        ssims.append(compute_ssim(pred, gt))
        
        if compute_lpips_flag:
            lpips_vals.append(lpips_metric.compute(pred, gt))
    
    metrics = {
        'psnr': np.mean(psnrs),
        'ssim': np.mean(ssims),
        'psnr_std': np.std(psnrs),
        'ssim_std': np.std(ssims),
    }
    
    if compute_lpips_flag:
        metrics['lpips'] = np.mean(lpips_vals)
        metrics['lpips_std'] = np.std(lpips_vals)
    
    return metrics


def get_rays(H, W, K, c2w):
    """Get ray origins and directions from camera parameters.
    
    Args:
        H: image height
        W: image width
        K: camera intrinsic matrix [3, 3]
        c2w: camera-to-world transformation [3, 4] or [4, 4]
    
    Returns:
        rays_o: [H, W, 3] ray origins
        rays_d: [H, W, 3] ray directions
    """
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H),
        indexing='ij'
    )
    i = i.t()
    j = j.t()
    
    # Convert K to tensor if needed
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float()
    
    dirs = torch.stack([
        (i - K[0][2]) / K[0][0],
        -(j - K[1][2]) / K[1][1],
        -torch.ones_like(i)
    ], -1)
    
    # Convert c2w to tensor if needed
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).float()
    
    # Rotate ray directions from camera frame to world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    
    # Translate camera origin to world frame
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    """Get ray origins and directions (NumPy version).
    
    Args:
        H: image height
        W: image width
        K: camera intrinsic matrix [3, 3]
        c2w: camera-to-world transformation [3, 4] or [4, 4]
    
    Returns:
        rays_o: [H, W, 3] ray origins
        rays_d: [H, W, 3] ray directions
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), 
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([
        (i - K[0][2]) / K[0][0],
        -(j - K[1][2]) / K[1][1],
        -np.ones_like(i)
    ], -1)
    
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    
    return rays_o, rays_d
