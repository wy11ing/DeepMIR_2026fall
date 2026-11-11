"""
Classifier-free guidance (CFG) for Stable Audio Open. Implement `cfg` below.

apg: adaptive projected guidance is implemented to improve quality
"""
import torch

class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_avg = 0
    
    def update(self, update_value: torch.Tensor):
        new_avg = self.momentum * self.running_avg
        self.running_avg = update_value + new_avg

def project(v0: torch.Tensor, v1: torch.Tensor):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])

    v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel

    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def apg(noise_pred_uncond, noise_pred_cond, guidance_scale, momentum_buffer: MomentumBuffer = None, eta: float = 0.0, norm_threshold: float = 0.0):
    diff = noise_pred_cond - noise_pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_avg
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    
    diff_parallel, diff_orthogonal = project(diff, noise_pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = noise_pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided

def cfg(noise_pred_uncond, noise_pred_cond, guidance_scale):
    """TODO: combine the unconditional and conditional model outputs of one denoising step.

    Args:
        noise_pred_uncond: tensor (1, C, L), model output with an empty (all-zero) condition.
        noise_pred_cond:   tensor (1, C, L), model output with the text prompt.
        guidance_scale:    float. 1.0 means no guidance, 0.0 means unconditional.
    Returns:
        tensor (1, C, L), the guided output passed to the scheduler.
    """
    guided_output = (1 - guidance_scale) * noise_pred_uncond + guidance_scale * noise_pred_cond
    return guided_output
