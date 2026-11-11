"""Classifier-free guidance (CFG) for Stable Audio Open. Implement `cfg` below."""


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
