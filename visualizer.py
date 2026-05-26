import torch
import torch.nn.functional as F


class Visualizer:
    def __init__(self, width=512, height=512, device='cuda'):
        self.width = width
        self.height = height
        self.device = device

        Y, X = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=device),
            torch.arange(width,  dtype=torch.float32, device=device),
            indexing='ij',
        )
        X_NORM = (X / (width  / 2)) - 1.0
        Y_NORM = (Y / (height / 2)) - 1.0
        self.R   = torch.sqrt(X_NORM ** 2 + Y_NORM ** 2)
        self.ANG = torch.atan2(Y_NORM, X_NORM)

        self._angle_accumulator = 0.0

    def apply_feedback(self, input_tensor, bands, config):
        """
        input_tensor: (1, 3, H, W) CUDA tensor in [0, 1].
        Returns same shape/device. Stays on GPU end-to-end via F.grid_sample.
        """
        bass, mids, highs = bands
        # Strong enough that default sliders punch through AI re-centering.
        # The UNet tries to redraw the prompt looking "correct" each frame, so
        # the geometric pull has to outpace that to be perceptible.
        zoom = config['zoom_base'] - (bass * 0.25 * config['zoom_sens'])
        rot  = config['rot_base']  + (mids * 0.15 * config['rot_sens'])

        new_r   = self.R * zoom
        new_ang = self.ANG + rot
        grid = torch.stack(
            (torch.cos(new_ang) * new_r, torch.sin(new_ang) * new_r),
            dim=-1,
        ).unsqueeze(0).to(input_tensor.dtype)

        warped = F.grid_sample(
            input_tensor, grid,
            mode='bilinear', padding_mode='reflection', align_corners=False,
        )

        decay = 0.92 + (highs * 0.06)
        warped = warped * decay

        return warped.clamp(0.0, 1.0)
