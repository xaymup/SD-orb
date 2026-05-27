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

        mode = config.get('kaleido_mode', 'Off')
        if mode != 'Off':
            band_val = {
                'Bass':  bass,
                'Mids':  mids,
                'Highs': highs,
            }.get(config.get('kaleido_band', 'Bass'), bass)
            amount = config.get('kaleido_base', 0.0) + band_val * 0.5 * config.get('kaleido_sens', 0.0)
            amount = max(0.0, min(1.0, amount))

            if mode in ('Mirror-fold', 'Stretch-in-folds'):
                # 2 → 16 segments. Mirror by folding into a wedge and taking abs.
                N = 2.0 + amount * 14.0
                wedge = torch.pi / N
                new_ang = torch.abs(((new_ang + wedge) % (2.0 * wedge)) - wedge)

            if mode in ('Anisotropic stretch', 'Stretch-in-folds'):
                # cos(2·ANG) elongates the sampling radius along one axis and
                # compresses the perpendicular axis, producing an audio-pulsed
                # squash/stretch. Use the pre-warp angle so the stretch axis
                # doesn't spin with the rotate effect.
                stretch = 0.5 if mode == 'Anisotropic stretch' else 0.3
                new_r = new_r * (1.0 + amount * stretch * torch.cos(2.0 * self.ANG))

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
