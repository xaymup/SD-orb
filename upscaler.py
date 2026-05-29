"""Optional GPU upscalers for the projector + recording paths.

Two profiles:
  fast — SRVGGNetCompact + realesr-general-x4v3 weights (~1.5M params, x4
         scale). Designed for video super-resolution. ~3-5 ms/frame.
  hq   — RRDBNet + RealESRGAN_x2plus weights (~16.7M params, x2). General
         super-resolution. ~10-15 ms/frame. Sharper on stills, slower per
         frame.

We vendor both architectures rather than depending on basicsr/realesrgan
PyPI packages — those drag in mmcv, opencv-contrib, tb-nightly, etc. for
two small networks whose forward passes fit in a screen of code.

Weights are pulled on demand from the official Real-ESRGAN GitHub
releases; users who never enable the upscaler don't pay the disk cost.
"""

from __future__ import annotations

import os
import threading
from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RRDBNet (Real-ESRGAN x2plus): heavy, high quality
# ---------------------------------------------------------------------------

def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    b, c, hh, hw = x.size()
    return (
        x.view(b, c, hh // scale, scale, hw // scale, scale)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(b, c * scale * scale, hh // scale, hw // scale)
    )


class _ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat,                    num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch,      num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch,  num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch,  num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch,  num_feat,    3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class _RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = _ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = _ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class _RRDBNet(nn.Module):
    def __init__(self, scale: int = 2, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale
        in_ch_actual = num_in_ch * (4 if scale == 2 else 16 if scale == 1 else 1)
        self.conv_first = nn.Conv2d(in_ch_actual, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[_RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr  = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = _pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = _pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# SRVGGNetCompact (Real-ESRGAN general-x4v3): light, fast
# ---------------------------------------------------------------------------

class _SRVGGNetCompact(nn.Module):
    """Compact VGG-style super-resolution network. Per-frame cost is
    dominated by num_conv inner convs — at num_conv=32 + num_feat=64 it's
    ~5-10× faster than RRDBNet at 23 blocks. Identity-style residual via
    nearest-neighbor base lets the network learn the high-freq residual."""

    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_conv: int = 32, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        layers: list[nn.Module] = [
            nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
            nn.PReLU(num_parameters=num_feat),
        ]
        for _ in range(num_conv):
            layers += [
                nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                nn.PReLU(num_parameters=num_feat),
            ]
        layers.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(layers)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        out = out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out


# ---------------------------------------------------------------------------
# Profiles (a profile is a model architecture + weights + scale + URL)
# ---------------------------------------------------------------------------

WEIGHTS_DIR = "weights"


PROFILES: dict[str, dict] = {
    "fast (x4)": {
        "filename": "realesr-general-x4v3.pth",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.5.0/realesr-general-x4v3.pth"
        ),
        "arch": "compact",
        "scale": 4,
    },
    "hq (x2)": {
        "filename": "RealESRGAN_x2plus.pth",
        "url": (
            "https://github.com/xinntao/Real-ESRGAN/releases/download/"
            "v0.2.1/RealESRGAN_x2plus.pth"
        ),
        "arch": "rrdbnet",
        "scale": 2,
    },
}

DEFAULT_PROFILE = "fast (x4)"


def _ensure_weights(filename: str, url: str, progress: bool = True) -> str:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, filename)
    if os.path.isfile(path):
        return path
    print(f"Downloading {url} → {path}...")

    def _hook(blocks: int, block_size: int, total: int):
        if not progress or total <= 0:
            return
        pct = min(100.0, blocks * block_size * 100.0 / total)
        print(f"  {pct:5.1f}%", end="\r", flush=True)

    tmp = path + ".part"
    urlretrieve(url, tmp, reporthook=_hook if progress else None)
    os.rename(tmp, path)
    print(f"\nSaved {path}")
    return path


def _build_model(profile: dict) -> nn.Module:
    if profile["arch"] == "rrdbnet":
        return _RRDBNet(scale=profile["scale"], num_block=23)
    if profile["arch"] == "compact":
        return _SRVGGNetCompact(num_conv=32, upscale=profile["scale"])
    raise ValueError(f"unknown upscaler arch: {profile['arch']}")


def _load_state_dict_loose(model: nn.Module, state) -> None:
    """Real-ESRGAN checkpoints wrap weights under 'params_ema' or 'params',
    sometimes ship as a raw dict. The compact-arch general-x4v3 checkpoint
    is a raw dict whose keys live under 'body.N.' — matches our module
    layout directly. Try the wrappers first then fall back to the dict."""
    if isinstance(state, dict):
        for key in ("params_ema", "params"):
            if key in state and isinstance(state[key], dict):
                model.load_state_dict(state[key], strict=True)
                return
    model.load_state_dict(state, strict=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Upscaler:
    """GPU upscaler with swappable model profiles. Loaded lazily — only
    consumes VRAM if the user actually enables it."""

    def __init__(self, profile: str = DEFAULT_PROFILE):
        if profile not in PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        self._profile_key = profile
        self._model: nn.Module | None = None
        self._lock = threading.Lock()
        self.enabled = False

    @property
    def profile(self) -> str:
        return self._profile_key

    @property
    def scale(self) -> int:
        return PROFILES[self._profile_key]["scale"]

    @staticmethod
    def list_profiles() -> list[str]:
        return list(PROFILES.keys())

    def is_loaded(self) -> bool:
        return self._model is not None

    def set_profile(self, profile: str) -> None:
        """Switch to a different model profile. If the previous one was
        loaded, the new one gets loaded too — keeps the upscaler in a
        consistent ready/not-ready state across the swap."""
        if profile == self._profile_key:
            return
        if profile not in PROFILES:
            raise ValueError(f"unknown profile: {profile}")
        was_loaded = self.is_loaded()
        self.unload()
        self._profile_key = profile
        if was_loaded:
            self.load()

    def load(self) -> None:
        """Build the model, download weights if needed, fp16+cuda+eval.
        Idempotent — calling load() while already loaded is a no-op."""
        with self._lock:
            if self._model is not None:
                return
            spec = PROFILES[self._profile_key]
            weights_path = _ensure_weights(spec["filename"], spec["url"])
            model = _build_model(spec)
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            _load_state_dict_loose(model, state)
            model.eval().half().to("cuda")
            self._model = model

    def unload(self) -> None:
        with self._lock:
            self._model = None
            torch.cuda.empty_cache()
        self.enabled = False

    @torch.no_grad()
    def upscale(self, tensor: torch.Tensor) -> torch.Tensor:
        """tensor: (1, 3, H, W) float16 CUDA in [0, 1].
        Returns: (1, 3, H*scale, W*scale) float16 CUDA in [0, 1]."""
        if self._model is None:
            raise RuntimeError("Upscaler.upscale() called before load()")
        with self._lock:
            out = self._model(tensor)
        return out.clamp(0.0, 1.0)
