"""DeepFloyd IF, pixel-space, with a selectable guidance stage.

Pixel space puts no VAE between the guided variable and the reader, which is
why Visual Anagrams and Images that Sound use this cascade.  The catch is
resolution: stage I runs at 64x64, too coarse for peripheral-drift structure.
`stage` therefore selects where the hook is applied:

    "base"  64x64   fast, likely too coarse -- the honest baseline
    "sr1"   256x256 guide the stage-II super-resolver (recommended)

Licence: DeepFloyd IF is released for NON-COMMERCIAL research use.  Report the
SDXL variant as the licence-clean headline and this as a quality comparison.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import Generator, GuidanceHook


class DeepFloydGenerator(Generator):
    name = "deepfloyd"
    stages = ("base", "sr1")

    def __init__(self, device: str = "cuda", stage: str = "sr1", steps: int = 50,
                 cfg_scale: float = 7.0, guide_from: float = 0.2,
                 guide_to: float = 0.9, guide_every: int = 1,
                 guide_lr: float = 2.0, grad_clip: float = 1.0,
                 out_size: int = 0, dtype=torch.float16):
        assert stage in self.stages, f"stage must be one of {self.stages}"
        self.device, self.stage, self.dtype = device, stage, dtype
        self.steps, self.cfg_scale = int(steps), float(cfg_scale)
        self.guide_from, self.guide_to = float(guide_from), float(guide_to)
        self.guide_every, self.guide_lr = int(guide_every), float(guide_lr)
        self.grad_clip = float(grad_clip)
        self.out_size = int(out_size)
        self._s1 = self._s2 = None

    def _load(self):
        if self._s1 is None:
            from diffusers import DiffusionPipeline
            self._s1 = DiffusionPipeline.from_pretrained(
                "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=self.dtype
            ).to(self.device)
            self._s1.set_progress_bar_config(disable=True)
            self._s1.unet.requires_grad_(False)
            if self.stage == "sr1":
                self._s2 = DiffusionPipeline.from_pretrained(
                    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16",
                    torch_dtype=self.dtype).to(self.device)
                self._s2.set_progress_bar_config(disable=True)
                self._s2.unet.requires_grad_(False)
        return self._s1, self._s2

    def _guided_loop(self, pipe, prompt_emb, neg_emb, shape, seed, hook,
                     extra_cond=None):
        steps = self.steps
        pipe.scheduler.set_timesteps(steps, device=self.device)
        ts = pipe.scheduler.timesteps
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        x = torch.randn(shape, generator=g, device=self.device, dtype=self.dtype)
        emb = torch.cat([neg_emb, prompt_emb])
        lo, hi = int(self.guide_from * steps), int(self.guide_to * steps)
        trace = {"step": [], "rho": [], "loss": []}
        for i, t in enumerate(ts):
            with torch.no_grad():
                inp = torch.cat([x] * 2)
                if extra_cond is not None:
                    inp = torch.cat([inp, torch.cat([extra_cond] * 2)], 1)
                inp = pipe.scheduler.scale_model_input(inp, t)
                out = pipe.unet(inp, t, encoder_hidden_states=emb).sample
                out, _ = out.split(x.shape[1], dim=1)
                nu, nc = out.chunk(2)
                noise = nu + self.cfg_scale * (nc - nu)
            if hook is not None and lo <= i < hi and (i - lo) % self.guide_every == 0:
                x = x.detach().requires_grad_(True)
                a = pipe.scheduler.alphas_cumprod.to(self.device)[t]
                x0 = ((x - (1 - a).sqrt() * noise) / a.sqrt())
                img = ((x0.float() / 2 + 0.5).clamp(0, 1))[0]
                loss, info = hook(img, i)
                grad, = torch.autograd.grad(loss, x)
                if self.grad_clip > 0:
                    n = grad.norm()
                    if float(n) > self.grad_clip:
                        grad = grad * (self.grad_clip / (n + 1e-8))
                x = (x - self.guide_lr * grad).detach().to(self.dtype)
                trace["step"].append(i); trace["rho"].append(info.get("rho", float("nan")))
                trace["loss"].append(info.get("loss", float("nan")))
            with torch.no_grad():
                x = pipe.scheduler.step(noise, t, x).prev_sample
        return x, trace

    def sample(self, prompt: str, seed: int = 0,
               hook: Optional[GuidanceHook] = None, **kw) -> Tuple[torch.Tensor, dict]:
        s1, s2 = self._load()
        pe, ne = s1.encode_prompt(prompt)
        h1 = self._guided_loop(s1, pe, ne, (1, 3, 64, 64), seed,
                               hook if self.stage == "base" else None)
        x1, tr = h1
        if self.stage == "base":
            img = (x1.float() / 2 + 0.5).clamp(0, 1)[0]
            return img.detach().cpu(), tr
        import torch.nn.functional as F
        up = F.interpolate(x1, size=(256, 256), mode="bilinear", align_corners=False)
        x2, tr2 = self._guided_loop(s2, pe, ne, (1, 3, 256, 256), seed + 1,
                                    hook, extra_cond=up)
        img = (x2.float() / 2 + 0.5).clamp(0, 1)[0]
        return img.detach().cpu(), tr2
