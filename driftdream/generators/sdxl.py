"""SDXL with guidance through the VAE decoder.

Resolution note -- this is a real design decision, not a detail.  Peripheral
drift is a HIGH-SPATIAL-FREQUENCY phenomenon: the percept lives in the precise
luminance step sequence across a few dozen pixels.  DeepFloyd IF generates at
64x64, where that structure does not yet exist.  SDXL generates at 1024x1024
(latent 128x128), so guiding through the differentiable VAE decoder gives the
reader something it can actually see, at the cost of whatever micro-structure
the VAE cannot represent.  Which wins is an empirical question -- run
scripts/day1_gate.py, which downsamples a known illusion to 64px and reports
whether rho survives.

No training: UNet, VAE and text encoders are frozen; gradients touch only the
latent x_t.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import Generator, GuidanceHook

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


class SDXLGenerator(Generator):
    name = "sdxl"
    stages = ("latent",)

    def __init__(self, device: str = "cuda", model_id: str = MODEL_ID,
                 steps: int = 50, cfg_scale: float = 6.0, size: int = 1024,
                 guide_from: float = 0.25, guide_to: float = 0.85,
                 guide_every: int = 2, guide_lr: float = 6.0,
                 grad_clip: float = 1.0, dtype=torch.float16):
        self.device, self.model_id, self.dtype = device, model_id, dtype
        self.steps, self.cfg_scale, self.size = int(steps), float(cfg_scale), int(size)
        self.guide_from, self.guide_to = float(guide_from), float(guide_to)
        self.guide_every, self.guide_lr = int(guide_every), float(guide_lr)
        self.grad_clip = float(grad_clip)
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from diffusers import DDIMScheduler, StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id, torch_dtype=self.dtype, variant="fp16",
                use_safetensors=True)
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.device)
            pipe.vae.requires_grad_(False)
            pipe.unet.requires_grad_(False)
            pipe.set_progress_bar_config(disable=True)
            self._pipe = pipe
        return self._pipe

    def _decode(self, lat):
        pipe = self._pipe
        img = pipe.vae.decode(lat / pipe.vae.config.scaling_factor).sample
        return (img / 2 + 0.5).clamp(0, 1)[0]

    @torch.no_grad()
    def _embed(self, pipe, prompt):
        pe, npe, pooled, npooled = pipe.encode_prompt(
            prompt=prompt, device=self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True, negative_prompt="")
        return pe, npe, pooled, npooled

    def sample(self, prompt: str, seed: int = 0,
               hook: Optional[GuidanceHook] = None, **kw) -> Tuple[torch.Tensor, dict]:
        pipe = self._load()
        steps = int(kw.get("steps", self.steps))
        pe, npe, pooled, npooled = self._embed(pipe, prompt)
        pipe.scheduler.set_timesteps(steps, device=self.device)
        ts = pipe.scheduler.timesteps

        g = torch.Generator(device=self.device).manual_seed(int(seed))
        lat_h = self.size // pipe.vae_scale_factor
        lat = torch.randn(1, pipe.unet.config.in_channels, lat_h, lat_h,
                          generator=g, device=self.device, dtype=self.dtype)
        lat = lat * pipe.scheduler.init_noise_sigma

        add_time = torch.tensor([[self.size, self.size, 0, 0, self.size, self.size]],
                                device=self.device, dtype=self.dtype)
        emb = torch.cat([npe, pe]); pool = torch.cat([npooled, pooled])
        added = {"text_embeds": pool, "time_ids": add_time.repeat(2, 1)}
        lo, hi = int(self.guide_from * steps), int(self.guide_to * steps)
        trace = {"step": [], "rho": [], "loss": []}

        for i, t in enumerate(ts):
            with torch.no_grad():
                inp = pipe.scheduler.scale_model_input(torch.cat([lat] * 2), t)
                noise = pipe.unet(inp, t, encoder_hidden_states=emb,
                                  added_cond_kwargs=added).sample
                nu, nc = noise.chunk(2)
                noise = nu + self.cfg_scale * (nc - nu)

            if hook is not None and lo <= i < hi and (i - lo) % self.guide_every == 0:
                lat = lat.detach().requires_grad_(True)
                a = pipe.scheduler.alphas_cumprod.to(self.device)[t]
                x0 = (lat - (1 - a).sqrt() * noise) / a.sqrt()
                img = self._decode(x0.to(self.dtype)).float()
                loss, info = hook(img, i)
                grad, = torch.autograd.grad(loss, lat)
                if self.grad_clip > 0:
                    n = grad.norm()
                    if float(n) > self.grad_clip:
                        grad = grad * (self.grad_clip / (n + 1e-8))
                lat = (lat - self.guide_lr * grad).detach().to(self.dtype)
                trace["step"].append(i)
                trace["rho"].append(info.get("rho", float("nan")))
                trace["loss"].append(info.get("loss", float("nan")))

            with torch.no_grad():
                lat = pipe.scheduler.step(noise, t, lat).prev_sample

        with torch.no_grad():
            img = self._decode(lat.to(self.dtype)).float()
        return img.detach().cpu(), trace
