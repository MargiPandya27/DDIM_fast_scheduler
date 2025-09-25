from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
import time
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


@dataclass
class SamplerResult:
    images: List["PIL.Image.Image"]
    steps_used: int
    seconds: float


class DDIMFastSampler:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(device)

    @torch.inference_mode()
    def sample(
        self,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        callback: Optional[Callable[[int, float], bool]] = None,
    ) -> SamplerResult:
        generator = torch.Generator(device=self.pipe.device)
        if seed is not None:
            generator = generator.manual_seed(int(seed))

        start = time.time()
        images = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
            output_type="pil",
            callback_on_step_end=self._make_callback(callback),
        ).images
        seconds = time.time() - start
        return SamplerResult(images=images, steps_used=num_inference_steps, seconds=seconds)

    def _make_callback(self, user_cb: Optional[Callable[[int, float], bool]]):
        if user_cb is None:
            return None

        last_metric = None

        def on_step_end(pipe, step_index: int, timestep: int, callback_kwargs):
            nonlocal last_metric
            latents = callback_kwargs.get("latents", None)
            metric = 0.0
            if latents is not None:
                metric = float(latents.var().item())
            delta = 0.0 if last_metric is None else metric - last_metric
            last_metric = metric
            should_stop = user_cb(step_index, delta)
            callback_kwargs["terminate"] = bool(should_stop)
            return callback_kwargs

        return on_step_end


