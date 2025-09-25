from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import json
from PIL import Image

from ..scheduler import AdaptiveStepScheduler, AdaptiveStepConfig
from ..sampler import DDIMFastSampler


def save_images(images: List[Image.Image], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(out_dir / f"{prefix}_{i:02d}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--steps", type=int, nargs="*", default=[10, 20, 50, 100])
    p.add_argument("--adaptive", action="store_true")
    p.add_argument("--out", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    args = p.parse_args()

    sampler = DDIMFastSampler(model_id=args.model)
    scheduler = AdaptiveStepScheduler()

    results = []
    out_dir = Path(args.out)

    if args.adaptive:
        adaptive_steps = scheduler.decide_steps(prompt=args.prompt)
        cb = scheduler.early_stop_callback()
        r = sampler.sample(
            prompt=args.prompt,
            num_inference_steps=adaptive_steps,
            seed=args.seed,
            height=args.height,
            width=args.width,
            callback=cb,
        )
        save_images(r.images, out_dir, f"adaptive_{adaptive_steps}")
        results.append({"mode": "adaptive", "steps": adaptive_steps, "seconds": r.seconds})

    for s in args.steps:
        r = sampler.sample(
            prompt=args.prompt,
            num_inference_steps=int(s),
            seed=args.seed,
            height=args.height,
            width=args.width,
        )
        save_images(r.images, out_dir, f"fixed_{s}")
        results.append({"mode": "fixed", "steps": int(s), "seconds": r.seconds})

    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()


