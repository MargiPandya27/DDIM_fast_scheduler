"""
Colab-optimized generation script with memory management
"""

import argparse
from pathlib import Path
import torch
import gc
from ddim_fast import AdaptiveStepScheduler, DDIMFastSampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=[10, 20, 50])
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--height", type=int, default=256)  # Smaller for Colab
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Initialize with memory-efficient settings
    sampler = DDIMFastSampler(
        model_id="runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16  # Use half precision
    )
    scheduler = AdaptiveStepScheduler()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    results = []

    # Generate with different step counts
    for steps in args.steps:
        print(f"Generating with {steps} steps...")
        
        result = sampler.sample(
            prompt=args.prompt,
            num_inference_steps=steps,
            height=args.height,
            width=args.width,
            seed=args.seed
        )
        
        # Save images
        for i, img in enumerate(result.images):
            img.save(out_dir / f"steps_{steps}_{i:02d}.png")
        
        results.append({
            "steps": steps,
            "seconds": result.seconds,
            "images": len(result.images)
        })
        
        # Clear memory between generations
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Adaptive generation
    if args.adaptive:
        print("Generating with adaptive steps...")
        adaptive_steps = scheduler.decide_steps(prompt=args.prompt)
        callback = scheduler.early_stop_callback()
        
        result = sampler.sample(
            prompt=args.prompt,
            num_inference_steps=adaptive_steps,
            height=args.height,
            width=args.width,
            seed=args.seed,
            callback=callback
        )
        
        for i, img in enumerate(result.images):
            img.save(out_dir / f"adaptive_{adaptive_steps}_{i:02d}.png")
        
        results.append({
            "steps": adaptive_steps,
            "seconds": result.seconds,
            "images": len(result.images),
            "adaptive": True
        })

    print(f"✅ Generated images saved to {out_dir}")
    print("Results:", results)

if __name__ == "__main__":
    main()
