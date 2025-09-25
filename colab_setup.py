"""
Colab-optimized setup for DDIM Fast Image Generation
Run this in a Google Colab cell to install and test the package
"""

# Install dependencies
!pip install -q diffusers>=0.30.0 torchvision>=0.15.0 Pillow>=9.5.0 imageio>=2.31.0 torchmetrics>=1.3.0 clean-fid>=0.1.35

# Install the package
!pip install -e .

# Test basic functionality
import torch
from ddim_fast import AdaptiveStepScheduler, DDIMFastSampler

print("✅ Installation successful!")

# Test with smaller settings for Colab
scheduler = AdaptiveStepScheduler()
sampler = DDIMFastSampler()

# Quick test generation (smaller image, fewer steps)
result = sampler.sample(
    prompt="a beautiful landscape painting",
    num_inference_steps=10,
    height=256,  # Smaller for Colab
    width=256,
    seed=42
)

print(f"✅ Generated {len(result.images)} images in {result.seconds:.2f}s")
print("Ready to use! Try: python -m ddim_fast.cli.generate --prompt 'your prompt' --steps 10 20 50")
