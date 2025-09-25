# DDIM Fast Image Generation

A Python package for fast image generation using DDIM (Denoising Diffusion Implicit Models) with adaptive step scheduling. This project demonstrates how to optimize diffusion models for speed while maintaining quality.

## Features

- **DDIM Sampling**: Fast image generation using DDIM scheduler
- **Adaptive Step Scheduling**: Dynamically chooses the number of denoising steps based on prompt complexity
- **Early Stopping**: Stops generation early when quality improvement plateaus
- **Quality Metrics**: FID and Inception Score evaluation
- **Visualization**: Generate comparison GIFs and quality vs speed plots

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Or (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### 1. Generate Images

```bash
# Generate with different step counts
python -m ddim_fast.cli.generate --prompt "a beautiful landscape with mountains and lakes" --steps 10 20 50 100

# Generate with adaptive step scheduling
python -m ddim_fast.cli.generate --prompt "a detailed portrait of a cat" --adaptive

# Generate with custom settings
python -m ddim_fast.cli.generate --prompt "futuristic city skyline" --steps 10 20 50 --height 512 --width 512 --seed 42
```

### 2. Evaluate Quality

```bash
# Compute FID and Inception Score
python -m ddim_fast.cli.evaluate --generated outputs --out metrics.json
```

### 3. Create Visualizations

```bash
# Plot quality vs speed trade-off
python -m ddim_fast.cli.plot --metrics metrics.json --out tradeoff.png

# Create comparison GIF
python -m ddim_fast.cli.gifify --dir outputs --out comparison.gif
```

## Google Colab

For running on Google Colab, use the optimized scripts:

```python
# In Colab cell
!python colab_setup.py
!python colab_optimized_generate.py --prompt "your prompt here" --steps 10 20 50 --adaptive
```

## API Usage

```python
from ddim_fast import AdaptiveStepScheduler, DDIMFastSampler

# Initialize
scheduler = AdaptiveStepScheduler()
sampler = DDIMFastSampler()

# Generate with adaptive steps
adaptive_steps = scheduler.decide_steps(prompt="a detailed painting")
result = sampler.sample(
    prompt="a detailed painting",
    num_inference_steps=adaptive_steps,
    height=512,
    width=512
)

print(f"Generated {len(result.images)} images in {result.seconds:.2f}s")
```

## Project Structure

```
ddim_fast/
├── __init__.py
├── scheduler.py      # Adaptive step scheduling
├── sampler.py        # DDIM sampling wrapper
└── cli/
    ├── generate.py   # Image generation CLI
    ├── evaluate.py   # Quality evaluation CLI
    ├── plot.py       # Visualization CLI
    └── gifify.py     # GIF creation CLI
```

## Performance

Expected performance on different hardware:

| Hardware | 10 steps | 20 steps | 50 steps | 100 steps |
|----------|----------|----------|----------|-----------|
| RTX 4090 | ~1s      | ~2s      | ~5s      | ~10s      |
| RTX 3080 | ~2s      | ~4s      | ~10s     | ~20s      |
| Colab T4 | ~3s      | ~6s      | ~15s     | ~30s      |

## License

MIT License
