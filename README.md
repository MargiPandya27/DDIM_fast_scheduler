# Prompt Adaptive DDIM Scheduler

A Python package for fast image generation using DDIM (Denoising Diffusion Implicit Models) with adaptive step scheduling. This project demonstrates how to optimize diffusion models for speed while maintaining quality.

## Features

- **DDIM Sampling**: Fast image generation using DDIM scheduler
- **Adaptive Step Scheduling**: Dynamically chooses the number of denoising steps based on prompt complexity
- **Early Stopping**: Stops generation early when quality improvement plateaus
- **Quality Metrics**: FID and Inception Score evaluation


 Prompt 1: a detailed portrait of a majestic lion in golden hour lighting with dramatic shadows
 <img width="1225" height="292" alt="demo1" src="https://github.com/user-attachments/assets/c3ed43c6-fcbc-47a4-9b46-9ced6480be47" />

Prompt 2: a cat
 <img width="1234" height="288" alt="demo2" src="https://github.com/user-attachments/assets/7b003dfc-8145-4251-af39-317756b9df10" />



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

1. Detail oriented prompt 
```bash
# Generate with different step counts
python -m ddim_fast.cli.generate --prompt "a detailed portrait of a majestic full body lion in golden hour lighting with dramatic shadows" --steps 10 20 50 100

# Generate with adaptive step scheduling
python -m ddim_fast.cli.generate --prompt "a detailed portrait of a majestic full body lion in golden hour lighting with dramatic shadows" --adaptive --height 512 --width 512 --seed 42

```

1. Simple less wordy prompt 
```bash
# Generate with adaptive step scheduling
python -m ddim_fast.cli.generate --prompt "a portrait of a cat" --adaptive --height 512 --width 512 --seed 42

```

### 2. Evaluate Quality

Note: FID and IS score works well for more number of generated images.
```bash
# Compute FID and Inception Score
python -m ddim_fast.cli.evaluate --generated outputs --out metrics.json
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
| Colab T4 | ~3s      | ~6s      | ~15s     | ~30s      |

