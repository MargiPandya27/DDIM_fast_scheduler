from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import numpy as np


@dataclass
class AdaptiveStepConfig:
    min_steps: int = 10
    max_steps: int = 100
    default_steps: int = 50
    complexity_weight: float = 0.7
    variance_weight: float = 0.3
    early_stop_threshold: float = 0.002
    early_stop_patience: int = 3


class AdaptiveStepScheduler:
    def __init__(self, config: Optional[AdaptiveStepConfig] = None):
        self.config = config or AdaptiveStepConfig()

    def estimate_complexity(self, prompt: Optional[str] = None, image_stats: Optional[dict] = None) -> float:
        prompt_complexity = 0.0
        if prompt:
            tokens = [t for t in prompt.replace("/", " ").replace("-", " ").split() if t]
            unique_ratio = len(set(tokens)) / max(1, len(tokens))
            prompt_complexity = min(1.0, 0.3 + 0.7 * unique_ratio)

        variance_score = 0.0
        if image_stats and "init_variance" in image_stats:
            variance_score = float(image_stats["init_variance"])  # expected in [0, 1]
            variance_score = float(np.clip(variance_score, 0.0, 1.0))

        complexity = (
            self.config.complexity_weight * prompt_complexity
            + self.config.variance_weight * variance_score
        )
        return float(np.clip(complexity, 0.0, 1.0))

    def decide_steps(self, prompt: Optional[str] = None, image_stats: Optional[dict] = None) -> int:
        c = self.estimate_complexity(prompt, image_stats)
        steps = int(
            self.config.min_steps
            + c * (self.config.max_steps - self.config.min_steps)
        )
        return int(np.clip(steps, self.config.min_steps, self.config.max_steps))

    def early_stop_callback(self) -> Callable[[int, float], bool]:
        improvements: list[float] = []

        def should_stop(step_index: int, metric_delta: float) -> bool:
            improvements.append(metric_delta)
            recent = improvements[-self.config.early_stop_patience :]
            if len(recent) < self.config.early_stop_patience:
                return False
            return all(abs(d) < self.config.early_stop_threshold for d in recent)

        return should_stop


