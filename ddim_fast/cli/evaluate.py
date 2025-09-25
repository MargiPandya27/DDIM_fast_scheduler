from __future__ import annotations

import argparse
from pathlib import Path
import json
import time
from typing import List

import numpy as np
from PIL import Image
from torchmetrics.image.inception import InceptionScore
from cleanfid import fid as cleanfid
import torch


def load_images_from_dir(img_dir: Path) -> List[Image.Image]:
    imgs = []
    for p in sorted(img_dir.glob("*.png")):
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def compute_is(images: List[Image.Image], device: str = "cuda" if torch.cuda.is_available() else "cpu") -> float:
    if len(images) == 0:
        return float("nan")
    tensor = torch.stack([torch.from_numpy(np.array(i)).permute(2, 0, 1).float() / 255.0 for i in images])
    metric = InceptionScore().to(device)
    with torch.no_grad():
        score = metric(tensor.to(device))
    return float(score[0].item())


def compute_fid(generated_dir: Path, ref_name: str = "cifar10_train") -> float:
    try:
        return float(cleanfid.compute_fid(generated_dir.as_posix(), dataset_name=ref_name))
    except Exception:
        return float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=str, default="outputs")
    p.add_argument("--ref", type=str, default="cifar10_train", help="cleanfid reference dataset name")
    p.add_argument("--out", type=str, default="metrics.json")
    args = p.parse_args()

    gen_dir = Path(args.generated)
    groups = {}
    for sub in gen_dir.iterdir():
        if sub.is_file():
            continue
        imgs = load_images_from_dir(sub)
        start = time.time()
        is_score = compute_is(imgs)
        fid_score = compute_fid(sub, ref_name=args.ref)
        seconds = time.time() - start
        groups[sub.name] = {
            "count": len(imgs),
            "inception_score": is_score,
            "fid": fid_score,
            "eval_seconds": seconds,
        }

    with open(Path(args.out), "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2)


if __name__ == "__main__":
    main()


