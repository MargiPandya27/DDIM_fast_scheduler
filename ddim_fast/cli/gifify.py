from __future__ import annotations

import argparse
from pathlib import Path
import imageio
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, required=True, help="Directory containing step-wise PNGs")
    p.add_argument("--out", type=str, default="comparison.gif")
    p.add_argument("--fps", type=int, default=2)
    args = p.parse_args()

    frames = []
    for img_path in sorted(Path(args.dir).glob("*.png")):
        frames.append(imageio.v3.imread(img_path.as_posix()))
    if not frames:
        raise SystemExit("No PNGs found in directory")
    imageio.mimsave(args.out, frames, fps=args.fps)


if __name__ == "__main__":
    main()


