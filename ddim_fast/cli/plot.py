from __future__ import annotations

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=str, required=True)
    p.add_argument("--out", type=str, default="tradeoff.png")
    p.add_argument("--title", type=str, default="Quality vs Steps")
    args = p.parse_args()

    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    steps = []
    fids = []
    iss = []
    for name, d in data.items():
        if name.startswith("fixed_"):
            step = int(name.split("_")[-1])
        elif name.startswith("adaptive_"):
            step = int(name.split("_")[-1])
        else:
            continue
        steps.append(step)
        fids.append(d.get("fid", None))
        iss.append(d.get("inception_score", None))

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.plot(steps, fids, "o-", color="tab:red", label="FID (lower better)")
    ax2.plot(steps, iss, "s-", color="tab:blue", label="IS (higher better)")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("FID", color="tab:red")
    ax2.set_ylabel("IS", color="tab:blue")
    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()


