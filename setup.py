from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
REQ_FILE = ROOT / "requirements.txt"

def read_requirements() -> list:
    if REQ_FILE.exists():
        with open(REQ_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="ddim-fast",
    version="0.1.0",
    description="Fast DDIM image generation with adaptive step scheduling",
    author="",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=read_requirements(),
    python_requires=">=3.9",
)
