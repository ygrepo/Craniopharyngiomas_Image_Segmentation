import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.util import (
    get_logger,
)

logger = get_logger(__name__)


def save_image(img: np.ndarray, path: Path):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    logger.info(f"Saving to {path}")
    plt.savefig(path, dpi=150)
    plt.close()
