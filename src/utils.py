from pathlib import Path

import numpy as np
import cv2

PROJECT_ROOT_PATH: Path = Path(__file__).parent.parent

def read_image(path: Path) -> np.ndarray:
    return cv2.imread(str(path))