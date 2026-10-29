from __future__ import annotations

import random


def set_seed(seed: int = 13) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

