import numpy as np

# inisialisasi bobot dengan nilai 0
def init_zero(shape: tuple) -> np.ndarray:
    return np.zeros(shape, dtype=float)

#inisialisasi bobot secara distribusi uniform
def init_uniform(shape: tuple, low: float = -0.05, high: float = 0.05, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=shape)

#inisialisasi bobot secara distribusi normal
def init_normal(shape: tuple, mean: float = 0.0, var: float = 1.0, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    std = np.sqrt(var)
    return np.random.normal(loc=mean, scale=std, size=shape)