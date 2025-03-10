import numpy as np

# mean squared error
def mse(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    return np.mean((y_pred - y_target)**2)

def binary_cross_entropy(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    eps = 1e-15
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    print(y_pred_clipped)
    return -np.mean(
        y_target * np.log(y_pred_clipped) + (1 - y_target) * np.log(1 - y_pred_clipped)
    )

def categorical_cross_entropy(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    eps = 1e-15 # agar tidak menghasilkan log(0) -> INF
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_target * np.log(y_pred_clipped), axis=1))