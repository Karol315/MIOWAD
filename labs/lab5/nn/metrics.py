from typing import Any

import numpy as np
import pandas as pd
from numpy import floating


def mse(y_true: np.ndarray | pd.Series | list , y_pred: np.ndarray | pd.Series |list) -> floating[Any]:
    return np.mean((y_true - y_pred) ** 2)
