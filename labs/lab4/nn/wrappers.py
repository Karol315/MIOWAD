import numpy as np
import pandas as pd

class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        """Oblicza wartości minimalne i zakres do późniejszego skalowania."""
        X_arr = np.asarray(X, dtype=float)

        self.min_ = np.min(X_arr, axis=0)
        max_val = np.max(X_arr, axis=0)
        self.scale_ = max_val - self.min_

        # Zabezpieczenie przed dzieleniem przez zero
        if isinstance(self.scale_, np.ndarray):
            self.scale_[self.scale_ == 0.0] = 1.0
        elif self.scale_ == 0.0:
            self.scale_ = 1.0

        return self

    def transform(self, X):
        """Skaluje cechy tak, aby mieściły się w przedziale [0, 1]."""
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Skaler nie został dopasowany. Wywołaj najpierw metodę 'fit'.")

        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.min_) / self.scale_

    def fit_transform(self, X):
        """Dopasowuje skaler do danych i je transformuje."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """Odwraża skalowanie, przywracając oryginalne wartości."""
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Skaler nie został dopasowany. Wywołaj najpierw metodę 'fit'.")

        X_arr = np.asarray(X_scaled, dtype=float)
        return (X_arr * self.scale_) + self.min_

class Identity:
    def fit(self, X):
        return self
    def transform(self, X):
        return X
    def inverse_transform(self, X):
        return X
    def fit_transform(self, X):
        return X