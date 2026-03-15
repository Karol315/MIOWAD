import numpy as np
import pandas as pd

class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None