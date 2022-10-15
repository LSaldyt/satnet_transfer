import torch
import numpy as np
from pathlib import Path

class SATDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path      = path
        # We do loading/transform at dataset init, which is atypical
        # However, it seems reasonable for our dataset
        self.solutions = np.load(self.path)

    def __len__(self):
        return len(self.solutions)

    def __getitem__(self, i):
        return (torch.from_numpy(self.solutions[f'input_{i}']),
                torch.from_numpy(self.solutions[f'label_{i}']))
