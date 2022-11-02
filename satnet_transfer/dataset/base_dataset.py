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
        return len(self.solutions) // 2

    def __getitem__(self, i):
        inp = torch.from_numpy(self.solutions[f'input_{i}']) # -1, 1 encoding
        lbl = torch.from_numpy(self.solutions[f'label_{i}'])
        inp_mask = torch.where(torch.abs(inp) > 0, 1, 0)     # 0, 1 mask encoding
        inp_mask = inp_mask.type(torch.IntTensor)
        inp      = torch.where(inp < 0., 0., inp)            # [0, 1] probability enc.
        lbl      = torch.where(lbl < 0., 0., lbl)            # [0, 1] probability enc.
        lbl      = lbl.type(torch.FloatTensor)
        return inp, inp_mask, lbl
