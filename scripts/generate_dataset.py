import numpy as np
import satnet
from pathlib import Path
from satnet_transfer.dataset import *
from functools import partial

def run():
    data_dir = Path('data/')
    data_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(2022)
    # filename = 'hanoi5.cnf'
    filename = 'CBS_k3_n100_m403_b10_999.cnf'
    mask_transform = partial(mask, rng=rng, n_masks=5, n_samples=5, mask_sym=0)
    dataset_path = generate_dataset(filename, data_dir, rng, transform=mask_transform, limit=10000, suffix='_masked.npz')
    print(dataset_path)
    dataset = SATDataset(dataset_path)
