import numpy as np
import satnet
from satnet_transfer.dataset import *
import torch
from satnet_transfer.loop import loop
from satnet_transfer.settings import Settings
from pathlib import Path

def run():
    s = Settings(batch_size=32, lr=2e-3, epochs=10, split=0.8, metrics_file='metrics/basic.csv')
    rng = np.random.default_rng(2022)

    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    dataset_path = Path('data/CBS_k3_n100_m403_b10_999_masked.npz')
    dataset = SATDataset(dataset_path)

    inp, inp_mask, lbl = dataset[0]
    n   = inp.shape[0]
    m   = 100
    aux = 10
    sat = satnet.SATNet(n, m, aux)

    optimizer = torch.optim.AdamW(sat.parameters(), lr=s.lr)
    loop(sat, dataset, optimizer, s)
