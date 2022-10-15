import numpy as np
import satnet
from satnet_transfer.dataset import *
import torch
from satnet_transfer.loop import loop
from satnet_transfer.settings import Settings
from pathlib import Path

def run():
    rng = np.random.default_rng(2022)
    s   = Settings(batch_size=32, lr=2e-3, epochs=10, split=0.8)

    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    data_dir = Path('data/')
    for problem_file in data_dir.glob('*.npz'):
        stem = problem_file.stem
        s.update(metrics_file=f'metrics/{stem}.csv')

        dataset = SATDataset(problem_file)

        inp, inp_mask, lbl = dataset[0]
        n   = inp.shape[0]
        m   = 1000
        aux = 1000
        sat = satnet.SATNet(n, m, aux)

        optimizer = torch.optim.AdamW(sat.parameters(), lr=s.lr)
        loop(sat, dataset, optimizer, s)
