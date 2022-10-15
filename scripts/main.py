import numpy as np
import satnet
from satnet_transfer.dataset import *

def run():
    rng = np.random.default_rng(2022)
    dataset_path = Path('data/CBS_k3_n100_m403_b10_999_solutions.npz')
    dataset = SATDataset(dataset_path)

    # Try making a SATNet and running a single example

    inp, inp_mask, lbl = dataset[0]

    n   = inp.shape[0]
    m   = 100
    aux = 100
    sat = satnet.SATNet(n, m, aux)
    inp      = inp.view(inp.shape[0]).unsqueeze(0)
    inp_mask = inp_mask.view(inp_mask.shape[0]).unsqueeze(0)
    print(inp.shape)
    print(inp_mask.shape)
    print(sat(inp, inp_mask))
