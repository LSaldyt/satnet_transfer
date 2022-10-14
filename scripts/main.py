import numpy as np
from satnet_transfer.dataset import *
import satnet

def run():
    rng = np.random.default_rng(2022)
    # filename = 'hanoi5.cnf'
    filename = 'CBS_k3_n100_m403_b10_999.cnf'
    for example in generate_from(filename, rng, n_masks=100, n_samples=100):
        print(example)
