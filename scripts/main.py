import numpy as np
import satnet
from time import time

def run():
    rng = np.random.default_rng(2022)
    # filename = 'hanoi5.cnf'
    filename = 'CBS_k3_n100_m403_b10_999.cnf'
    start = time()
    count = 0
    for example in generate_from(filename, rng, n_masks=100, n_samples=100):
        print(example)
        count += 1
    end = time()
    print(f'Found {count} examples in {end - start} seconds')
