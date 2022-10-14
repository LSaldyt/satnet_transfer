from pysat.formula import CNF
from pysat.solvers import Solver
import numpy as np
import random, copy

def encode(solution):
    return np.sign(solution, dtype=np.int32) # {-1, 1} vector

def mask(encoding, rng, n_masks, n_samples, mask_sym=0):
    ''' Mask the solved CNF instance '''
    for _ in range(n_samples):
        mask     = rng.choice(len(encoding), size=(n_masks,))
        instance = encoding.copy()
        instance[mask] = mask_sym # Now a {-1, 0, 1} vector
        yield instance

def run():
    # filename = 'hanoi5.cnf'
    filename = 'CBS_k3_n100_m403_b10_999.cnf'
    cnf = CNF(from_file=f'cnf/{filename}')
    solver = Solver(name='g4')
    solver.append_formula(cnf.clauses)
    solver.solve()

    rng = np.random.default_rng(2022)
    for model in solver.enum_models():
        print('solution', model)
        encoding = encode(model)
        print('encoding', encoding)
        for instance in mask(encoding, rng, 4, 4):
            print('instance', instance)
        break
