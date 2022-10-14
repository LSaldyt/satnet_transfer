from pysat.formula import CNF
from pysat.solvers import Solver
import numpy as np

def encode(solution):
    return np.sign(solution, dtype=np.int32) # {-1, 1} vector

def mask(encoding, rng, n_masks, n_samples, mask_sym=0):
    ''' Mask the solved CNF instance '''
    for _ in range(n_samples):
        mask     = rng.choice(len(encoding), size=(n_masks,))
        instance = encoding.copy()
        instance[mask] = mask_sym # Now a {-1, 0, 1} vector
        yield instance

def generate_from(cnf_filename, rng, n_masks, n_samples):
    cnf = CNF(from_file=f'cnf/{cnf_filename}')
    solver = Solver(name='g4')
    solver.append_formula(cnf.clauses)
    solver.solve()
    for model in solver.enum_models():
        yield from mask(encode(model), rng, n_masks=n_masks, n_samples=n_samples)
