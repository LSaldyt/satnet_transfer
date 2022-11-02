from pysat.formula import CNF
from pysat.solvers import Solver
import numpy as np
from pathlib import Path
import itertools

from contextlib import contextmanager

class SolverFailedError(Exception): # We should never see this
    pass

def encode(solution):
    return np.sign(solution, dtype=np.int32) # {-1, 1} vector

def mask(encoding, rng, n_masks, n_samples, mask_sym=0):
    ''' Mask the solved CNF instance '''
    for _ in range(n_samples):
        mask     = rng.choice(len(encoding), size=(n_masks,))
        instance = encoding.copy()
        instance[mask] = mask_sym # Now a {-1, 0, 1} vector
        yield instance

@contextmanager
def solver_of(cnf_filename):
    with Solver(name='mcb') as solver:
        cnf = CNF(from_file=cnf_filename)
        clauses = [c for c in cnf.clauses if c] # Filter empty clauses
        solver.append_formula(clauses)
        yield solver

def generate_from(cnf_filename, rng, limit=10000):
    with solver_of(cnf_filename) as solver:
        if not solver.solve():
            raise SolverFailedError
        for solution in itertools.islice(solver.enum_models(), limit):
            yield encode(solution)

def generate_dataset(cnf_filename, data_dir, rng, transform=lambda inp : inp, limit=10000,
                     suffix='_solutions.npz'):
    # Solve a given CNF filename up to limit solutions
    inputs = generate_from(cnf_filename, rng, limit=limit)
    with solver_of(cnf_filename) as solver:
        # Map transform across inputs and flatten the result (up to limit)
        pairs = itertools.islice(((t, inp) for inp in inputs for t in transform(inp)), limit)

        # Filter inputs for those that are solvable
        pairs = (p for p in pairs if solver.solve(
                      assumptions=[int((i + 1) * v) for i, v in enumerate(p[0]) if v != 0]))
        inputs, labels = zip(*pairs)
    # Separate inputs and labels and save them to file as named arrays
    dest = data_dir / Path('{}{}'.format(Path(cnf_filename).stem, suffix))
    np.savez_compressed(dest, **{f'input_{i}' : inp for i, inp in enumerate(inputs)},
                              **{f'label_{i}' : lbl for i, lbl in enumerate(labels)})
    return dest

