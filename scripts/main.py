from pysat.formula import CNF
from pysat.solvers import Solver


def encode(pysat_cnf, masked_solution):
  satnet_input = np.zeros(pysat_cnf.nv, dtype=np.int32)
  for assignment in masked_solution:
    if assignment == 0:
      continue
    else:
      satnet_input[abs(assignment) - 1] = np.sign(assignment)
  mask = np.where(np.abs(satnet_input) > 0, 1, 0)
  return satnet_input, mask

# masked_solution = [-1, -2, 0] # suppose the assignment to 3 has been masked
# encode(cnf, masked_solution)

import random, copy

def mask_solved_cnf(solved_cnf, num_masks, num_samples=100, mask_sym=0):
  """ Mask the solved CNF instance """
  masked = []
  for _ in range(num_samples):
    sampled_inds = random.sample(list(range(len(solved_cnf))), num_masks)
    masked_instance = copy.deepcopy(solved_cnf)
    for i in sampled_inds:
      masked_instance[i] = mask_sym
    masked.append(masked_instance)
  return masked

def run():
    # filename = 'hanoi5.cnf'
    filename = 'CBS_k3_n100_m403_b10_999.cnf'
    cnf = CNF(from_file=f'cnf/{filename}')
    solver = Solver(name='g4')
    solver.append_formula(cnf.clauses)
    solver.solve()
    for model in solver.enum_models():
        print(model)
