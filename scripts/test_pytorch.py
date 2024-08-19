import numpy as np
import torch
import evostencils.code_generation.flexible_mg_torch as Solver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 2**7
x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)
def g(a, x):
    return (x**a) * (1 - x) ** a


def h(a, x):
    return a * (a - 1) * (1 - 2 * x) ** 2 - 2 * a * x * (1 - x)
a=10 #randint(1, 20)
physical_rhs = (2 ** (4 * a)) * g(a, Y) * g(a - 2, X) * h(a, X)
physical_rhs = torch.from_numpy(physical_rhs[np.newaxis, np.newaxis, :, :].astype(np.float64))
solver = Solver.Solver(physical_rhs, [-1, -1, -1, -1, 0, 1, 1, 1, 1],
                        [2, 2, 2, 2, 2, 2, 2, 2, 2], 
                        [1, 1, 1, 1, 1, 1, 1, 1, 1], trainable = False, trainable_omega=0.7*torch.ones(9, 10, dtype=torch.double).to(device))
print("########################################")
print("run time: ", solver.run_time)
print("Convergence factor: ", solver.convergence_factor)
print("Number of iterations: ", solver.n_iterations)
print("########################################")