import numpy as np
import torch
import evostencils.code_generation.flexible_mg_torch_polar as Solver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 2**7
r, theta = np.linspace(0, 1, N), np.linspace(0, 2*np.pi, N)
R, Theta = np.meshgrid(r, theta)
# u = r^3 cos(theta)
physical_rhs = -7 * R * np.cos(Theta) + 3 * R * np.sin(Theta)
physical_rhs = torch.from_numpy(physical_rhs[np.newaxis, np.newaxis, :, :].astype(np.float64))
solver = Solver.Solver(physical_rhs, [-1, -1, -1, -1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1], 
                        [1, 1, 1, 1, 1, 1, 1, 1, 1], trainable = False, trainable_omega=1*torch.ones(9, 10, dtype=torch.double).to(device))
print("########################################")
print("run time: ", solver.run_time)
print("Convergence factor: ", solver.convergence_factor)
print("Number of iterations: ", solver.n_iterations)
print("########################################")