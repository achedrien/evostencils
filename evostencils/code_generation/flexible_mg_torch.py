import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class FlexibleMGTorch(nn.Module):
    def __init__(self, intergrid_operators, smoother, weight, device=None):
        super(FlexibleMGTorch, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stencil = torch.tensor([[0.0, 1.0, 0.0],
                                     [1.0, -4.0, 1.0],
                                     [0.0, 1.0, 0.0]], dtype=torch.float64).to(self.device)
        self.stencil = nn.Parameter(data = self.stencil.unsqueeze(0).unsqueeze(0))
        self.intergrid_operators = intergrid_operators
        self.smoother = smoother
        self.weight = weight
        
        N = 2**6
        x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        physical_rhs = -2 * (np.pi**2) * np.sin(np.pi * X) * np.sin(np.pi * Y)
        self.f = torch.from_numpy(physical_rhs[np.newaxis, np.newaxis, :, :].astype(np.float64)).to(self.device)
        
        self.run_time, self.convergence_factor, self.n_iterations = self.solve_poisson(self.f, self.stencil, 1, 1e-3)

    def weighted_jacobi_smoother(self, u, f, stencil, omega):
        h = 1 / np.shape(u)[-1]
        stencil = (1 / h**2) * stencil
        central_coeff = stencil[0, 0, 1, 1]
        u_conv = F.conv2d(u, stencil, padding=0)
        u_conv = F.pad(u_conv, (1, 1, 1, 1), "constant", 0)
        u = u + omega * (f - u_conv) / central_coeff
        u[:, :, :, 0] = 0
        u[:, :, :, -1] = 0
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        return u

    def restrict(self, u):
        u = F.avg_pool2d(u, 2)
        u[:, :, :, 0] = 0
        u[:, :, :, -1] = 0
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        return u

    def prolongate(self, u):
        u = F.interpolate(u, scale_factor=2, mode='bilinear', align_corners=True)
        u[:, :, :, 0] = 0
        u[:, :, :, -1] = 0
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        return u

    def cgs(self, u, f):
        u = u.detach().numpy()[0, 0, :, :].flatten()
        f = f.detach().numpy()[0, 0, :, :].flatten()
        N = int(len(u)**0.5)
        A = np.kron(np.diag(np.ones(N - 1), -1) + np.diag(np.ones(N - 1), 1), - np.eye(N))
        D = -1.0 * np.diag(np.ones(N - 1), -1) + -1.0 * np.diag(np.ones(N - 1), 1) + 4 * np.diag(np.ones(N), 0)
        A += np.kron(np.diag(np.ones(N), 0), D)
        A = -A*int(len(u)**0.5)**2
        u = np.linalg.inv(A) @ f
        return torch.from_numpy(u.reshape(int(np.shape(u)[-1]**0.5), int(np.shape(u)[-1]**0.5))[np.newaxis, np.newaxis, :, :].astype(np.float64))

    def flex_cycle(self, u, f, stencil, levels, intergrid_operators, smoothers, weights):
        udictionary = {}
        fdictionary = {}
        udictionary[levels] = u
        fdictionary[levels] = f
        for i, operator in enumerate(intergrid_operators):
            if operator == -1:
                if smoothers[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], stencil, weights[i])
                conv_holder = F.conv2d(udictionary[levels], (1 / (1 / np.shape(udictionary[levels])[-1])**2) * stencil, padding=0)
                conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
                residual = fdictionary[levels] - conv_holder
                fdictionary[levels - 1] = self.restrict(residual)
                udictionary[levels - 1] = torch.zeros_like(fdictionary[levels - 1])
                levels -= 1
            elif operator == 1:
                if smoothers[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], stencil, weights[i])
                udictionary[levels + 1] += self.prolongate(udictionary[levels])
                levels += 1
            else:
                if smoothers[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], stencil, weights[i])
                elif smoothers[i] == 0:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
        return udictionary[levels]

    def solve_poisson(self, f, stencil, levels, tol):
        u = torch.zeros_like(f)
        conv_holder = F.conv2d(u, (1 / (1 / np.shape(u)[-1])**2) * stencil, padding=0)
        conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
        res = np.inner((f - conv_holder).detach().numpy()[0, 0, :, :].flatten(),
                       (f - conv_holder).detach().numpy()[0, 0, :, :].flatten())
        prevres = res
        iter = 0
        convfactorlist = []
        while res > tol and iter < 100:
            start_time = time.time()
            u = self.flex_cycle(u, f, stencil, levels, self.intergrid_operators, self.smoother, self.weight)
            end_time = time.time()
            conv_holder = F.conv2d(u, (1 / (1 / np.shape(u)[-1])**2) * stencil, padding=0)
            conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
            res = np.inner((f - conv_holder).detach().numpy()[0, 0, :, :].flatten(),
                           (f - conv_holder).detach().numpy()[0, 0, :, :].flatten())
            resratio = res / prevres
            convfactorlist.append(resratio)
            prevres = res
            # print(f'Iteration: {iter}; Residual: {res}; Conv Factor: {resratio}')
            iter += 1
        # print(f'solve time: {end_time - start_time}, convergence factor: {np.mean(convfactorlist)}, n iterations:  {iter}')
        return end_time - start_time, np.mean(convfactorlist), iter

# Example usage
#intergrid_operators = [0, -1, 0, 1, 0]  # Example operators
#smoother = [1, 1, 0, 1, 1]  # Example smoothers
#weight = [0.8, 0.8, 0.8, 0.8, 1.2]  # Example weights

#model = FlexibleMGTorch(intergrid_operators, smoother, weight)
