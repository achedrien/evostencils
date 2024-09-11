import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from random import randint
import os

class Solver(nn.Module):
    def __init__(self, physical_rhs, intergrid_operators, smoother, weight, trainable=True, device=None,
                 trainable_stencil = nn.Parameter(torch.tensor([[[[0.0, 1.0, 0.0],
                                                                [1.0, -4.0, 1.0],
                                                                [0.0, 1.0, 0.0]]]], dtype=torch.float64)),
                 trainable_weight = nn.Parameter(0.1*torch.rand(1, 1, 1, 1, generator=torch.Generator(), dtype=torch.double, requires_grad=True)),
                 trainable_omega = nn.Parameter(torch.rand(1, 1, 1, 10, dtype=torch.double),  requires_grad=True)):
        super(Solver, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed_stencil = torch.tensor([[0.0, 1.0, 0.0],
                                           [1.0, -4.0, 1.0],
                                           [0.0, 1.0, 0.0]], dtype=torch.float64).to(self.device).unsqueeze(0).unsqueeze(0)
        self.trainable = trainable
        if self.trainable:
            self.trainable_stencil = self.fixed_stencil # nn.Parameter(trainable_stencil.to(self.device))
            # self.trainable_stencil = nn.Parameter(4*torch.rand_like(self.fixed_stencil, dtype=torch.double, requires_grad=True)).to(self.device)
            self.trainable_weight = 0 # nn.Parameter(trainable_weight.to(self.device)).clamp(0, 1)
            self.trainable_omega = 1 # nn.Parameter(torch.rand(len(intergrid_operators), 10, dtype=torch.double))# , requires_grad=True)).to(self.device)
            self.restriction_stencils = nn.Parameter(torch.tensor([[[[0.25, 0.25],
                                                                      [0.25, 0.25]]]], dtype=torch.float64).to(self.device))
            self.prolongation_stencils = nn.Parameter(torch.tensor([[[[0.25, 0.5, 0.25],
                                                                      [0.5, 1, 0.5],
                                                                      [0.25, 0.5, 0.25]]]], dtype=torch.float64).to(self.device))
        else:
            self.trainable_weight = 1
            self.trainable_omega = trainable_omega
        self.intergrid_operators = intergrid_operators
        self.smoother = smoother
        self.weight = weight
        self.max_iter = 100
        self.f = physical_rhs.to(self.device)
        u, res, self.run_time, self.convergence_factor, self.n_iterations = self.solve_poisson(1e-3)

    def weighted_jacobi_smoother(self, u, f, omega):
        h = 1 / np.shape(u)[-1]
        fixed_stencil = (1 / h**2) * self.fixed_stencil
        fixed_central_coeff = fixed_stencil[0, 0, 1, 1]
        u_conv_fixed = F.conv2d(u, fixed_stencil, padding=0)
        u_conv_fixed = F.pad(u_conv_fixed, (1, 1, 1, 1), "constant", 0)
        if self.trainable:
            trainable_stencil = (1 / h**2) * self.trainable_stencil
            trainable_central_coeff = trainable_stencil[0, 0, 1, 1]
            u_conv_trainable = F.conv2d(u, trainable_stencil, padding=0)
            u_conv_trainable = F.pad(u_conv_trainable, (1, 1, 1, 1), "constant", 0)
            u = u + ( 1 - self.trainable_weight) * omega * (f - u_conv_fixed) / fixed_central_coeff + self.trainable_weight * omega * (f - u_conv_trainable) / trainable_central_coeff
        else:
            u = u + omega * (f - u_conv_fixed) / fixed_central_coeff
        return u

    def chebyshev_smoother(self, u, f):
        h = 1 / np.shape(u)[-1]
        fixed_stencil = (1 / h**2) * self.fixed_stencil
        fixed_central_coeff = fixed_stencil[0, 0, 1, 1]
        for i in range(10):
            u_conv_fixed = F.conv2d(u, fixed_stencil, padding=0)
            u_conv_fixed = F.pad(u_conv_fixed, (1, 1, 1, 1), "constant", 0)
            u = u + ((f - u_conv_fixed) / fixed_central_coeff).mul(self.trainable_omega[self.n_operations, i])
        return u
    
    def restrict(self, u):
        # print(u.size())
        u = F.conv2d(u, self.restriction_stencils, padding=0, stride=(2, 2))
        # u = F.pad(u, (1, 1, 1, 1), "constant", 0)
        # print(u.size())
        return u

    def prolongate(self, u):
        u = F.interpolate(u, scale_factor=2, mode='nearest') # , align_corners=True)
        u = F.conv2d(u, self.prolongation_stencils, padding=0, stride=1)
        u = F.pad(u, (1, 1, 1, 1), "constant", 0)
        return u

    def cgs(self, u, f):
        u_np = u.cpu().detach().numpy()[0, 0, :, :].flatten()
        f_np = f.cpu().detach().numpy()[0, 0, :, :].flatten()
        N = int(len(u_np)**0.5)
        A = np.kron(np.diag(np.ones(N - 1), -1) + np.diag(np.ones(N - 1), 1), - np.eye(N))
        D = -1.0 * np.diag(np.ones(N - 1), -1) + -1.0 * np.diag(np.ones(N - 1), 1) + 4 * np.diag(np.ones(N), 0)
        A += np.kron(np.diag(np.ones(N), 0), D)
        A = -A*int(len(u_np)**0.5)**2
        u_np = np.linalg.inv(A) @ f_np
        return torch.from_numpy(u_np.reshape(int(np.shape(u_np)[-1]**0.5), int(np.shape(u_np)[-1]**0.5))[np.newaxis, np.newaxis, :, :].astype(np.float64)).to(self.device)

    def flex_cycle(self, u):
        levels=0
        udictionary = {}
        fdictionary = {}
        udictionary[levels] = u.clone().to(self.device)
        fdictionary[levels] = self.f.clone().to(self.device)
        for i, operator in enumerate(self.intergrid_operators):
            if operator == -1:
                if self.smoother[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], self.weight[i])
                elif self.smoother[i] == 2:
                    udictionary[levels] = self.chebyshev_smoother(udictionary[levels], fdictionary[levels])
                elif self.smoother[i] == 3:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
                conv_holder = F.conv2d(udictionary[levels].clone(), (1 / (1 / np.shape(udictionary[levels].clone())[-1])**2) * self.fixed_stencil.clone(), padding=0)
                conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
                residual = fdictionary[levels] - conv_holder
                fdictionary[levels + 1] = self.restrict(residual)
                udictionary[levels + 1] = torch.zeros_like(fdictionary[levels + 1])
                levels += 1
            elif operator == 1:
                udictionary[levels - 1] += self.prolongate(udictionary[levels])
                levels -= 1
                if self.smoother[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], self.weight[i])
                elif self.smoother[i] == 2:
                    udictionary[levels] = self.chebyshev_smoother(udictionary[levels], fdictionary[levels])
                elif self.smoother[i] == 3:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
            else:
                if self.smoother[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], self.weight[i])
                elif self.smoother[i] == 2:
                    udictionary[levels] = self.chebyshev_smoother(udictionary[levels], fdictionary[levels])
                elif self.smoother[i] == 3:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
            self.n_operations += 1
        result = udictionary[0]
        del(fdictionary, udictionary)
        return result

    def solve_poisson(self, tol):
        start_time = time.time()
        u = torch.zeros_like(self.f).to(self.device)
        conv_holder = F.conv2d(u, (1 / (1 / np.shape(u)[-1])**2) * self.fixed_stencil, padding=0)
        conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
        res = torch.sum(torch.abs(self.f - conv_holder))/torch.sum(torch.abs(self.f)) 
        prevres = res
        iter = 0
        convfactorlist = []
        while res > tol and iter < self.max_iter:
            self.n_operations = 0
            u = self.flex_cycle(u)
            conv_holder = F.conv2d(u, (1 / (1 / np.shape(u)[-1])**2) * self.fixed_stencil, padding=0)
            conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
            res = torch.sum(torch.abs(self.f - conv_holder))/torch.sum(torch.abs(self.f)) 
            resratio = res / prevres
            convfactorlist.append(resratio)
            prevres = res
            # print(f'Iteration: {iter}; Residual: {res}; Conv Factor: {resratio}')
            iter += 1
        end_time = time.time()
        # print(f'solve time: {end_time - start_time}, convergence factor: {torch.mean(torch.stack(convfactorlist))}') 
        return u, res, end_time - start_time, torch.mean(torch.stack(convfactorlist)).item(), iter

    def forward(self, f, tol, optimizer):
        if f.dim() == 6:
            self.f = f.double().to(self.device)[0, 0, :, :, :, :]
        else:
            self.f = f.double().to(self.device)
        self.trainable = True
        self.max_iter = 1
        u = torch.zeros_like(self.f)
        optimizer.zero_grad()
        with torch.enable_grad():
            u, res, time, conv_factor, iter = self.solve_poisson(tol)
        return u, res, time, conv_factor, iter, self.prolongation_stencils, self.restriction_stencils
    
    def save(self, checkpoint_path: str, epoch: int):
        save_dir: str = os.path.join(checkpoint_path, 'pth')
        os.makedirs(save_dir, exist_ok=True)
        save_path: str = os.path.join(save_dir, f'epoch_{epoch}.pth')
        torch.save(self.trainable_stencil, save_path)
