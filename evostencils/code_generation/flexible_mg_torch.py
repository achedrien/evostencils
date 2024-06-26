import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from random import randint
import os

class Solver(nn.Module):
    def __init__(self, physical_rhs, intergrid_operators, smoother, weight, device=None):
        super(Solver, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fixed_stencil = torch.tensor([[0.0, 1.0, 0.0],
                                           [1.0, -4.0, 1.0],
                                           [0.0, 1.0, 0.0]], dtype=torch.float64).to(self.device).unsqueeze(0).unsqueeze(0)
        self.trainable = True
        if self.trainable:
            self.trainable_stencil = nn.Parameter(torch.tensor([[0.0, 1.0, 0.0],
                                                                [1.0, -4.0, 1.0],
                                                                [0.0, 1.0, 0.0]], dtype=torch.float64).to(self.device).unsqueeze(0).unsqueeze(0))
        
        self.intergrid_operators = intergrid_operators
        self.smoother = smoother
        self.weight = weight
        self.f = physical_rhs # torch.from_numpy(physical_rhs[np.newaxis, np.newaxis, :, :].astype(np.float64)).to(self.device)
        u, self.run_time, self.convergence_factor, self.n_iterations = self.solve_poisson(1e-3)
        torch.autograd.set_detect_anomaly(True)

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
            tau = 0.01
            u = u + ( 1 - tau ) * omega * (f - u_conv_fixed) / fixed_central_coeff + tau * omega * (f - u_conv_trainable) / trainable_central_coeff
        else:
            u = u + omega * (f - u_con_fixedv) / fixed_central_coeff
            
        u = u.clone()
        u[:, :, :, 0] = 0
        u[:, :, :, -1] = 0
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        return u

    def restrict(self, u):
        u = F.avg_pool2d(u, 2)
        u = u.clone()
        u[:, :, :, 0] = 0
        u[:, :, :, -1] = 0
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        return u

    def prolongate(self, u):
        u = F.interpolate(u, scale_factor=2, mode='bilinear', align_corners=True)
        u = u.clone()
        u[:, :, :, 0] = 0
        u[:, :, :, -1] = 0
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        return u

    def cgs(self, u, f):
        u_np = u.detach().numpy()[0, 0, :, :].flatten()
        f_np = f.detach().numpy()[0, 0, :, :].flatten()
        N = int(len(u_np)**0.5)
        A = np.kron(np.diag(np.ones(N - 1), -1) + np.diag(np.ones(N - 1), 1), - np.eye(N))
        D = -1.0 * np.diag(np.ones(N - 1), -1) + -1.0 * np.diag(np.ones(N - 1), 1) + 4 * np.diag(np.ones(N), 0)
        A += np.kron(np.diag(np.ones(N), 0), D)
        A = -A*int(len(u_np)**0.5)**2
        u_np = np.linalg.inv(A) @ f_np
        return torch.from_numpy(u_np.reshape(int(np.shape(u_np)[-1]**0.5), int(np.shape(u_np)[-1]**0.5))[np.newaxis, np.newaxis, :, :].astype(np.float64))

    def flex_cycle(self, u):
        levels=0
        udictionary = {}
        fdictionary = {}
        udictionary[levels] = u.clone()
        fdictionary[levels] = self.f.clone()
        for i, operator in enumerate(self.intergrid_operators):
            # print(levels)
            if operator == -1:
                if self.smoother[i] == 1:
                    # print(f'smoother, level: {levels}')
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], self.weight[i])
                elif self.smoother[i] == 2:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
                conv_holder = F.conv2d(udictionary[levels].clone(), (1 / (1 / np.shape(udictionary[levels].clone())[-1])**2) * self.fixed_stencil.clone(), padding=0)
                conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
                residual = fdictionary[levels] - conv_holder
                fdictionary[levels + 1] = self.restrict(residual)
                udictionary[levels + 1] = torch.zeros_like(fdictionary[levels + 1])
                levels += 1
            elif operator == 1:
                if self.smoother[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], self.weight[i])
                elif self.smoother[i] == 2:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
                udictionary[levels - 1] += self.prolongate(udictionary[levels])
                levels -= 1
            else:
                if self.smoother[i] == 1:
                    udictionary[levels] = self.weighted_jacobi_smoother(udictionary[levels], fdictionary[levels], self.weight[i])
                elif self.smoother[i] == 2:
                    udictionary[levels] = self.cgs(udictionary[levels], fdictionary[levels])
        result = udictionary[0]
        del(fdictionary, udictionary)
        return result

    def solve_poisson(self, tol):
        start_time = time.time()
        u = torch.zeros_like(self.f)
        conv_holder = F.conv2d(u, (1 / (1 / np.shape(u)[-1])**2) * self.fixed_stencil, padding=0)
        conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
        res = torch.sum(torch.abs(self.f - conv_holder))/torch.sum(torch.abs(self.f)) #np.inner((f - conv_holder).detach().cpu().numpy()[0, 0, :, :].flatten(),
              #         (f - conv_holder).detach().cpu().numpy()[0, 0, :, :].flatten())
        prevres = res
        iter = 0
        convfactorlist = []
        while res > tol and iter < 100:
            u = self.flex_cycle(u)
            conv_holder = F.conv2d(u, (1 / (1 / np.shape(u)[-1])**2) * self.fixed_stencil, padding=0)
            conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
            res = torch.sum(torch.abs(self.f - conv_holder))/torch.sum(torch.abs(self.f)) #np.inner((f - conv_holder).detach().cpu().numpy()[0, 0, :, :].flatten(),
                  #         (f - conv_holder).detach().cpu().numpy()[0, 0, :, :].flatten())
            # print(res.detach().cpu().numpy())
            resratio = res / prevres
            convfactorlist.append(resratio)
            prevres = res
            # print(f'Iteration: {iter}; Residual: {res}; Conv Factor: {resratio}')
            iter += 1
        end_time = time.time()
        # print(f'solve time: {end_time - start_time}, convergence factor: {torch.mean(torch.stack(convfactorlist))}') # np.mean(convfactorlist)}, n iterations:  {iter}')
        # del(conv_holder, f, stencil)
        # torch.cuda.empty_cache()
        return u, end_time - start_time, torch.mean(torch.stack(convfactorlist)).item(), iter

    def forward(self, tol):
        u = torch.zeros_like(self.f)
        u, time, conv_factor, iter = self.solve_poisson(tol) # self.intergrid_operators, self.smoother, self.weight)
        return u, time, conv_factor, iter
    
    def save(self, checkpoint_path: str, epoch: int):
        save_dir: str = os.path.join(checkpoint_path, 'pth')
        os.makedirs(save_dir, exist_ok=True)
        save_path: str = os.path.join(save_dir, f'epoch_{epoch}.pth')
        print(self.trainable_stencil)
        torch.save(self.trainable_stencil, save_path)
# 3rd try with more rigorous optimization
# intergrid_operators = [-1,-1,0,-1,-1,1,1,-1,1,1,1,0,0]  # Example operators
# smoother = [0,0,1,1,0,2,1,0,1,0,1,1,1,1]  # Example smoothers
# weight = [0,0,1.75,1.8,0,1,1.65,0,1.55,0,1.9,1.3,0.5499999999999999,1.9]  # Example weights

# model = FlexibleMGTorch(intergrid_operators, smoother, weight)
