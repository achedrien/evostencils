import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from random import randint
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
class Solver(nn.Module):
    def __init__(self, physical_rhs, intergrid_operators, smoother, weight, trainable=True, device=None,
                 trainable_stencil = nn.Parameter(torch.tensor([[[[0.0, 1.0, 0.0],
                                                                [1.0, -4.0, 1.0],
                                                                [0.0, 1.0, 0.0]]]], dtype=torch.float64)),
                 trainable_weight = nn.Parameter(0.1*torch.rand(1, 1, 1, 1, generator=torch.Generator(), dtype=torch.double, requires_grad=True)),
                 trainable_omega = nn.Parameter(torch.rand(1, 1, 1, 10, dtype=torch.double),  requires_grad=True)):
        super(Solver, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainable = trainable
        if self.trainable:
            self.trainable_stencil = self.fixed_stencil # nn.Parameter(trainable_stencil.to(self.device))
            # self.trainable_stencil = nn.Parameter(4*torch.rand_like(self.fixed_stencil, dtype=torch.double, requires_grad=True)).to(self.device)
            self.trainable_weight = 0 # nn.Parameter(trainable_weight.to(self.device)).clamp(0, 1)
            self.trainable_omega = nn.Parameter(torch.rand(len(intergrid_operators), 10, dtype=torch.double))# , requires_grad=True)).to(self.device)
            print(self.trainable_omega)
        else:
            self.trainable_weight = 1
            self.trainable_omega = trainable_omega
        self.intergrid_operators = intergrid_operators
        self.smoother = smoother
        self.weight = weight
        self.max_iter = 100
        self.f = physical_rhs.to(self.device)
        u, res, self.run_time, self.convergence_factor, self.n_iterations = self.solve_poisson(1e-3)

    def conv2d_polar(self, input):
        output = input[:, :, 1:-1, 1:-1].clone()
        delta_r = 1 / input.size(2)
        delta_theta = 2 * np.pi / input.size(3)
        for i in range(1, input.size(2)-2):
            r = i * delta_r
            r_p_12 = r + delta_r/2
            r_m_12 = r - delta_r/2
            for j in range(input.size(3)-2):
                kernel = torch.tensor([[0.0, r_p_12/(r*delta_r**2), 0.0],
                                       [1/((r**2)*(delta_theta**2)), -r_p_12/(r*delta_r**2)-r_m_12/(r*delta_r**2)-2/((r**2)*(delta_theta**2)), 1/((r**2)*(delta_theta**2))],
                                       [0.0, r_m_12/(r*delta_r**2), 0.0]], dtype=torch.float64).to(self.device) #.unsqueeze(0).unsqueeze(0)
                output[:, :, i-1, j] = input[:, :, i-1:i+2, j:j+3].mul(kernel).sum()
        # output[:, :, 0, :] = 0
        # print(output.size())
        return output
    
    def weighted_jacobi_smoother(self, u, f, omega):
        h = 1 / np.shape(u)[-1]
        u_conv_fixed = self.conv2d_polar(u)
        u_conv_fixed = F.pad(u_conv_fixed, (1, 1, 1, 1), "replicate")
        delta_r = 1 / u.size(2)
        delta_theta = 2 * np.pi / u.size(3)
        for i in range(1, u.size(2)):
            r = i * delta_r
            r_p_12 = r + delta_r/2
            r_m_12 = r - delta_r/2
            for j in range(u.size(3)):
                u[:, :, i, j] = u[:, :, i, j] + omega * (f[:, :, i, j] - u_conv_fixed[:, :, i, j]) / ( -r_p_12/(r*delta_r**2)-r_m_12/(r*delta_r**2)-2/((r**2)*(delta_theta**2)) )
        # u = u + omega * (f - u_conv_fixed) / fixed_central_coeff
        # Get the values of u
        u = u.clone().to(self.device)
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        u[:, :, :, -1] = u[:, :, :, 0]
        
        # u_values = u.cpu().detach().numpy()[0, 0, :, :]

        # Create a meshgrid for the x and y coordinates
        # N = np.shape(u_values)[-1]
        # r, theta = np.linspace(0, 1, N), np.linspace(0, 2*np.pi, N)
        # R, Theta = np.meshgrid(r, theta)
        # X, Y = R*np.cos(Theta), R*np.sin(Theta)

        # # Create a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # surf = ax.plot_surface(X, Y, u_values, cmap=cm.coolwarm,
        #                linewidth=0, antialiased=False)

        # # Set labels and title
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('u')
        # ax.set_title('Surface Plot of u')
        # ax.view_init(elev=30, azim=45, roll=15)
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # # Show the plot
        # plt.show()
        return u

    def chebyshev_smoother(self, u, f):
        h = 1 / np.shape(u)[-1]
        fixed_stencil = (1 / h**2) * self.fixed_stencil
        fixed_central_coeff = fixed_stencil[0, 0, 1, 1]
        for i in range(10):
            u_conv_fixed = self.conv2d_polar(u)
            u_conv_fixed = F.pad(u_conv_fixed, (1, 1, 1, 1), "replicate")
            u = u + ((f - u_conv_fixed) / fixed_central_coeff).mul(self.trainable_omega[self.n_operations, i])
        return u
    
    def restrict(self, u):
        u = F.interpolate(u, scale_factor=0.5, mode='bilinear', align_corners=True)
        u = u.clone().to(self.device)
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        u[:, :, :, -1] = u[:, :, :, 0]
        return u

    def prolongate(self, u):
        u = F.interpolate(u, scale_factor=2, mode='bilinear', align_corners=True)
        u = u.clone().to(self.device)
        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 0
        u[:, :, :, -1] = u[:, :, :, 0]
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
                conv_holder = self.conv2d_polar(udictionary[levels].clone())
                conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "replicate")
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
        conv_holder = self.conv2d_polar(u)
        conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "replicate")
        res = torch.sum(torch.abs(self.f - conv_holder))/torch.sum(torch.abs(self.f)) 
        prevres = res
        iter = 0
        convfactorlist = []
        while res > tol and iter < self.max_iter:
            self.n_operations = 0
            u = self.flex_cycle(u)
            conv_holder = self.conv2d_polar(u)
            conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "replicate")
            diff = self.f - conv_holder
            max_value, max_index = torch.max(diff.view(-1), 0)
            print(f'Max value: {max_value.item()}, Max index:{max_index.detach().cpu().numpy()}')
            res = torch.sum(torch.abs(self.f - conv_holder))/torch.sum(torch.abs(self.f)) 
            resratio = res / prevres
            convfactorlist.append(resratio)
            prevres = res
            print(f'Iteration: {iter}; Residual: {res}; Conv Factor: {resratio}')
            iter += 1
            u_values = u.cpu().detach().numpy()[0, 0, :, :]
            N = np.shape(u_values)[-1]
            r, theta = np.linspace(0, 1, N), np.linspace(0, 2*np.pi, N)
            R, Theta = np.meshgrid(r, theta)
            X, Y = R*np.cos(Theta), R*np.sin(Theta)

            # Create a 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, u_values, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

            # Set labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('u')
            ax.set_title('Surface Plot of u')
            ax.view_init(elev=30, azim=45, roll=15)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            # Show the plot
            plt.show()
        end_time = time.time()
        print(f'solve time: {end_time - start_time}, convergence factor: {torch.mean(torch.stack(convfactorlist))}') 
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
        return u, res, time, conv_factor, iter, self.trainable_stencil, self.trainable_weight, self.trainable_omega
    
    def save(self, checkpoint_path: str, epoch: int):
        save_dir: str = os.path.join(checkpoint_path, 'pth')
        os.makedirs(save_dir, exist_ok=True)
        save_path: str = os.path.join(save_dir, f'epoch_{epoch}.pth')
        torch.save(self.trainable_stencil, save_path)
