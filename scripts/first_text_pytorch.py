import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, spdiags, csr_matrix
from scipy.sparse.linalg import spsolve
import time

def plot(X, Y, unumpy, resnumpy, exact):
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    contourf1 = axs[0, 0].contourf(X, Y, unumpy, cmap='viridis')
    axs[0, 0].set_title('Approximate solution')
    fig.colorbar(contourf1, ax=axs[0, 0])
    contourf2 = axs[0, 1].contourf(X, Y, resnumpy, cmap='viridis')
    axs[0, 1].set_title('Residual')
    fig.colorbar(contourf2, ax=axs[0, 1])
    contourf3 = axs[1, 0].contourf(X, Y, exact, cmap='viridis')
    axs[1, 0].set_title('Exact solution')
    fig.colorbar(contourf3, ax=axs[1, 0])
    contourf4 = axs[1, 1].contourf(X, Y, np.abs(unumpy - exact), cmap='viridis')
    axs[1, 1].set_title('Absolute error')
    fig.colorbar(contourf4, ax=axs[1, 1])
    fig.tight_layout()
    plt.show()

def weighted_jacobi_smoother_pytorch(u, f, stencil, omega, iterations):
    h = 1 / np.shape(u)[-1]
    stencil = (1 / h**2) * stencil
    central_coeff = stencil[0, 0, 1, 1]
    for _ in range(iterations):
        u_conv = F.conv2d(u, stencil, padding=0)
        u_conv = F.pad(u_conv, (1, 1, 1, 1), "constant", 0)
        u = u + omega * (f - u_conv) / central_coeff
    return u

def restrict_pytorch(u):
    u = F.interpolate(u, scale_factor=0.5, mode='bilinear', align_corners=True)
    return u

def prolongate_pytorch(u):
    u = F.interpolate(u, scale_factor=2, mode='bilinear', align_corners=True)
    return u

def cgs_pytorch(u, f):
    u_np = u.cpu().detach().numpy()[0, 0, :, :].flatten()
    f_np = f.cpu().detach().numpy()[0, 0, :, :].flatten()
    N = int(len(u_np)**0.5)
    A = np.kron(np.diag(np.ones(N - 1), -1) + np.diag(np.ones(N - 1), 1), - np.eye(N))
    D = -1.0 * np.diag(np.ones(N - 1), -1) + -1.0 * np.diag(np.ones(N - 1), 1) + 4 * np.diag(np.ones(N), 0)
    A += np.kron(np.diag(np.ones(N), 0), D)
    A = -A*int(len(u_np)**0.5)**2
    u_np = np.linalg.inv(A) @ f_np
    return torch.from_numpy(u_np.reshape(int(np.shape(u_np)[-1]**0.5), int(np.shape(u_np)[-1]**0.5))[np.newaxis, np.newaxis, :, :].astype(np.float64)).to(device)


def v_cycle_pytorch(u, f, stencil, presmoothing, postsmoothing, level,
                    depth, tol):
    if level == 0:
        return cgs_pytorch(u, f)

    u_new = weighted_jacobi_smoother_pytorch(u, f, stencil, 1, presmoothing)
    conv_holder = F.conv2d(u_new, (1 / (1 / np.shape(u)[-1])**2) * stencil,
                           padding=0)
    conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
    residual = f - conv_holder
    f_new = restrict_pytorch(residual)
    u_coarse = torch.zeros_like(f_new)
    u_coarse = v_cycle_pytorch(u_coarse, f_new, stencil, presmoothing,
                       postsmoothing, level - 1, depth, tol)
    u_new += prolongate_pytorch(u_coarse)
    u_new = weighted_jacobi_smoother_pytorch(u_new, f, stencil, 1, postsmoothing)
    return u_new

def flex_cycle_pytorch(u, f, stencil, levels, intergrid_operators, smoothers, weights):
    # u: initial guess
    # f: right hand side
    # stencil: finite difference stencil
    # levels: number of levels
    # intergrid_operators: list of intergrid operators (-1: restrict, 0: stay at same level, 1: prolongation)
    # smoothers: list of smoothers (1: weighted jacobi smoother_pytorch, 0: CGS)
    # weights: list of weights for smoothers
    udictionary = {}
    fdictionary = {}
    udictionary[levels] = u
    fdictionary[levels] = f
    for i, operator in enumerate(intergrid_operators):
        if operator == -1:
            conv_holder = F.conv2d(udictionary[levels], (1 / (1 / np.shape(udictionary[levels])[-1])**2) * stencil,
                                   padding=0)
            conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
            residual = f - conv_holder
            fdictionary[levels - 1] = restrict_pytorch(residual)
            udictionary[levels - 1] = torch.zeros_like(fdictionary[levels - 1])
            levels -= 1
        elif operator == 1:
            udictionary[levels + 1] += prolongate_pytorch(udictionary[levels])
            levels += 1
        else:
            if smoothers[i] == 1:
                udictionary[levels] = weighted_jacobi_smoother_pytorch(udictionary[levels], fdictionary[levels], stencil, weights[i])
            elif smoothers[i] == 0:
                udictionary[levels] = cgs_pytorch(udictionary[levels], fdictionary[levels])
    return udictionary[levels]

def solve_poisson_pytorch(f, stencil, presmoothing, postsmoothing, levels, tol):
    u_pytorch = torch.zeros_like(f)
    h = 1 / np.shape(u_pytorch)[-1]
    conv_holder = F.conv2d(u_pytorch, (1 / (1 / np.shape(u_pytorch)[-1])**2) * stencil, padding=0)
    conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
    res = torch.sum(torch.abs(f - conv_holder))/torch.sum(torch.abs(f))
    initres = res
    prevres = res
    iter = 0
    # x, y = np.linspace(0, 1, np.shape(u_pytorch)[-1]), np.linspace(0, 1, np.shape(u_pytorch)[-1])
    # X, Y = np.meshgrid(x, y)
    # exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    while res > tol and iter < 100:
        u_pytorch = v_cycle_pytorch(u_pytorch, f, stencil, presmoothing,
                    postsmoothing, levels, levels, tol)
        conv_holder = F.conv2d(u_pytorch, (1 / (1 / np.shape(u_pytorch)[-1])**2) * stencil, padding=0)
        conv_holder = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
        res = torch.sum(torch.abs(f - conv_holder))/torch.sum(torch.abs(f))
        resratio = res / prevres
        prevres = res
        print(
             f'Iteration: {iter}; Residual: {res}; Conv Factor: {resratio}')
        iter += 1
    # plot(X, Y, unumpy_pytorch, resnumpy, exact)
    return u_pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("\n######################### Multigrid with Pytorch ############################\n")

for size in [3, 4, 5, 6, 7, 8, 9]: #, 9, 10]:
    print(f"\n##########   size: {(2**size)**2} #############\n")

    stencil = torch.tensor([[0.0, 1.0, 0.0],
                            [1.0, -4.0, 1.0],
                            [0.0, 1.0, 0.0]], dtype=torch.float64).to(device)
    stencil = stencil.unsqueeze(0).unsqueeze(0)
    N = 2**size
    x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    physical_rhs = np.sin(np.pi * X) * np.sin(2 * np.pi * Y)
    f = torch.from_numpy(
        physical_rhs[np.newaxis, np.newaxis, :, :].astype(np.float64)).to(device)
    start_time = time.time()
    u_pytorch = solve_poisson_pytorch(f, stencil, 2, 2, size - 2, 1e-3)
    end_time = time.time()

    print("Test" + str(size) + " Computation time for u_pytorch:", end_time - start_time, "seconds")
    u = u_pytorch.detach().cpu().numpy()
    np.savetxt(f"u_size_{N}.txt", np.column_stack((X.flatten(), Y.flatten(), u.flatten())), delimiter=',')
