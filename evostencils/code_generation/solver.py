import os
import typing
import torch.nn.functional as F
import numpy as np
import torch

from evostencils.code_generation.flexible_mg_torch import Flexible_mg_torch

def absolute_residue(x: torch.Tensor,
                     bc_mask: torch.Tensor,
                     f: typing.Optional[torch.Tensor],
                     reduction: str = 'norm') -> torch.Tensor:
    """
    For a linear system Ax = f,
    the absolute residue is r = f - Ax,
    the absolute residual (norm) error eps = ||f - Ax||.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    laplace_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32
                              ).view(1, 1, 3, 3).to(device)

    # eps of size (batch_size, channel (1), image_size, image_size)
    eps = F.conv2d(x, laplace_kernel, padding=1)

    if f is not None:
        eps = eps - f

    eps = eps * (1 - bc_mask)
    eps = eps.view(eps.size(0), -1)            # of size (batch_size, image_size ** 2)

    if reduction == 'norm':
        y = eps.view(eps.size(0), -1)
        error = (y * y).sum(dim=1).sqrt()                      # of size (batch_size,)
    elif reduction == 'mean':
        error = torch.abs(eps).mean(dim=1)     # of size (batch_size,)
    elif reduction == 'max':
        error = torch.abs(eps).max(dim=1)[0]   # of size (batch_size,)
    elif reduction == 'none':
        error = -eps                           # of size (batch_size, image_size ** 2)
    else:
        raise NotImplementedError

    return error

class Solver:
    def __init__(self, physical_rhs, intergrid_operators, smoother, weight, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.physical_rhs = physical_rhs
        self.intergrid_operators = intergrid_operators
        self.smoother = smoother
        self.weight = weight
        self.is_train: bool = True
        self.iterator = Flexible_mg_torch(physical_rhs, intergrid_operators, smoother, weight, device=None).to(self.device)
        
    def __call__(self,
                 x: typing.Optional[torch.Tensor],
                 bc_value: torch.Tensor,
                 bc_mask: torch.Tensor,
                 f: typing.Optional[torch.Tensor],
                 rel_tol: typing.Optional[float] = None) \
            -> typing.Tuple[torch.Tensor, int]:
        if not self.is_train:
            if rel_tol is None:
                rel_tol: float = self.relative_tolerance

            with torch.no_grad():
                rhs = bc_value

                if f is not None:
                    rhs = rhs + f

                y = rhs.view(rhs.size(0), -1)
                rhs_norm: torch.Tensor = (y * y).sum(dim=1).sqrt()
                abs_tol: torch.Tensor = rel_tol * rhs_norm
        self.initial_guess = torch.zeros_like(y).to(self.device)
        if x is None:
            x: torch.Tensor = self.initial_guess(bc_value, bc_mask)

        # # TODO: UGrid benchmark
        # np.save(f'var/conv/UGrid/tmp/x0.npy',
        #         x.detach().squeeze().cpu().numpy())

        for iteration in range(1, self.num_iterations + 1):
            x: torch.Tensor = self.iterator(x, bc_value, bc_mask, f)

            # # TODO: UGrid benchmark
            # np.save(f'var/conv/UGrid/tmp/x{iteration}.npy',
            #         x.detach().squeeze().cpu().numpy())

            if not self.is_train:
                with torch.no_grad():
                    if iteration % 4 == 0 and \
                            torch.all(absolute_residue(x, bc_mask, f, reduction='norm') <= abs_tol):
                        break

        # noinspection PyUnboundLocalVariable
        return x, iteration

    def train(self):
        self.is_train = True
        self.iterator.train()

    def eval(self):
        self.is_train = False
        self.iterator.eval()

    def parameters(self):
        return self.iterator.parameters()

    def load(self, checkpoint_path: str, epoch: int):
        checkpoint_pth_root: str = os.path.join(checkpoint_path, 'pth')

        if epoch == -1:
            cnt = 0
            for name in os.listdir(checkpoint_pth_root):
                if os.path.isfile(os.path.join(checkpoint_pth_root, name)):
                    cnt += 1
            epoch = cnt

        load_path: str = os.path.join(checkpoint_path, 'pth', f'epoch_{epoch}.pth')
        self.iterator.load_state_dict(torch.load(load_path))

        return load_path

    def save(self, checkpoint_path: str, epoch: int):
        save_dir: str = os.path.join(checkpoint_path, 'pth')
        os.makedirs(save_dir, exist_ok=True)
        save_path: str = os.path.join(save_dir, f'epoch_{epoch}.pth')
        torch.save(self.iterator.state_dict(), save_path)