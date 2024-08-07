import typing
import os
from torch.utils.data import Dataset
import torch
import torch.utils.data.dataloader
import numpy as np
import torch.nn.functional as F
import typing
import random

def norm(x: torch.Tensor) -> torch.Tensor:
    """
    Vector norm on each batch.
    Note: We only deal with cases where channel == 1!
    :param x: (batch_size, channel, image_size, image_size)
    :return: (batch_size,)
    """
    y = x.view(x.size(0), -1)
    return (y * y).sum(dim=1).sqrt()

def square_residue(x: torch.Tensor,
                     bc_mask: torch.Tensor,
                     f: typing.Optional[torch.Tensor],
                     reduction: str = 'norm') -> torch.Tensor:
    """
    For a linear system Ax = f,
    the absolute residue is r = f - Ax,
    the absolute residual (norm) error eps = ||f - Ax||.
    """
    # eps of size (batch_size, channel (1), image_size, image_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    laplace_kernel = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float64
                              ).view(1, 1, 3, 3).to(device)
    eps = F.conv2d(x, laplace_kernel, padding=1)

    if f is not None:
        eps = eps - f
    eps = eps * (1 - bc_mask)
    eps = eps.view(eps.size(0), -1)            # of size (batch_size, image_size ** 2)

    if reduction == 'norm':
        error = norm(eps)                      # of size (batch_size,)
    elif reduction == 'mean':
        error = torch.abs(eps).mean(dim=1)     # of size (batch_size,)
    elif reduction == 'max':
        error = torch.abs(eps).max(dim=1)[0]   # of size (batch_size,)
    elif reduction == 'none':
        error = -eps                           # of size (batch_size, image_size ** 2)
    else:
        raise NotImplementedError

    return error**2

class SynDat(Dataset):
    def __init__(self, dataset_root: str):
        super().__init__()
        self.dataset_root: str = dataset_root
        self.instance_path_lst: typing.List[str] = []

        for dirpath, dirnames, filenames in os.walk(self.dataset_root):
            for filename in sorted(filenames):
                if ".npy" in filename:
                    self.instance_path_lst.append(os.path.join(dirpath, filename))

    def __getitem__(self, index: int):
        data: np.ndarray = np.load(self.instance_path_lst[index], allow_pickle=True)
        x, b = data
        x: torch.Tensor = torch.from_numpy(x).float().unsqueeze(0)
        b: torch.Tensor = torch.from_numpy(b).float().unsqueeze(0)
        return {'x': x,
                'b': b}

    def __len__(self):
        return len(self.instance_path_lst) // 100

class Trainer:
    def __init__(self,
                 experiment_name: str, experiment_checkpoint_path: str, device: torch.device,
                 model, logger,
                 optimizer: str, scheduler: str, initial_lr: float, lambda_1: float, lambda_2: float,
                 start_epoch: int, max_epoch: int, save_every: int, evaluate_every: int,
                 dataset_root: str, num_workers: int, batch_size: int, verbose: int = 0):
        self.experiment_name: str = experiment_name
        self.experiment_checkpoint_path: str = experiment_checkpoint_path

        self.model = model
        self.logger = logger

        self.device: torch.device = device

        self.initial_lr: float = initial_lr
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2

        self.start_epoch: int = start_epoch
        self.max_epoch: int = max_epoch
        self.save_every: int = save_every
        self.evaluate_every: int = evaluate_every

        self.dataset_root: str = dataset_root
        self.num_workers: int = num_workers
        self.batch_size: int = batch_size

        self.verbose: int = verbose
        self.best_loss: float = float('inf')
        self.epochs_no_improve: int = 0

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_lr)
        elif optimizer == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.initial_lr)
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.initial_lr)
        elif optimizer == 'LBFGS':
            # for param in model.parameters():
                # print(param, param.size())
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.initial_lr, max_iter=50, max_eval=None, tolerance_grad=1e-15, tolerance_change=1e-15, history_size=101, line_search_fn='strong_wolfe')
        else:
            raise NotImplementedError

        if 'step' in scheduler:
            _, step_size, gamma = scheduler
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, int(step_size), float(gamma))
        else:
            raise NotImplementedError

        train_dataset_path: str = os.path.join(dataset_root, 'train')
        self.train_dataset = SynDat(train_dataset_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        num_workers=self.num_workers,
                                                        batch_size=batch_size,
                                                        pin_memory=True,
                                                        shuffle=True)

        if verbose > 0:
            logger.info(f'[Trainer] {len(self.train_dataset)} training data loaded from {train_dataset_path}')

        evaluate_dataset_path: str = os.path.join(dataset_root, 'evaluate')
        self.evaluate_dataset = SynDat(evaluate_dataset_path)
        self.evaluate_loader = torch.utils.data.DataLoader(self.evaluate_dataset,
                                                           num_workers=self.num_workers,
                                                           batch_size=batch_size,
                                                           pin_memory=True)

        if verbose > 0:
            logger.info(f'[Trainer] {len(self.evaluate_dataset)} evaluation data loaded from {evaluate_dataset_path}\n')

    def train(self):
        for epoch in range(self.start_epoch, self.max_epoch):
            self.model.train()
            train_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

            for batch in self.train_loader:
                u: typing.Optional[torch.Tensor] = batch['x'][0, 0, :, :, :, :].to(self.device)
                f: typing.Optional[torch.Tensor] = batch['b'][0, 0, :, :, :, :].to(self.device)
                # print(u, f)
                fixed_stencil = torch.tensor([[0.0, 1.0, 0.0],
                                  [1.0, -4.0, 1.0],
                                  [0.0, 1.0, 0.0]], dtype=torch.float64).to(self.device).unsqueeze(0).unsqueeze(0)
                conv_holder = F.conv2d(u.type(torch.float64), (1 / (1 / np.shape(u)[-1])**2) * fixed_stencil, padding=0)
                f = F.pad(conv_holder, (1, 1, 1, 1), "constant", 0)
                tup: typing.Tuple[torch.Tensor, int] = self.model(f, 1e-3, self.optimizer)
                y, res, time, conv_factor, iterations_used, trainable_stencils, trainable_weight, trainable_omega = tup
                residue: torch.Tensor = square_residue(y, f.to(self.device), f.to(self.device), reduction='none')

                def closure():
                    with torch.enable_grad():
                        self.optimizer.zero_grad()
                        tup = self.model(batch['b'], 1e-3, self.optimizer)
                        y, res, time, conv_factor, iterations_used, trainable_stencils, trainable_weight, trainable_omega = tup
                        # residue = square_residue(y, batch['x'].to(self.device), f, reduction='none')
                        # loss_x = torch.tensor((torch.mean(residue)**0.5)/torch.mean(batch['b']), requires_grad=True).to(self.device)
                        loss_x = F.mse_loss(y, batch['x'][0, 0, :, :, :, :].double().to(self.device)) #torch.tensor(conv_factor, requires_grad=True).to(self.device)
                        with torch.autograd.set_detect_anomaly(True):
                            loss_x.backward() #retain_graph=True)
                        # print(loss_x.grad)
                        # print(loss_x.grad_fn)
                        # print(trainable_omega)
                    return loss_x
                # loss_x: torch.Tensor =  norm(residue).mean().to(self.device) # 
                self.optimizer.step(closure=closure)

            if 0 < self.evaluate_every and (epoch + 1) % self.evaluate_every == 0:
                with torch.no_grad():
                    self.model.eval()
                    evaluate_loss_dict: typing.Dict[str, typing.List[torch.Tensor]] = {}

                    for batch in self.evaluate_loader:
                        x: typing.Optional[torch.Tensor] = None
                        bc_value: torch.Tensor = batch['bc_value'].to(self.device)
                        bc_mask: torch.Tensor = batch['bc_mask'].to(self.device)
                        f: typing.Optional[torch.Tensor] = None

                        tup: typing.Tuple[torch.Tensor, int] = self.model(x, bc_value, bc_mask, f)
                        y, iterations_used = tup
                    if loss_x < self.best_loss:
                        self.best_loss = loss_x
                        self.epochs_no_improve = 0
                    else:
                        self.epochs_no_improve += 1
                        if self.epochs_no_improve == self.patience:
                            self.max_epoch = epoch
                            return time, conv_factor, iterations_used, trainable_stencils, trainable_weight, trainable_omega
                    self.model.train()
            if self.verbose > 0:
                self.logger.info('trainable omega = {trainable_omega}, trainable weight = {trainable_weight}, time = {time}, conv_factor = {conv_factor}, iterations_used = {iterations_used}'.format(trainable_omega=trainable_omega, trainable_weight=trainable_weight, time=time, conv_factor=conv_factor, iterations_used=iterations_used))
            self.scheduler.step()
            if (epoch + 1) % self.save_every == 0 or epoch == self.max_epoch - 1:
                self.model.save(self.experiment_checkpoint_path, epoch + 1)
        return time, conv_factor, iterations_used, trainable_stencils, trainable_weight, trainable_omega