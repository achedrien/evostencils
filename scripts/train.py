import argparse
import datetime
import os
import torch
import numpy as np
from loguru import logger
import torch
import torch.backends.cudnn
import evostencils.code_generation.flexible_mg_torch as Solver
import evostencils.code_generation.trainer as Trainer

class TrainArg():
    def __init__(self) -> None:
        super().__init__()

        self.structure = 'unet'
        self.downsampling_policy='lerp'
        self.upsampling_policy='lerp'
        self.num_iterations=16
        self.relative_tolerance=1e-6
        self.initialize_x0='random'
        self.num_mg_layers=6
        self.num_mg_pre_smoothing=2
        self.num_mg_post_smoothing=2
        self.activation="none"
        self.initialize_trainable_parameters= 'default'
        self.optimizer='adam'
        self.scheduler=["step", "50", "0.1"] 
        self.initial_lr=1e-3
        self.lambda_1=1
        self.lambda_2=1
        self.start_epoch=0
        self.max_epoch=300
        self.save_every=10
        self.evaluate_every=1
        self.dataset_root='/home/hadrien/Applications/mg_pytorch/evostencils/scripts/data/'
        self.num_workers=24
        self.batch_size=10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_root='/home/hadrien/Applications/mg_pytorch/evostencils/scripts/train/checkpoints'
        self.load_experiment="None"
        self.load_epoch="None"
        self.seed=42 # 9590589012167207234
        self.deterministic=False
        
# noinspection DuplicatedCode
def main() -> None:
    # args
    opt: argparse.Namespace = TrainArg()

    # checkpoint
    experiment_name: str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    experiment_checkpoint_path: str = os.path.join(opt.checkpoint_root, experiment_name)
    os.makedirs(experiment_checkpoint_path, exist_ok=True)
    np.save(os.path.join(experiment_checkpoint_path, 'opt_old.npy'), opt)

    # logger
    logger.add(os.path.join(experiment_checkpoint_path, 'train.log'))
    logger.info('======================== Args ========================')
    for k, v in vars(opt).items():
        logger.info(f'{k}\t\t{v}')
    logger.info('======================================================\n')

    # backend
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'[Train] Using device {device}')

    if opt.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f'[Train] Enforce deterministic algorithms, cudnn benchmark disabled')
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f'[Train] Do not enforce deterministic algorithms, cudnn benchmark enabled')

    if opt.seed is not None:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        logger.info(f'[Train] Manual seed PyTorch with seed {opt.seed}\n')
    else:
        seed: int = torch.seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f'[Train] Using random seed {seed} for PyTorch\n')

    # torch.autograd.set_detect_anomaly(True)  # for debugging only, do NOT use testcase training!

    # training
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
    solver = Solver.Solver(physical_rhs, [-1,-1,-1,-1,1,1,-1,1,1,0,1,0,-1,0,-1,-1,1,-1,1,1,1,0,0],
                          [0,0,1,0,2,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1], 
                          [0,0,1.9,0,1,1.8,1.7,1.75,0,1.45,1.35,0.7499999999999999,1.35,1.7,1.7,0,1.8,1.7,1.9,0,0.8499999999999999,0.5499999999999999,1.65,0.8999999999999999], device=device)
    trainer = Trainer.Trainer(experiment_name, experiment_checkpoint_path, device,
                            solver, logger,
                            opt.optimizer, opt.scheduler, opt.initial_lr, opt.lambda_1, opt.lambda_2,
                            opt.start_epoch, opt.max_epoch, opt.save_every, opt.evaluate_every,
                            opt.dataset_root, opt.num_workers, opt.batch_size)
    trainer.train()


if __name__ == '__main__':
    main()
