# pylint: disable=invalid-name
"""
This module contains the necessary tools to interract with hyteg
"""
import torch
import subprocess
import re
import os
import shutil
from enum import Enum
import numpy as np
import evostencils
import evostencils.code_generation.flexible_mg_torch as Solver
import evostencils.code_generation.trainer as Trainer
from loguru import logger

class InterGridOperations(Enum):
    """
    Enum class specifying cycle structure.
    """
    Restriction = -1
    Interpolation = 1
    AltSmoothing = 0

class CorrectionTypes(Enum):
    """
    Enum class specifying correction types.
    """
    Smoothing = 1
    CoarseGridCorrection = 0
class Smoothers(Enum):
    """
    Enum class specifying different smoothers used across the MG cycle.
    """
    WeightedJacobi = 1
    CGS_GE = 2
    NoSmoothing = 0

class ProgramGenerator:
    """
    This class generates a program based on the given parameters.
    """
    def __init__(self,min_level, max_level, mpi_rank=0,cgs_level=0, use_mpi=False) -> None:
        # INPUT
        self.min_level = min_level
        self.cgs_level = cgs_level
        self.max_level = max_level
        self.mpi_rank = mpi_rank
        self.use_mpi = use_mpi
        cwd = os.path.dirname(os.path.dirname(evostencils.__file__))
        # self.template_path = f"{cwd}/flexible_mg_pytorch/"
        # self.problem = "flexible_mg_pytorch.py"
        self.uses_FAS = False
        # generate build path
        # self.build_path = f"{self.template_path}_{self.mpi_rank}/"
        # os.makedirs(self.build_path,exist_ok=True)
        # i. Get a list of all files in the template directory
        # files = os.listdir(self.template_path)
        # files = [file for file in files if os.path.isfile(os.path.join(self.template_path, file))]
        # for file in files:
        #     source_path = os.path.join(self.template_path,file)
        #     destination_path = os.path.join(self.build_path,file)
        #     shutil.copy(source_path,destination_path) # copy from source to destination
        # TEMP OBJECTS
        self.list_states = []
        self.cycle_objs = []
        self.n_individuals = 0

        # MG PARAMETERS
        self.intergrid_ops = []  # sequence of inter-grid operations in the multigrid solver
        # -> describes the cycle structure.
        self.smoothers = []  # sequence of different smoothers used across the MG cycle.
        self.num_sweeps = []  # number of sweeps for each smoother.
        self.relaxation_weights = []  # sequence of relaxation factors for each smoother.
        self.cgc_weights = []  # sequence of relaxations weights at intergrid transfer steps
        # (meant for correction steps, weights in restriction steps is typically set to 1)
        self.cgs_tolerance = None

        #OUTPUT
        self.mgcycle= "" # the command line arguments for MG specification in hyteg.

    @property
    def uses_fas(self):
        """
        Returns whether FAS (Full Approximation Scheme) is used.
        """
        return False

    def reset(self):
        """
        Resets the state of the ProgramGenerator object.
        """
        self.list_states.clear()
        self.cycle_objs.clear()
        self.intergrid_ops.clear()
        self.smoothers.clear()
        self.num_sweeps.clear()
        self.relaxation_weights.clear()
        self.cgc_weights.clear()
        self.mgcycle = []

    def traverse_graph(self, expression):
        """
        Traverses the graph of the given expression.
        """
        expr_type = type(expression).__name__
        cur_lvl = expression.grid[0].level
        list_states = []
        cur_state = {
            'level': cur_lvl,
            'correction_type': None,
            'component': None,
            'relaxation_factor': None,
            'additional_info': None
        }
        if expr_type == "Cycle" and expression not in self.cycle_objs:
            self.cycle_objs.append(expression)
            list_states = self.traverse_graph(expression.approximation) + \
                          self.traverse_graph(expression.correction)
            correction_expr_type = type(expression.correction.operand1).__name__
            if correction_expr_type == "Prolongation":
                cur_state['correction_type'] = CorrectionTypes.CoarseGridCorrection
                cur_state['component'] = -1
            elif correction_expr_type == "Inverse":
                smoothing_operator = expression.correction.operand1.operand
                cur_state['correction_type'] = CorrectionTypes.Smoothing
                cur_state['component'] = smoothing_operator.smoother_type
            cur_state['relaxation_factor'] = expression.relaxation_factor
            list_states.append(cur_state)
            return list_states
        if expr_type == "Multiplication":
            list_states = self.traverse_graph(expression.operand2)
            op_type = type(expression.operand1).__name__
            if op_type == "CoarseGridSolver":
                cur_state['correction_type'] = CorrectionTypes.Smoothing
                cur_state['component'] = Smoothers.CGS_GE
                cur_state['relaxation_factor'] = 1
                cur_state['additional_info'] = expression.operand1.additional_info
                list_states.append(cur_state)
            return list_states
        if "Residual" in expr_type:
            list_states = self.traverse_graph(expression.approximation) + \
                          self.traverse_graph(expression.rhs)
            return list_states
        # else:
        return list_states

    def set_mginputs(self):
        """
        Sets the multigrid inputs.
        """
        cur_lvl = self.max_level # finest level
        first_state_lvl = self.list_states[0]['level']
        # restrict from the finest level until first_state_lvl is reached
        # n_cgs = 0
        while cur_lvl > first_state_lvl:
            self.smoothers.append(Smoothers.NoSmoothing)
            self.relaxation_weights.append(0)
            self.num_sweeps.append(0)
            self.intergrid_ops.append(InterGridOperations.Restriction)
            self.cgc_weights.append(1)
            cur_lvl -=1
        # loop through list_states
        for index,state in enumerate(self.list_states):
            state_lvl = state['level']
            assert state_lvl >= cur_lvl
            if state['correction_type']==CorrectionTypes.Smoothing: # smoothing correction
                if state['component'] == Smoothers.CGS_GE:
                    if state['additional_info']:
                        self.cgs_level = state['additional_info']['CGSlvl']
                        self.cgs_tolerance = state['additional_info']['CGStol']
                    while cur_lvl > self.cgs_level:
                        self.smoothers.append(Smoothers.GaussSeidel)
                        self.num_sweeps.append(1)
                        self.relaxation_weights.append(1)
                        self.intergrid_ops.append(InterGridOperations.Restriction)
                        self.cgc_weights.append(1)
                        cur_lvl -=1
                    self.smoothers.append(Smoothers.CGS_GE)
                    self.num_sweeps.append(1)
                    self.relaxation_weights.append(1)
                    while cur_lvl < state_lvl:
                        self.relaxation_weights.append(1)
                        self.intergrid_ops.append(InterGridOperations.Interpolation)
                        self.cgc_weights.append(1)
                        self.smoothers.append(Smoothers.GaussSeidel)
                        self.num_sweeps.append(1)
                        cur_lvl +=1
                else:
                    self.smoothers.append(state['component'])
                    self.relaxation_weights.append(state['relaxation_factor'])
                    self.num_sweeps.append(1)
            elif state['correction_type'] == CorrectionTypes.CoarseGridCorrection:
                self.intergrid_ops.append(InterGridOperations.Interpolation)
                self.cgc_weights.append(state['relaxation_factor'])
            cur_lvl = state_lvl
            if index+1 < len(self.list_states):
                next_state_lvl = self.list_states[index+1]['level']
                next_state_correction_type = self.list_states[index+1]['correction_type']
                if next_state_lvl < cur_lvl:
                    if state['correction_type']==CorrectionTypes.CoarseGridCorrection:
                        self.smoothers.append(Smoothers.NoSmoothing)
                        self.num_sweeps.append(0)
                        self.relaxation_weights.append(0)
                    while cur_lvl > next_state_lvl:
                        self.intergrid_ops.append(InterGridOperations.Restriction)
                        self.cgc_weights.append(1)
                        self.smoothers.append(Smoothers.NoSmoothing)
                        self.num_sweeps.append(0)
                        self.relaxation_weights.append(0)
                        cur_lvl -=1
                    self.smoothers.pop()
                    self.num_sweeps.pop()
                    self.relaxation_weights.pop()
                # if consecutive coarse grid corrections are performed
                elif next_state_lvl > cur_lvl and state['correction_type'] == \
                    next_state_correction_type and \
                    next_state_correction_type == CorrectionTypes.CoarseGridCorrection:
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)
                # if consecutive smoothing steps are performed at the same level.
                elif next_state_lvl == cur_lvl and state['correction_type'] == \
                    next_state_correction_type == CorrectionTypes.Smoothing:
                    self.intergrid_ops.append(InterGridOperations.AltSmoothing)
                    self.cgc_weights.append(0)
            elif index == len(self.list_states)-1:
                if state['correction_type']==CorrectionTypes.CoarseGridCorrection:
                    self.smoothers.append(Smoothers.NoSmoothing)
                    self.num_sweeps.append(0)
                    self.relaxation_weights.append(0)

    def generate_cmdline_args(self):
        """
        Generate command line arguments.
        """
        # assert checks
        # sum of elements in intergrid_ops is zero, converting the enum to int
        assert sum([i.value for i in self.intergrid_ops]) == 0, \
            "The sum of intergrid operations should be zero"
        # the grid hierarchy should be for self.max_level levels.
        assert min([sum([i.value for i in self.intergrid_ops[:j+1]])
                    for j in range(len(self.intergrid_ops))]) \
                        + self.max_level -self.cgs_level ==0, \
                            "The grid hierarchy should be for self.max_level - self.cgs_levels"
        # length of intergrid_ops is one less than length of smoothers
        assert len(self.intergrid_ops) == len(self.smoothers) - 1, \
              "The number of intergrid operations should be one less\
                  than the number of nodes in the mg cycle"
        # length of smoothing weights is equal to length of smoothers and num_sweeps
        assert len(self.smoothers) == len(self.relaxation_weights) == len(self.num_sweeps),\
              "The number of smoothing weights should be equal to the\
                  number of nodes in the mg cycle"
        # length of cgc weights is equal to length of intergrid_ops
        assert len(self.intergrid_ops) == len(self.cgc_weights), \
            "The number of coarse grid correction weights should be equal\
                  to the number of intergrid operations in the mg cycle"
        # list to comma separated string
        def list_to_string(list):
            string = ""
            for item in list:
                # check if item is an enum
                if type(item).__name__ == 'Smoothers' or type(item).__name__ == 'InterGridOperations':
                    string += str(item.value) + ","
                else:
                    string += str(item) + ","
            return string[:-1]

        # generate the MG cycle string
        self.mgcycle = []
        self.mgcycle.append("-cycleStructure")
        self.mgcycle.append(list_to_string(self.intergrid_ops))
        self.mgcycle.append("-smootherTypes")
        self.mgcycle.append(list_to_string(self.smoothers))
        self.mgcycle.append("-smootherWeights")
        self.mgcycle.append(list_to_string(self.relaxation_weights))
        if not self.cgs_tolerance is None:
            self.mgcycle.append("-coarseGridResidualTolerance")
            self.mgcycle.append(str(self.cgs_tolerance))
            self.mgcycle.append("-minLevel")
            self.mgcycle.append(str(self.cgs_level))

    def execute_code(self, intergrid_operators, smoother, weight):
        """
        Execute the code with the given command line arguments.

        Args:
            cmd_args (list): List of command line arguments. Defaults to None.
        """
        # run the code and pass the command line arguments from the input list
        # output = subprocess.run(["mpiexec", "--map-by", "ppr:1:core", "--bind-to",
        # "core", self.problem] + cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        # timeout=30) # capture_output=True, text=True, cwd=self.build_path, timeout=30)
        N = 2**6
        x, y = np.linspace(0, 1, N), np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        def g(a, x):
            return (x**a) * (1 - x) ** a


        def h(a, x):
            return a * (a - 1) * (1 - 2 * x) ** 2 - 2 * a * x * (1 - x)
        a=10 #randint(1, 20)
        physical_rhs = (2 ** (4 * a)) * g(a, Y) * g(a - 2, X) * h(a, X)
        physical_rhs = torch.from_numpy(physical_rhs[np.newaxis, np.newaxis, :, :].astype(np.float64))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.add(os.path.join('/home/hadrien/Applications/mg_pytorch/evostencils/scripts/train/checkpoints', 'train.log'))

        # backend
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'[Train] Using device {device}')

        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f'[Train] Do not enforce deterministic algorithms, cudnn benchmark enabled')

        solver = Solver.Solver(physical_rhs, intergrid_operators, smoother, weight, device = self.device, trainable = True)
        trainer = Trainer.Trainer("train", '/home/hadrien/Applications/mg_pytorch/evostencils/scripts/train/checkpoints', self.device,
                                solver, logger,
                                'adam', ["step", "1", "0.99"] , 0.5, 1, 1,
                                0, 300, 300, 5, '/home/hadrien/Applications/mg_pytorch/evostencils/scripts/data/',
                                 10, 50)
        run_time, convergence_factor, n_iterations, trainable_stencils, trainable_weight = trainer.train()



        # model = Solver(physical_rhs, intergrid_operators, smoother, weight, trainable = True, trainable_stencils=trainable_stencils, trainable_weight=trainable_weight) # FlexibleMGTorch(self, cmd_args)

        # run_time, convergence_factor, n_iterations = model.run_time, model.convergence_factor, model.n_iterations
        if n_iterations == 100 or convergence_factor>1:
            run_time, convergence_factor, n_iterations = 1e100, 1e100, 1e100
        print(type(run_time), type(convergence_factor), type(n_iterations))
        # del(model)

        return run_time, convergence_factor, n_iterations

    def generate_and_evaluate(self, *args, **kwargs):
        """
        Generate and evaluate the code based on the given arguments and keyword arguments.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        expression_list = []
        time_solution_list = []
        convergence_factor_list = []
        n_iterations_list = []
        cmdline_args = [] # f"{self.build_path}MultigridStudies.prm"]
        evaluation_samples = 1
        for arg in args:
            # get expression list from the input arguments
            if type(arg).__name__ == 'list':
                for cycle in arg:
                    if type(cycle).__name__ == 'Cycle':
                        expression_list.append(cycle)
            elif type(arg).__name__ == 'Cycle':
                expression_list.append(arg)

        self.n_individuals = len(expression_list)
        if 'evaluation_samples' in kwargs:
            evaluation_samples = kwargs['evaluation_samples']

        # print(expression_list)
        for expression in expression_list:
            self.reset()
            self.list_states = self.traverse_graph(expression)

            # fill in the MG parameter list based on the sequence
            # of MG states visited in the GP tree.
            self.set_mginputs()

            # generate cmd line arguments to set mg inputs
            self.generate_cmdline_args()
            cmdline_args += self.mgcycle

        # run the code and pass the command line arguments from the list
        for _ in range(evaluation_samples):
            # print([float(x) for x in cmdline_args[1].split(',')])
            run_time, convergence, n_iterations = self.execute_code([float(x) for x in cmdline_args[1].split(',')], 
                                                                    [float(x) for x in cmdline_args[3].split(',')],
                                                                    [float(x) for x in cmdline_args[5].split(',')])
            time_solution_list.append(run_time)
            convergence_factor_list.append(convergence)
            if type(n_iterations) == torch.Tensor:
                n_iterations = n_iterations.cpu().detach()
            n_iterations_list.append(n_iterations)


        array_mean_time = np.atleast_1d(np.mean(time_solution_list,axis=0))
        array_mean_convergence = np.atleast_1d(np.mean(convergence_factor_list,axis=0))
        array_mean_iterations = np.atleast_1d(np.mean(n_iterations_list,axis=0))

        assert (array_mean_time.shape == array_mean_convergence.shape ==\
                 array_mean_iterations.shape), "The shape of the output arrays \
                    with solver metrics (runtime, convergence, n_iterations) should be the same"
        return array_mean_time, array_mean_convergence, array_mean_iterations

    def generate_cycle_function(self, *args):
        """
        Generate the cycle function based on the given arguments.
        
        Args:
            *args: Variable length argument list.
        
        Returns:
            str: The generated cycle function.
        """
        expression = None
        for arg in args:
            if type(arg).__name__ == 'Cycle':
                expression = arg

        self.reset()
        self.list_states = self.traverse_graph(expression)
        self.set_mginputs()
        self.generate_cmdline_args()
        return self.mgcycle  # str(self.mgcycle)

    # dummy functions to maintain compatibility in the optimisation pipeline
    def generate_storage(self, *args):
        """
        Generate the storage based on the given arguments.
        
        Args:
            *args: Variable length argument list.
        
        Returns:
            list: The generated storage.
        """
        empty_list = []
        return empty_list

    def initialize_code_generation(self, *args):
        """
        Initialize the code generation based on the given arguments.
        
        Args:
            *args: Variable length argument list.
        """
        pass