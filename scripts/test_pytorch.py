from evostencils.optimization.program import Optimizer
from evostencils.code_generation.mg_torch import ProgramGenerator
import os
import sys
from mpi4py import MPI
import evostencils
import numpy as np

stencils = np.array([[0.0, 1.0, 0.0],
                     [1.0, -4.0, 1.0],
                     [0.0, 1.0, 0.0]])
program_generator = ProgramGenerator(0, 4, stencils) #, mpi_rank=0, cgs_level=0, use_mpi=False)
cwd = f'{os.getcwd()}'
dimension = 2# program_generator.dimension  # Dimensionality of the problem
finest_grid = 0# program_generator.finest_grid  # Representation of the finest grid
coarsening_factors = 0 #program_generator.coarsening_factor
min_level = program_generator.min_level  # Minimum discretization level
max_level = program_generator.max_level  # Maximum discretization level
# equations = # program_generator.equations  # System of PDEs in SymPy
# operators = # program_generator.operators  # Discretized differential operators
# fields = # program_generator.fields  # Variables that occur within system of PDEs
problem_name = "2dpoisson" #program_generator.problem_name
convergence_evaluator = None
performance_evaluator = None

comm = None
nprocs = 1 #comm.Get_size()
mpi_rank = 0# comm.Get_rank()
checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints'
# Create optimizer object
optimizer = Optimizer(dimension, finest_grid, coarsening_factors, min_level, max_level,
                        mpi_comm=comm, mpi_rank=mpi_rank, number_of_mpi_processes=nprocs,
                        program_generator=program_generator,
                        convergence_evaluator=convergence_evaluator,
                        performance_evaluator=performance_evaluator,
                        checkpoint_directory_path=checkpoint_directory_path)
# Option to split the optimization into multiple runs,
# where each run is only performed on a subrange of the discretization hierarchy starting at the top (finest grid)
# (Not recommended for code-generation based model_based_prediction)
levels_per_run = max_level - min_level
if model_based_estimation:
    # Model-based estimation only feasible for up to 2 levels per run
    levels_per_run = 1
assert levels_per_run <= 5, "Can not optimize more than 5 levels"
# Choose optimization method
optimization_method = optimizer.NSGAII
if len(sys.argv) > 1:
    # Multi-objective (mu+lambda)-EA with NSGA-II non-dominated sorting-based selection
    if sys.argv[1].upper() == "NSGAII":
        optimization_method = optimizer.NSGAII
    # Multi-objective (mu+lambda)-EA with NSGA-III non-dominated sorting-based selection
    elif sys.argv[1].upper() == "NSGAIII":
        optimization_method = optimizer.NSGAIII
    # Classic single-objective (mu+lambda)-EA with binary tournament selection
    elif sys.argv[1].upper() == "SOGP":
        optimization_method = optimizer.SOGP
# Option to use random search instead of crossover and mutation to create new individuals
use_random_search = False

mu_ = 8  # Population size
lambda_ = 8  # Number of offspring
generations = 50  # Number of generations
population_initialization_factor = 4  # Multiply mu_ by this factor to set the initial population size

# Number of generations after which a generalization is performed
# This is achieved by incrementing min_level and max_level within the optimization
# Such that a larger (and potentially more difficult) instance of the same problem is considered in subsequent generations
generalization_interval = 50
crossover_probability = 0.7
mutation_probability = 1.0 - crossover_probability
node_replacement_probability = 0.1  # Probability to perform mutation by altering a single node in the tree
evaluation_samples = 3  # Number of evaluation samples
maximum_local_system_size = 4  # Maximum size of the local system solved within each step of a block smoother
# Option to continue from the checkpoint of a previous optimization
# Warning: So far no check is performed whether the checkpoint is compatible with the current optimization setting
continue_from_checkpoint = False

# Return values of the optimization
# program: ExaSlang program string representing the multigrid solver functions
# pops: Populations at the end of each optimization run on the respective subrange of the discretization hierarchy
# stats: Statistics structure (data structure provided by the DEAP framework)
# hofs: Hall-of-fames at the end of each optimization run on the respective subrange of the discretization hierarchy
program, dsl_code, pops, stats, hofs = optimizer.evolutionary_optimization(optimization_method=optimization_method,
                                                                    use_random_search=use_random_search,
                                                                    mu_=mu_, lambda_=lambda_,
                                                                    population_initialization_factor=population_initialization_factor,
                                                                    generations=generations,
                                                                    generalization_interval=generalization_interval,
                                                                    crossover_probability=crossover_probability,
                                                                    mutation_probability=mutation_probability,
                                                                    node_replacement_probability=node_replacement_probability,
                                                                    levels_per_run=levels_per_run,
                                                                    evaluation_samples=evaluation_samples,
                                                                    maximum_local_system_size=maximum_local_system_size,
                                                                    model_based_estimation=model_based_estimation,
                                                                    pde_parameter_values=pde_parameter_values,
                                                                    continue_from_checkpoint=continue_from_checkpoint)
# Print the outcome of the optimization and store the data and statistics
if mpi_rank == 0:
    print(f'\nExaSlang Code:\n{dsl_code}\n', flush=True)
    print(f'\nGrammar representation:\n{program}\n', flush=True)
    if not os.path.exists(f'./{problem_name}'):
        os.makedirs(f'./{problem_name}')
    j = 0
    log_dir_name = f'./{problem_name}/data_{j}'
    while os.path.exists(log_dir_name):
        j += 1
        log_dir_name = f'./{problem_name}/data_{j}'
    os.makedirs(log_dir_name)
    for i, log in enumerate(stats):
        optimizer.dump_data_structure(log, f"{log_dir_name}/log_{i}.p")
    for i, pop in enumerate(pops):
        optimizer.dump_data_structure(pop, f"{log_dir_name}/pop_{i}.p")
    for i, hof in enumerate(hofs):
        hof_dir = f'{log_dir_name}/hof_{i}'
        os.makedirs(hof_dir)
        for j, ind in enumerate(hof):
            with open(f'{hof_dir}/individual_{j}.txt', 'w') as grammar_file:
                grammar_file.write(str(ind) + '\n')