"""
Script for Hyteg optimization.
"""

import os
import datetime
from mpi4py import MPI
from evostencils.optimization.program import Optimizer
from evostencils.code_generation.mg_torch import ProgramGenerator

banner = """
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
░        ░░  ░░░░  ░░░      ░░░░      ░░░        ░░        ░░   ░░░  ░░░      ░░░        ░░  ░░░░░░░░░      ░░
▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒  ▒▒  ▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒    ▒▒  ▒▒  ▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒  ▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒
▓      ▓▓▓▓▓  ▓▓  ▓▓▓  ▓▓▓▓  ▓▓▓      ▓▓▓▓▓▓  ▓▓▓▓▓      ▓▓▓▓  ▓  ▓  ▓▓  ▓▓▓▓▓▓▓▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓▓▓▓▓▓      ▓▓
█  ██████████    ████  ████  ████████  █████  █████  ████████  ██    ██  ████  █████  █████  ██████████████  █
█        █████  ██████      ████      ██████  █████        ██  ███   ███      ███        ██        ███      ██
██████████████████████████████████████████████████████████████████████████████████████████████████████████████
"""

def main():
    cwd = f'{os.getcwd()}'
    eval_software = "hyteg"

    # Set up MPI
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    mpi_rank = comm.Get_rank()
    if nprocs > 1:
        tmp = "processes"
        use_mpi = True
    else:
        tmp = "process"
        use_mpi = False
    if mpi_rank == 0:
        print(f"Running {nprocs} MPI {tmp}")
        print(banner)

    # problem specifications
    flexmg_min_level = 0
    flexmg_max_level = 4
    cgs_level = 0
    assert flexmg_min_level < flexmg_max_level
    assert flexmg_min_level >= cgs_level
    assert flexmg_max_level - flexmg_min_level < 5
    problem_name = "2dpoisson"
    program_generator = ProgramGenerator(flexmg_min_level,flexmg_max_level,
                                         mpi_rank, cgs_level, use_mpi=use_mpi)

    if mpi_rank == 0 and not os.path.exists(f'{cwd}/{problem_name}'):
        # Create directory for checkpoints and output data
        os.makedirs(f'{cwd}/{problem_name}')
    # Path to directory for storing checkpoints
    now = datetime.datetime.now()
    date_and_time = now.strftime("%d_%m_%y-%H:%M")
    checkpoint_directory_path = f'{cwd}/{problem_name}/checkpoints_{date_and_time}'
    # Create optimizer object
    optimizer = Optimizer(flexmg_min_level, flexmg_max_level, mpi_comm=comm, mpi_rank=mpi_rank,
                          number_of_mpi_processes=nprocs, program_generator=program_generator,
                          checkpoint_directory_path=checkpoint_directory_path)

    # IV. optimization parameters
    optimization_method = optimizer.NSGAII


    mu_ = 24 # Population size
    lambda_ = 12 # Number of offspring
    generations = 25 # 250 # 0 # Number of generations
    population_initialization_factor = 8  # Multiplicator of the initial population size
    generalization_interval = 1e100
    crossover_probability = 2/3
    mutation_probability = 1.0 - crossover_probability
    node_replacement_probability = 0.2  # Probability to perform mutation
    evaluation_samples = 1 # 32  # Number of evaluation samples
    maximum_local_system_size = 8  # Maximum size of the local system solved
    continue_from_checkpoint = False
    lambda_prime = int(lambda_ / nprocs)

    program, dsl_code, pops, stats, hofs, fitnesses = \
        optimizer.evolutionary_optimization(optimization_method=optimization_method,
                                            use_random_search=False,
                                            mu_=mu_, lambda_=lambda_prime,
                                            population_initialization_factor=\
                                                population_initialization_factor,
                                            generations=generations,
                                            generalization_interval=generalization_interval,
                                            crossover_probability=crossover_probability,
                                            mutation_probability=mutation_probability,
                                            node_replacement_probability=\
                                                node_replacement_probability,
                                            levels_per_run=flexmg_max_level - flexmg_min_level,
                                            evaluation_samples=evaluation_samples,
                                            maximum_local_system_size=maximum_local_system_size,
                                            continue_from_checkpoint=continue_from_checkpoint)
    # Print the outcome of the optimization and store the data and statistics
    if mpi_rank == 0:
        print(f'\n{eval_software} specifications:\n{dsl_code}\n', flush=True)
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
                with open(f'{hof_dir}/individual_{j}.txt', 'w', encoding='utf-8') as grammar_file:
                    grammar_file.write(str(ind) + '\n')
        optimizer.dump_data_structure(fitnesses, f"{log_dir_name}/fitnesses.p")


if __name__ == "__main__":
    main()