from evostencils.expressions import base, partitioning as part, system
from evostencils.initialization import multigrid, parser
import os
import subprocess
import math
import sympy


class CycleStorage:
    def __init__(self, equations: [multigrid.EquationInfo], fields: [sympy.Symbol], grid):
        self.grid = grid
        self.solution = [Field(f'{symbol.name}', g.level, self) for g, symbol in zip(grid, fields)]
        self.rhs = [Field(f'{eq_info.rhs_name}', g.level, self) for g, eq_info in zip(grid, equations)]
        self.residual = [Field(f'Residual_{symbol.name}', g.level, self) for g, symbol in zip(grid, fields)]
        self.correction = [Field(f'Correction_{symbol.name}', g.level, self) for g, symbol in zip(grid, fields)]


class Field:
    def __init__(self, name=None, level=None, cycle_storage=None):
        self.name = name
        self.level = level
        self.cycle_storage = cycle_storage
        self.valid = False

    def to_exa3(self):
        if self.level > 0:
            return f'{self.name}@(finest - {self.level})'
        else:
            return f'{self.name}@finest'


class ProgramGenerator:
    def __init__(self, absolute_compiler_path: str, base_path: str, settings_path: str, knowledge_path: str,
                 platform='linux'):
        self._absolute_compiler_path = absolute_compiler_path
        self._base_path = base_path
        self._knowledge_path = knowledge_path
        self._settings_path = settings_path
        self._dimension, self._min_level, self._max_level = \
            parser.extract_knowledge_information(base_path, knowledge_path)
        self._base_path_prefix, self._problem_name, self._debug_l3_path, self._output_path = \
            parser.extract_settings_information(base_path, settings_path)
        self._platform = platform
        self.run_exastencils_compiler()
        self._equations, self._operators, self._fields = \
            parser.extract_l2_information(f'{base_path}/{self._debug_l3_path}', self.dimension)
        size = 2 ** self._max_level
        grid_size = tuple([size] * self.dimension)
        h = 1 / (2 ** self._max_level)
        step_size = tuple([h] * self.dimension)
        tmp = tuple([2] * self.dimension)
        self._coarsening_factor = [tmp for _ in range(len(self.fields))]
        self._finest_grid = [base.Grid(grid_size, step_size, self.max_level) for _ in range(len(self.fields))]
        self._compiler_available = False
        if os.path.exists(absolute_compiler_path):
            if os.path.isfile(absolute_compiler_path):
                self._compiler_available = True

    @property
    def absolute_compiler_path(self):
        return self._absolute_compiler_path

    @property
    def knowledge_path(self):
        return self._knowledge_path

    @property
    def settings_path(self):
        return self._settings_path

    @property
    def problem_name(self):
        return self._problem_name

    @property
    def compiler_available(self):
        return self._compiler_available

    @property
    def base_path(self):
        return self._base_path

    @property
    def output_path(self):
        return self._output_path

    @property
    def platform(self):
        return self._platform

    @property
    def dimension(self):
        return self._dimension

    @property
    def finest_grid(self):
        return self._finest_grid

    @property
    def equations(self):
        return self._equations

    @property
    def operators(self):
        return self._operators

    @property
    def fields(self):
        return self._fields

    @property
    def coarsening_factor(self):
        return self._coarsening_factor

    @property
    def min_level(self):
        return self._min_level

    @property
    def max_level(self):
        return self._max_level

    def generate_global_weight_initializations(self, weights):
        # Hack to change the weights after generation
        weights = reversed(weights)
        path_to_file = f'{self.base_path}/{self.output_path}/Globals/Globals_initGlobals.cpp'
        subprocess.run(['cp', path_to_file, f'{path_to_file}.backup'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(path_to_file, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1]
            lines = lines[:-1]
        content = ''
        for line in lines:
            content += line
        for i, weight in enumerate(weights):
            lines.append(f'\tomega_{i} = {weight};\n')
            content += lines[-1]
        content += last_line
        with open(path_to_file, 'w') as file:
            file.write(content)

    def restore_global_initializations(self):
        # Hack to change the weights after generation
        path_to_file = f'{self.base_path}/{self.output_path}/Globals/Globals_initGlobals.cpp'
        subprocess.run(['cp', f'{path_to_file}.backup', path_to_file],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def generate_cycle_function(self, expression, storages, use_global_weights=False):
        base_level = 0
        for i, storage in enumerate(storages):
            if expression.grid.size == storage.grid.size:
                expression.storage = storage.solution
                base_level = i
                break
        self.assign_storage_to_subexpressions(expression, storages, base_level)
        program = f'Function Cycle@(finest - {base_level}) {{\n'
        program += self.generate_multigrid(expression, storages, use_global_weights)
        program += '}\n'
        return program

    def run_exastencils_compiler(self):

        current_path = os.getcwd()
        os.chdir(self.base_path)
        result = subprocess.run(['java', '-cp',
                                 self.absolute_compiler_path, 'Main',
                                 f'{self.base_path}/{self.settings_path}',
                                 f'{self.base_path}/{self.knowledge_path}',
                                 f'{self.base_path}/lib/{self.platform}.platform'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

        os.chdir(current_path)
        return result.returncode

    def run_c_compiler(self):
        result = subprocess.run(['make', '-j4', '-s', '-C', f'{self.base_path}/{self.output_path}'],
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return result.returncode

    def evaluate(self, infinity=1e100, number_of_samples=1, only_weights_adapted=False):
        if not only_weights_adapted:
            return_code = self.run_exastencils_compiler()
            if not return_code == 0:
                return infinity, infinity
        return_code = self.run_c_compiler()
        if not return_code == 0:
            return infinity, infinity
        total_time = 0
        sum_of_convergence_factors = 0
        for i in range(number_of_samples):
            result = subprocess.run([f'{self.base_path}/{self.output_path}/exastencils'],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if not result.returncode == 0:
                return infinity, infinity
            output = result.stdout.decode('utf8')
            time_to_solution, convergence_factor = self.parse_output(output)
            if math.isinf(convergence_factor) or math.isnan(convergence_factor) or not convergence_factor < 1:
                return infinity, infinity
            total_time += time_to_solution
            sum_of_convergence_factors += convergence_factor
        return total_time / number_of_samples, sum_of_convergence_factors / number_of_samples

    @staticmethod
    def parse_output(output: str):
        lines = output.splitlines()
        convergence_factor = 1
        count = 0
        for line in lines:
            if 'convergence factor' in line:
                tmp = line.split('convergence factor is ')
                convergence_factor *= float(tmp[-1])
                count += 1
        convergence_factor = math.pow(convergence_factor, 1/count)
        tmp = lines[-1].split(' ')
        time_to_solution = float(tmp[-2])
        return time_to_solution, convergence_factor

    def generate_storage(self, min_level, max_level, finest_grid):
        storage = []
        grid = finest_grid
        for i in range(min_level, max_level):
            storage.append(CycleStorage(self.equations, self.fields, grid))
            grid = system.get_coarse_grid(grid, self.coarsening_factor)
        return storage

    @staticmethod
    def needs_storage(expression: base.Expression):
        return expression.shape[1] == 1

    # Warning: This function modifies the expression passed to it
    @staticmethod
    def assign_storage_to_subexpressions(expression: base.Expression, storages: [CycleStorage], i: int):
        if expression.storage is not None:
            return None

    def generate_multigrid(self, expression: base.Expression, storages, use_global_weights=False):
        # import decimal
        # if expression.program is not None:
        #     return expression.program
        program = ''
        if expression.storage is not None:
            expression.storage.valid = False

    @staticmethod
    def invalidate_storages(storages: [CycleStorage]):
        for storage in storages:
            for residual, rhs, solution, correction in zip(storage.residual, storage.rhs, storage.solution, storage.correction):
                residual.valid = False
                rhs.valid = False
                solution.valid = False
                correction.valid = False

