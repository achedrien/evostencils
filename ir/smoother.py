from evostencils.ir import base, system
from evostencils.stencils import multiple
from evostencils.code_generation.hypre import Smoothers as hypre_smoothers
from evostencils.code_generation.composyx import Smoothers as hyteg_smoothers

def generate_decoupled_jacobi(operator: system.Operator):
    return system.Diagonal(operator)


def generate_collective_jacobi(operator: system.Operator):
    return system.ElementwiseDiagonal(operator)


def generate_collective_block_jacobi(operator: system.Operator, block_sizes: [tuple]):
    entries = []
    for i, row in enumerate(operator.entries):
        entries.append([])
        for j, entry in enumerate(row):
            stencil = entry.generate_stencil()
            block_diagonal = multiple.block_diagonal(stencil, block_sizes[i])
            new_entry = base.Operator(f'{operator.name}_{i}{j}_block_diag', entry.grid, base.ConstantStencilGenerator(block_diagonal))
            entries[-1].append(new_entry)
    return system.Operator(f'{operator.name}_block_diag', entries)


def generate_decoupled_block_jacobi(operator: system.Operator, block_sizes: [tuple]):
    entries = []
    for i, row in enumerate(operator.entries):
        entries.append([])
        for j, entry in enumerate(row):
            if i == j:
                stencil = entry.generate_stencil()
                block_diagonal = multiple.block_diagonal(stencil, block_sizes)
                new_entry = base.Operator(f'{operator.name}_{i}{j}_block_diag', entry.grid,
                                          base.ConstantStencilGenerator(block_diagonal))
            else:
                new_entry = base.ZeroOperator(entry.grid)
            entries[-1].append(new_entry)
    return system.Operator(f'{operator.name}_block_diag', entries)


def generate_jacobi_picard(operator: system.Operator):
    return system.ElementwiseDiagonal(operator)


def generate_jacobi_newton(operator: system.Operator, n_newton_steps: int):
    return base.Addition(system.ElementwiseDiagonal(operator), system.Jacobian(operator, n_newton_steps))

# hypre smoothers
def generate_jacobi(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hypre_smoothers.Jacobi
    return op
def generate_GS_forward(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hypre_smoothers.GS_Forward
    return op
def generate_GS_backward(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hypre_smoothers.GS_Backward
    return op


# hyteg smoothers
def generate_sor(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.SOR
    return op
def generate_weightedjacobi(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.WeightedJacobi
    return op
def generate_symmetricsor(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.SymmtericSOR
    return op
def generate_gaussseidel(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.GaussSeidel
    return op
def generate_symmetricgaussseidel(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.SymmetricGaussSeidel
    return op
def generate_chebyshev(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.Chebyshev
    return op
def generate_uzawa(operator: system.Operator):
    op = system.ElementwiseDiagonal(operator)
    op.smoother_type = hyteg_smoothers.Uzawa
    return op