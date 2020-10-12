"""Utility function and classes.

Contains various utility functions and classes that are used by the
B^2 algorithm.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import math
import random
import cplex
import bisect
from cplex import SparsePair
from b2_sddp_settings import B2Settings

class CutData:
    """Data for a single linear cut.

    Parameters
    ----------
    indices : List[int]
        Indices of nonzero cut elements.

    elements : List[float]
        Values of nonzero cut elements.

    sense : {'E', 'L', 'G'}
        Direction of the inequality.

    rhs : float
        Right-hand side of the inequality.

    name : string
        Mnemonic name of the inequality.
    """
    def __init__(self, indices, elements, sense, rhs, name = 'cut'):
        assert(len(indices) == len(elements))
        assert(sense in {'E', 'L', 'G'})
        self.indices = indices
        self.elements = elements
        self.sense = sense
        self.rhs = rhs
        self.name = name

# -- end class CutData

class InverseCDF:
    """Inverse cumulative distribution function.

    Generate the empirical inverse cumulative distribution function
    given the distribution of a set of events.

    Parameters
    ----------
    prob : List[float]
        Set of probabilities of the events.
    """
    def __init__(self, prob):
        """Constructor.
        """
        self.k = len(prob)
        self.prob = prob
        self.cdf = [math.fsum(prob[:(i+1)]) for i in range(self.k)]

    def at(self, value):
        """Compute inverse CDF for a given value.

        Parameters
        ----------
        value : float
            The value for which we want to compute the inverse CDF.

        Returns
        -------
        int
            The point that has CDF equal to value.
        """
        assert(0 <= value <= 1)
        return bisect.bisect_right(self.cdf, value)

# -- end class InverseCDF

def col_matrix_vector_product(m, n, col_matrix, vector):
    """Product between a matrix (in column form) and a vector.

    Compute the standard matrix/vector multiplication, with a matrix
    given in sparse column format and a vector given in dense format.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.

    n : int
        Number of columns in the matrix.

    col_matrix : List[List[int], List[float]]
        Matrix in sparse column format (indices and elements).

    vector : List[float]
        Vector in dense format.

    Returns
    -------
    List[float]
        The result of the multiplication.
    """
    assert(len(col_matrix) == n)
    assert(len(vector) == n)
    product = [0] * m
    for j in range(n):
        indices, elements = col_matrix[j]
        for (i, val) in enumerate(elements):
            product[indices[i]] += val * vector[j]
    return product

# -- end function

def vector_col_matrix_product(m, n, vector, col_matrix):
    """Product between a vector and matrix (in column form).

    Compute the standard row vector/matrix multiplication, with a
    vector given in dense format and a matrix given in sparse column
    format.

    Parameters
    ----------
    m : int
        Number of rows in the matrix.

    n : int
        Number of columns in the matrix.

    vector : List[float]
        Row vector in dense format.

    col_matrix : List[List[int], List[float]]
        Matrix in sparse column format (indices and elements).

    Returns
    -------
    List[float]
        The result of the multiplication.
    """
    assert(len(col_matrix) == n)
    assert(len(vector) == m)
    product = [sum(col[1][i] * vector[col[0][i]] for i in range(len(col[0])))
               for col in col_matrix]
    return product

# -- end function

def row_matrix_to_col_matrix(m, n, row_matrix):
    """Compute sparse column representation of a sparse row matrix.

    Parameters
    ----------
    m : int
        Number of rows.

    n : int
        Number of columns.

    row_matrix : List[List[int], List[float]]
        Matrix in sparse row format (indices and elements).

    Returns
    -------
    List[List[int], List[float]]
        Matrix in sparse column format (indices and elements).
    """
    assert(len(row_matrix) == m)
    # Store new matrix here
    col_matrix = list()
    # Length of each row
    row_length = [len(ind) for (ind, elem) in row_matrix]
    # Pointer to first element in each row
    curr_pos = [0] * m 
    first_elem = [ind[0] for (ind, elem) in row_matrix]
    for j in range(n):
        indices = [i for i in range(m) if curr_pos[i] < row_length[i] 
                   and row_matrix[i][0][curr_pos[i]] == j]
        elements = [row_matrix[i][1][curr_pos[i]] for i in indices]
        for i in indices:
            curr_pos[i] += 1
        col_matrix.append([indices, elements])
    return col_matrix

# -- end function

def sparse_matrix_diff(matrix_a, matrix_b):
    """Compute the difference between two sparse matrices.

    Compute (matrix_a - matrix_b), given in sparse format, assuming
    that both matrices are nonempty, have exactly the same size, and
    the indices are sorted in increasing order. The result may be
    unpredictable if the assumptions are not true. The matrices can be
    in either row or column format, but both should be in the same
    format.

    Parameters
    ----------
    matrix_a : List[List[int], List[float]]
        First matrix in sparse format (indices and elements).

    matrix_b : List[List[int], List[float]]
        Second matrix in sparse format (indices and elements).

    Returns
    -------
    List[List[int], List[float]]
        The matrix (matrix_a - matrix_b), in sparse format.
    """
    assert(len(matrix_a) == len(matrix_b))
    # Store difference matrix here
    diff = list()
    for (i, row_a) in enumerate(matrix_a):
        ind_a, elem_a = row_a
        ind_b, elem_b = matrix_b[i]
        # Make a copy of the row of matrix_a
        indices = [val for val in ind_a]
        elements = [val for val in elem_a]
        for (k, j) in enumerate(ind_b):
            # Find if the k-th element of the row of matrix_b appears
            # in a position that is nonzero in the row of matrix_a. 
            pos = bisect.bisect_left(indices, j)
            if (pos >= len(indices) or indices[pos] != j):
                # The element was zero: insert new element
                indices.insert(pos, j)
                elements.insert(pos, -elem_b[k])
            else:
                # The element was already present: do the difference
                elements[pos] -= elem_b[k]
        diff.append([indices, elements])
    return diff
# -- end function


def purge_cuts(settings, m, p, q, cut_activity, master, slave):
    """Purge inactive cuts from LPs.

    Cycle through the cut activity and remove all inactive cuts from
    the master and the slave.

    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.

    m : int
        Number of rows in the first set of constraints
        A x + G y >= b + T y_{t-1}.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    cut_activity : List[int]
        Number of consecutive rounds that a cut has been inactive.

    master : cplex.Cplex
        Master LP.

    slave : cplex.Cplex
        Slave LP.

    Returns
    -------
    List[int]
        Cut activity for the cuts that have not been purged.
    """
    assert(isinstance(settings, B2Settings))
    del_rows_master = list()
    del_rows_slave = list()
    new_cut_activity = list()
    for (i, rounds_inactive) in enumerate(cut_activity):
        if (rounds_inactive >= settings.rounds_before_cut_purge):
            del_rows_master.append(i + m + p + q)
            del_rows_slave.append(i + m + p + q + 1)
        else:
            new_cut_activity.append(rounds_inactive)
    master.linear_constraints.delete(del_rows_master)
    slave.linear_constraints.delete(del_rows_slave)
    return new_cut_activity

# -- end function

def add_cuts(problem, cuts):
    """Add cuts to a Cplex problem.

    Parameters
    ----------
    problem : cplex.Cplex
        LP to which cuts should be added.

    cuts : List[CutData]
        Cuts to be added.
    """
    # Add new cuts to the master
    for cut in cuts:
        lp_cut = SparsePair(ind=cut.indices,
                            val=cut.elements)
        problem.linear_constraints.add(lin_expr=[lp_cut],
                                       senses=[cut.sense],
                                       rhs=[cut.rhs],
                                       names=[cut.name])
# -- end function

def is_problem_optimal(settings, problem):
    """Determine optimality status.

    Verify the solution status of a Cplex problem, and return True if
    optimal, False if not.

    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.

    problem : cplex.Cplex
        A Cplex problem.

    Returns
    -------
    bool
        True if optimal, False if not.
    """
    if (problem.solution.get_status() == problem.solution.status.optimal):
        return True
    else:
        if (settings.print_cplex_output):
            status = problem.solution.status[problem.solution.get_status()]
            print('Not optimal Cplex problem returned status' +
                  '{:s}'.format(status))
        return False

# -- end function

def print_LP_solution(settings, n1, n2, k, problem,
                      tag = None, is_slave = False,
                      output_stream = sys.stdout):
    """Print an LP solution to screen.

    Print the solution of the master or slave LP.

    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.

    n1 : int
        Number of x variables of the problem.

    n2 : int
        Number of y variables of the problem.

    k : int
        Number of scenarios.

    problem : cplex.Cplex
        Cplex problem.

    tag : string
        A string to be printed before the solution.

    is_slave : bool
        True if this is the slave problem (print more statistics)

    output_stream : file
        Where to print.
    """

    assert(is_problem_optimal(settings,problem))
    if (tag is not None):
        print(tag, file = output_stream)
    x = problem.solution.get_values(0, n1 - 1)
    y = problem.solution.get_values(n1, n1 + n2 - 1)
    z = problem.solution.get_values(n1 + n2, n1 + n2 + k - 1)
    print('Objective {:f}'.format(problem.solution.get_objective_value()),
          file = output_stream)
    print('x:', end = ' ', file = output_stream)
    print(x, file = output_stream)
    print('y:', end = ' ', file = output_stream)
    print(y, file = output_stream)
    print('z:', end = ' ', file = output_stream)
    print(z, file = output_stream)
    if (is_slave):
        print('r:', end = ' ', file = output_stream)
        print(problem.solution.get_values(n1 + n2 + k - 1), 
              file = output_stream)
        print('Dual:', end = ' ', file = output_stream)
        print(problem.solution.get_dual_values(), file = output_stream)

# -- end function

