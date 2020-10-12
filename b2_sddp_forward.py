"""Functions for the forward pass of the algorithm.

This module contains routines to perform the forward pass of the B^2
algorithm.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import math
import random
import cplex
import b2_sddp_utility as util
from cplex import SparsePair
from b2_sddp_settings import B2Settings


def generate_sample_path(settings, tau, inverse_cdf):
    """Generate a sample path.

    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.

    tau : int
        Length of the time horizon.

    inverse_cdf : b2_sddp_utility.InverseCDF
        Inverse CDF of the probability distribution of the scenarios.

    Returns
    -------
    List[int]
        A sample path, indicating the scenario index at each stage.
    """
    assert(isinstance(inverse_cdf, util.InverseCDF))
    return [inverse_cdf.at(random.uniform(0,1)) for t in range(tau)]

# -- end function


def forward_pass(settings, n1, n2, k, m, p, q, tau, T_cols, b, d, w, scenarios,
                 prob, inverse_cdf, master, current_pass, cut_activity):
    """Perform the forward pass of the B^2 SDDP algorithm.

    Generate sample path and compute the corresponding LP solutions
    moving forward in time. LP solutions could be aggregated over
    several samples, but at the moment we do not do it.

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

    m : int
        Number of rows in the first set of constraints
        A x + G y >= b + T y_{t-1}.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    tau : int
        The current length of the time horizon.

    T_cols : List[List[int], List[float]]
        The coefficients of the columns of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    b : List[float]
        The rhs of the master.

    d : List[float]
        The rhs of the second set of constraints in the master.

    w : List[float]
        The rhs of the third set of constraints in the master.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    inverse_cdf : b2_sddp_utility.InverseCDF
        Inverse CDF of the probability distribution of the scenarios.

    master : cplex.Cplex
        Master problem.

    current_pass : int
        Identifier of the current pass.

    cut_activity : List[int]
        Number of consecutive rounds that a cut has been
        inactive. This will be updated at the end of the forward pass.

    Returns
    -------
    (List[List[float]], List[List[float]], List[List[float]])
        A triple containing the x, y, z components of the LP
        solutions. Each list will have length equal to tau, and
        contain the corresponding values of the solutions.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(b) == m)
    assert(len(d) == p)
    assert(len(w) == q)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)
    assert(isinstance(inverse_cdf, util.InverseCDF))
    assert(current_pass < settings.num_passes)

    # Numbering system for LPs written to file, for debugging
    lp_id = 0

    # Adjust length of cut activity vector
    num_rows = master.linear_constraints.get_num()
    if (len(cut_activity) < num_rows - (m + p + q)):
        cut_activity.extend([0] * (num_rows - (m + p + q) - len(cut_activity)))

    # Consider sample paths up to this stage
    sample_path = generate_sample_path(settings, tau, inverse_cdf)
    # Store solutions in these lists
    x = list()
    y = list()
    z = list()
	
    for t in range(tau):
        # Find current scenario index
        j = sample_path[t]
        if (t == 0):
            # The rhs value is the initial one, b
            master.linear_constraints.set_rhs([(i, b[i])
                                               for i in range(m)])
            master.linear_constraints.set_rhs([(m + i, d[i])
                                               for i in range(p)])
            master.linear_constraints.set_rhs([(m + p + i, w[i])
                                               for i in range(q)])
        else:
            # Get rhs from the corresponding scenario, adding carried
            # over resources
            carry_over = util.col_matrix_vector_product(m, n2, T_cols, y[-1])
            rhs = ([scenarios[j][i] + carry_over[i] for i in range(m)] +
                   scenarios[j][m:])
            master.linear_constraints.set_rhs([(i, rhs[i])
                                               for i in range(m + p + q)])
        # Save problem for debugging
        if (settings.debug_save_lp):
            print('Writing problem master_' + str(tau) + '_' +
                  str(current_pass) + '_' + str(lp_id) + '.lp')
            master.write('master_' + str(tau) + '_' + str(current_pass) +
                         '_' + str(lp_id) + '.lp', 'lp')
            lp_id += 1

        master.solve()
        if (util.is_problem_optimal(settings, master)):
            # Save all solutions
            x.append(master.solution.get_values(0, n1 - 1))
            y.append(master.solution.get_values(n1, n1 + n2 - 1))
            z.append(master.solution.get_values(n1 + n2, n1 + n2 + k - 1))
            # Check cut activity
            dual = master.solution.get_dual_values(m + p + q, num_rows - 1)
            for (i, val) in enumerate(dual):
                if (abs(val) <= settings.eps_activity):
                    cut_activity[i] += 1
                else:
                    cut_activity[i] = 0

            if (settings.print_lp_solution_forward):
                util.print_LP_solution(settings, n1, n2, k, master,
                                       'FORWARD At stage ' +
                                       '{:d}, '.format(t) +
                                       'scenario {:d}'.format(j))
        else:
            print('Cannot  solve master problem; abort.')
            sys.exit()
	
    return (x, y, z)

# -- end function

def single_forward_step(settings, n1, n2, k, m, p, q, T_cols, carry,
                        scenarios, prob, inverse_cdf, master, cut_activity):
    """Perform a single forward step of the B^2 SDDP algorithm.

    Sample a scenario and compute the corresponding LP solution.

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

    m : int
        Number of rows in the first set of constraints
        A x + G y >= b + T y_{t-1}.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    T_cols : List[List[int], List[float]]
        The coefficients of the columns of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    carry : List[float]
        Carried over inventory from previous time period, i.e. y_{t-1}.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    inverse_cdf : b2_sddp_utility.InverseCDF
        Inverse CDF of the probability distribution of the scenarios.

    master : cplex.Cplex
        Master problem.

    cut_activity : List[int]
        Number of consecutive rounds that a cut has been
        inactive. This will be updated at the end of the forward pass.

    Returns
    -------
    (List[float], List[float], List[float])
        A triple containing the x, y, z components of the LP
        solutions. 
    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(carry) == n2)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)
    assert(isinstance(inverse_cdf, util.InverseCDF))

    # Adjust length of cut activity vector
    num_rows = master.linear_constraints.get_num()
    if (len(cut_activity) < num_rows - (m + p + q)):
        cut_activity.extend([0] * (num_rows - (m + p + q) - len(cut_activity)))

    # Sample current scenario index
    j = inverse_cdf.at(random.uniform(0,1))
    # Get rhs from the corresponding scenario, adding
    # carried over resources
    carry_over = util.col_matrix_vector_product(m, n2, T_cols, carry)
    rhs = ([scenarios[j][i] + carry_over[i] for i in range(m)] +
           scenarios[j][m:])
             
    master.linear_constraints.set_rhs([(i, rhs[i]) for i in range(m + p + q)])
    
    if (settings.debug_save_lp):
        print('Writing problem master_single_fp.lp')
        master.write('master_single_fp.lp', 'lp')

    master.solve()
    if (util.is_problem_optimal(settings, master)):
        # Save all solutions
        x = master.solution.get_values(0, n1 - 1)
        y = master.solution.get_values(n1, n1 + n2 - 1)
        z = master.solution.get_values(n1 + n2, n1 + n2 + k - 1)
        # Check cut activity
        dual = master.solution.get_dual_values(m + p + q, num_rows - 1)
        for (i, val) in enumerate(dual):
            if (abs(val) <= settings.eps_activity):
                cut_activity[i] += 1
            else:
                cut_activity[i] = 0

        # Report progress status
        if (settings.print_lp_solution_forward):
            util.print_LP_solution(settings, n1, n2, k, master,
                                   'FORWARD At stage ' +
                                   '{:d}, '.format(t) +
                                   'scenario {:d}'.format(j))
    else:
        print('Cannot  solve master problem; abort.')
        sys.exit()
	
    return (x, y, z)

# -- end function
