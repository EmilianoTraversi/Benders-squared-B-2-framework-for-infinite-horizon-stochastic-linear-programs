"""Functions for the backward pass of the algorithm.

This module contains routines to perform the backward pass of the B^2
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


def generate_aggregate_cut(settings, n1, n2, k, prob, cuts, cut_scenario,
                           name = 'aggr_cut'):
    """Generate an aggregate cut.

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

    prob : List[float]
        The probability of each scenario.

    cuts : List[b2_sddp_utility.CutData]
        The cuts to aggregate.

    cut_scenario : List[int]
        The index of the scenario that each cut was generated from.

    name : string
        Name of the cut.

    Returns
    -------
    b2_sddp_utility.CutData
        An aggregated cut.

    Raises
    ------
    ValueError
        If some cuts have the wrong sense.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(prob) == k)
    assert(cuts)
    assert(len(cuts) == len(cut_scenario))
    cut_ind = [i for i in range(n1, n1 + n2 + k)]
    cut_elem = [0 for i in range(n1, n1 + n2 + k)]
    sense = cuts[0].sense
    for (i, cut) in enumerate(cuts):
        for (pos, cut_index) in enumerate(cut.indices):
            cut_elem[cut_index - n1] += prob[cut_scenario[i]]*cut.elements[pos]
        if (cut.sense != sense):
            raise ValueError('Cut cannot be aggregated: wrong sense!')
    cut_rhs = math.fsum(prob[cut_scenario[i]]*cuts[i].rhs 
                        for i in range(len(cuts)))
    return util.CutData(cut_ind, cut_elem, sense, cut_rhs, name)

# -- end function

def add_master_cut_to_slave(settings, n1, n2, k, cut, slave):
    """Add a cut valid for the master to the slave problem.

    The function takes a cut that is valid for the master problem,
    performs the necessary modification to make it valid for the
    slave, and adds it. In particular, the modification requires
    adding the coefficient for the rho ('r') variable that represents
    the maximum constraint violation.

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

    cut : b2_sddp_utility.CutData
        The cut generated for the master.

    slave : cplex.Cplex
        The slave problem.
    """
    assert(isinstance(settings, B2Settings))
    assert(slave.variables.get_num() == (n1 + n2 + k + 1))
    slave_cut = SparsePair(ind = (cut.indices + [n1 + n2 + k]),
                           val = cut.elements + [-1])
    slave.linear_constraints.add(lin_expr = [slave_cut], senses = [cut.sense],
                                 rhs = [cut.rhs], names = [cut.name])

# -- end function

def backward_pass(settings, n1, n2, k, m, p, q, tau, T_cols, scenarios,
                  prob, x, y, z, slave, current_pass):
    """Perform the backward pass of the B^2 SDDP algorithm.

    Generate Benders cuts moving backward in time using the given
    sequence of LP solutions. Cuts can be aggregated depending on the
    settings. They are automatically added to the slave problem.

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

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    x : List[List[float]]
        The values of the x component of the solution in the forward pass.

    y : List[List[float]]
        The values of the y component of the solution in the forward pass.

    z : List[List[float]]
        The values of the z component of the solution in the forward pass.

    slave : cplex.Cplex
        Cut generating problem (i.e. slave).

    current_pass : int
        Identifier of the current pass.

    Returns
    -------
    (List[b2_sddp_utility.CutData])
        A list of generated cuts, that should be added to the master.

    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)
    assert(len(x) == tau)
    assert(len(y) == tau)
    assert(len(z) == tau)
    assert(current_pass < settings.num_passes)

    # Numbering system for LPs written to file, for debugging
    lp_id = 0

    # Store cuts for the master here
    pool = list()
    # Backward pass: collect Benders cuts
    for t in reversed(range(1, tau + 1)):
        # Pool of cuts from the current pass
        local_pool = list()
        # Scenario a cut was generated from
        local_pool_scenario = list()
        for j in range(k):
            # Construct rhs of the slave, remembering that we have the
            # optimality cut together with the resource constraints
            carry_over = util.col_matrix_vector_product(m, n2, T_cols, y[t-1])
            rhs = ([scenarios[j][i] + carry_over[i] for i in range(m)] +
                   scenarios[j][m:])
            slave.linear_constraints.set_rhs([(i, rhs[i])
                                              for i in range(m + p + q)])
            slave.linear_constraints.set_rhs(m + p + q, -z[t-1][j])
            m_with_cuts = slave.linear_constraints.get_num()
            rhs_with_cuts = slave.linear_constraints.get_rhs()
            # Save problem for debugging
            if (settings.debug_save_lp):
                print('Writing problem slave_' + str(tau) + '_' + 
                      str(current_pass) + '_' + str(lp_id) + '.lp')
                slave.write('slave_' + str(tau) + '_' + str(current_pass) + 
                            '_' + str(lp_id) + '.lp', 'lp')
                lp_id += 1
            slave.solve()
            if (util.is_problem_optimal(settings, slave) and
                slave.solution.get_objective_value() >= settings.eps_opt):
                # The problem is infeasible; collect a cut in >= form.
                dual = slave.solution.get_dual_values()
                # The cut rhs depends on the scenario's rhs, and on
                # the part that comes from the Benders cuts.
                cut_rhs = sum(dual[i] * scenarios[j][i]
                              for i in range(m + p + q))
                cut_rhs += sum(dual[i] * rhs_with_cuts[i]
                               for i in range(m + p + q + 1, m_with_cuts))
                cut_coeff = util.vector_col_matrix_product(m, n2, dual[:m], 
                                                           T_cols)
                # We usually normalize by the dual variable that
                # multiplies the z variable for the current scenario,
                # but if such variable is zero, we simply do not
                # normalize the cut.
                if (abs(dual[m + p + q]) > settings.eps_zero):
                    cut_rhs /= dual[m + p + q]
                    # Now generate the cut coefficients
                    cut_ind = [i for i in range(n1, n1 + n2)] + [n1 + n2 + j]
                    cut_elem = ([-cut_coeff[i]/dual[m + p + q] 
                                 for i in range(n2)] + [1])
                else:
                    cut_ind = [i for i in range(n1, n1 + n2)]
                    cut_elem = [-cut_coeff[i] for i in range(n2)] 
                cut = util.CutData(cut_ind, cut_elem, 'G', cut_rhs,
                                   'c_' + str(t) + '_' + str(j))

                local_pool.append(cut)
                local_pool_scenario.append(j)

                # Check violation
                if (settings.debug_check_violation):
                    primal_sol = x[t - 1] + y[t - 1] + z[t - 1]
                    lhs = sum(primal_sol[cut_ind[i]] * cut_elem[i]
                              for i in range(len(cut_ind)))
                    if (lhs - cut_rhs >= 0):
                        print('Cut not violated')
                        print('Cut {:f} >= {:f}'.format(lhs, cut_rhs))
                        print(local_pool[-1])
                        sys.exit()
                if (settings.print_lp_solution_backward and
                    util.is_problem_optimal(settings, slave)):
                    util.print_LP_solution(settings, n1, n2, k, slave,
                                           'BACKWARD At stage ' +
                                           '{:d}, scenario {:d}'.format(t, j),
                                           True)

                # Add cut to the slave
                if (settings.add_cuts_immediately and 
                    not settings.aggregate_cuts):
                    add_master_cut_to_slave(settings, n1, n2, k, cut, slave)

        if (settings.aggregate_cuts):
            # Define the aggregate cut as the surrogate of all the cuts
            if (local_pool):
                aggr_cut = generate_aggregate_cut(settings, n1, n2, k,
                                                  prob, local_pool, 
                                                  local_pool_scenario,
                                                  'c_aggr_' + str(t))
                pool.append(aggr_cut)
                add_master_cut_to_slave(settings, n1, n2, k, aggr_cut, slave)
        else:
            # If the cut was not already added, we should add it now
            if (not settings.add_cuts_immediately):
                for cut in local_pool:
                    add_master_cut_to_slave(settings, n1, n2, k, cut, slave)
            pool.extend(local_pool)
    return pool

# -- end function
