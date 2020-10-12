"""Pereira-Pinto algorithm for stochastic infinite-horizon DP.

This module contains an implementation of the Pereira-Pinto SDDSP
algorithm to solve stochastic infinite-horizon linear programs, in a
classical dynamic programming fashion. The algorithm uses a Benders
decomposition approach.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import math
import random
import time
import cplex
import b2_sddp_primal_bound as pb
import lotsizing as ls
import b2_sddp_utility as util
from cplex import SparsePair
from b2_sddp_settings import B2Settings
from b2_sddp import create_master_problem, create_slave_problem
from b2_sddp_forward import forward_pass, single_forward_step
from b2_sddp_backward import backward_pass

def get_time_horizon_length(delta, ub, max_error):
    """Determine length of the time horizon.

    Determine how many stages of the problem are necessary to reduce
    the error below the given value, based on the cost of a feasible
    primal solution.

    Parameters
    ----------
    delta : float
        Discount factor.

    ub : float
        Expected cost of a primal solution for a single stage.

    max_error : float
        Maximum absolute error allowed.
    
    Returns
    -------
    int
        The length of the time horizon.
    """
    assert(0 <= delta < 1)
    assert(max_error > 0)
    tau = math.log(max_error * (1 - delta) / abs(ub), delta) - 1
    return int(math.ceil(tau))
# -- end function

def pp_sddp(settings, n1, n2, k, m, p, q, x_obj, y_obj, A_rows, G_rows,
            T_rows, D_rows, W_rows, b, d, w, scenarios, prob, z_lower, 
            delta, output_stream = sys.stdout):

    """Apply the Pereira-Pinto SDDP algorithm for dynamic programming.

    Solve an infinite horizon stochastic dynamic program using the
    Pereira-Pinto SDDP algorithm, given the initial data. It is
    assumed that this is a minimization problem and the constraints
    are in >= form. The problem is written as:

    .. math ::

    \min \{\sum_{t=0}^\infty c^T x_t + h^T y_t : \forall t A x_t + G
    y_t \ge b + Ty_{t-1}, D x_t \ge d, W y_t \ge w\}


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

    x_obj : List[float]
        Cost coefficient for the x variables.

    y_obj : List[float]
        Cost coefficient for the y (state) variables.

    A_rows : List[List[int], List[float]]
        The coefficients of the rows of the A matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    G_rows : List[List[int], List[float]]
        The coefficients of the rows of the G matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    T_rows : List[List[int], List[float]]
        The coefficients of the rows of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    D_rows : List[List[int], List[float]]
        The coefficients of the rows of the D matrix in D x >= d.
        These are in sparse form: indices and elements.

    W_rows : List[List[int], List[float]]
        The coefficients of the rows of the W matrix in W y >= w.
        These are in sparse form: indices and elements.

    b : List[float]
        The rhs of the first set of constraints in the master.

    d : List[float]
        The rhs of the second set of constraints in the master.

    w : List[float]
        The rhs of the third set of constraints in the master.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    z_lower : List[float]
        Lower bounds on the objective value of each scenario.

    delta : float
        Discount factor, 0 <= delta < 1.

    output_stream : file
        Output stream. Must have a 'write' and 'flush' method.

    Returns
    -------
    (cplex.Cplex, float, float, float, int, float)
        The master problem after convergence, the best upper bound,
        the best lower bound, the final gap (as a fraction, i.e. not
        in percentage point), the final length of the time horizon
        tau, the total CPU time in seconds.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(x_obj) == n1)
    assert(len(y_obj) == n2)
    assert(len(A_rows) == m)
    assert(len(G_rows) == m)
    assert(len(T_rows) == m)
    assert(len(D_rows) == p)
    assert(len(W_rows) == q)
    assert(len(b) == m)
    assert(len(d) == p)
    assert(len(w) == q)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)
    assert(len(z_lower) == k)
    assert(0 <= delta < 1)

    # Start counting time
    start_time = time.clock()

    # Initialize inverse CDF of the probability distribution of the
    # scenarios
    inverse_cdf = util.InverseCDF(prob)
    # Compute column representation of T
    T_cols = util.row_matrix_to_col_matrix(m, n2, T_rows)

    # Create the master and slave
    master_orig = create_master_problem(settings, n1, n2, k, m, p, q,
                                        x_obj, y_obj, A_rows, G_rows,
                                        D_rows, W_rows, b, d, w, scenarios,
                                        prob, z_lower, delta)
    slave_orig = create_slave_problem(settings, n1, n2, k, m, p, q,
                                      x_obj, y_obj, A_rows, G_rows, D_rows,
                                      W_rows, d, w, prob, z_lower, delta)
	
    # Store solutions for all forward passes
    fwpass_sol = list()
    # Store cost of primal solutions, i.e. sample paths
    cost_sol = list()
    # The best primal bound available so far
    best_ub = cplex.infinity
    # Is the stopping criterion satisfied?
    stop = False

    # Create master problem with fixed y variables
    pb_prob = pb.create_p_fixed_y_problem(settings, n1, n2, k, m, p, q, 
                                          x_obj, y_obj, A_rows, G_rows, 
                                          T_rows, D_rows, W_rows, d, w, 
                                          scenarios)

    if (not settings.print_cplex_output):
        pb_prob.set_results_stream(None)
    else:
        pb_prob.set_results_stream(output_stream)

    # Compute cost of a single stage, using a fixed y policy
    ub_tail = pb.cost_tail_fixed_y(settings, n1, n2, k, m, p, q, T_cols, [0]*n2,
                                   0, delta, scenarios, prob, 0, pb_prob)
    # Fraction of the absolute gap used to upper bound the error in
    # the infinite tail
    gap_frac = 0.9
    # Length of the time horizon
    tau = get_time_horizon_length(delta, ub_tail, 
                                  settings.gap_absolute * gap_frac)
    print('Pereira-Pinto time horizon of length {:d}'.format(tau),
          file = output_stream)
    output_stream.flush()

    # Make copies of master and slave
    master = [cplex.Cplex(master_orig) for t in range(tau)]
    slave = [cplex.Cplex(slave_orig) for t in range(tau)]
    # Cut activity levels
    cut_activity = [list() for t in range(tau)]

    for t in range(tau):
        if (not settings.print_cplex_output):
            master[t].set_results_stream(None)
            slave[t].set_results_stream(None)
        else:
            master[t].set_results_stream(output_stream)
            slave[t].set_results_stream(output_stream)

    # Start the PP SDDP algorithm!
    while (not stop):
        print('*** Major iteration after {:d} passes'.format(len(fwpass_sol)),
              file = output_stream)
        output_stream.flush()
        for num_pass in range(settings.num_passes):
            # Collect forward solution
            x, y, z = single_forward_step(settings, n1, n2, k, m, p, q, 
                                          T_cols, [0]*n2, scenarios, 
                                          prob, inverse_cdf, master[0],
                                          cut_activity[0])
            sol_x, sol_y, sol_z = [x], [y], [z]
            for t in range(1, tau):
                # One step at a time
                x, y, z = single_forward_step(settings, n1, n2, k, m, p, q, 
                                              T_cols, sol_y[-1], scenarios, 
                                              prob, inverse_cdf, master[t],
                                              cut_activity[t])
                sol_x.append(x)
                sol_y.append(y)
                sol_z.append(z)
            # Store all LP solutions
            fwpass_sol.append((sol_x, sol_y, sol_z))
            # Computation of upper bounds
            cost_sol.append(pb.cost_primal_sol(sol_x, sol_y, x_obj, y_obj, 
                                               tau, delta))
            # Backward pass: collect Benders cuts
            for t in reversed(range(tau)):
                cuts = backward_pass(settings, n1, n2, k, m, p, q, 1,
                                     T_cols, scenarios, prob, 
                                     [sol_x[t]], [sol_y[t]], [sol_z[t]],
                                     slave[t], num_pass)
                util.add_cuts(master[t], cuts)
        # -- end passes

        # Clean up LPs
        for t in range(1, tau):
            cut_activity[t] = util.purge_cuts(settings, m, p, q, 
                                              cut_activity[t],
                                              master[t], slave[t])

        # Primal bound: compute cost based on samples
        ub = pb.primal_bound(settings, cost_sol)

        # Get current dual bound and compute optimality gap
        master[0].linear_constraints.set_rhs([(i, b[i]) for i in range(m)])
        master[0].solve()
        lb = master[0].solution.get_objective_value()
        gap =  ((ub + settings.gap_absolute * gap_frac - lb) / 
                (ub + 1.0e-10))

        # Report progress status
        print('Primal bound: {:f}  '.format(ub) +
              'Dual bound: {:f}  '.format(lb) +
              'gap {:.2f} %'.format(100*gap),
              file = output_stream)
        output_stream.flush()
        cpu_time = time.clock() - start_time

        # Stop when the optimality gap is small enough
        if (tau >= settings.max_tau or 
            cpu_time >= settings.max_cpu_time or
            gap <= settings.gap_relative or 
            (ub - lb) <= settings.gap_absolute * (1 - gap_frac)):
            stop = True

    # -- end main loop

    # Print last solution and exit
    util.print_LP_solution(settings, n1, n2, k, master[0],
                           'Final solution of the master:',
                           output_stream = output_stream)

    total_time = time.clock() - start_time
    print(file = output_stream)
    print('Summary:', end = ' ', file = output_stream)
    print('ub {:.3f} lb {:.3f} '.format(ub + settings.gap_absolute * gap_frac,
                                        lb) + 
          'gap {:.2f} % tau {:d} '.format(100 * gap, tau) + 
          'time {:.4f}'.format(total_time),
          file = output_stream)
    output_stream.flush()

    return (master[0], ub + settings.gap_absolute * gap_frac, lb, gap,
            tau, total_time)

# -- end function

if (__name__ == "__main__"):
    # Test the algorithm on a small problem
    if (sys.version_info[0] >= 3):
        print('Error: Python 3.* is currently not supported.')
        print('Please use Python 2.7')
        sys.exit()

    # Settings for the problem
    settings = B2Settings(add_cuts_immediately = False,
                          aggregate_cuts = False,
                          rounds_before_cut_purge = 20,
                          gap_absolute = 100)

    # Define problem - input
    m = 2
    n1 = 2
    n2 = 2
    k = 2
    p = 2
    q = 2
    delta = 0.99
    x_obj = [3, 2]
    y_obj = [1.75, 1.75]
    A_rows = [[[0], [1]], [[0, 1], [1, 2]]]
    G_rows = [[[0], [1]], [[1], [1]]]
    T_rows = [[[0], [1]], [[1], [1]]]
    D_rows = [[[0], [1]], [[1], [1]]]
    W_rows = [[[0], [1]], [[1], [1]]]
    b = [1, 1]
    d = [0, 0]
    w = [0, 0]
    scenarios = [[2, 1] + d + w, [1, 4] + d + w]
    prob = [1 / 2, 1 / 2]
    z_lower = [0, 0]

    # Call main function here
    master = pp_sddp(settings, n1, n2, k, m, p, q, x_obj, y_obj, A_rows,
                     G_rows, T_rows, D_rows, W_rows, b, d, w,
                     scenarios, prob, z_lower, delta)
