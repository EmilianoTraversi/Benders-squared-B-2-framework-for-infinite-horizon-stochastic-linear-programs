"""B^2 algorithm for stochastic infinite-horizon dynamic programming.

This module contains an implementation of the B^2 algorithm to solve
stochastic infinite-horizon linear programs, in a classical dynamic
programming fashion. The algorithm uses a recursive Benders
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
import rebalancing as rb
import b2_sddp_utility as util
from cplex import SparsePair
from b2_sddp_settings import B2Settings
from b2_sddp_forward import forward_pass, single_forward_step
from b2_sddp_backward import backward_pass

def update_time_horizon(settings, tau):
    """Update length of the time horizon.

    Update the length of the time horizon in the B^2 algorithm. For
    now, we only employ the simple strategy of incrementing the length
    by one at each major iteration.

    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.

    tau : int
        Length of the time horizon.

    Returns
    -------
    int
        The new length of the time horizon.
    """
    assert(isinstance(settings, B2Settings))
    return tau + 1

# -- end function

def create_master_problem(settings, n1, n2, k, m, p, q, x_obj, y_obj,
                          A_rows, G_rows, D_rows, W_rows,
                          b, d, w, scenarios, prob, z_lower, delta):
    """Create master problem.

    Create the master problem for the B^2 algorithm in Cplex, given
    the initial data. It is assumed that this is a minimization
    problem and the constraints are in >= form. The problem as described by
    the input data is written as:

    .. math ::

    \min \{c^T x + h^T y : A x + G y \ge b + Ty_{t-1}, D x \ge d,
    W y \ge w\}


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
        The rhs vectors b, d, w for each scenario.

    prob : List[float]
        The probability of each scenario.

    z_lower : List[float]
        Lower bounds on the objective value of each scenario.

    delta : float
        Discount factor, 0 <= delta < 1.

    Returns
    -------
    cplex.Cplex
        A cplex problem describing the master.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(x_obj) == n1)
    assert(len(y_obj) == n2)
    assert(len(A_rows) == m)
    assert(len(G_rows) == m)
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


    # Create empty problem
    master = cplex.Cplex()
    master.objective.set_sense(master.objective.sense.minimize)

    # Add original variables
    master.variables.add(obj = x_obj, lb = [-cplex.infinity] * n1,
                         ub = [cplex.infinity] * n1,
                         names=['x' + str(i) for i in range(n1)])

    # Now add the state variables.
    master.variables.add(obj = y_obj, lb = [-cplex.infinity] * n2,
                         ub = [cplex.infinity] * n2,
                         names=['y' + str(i) for i in range(n2)])

    # Add rows Ax + Gy >= b
    rows = [SparsePair(ind = A_rows[i][0] + [n1 + j for j in G_rows[i][0]],
                       val = A_rows[i][1] + G_rows[i][1])
            for i in range(m)]
    master.linear_constraints.add(lin_expr = rows, senses=['G'] * m, rhs = b,
                                  names = ['R1_' + str(i) for i in range(m)])

    # Add rows D x >= d
    rows = [SparsePair(ind = D_rows[i][0], val = D_rows[i][1])
            for i in range(p)]
    master.linear_constraints.add(lin_expr = rows, senses=['G'] * p, rhs = d,
                                  names = ['R2_' + str(i) for i in range(p)])

    # Add rows W y >= w
    rows = [SparsePair(ind = [n1 + j for j in W_rows[i][0]],
                       val = W_rows[i][1]) for i in range(q)]
    master.linear_constraints.add(lin_expr = rows, senses=['G'] * q, rhs = w,
                                  names = ['R3_' + str(i) for i in range(q)])

    # Weight probabilities by delta
    delta_prob = [val * delta for val in prob]

    # Finally, add the scenario variables z that estimate the cost for
    # each of the scenarios
    master.variables.add(obj = delta_prob,
                         lb = z_lower, ub = [cplex.infinity] * k,
                         names=['z' + str(i) for i in range(k)])

    # Save problem for debugging
    if (settings.debug_save_lp):
        master.write('master_orig.lp', 'lp')
            
    return (master)

# -- end function

def create_slave_problem(settings, n1, n2, k, m, p, q, x_obj, y_obj,
                         A_rows, G_rows, D_rows, W_rows,
                         d, w, prob, z_lower, delta):
    """Create the slave problem.

    Create the slave problem for the B^2 algorithm in Cplex, given the
    initial data. It is assumed that this is a minimization problem
    and the constraints are in >= form. The problem as described by
    the input data is written as:

    .. math ::

    \min \{c^T x + h^T y : A x + G y \ge b + Ty_{t-1}, D x \ge d,
    W y \ge w\}

    The slave problem created here corresponds to the minimization of
    the absoltute violation of the constraints.

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
        A x + G y >= b + T y_{t-1.

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

    D_rows : List[List[int], List[float]]
        The coefficients of the rows of the D matrix in D x >= d.
        These are in sparse form: indices and elements.

    W_rows : List[List[int], List[float]]
        The coefficients of the rows of the W matrix in W y >= w.
        These are in sparse form: indices and elements.

    d : List[float]
        The rhs of the second set of constraints in the master.

    w : List[float]
        The rhs of the third set of constraints in the master.

    prob : List[float]
        The probability of each scenario.

    z_lower : List[float]
        Lower bounds on the objective value of each scenario.

    delta : float
        Discount factor, 0 <= delta < 1.

    Returns
    -------
    cplex.Cplex
        A cplex problem describing the master.

    """
    assert(isinstance(settings, B2Settings))
    assert(len(x_obj) == n1)
    assert(len(y_obj) == n2)
    assert(len(A_rows) == m)
    assert(len(G_rows) == m)
    assert(len(D_rows) == p)
    assert(len(W_rows) == q)
    assert(len(d) == p)
    assert(len(w) == q)
    assert(len(prob) == k)
    assert(len(z_lower) == k)
    assert(0 <= delta < 1)

    # Create empty problem
    slave = cplex.Cplex()
    slave.objective.set_sense(slave.objective.sense.minimize)

    # Add original variables
    slave.variables.add(lb = [-cplex.infinity] * n1,
                        ub = [cplex.infinity] * n1,
                        names=['x' + str(i) for i in range(n1)])

    # Now add the state variables.
    slave.variables.add(lb = [-cplex.infinity] * n2,
                        ub = [cplex.infinity] * n2,
                        names=['y' + str(i) for i in range(n2)])

    # Add rows Ax + Gy >= b
    rows = [SparsePair(ind = A_rows[i][0] + [n1 + j for j in G_rows[i][0]],
                       val = A_rows[i][1] + G_rows[i][1])
            for i in range(m)]
    slave.linear_constraints.add(lin_expr = rows, senses=['G'] * m,
                                 rhs = [0] * m,
                                 names = ['R1_' + str(i) for i in range(m)])

    # Add rows D x >= d
    rows = [SparsePair(ind = D_rows[i][0], val = D_rows[i][1])
            for i in range(p)]
    slave.linear_constraints.add(lin_expr = rows, senses=['G'] * p, rhs = d,
                                  names = ['R2_' + str(i) for i in range(p)])

    # Add rows W y >= w
    rows = [SparsePair(ind = [n1 + j for j in W_rows[i][0]],
                       val = W_rows[i][1]) for i in range(q)]
    slave.linear_constraints.add(lin_expr = rows, senses=['G'] * q, rhs = w,
                                  names = ['R3_' + str(i) for i in range(q)])

    # Weight probabilities by delta
    delta_prob = [val * delta for val in prob]

    # Add the scenario variables z that estimate the cost for each of
    # the scenarios
    slave.variables.add(lb = z_lower, ub = [cplex.infinity] * k,
                        names = ['z' + str(i) for i in range(k)])

    # Add the optimality cut
    oc_row = SparsePair(ind = [i for i in range(n1 + n2 + k)],
                        val = [-v for v in x_obj + y_obj + delta_prob])
    slave.linear_constraints.add(lin_expr = [oc_row], senses = ['G'],
                                 rhs = [0], names = ['OC'])

    # Add the constraint relaxation variable that corresponds to the
    # normalization constraint of the dual, to ensure that we truncate
    # the infeasibility rays of the dual cone.
    col = SparsePair(ind = [i for i in range(m + p + q + 1)],
                     val = [-1.0] * (m + p + q + 1))
    slave.variables.add(obj = [-1.0], lb = [-cplex.infinity],
                        ub = [cplex.infinity], names = ['r'], columns = [col])

    # Save problem for debugging
    if (settings.debug_save_lp):
        slave.write('slave_orig.lp', 'lp')

    return (slave)

# -- end function


def b2_sddp(settings, n1, n2, k, m, p, q, x_obj, y_obj, A_rows, G_rows,
            T_rows, D_rows, W_rows, b, d, w, scenarios, prob, z_lower, 
            delta, output_stream = sys.stdout):
    """Apply the B^2 algorithm for dynamic programming.

    Solve an infinite horizon stochastic dynamic program using the B^2
    algorithm, given the initial data. It is assumed that this is a
    minimization problem and the constraints are in >= form. The
    problem is written as:

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
        The rhs vector b, d, w for each scenario.

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
    master = create_master_problem(settings, n1, n2, k, m, p, q, x_obj, y_obj,
                                   A_rows, G_rows, D_rows, W_rows,
                                   b, d, w, scenarios, prob, z_lower, delta)
    slave = create_slave_problem(settings, n1, n2, k, m, p, q, x_obj, y_obj, 
                                 A_rows, G_rows, D_rows, W_rows, 
                                 d, w, prob, z_lower, delta)
    if (settings.primal_bound_method == 'no_y'):
        # Create master problem without the y variables
        pb_prob = pb.create_p_no_y_problem(settings, n1, k, m, p, q, 
                                           x_obj, A_rows, D_rows, b, d)
    elif (settings.primal_bound_method == 'fixed_y'):
        # Create master problem with fixed y variables
        pb_prob = pb.create_p_fixed_y_problem(settings, n1, n2, k, m, p, q, 
                                              x_obj, y_obj, A_rows, G_rows, 
                                              T_rows, D_rows, W_rows, d, w, 
                                              scenarios)
    elif (settings.primal_bound_method == 'p_heur'):
        # Create the p_heur problem
        pb_prob = pb.create_p_heur_problem(settings, n1, n2, k, m, p, q, 
                                           x_obj, y_obj, A_rows, G_rows, 
                                           T_rows, D_rows, W_rows, d, w, 
                                           scenarios)
    elif (settings.primal_bound_method == 'p_rebalance'):
        # Create an empty p_heur problem
        pb_prob = pb.create_p_rebalance()
    else:
        raise ValueError('Primal bound method ' +
                         '{:s}'.format(settings.primal_bound_method) +
                         ' not implemented')
	
    if (not settings.print_cplex_output):
        master.set_results_stream(None)
        slave.set_results_stream(None)
        pb_prob.set_results_stream(None)
    else:
        master.set_results_stream(output_stream)
        slave.set_results_stream(output_stream)
        pb_prob.set_results_stream(output_stream)

    # Store solutions for all forward passes
    fwpass_sol = list()
    # Store cost of primal solutions, i.e. sample paths
    cost_sol = list()
    # The best primal bound available so far
    best_ub = cplex.infinity
    # Is the stopping criterion satisfied?
    stop = False
    # Length of the time horizon
    tau = 0
    # Cut activity levels
    cut_activity = list()

    
    # Comput single-stage cost for the infinite tail, if necessary
    if (settings.primal_bound_method == 'no_y'):
        ub_tail = pb.cost_tail_no_y(settings, n1, n2, k, m, p, q,
                                    T_cols, [0]*n2, 0, delta,
                                    scenarios, prob, 0, pb_prob)
    elif (settings.primal_bound_method == 'fixed_y'):
        ub_tail = pb.cost_tail_fixed_y(settings, n1, n2, k, m, p, q,
                                       T_cols, [0]*n2, 0, delta,
                                       scenarios, prob, 0, pb_prob)
    elif (settings.primal_bound_method == 'p_rebalance'):
        ub_tail = pb.cost_tail_rebalance(settings, n1, n2, k, m,
                                         p, q, x_obj, T_cols, [0]*n2,
                                         0, delta, scenarios, 
                                         prob,0)
      
    # Start the B^2 algorithm!
    while (not stop):
        # Increment length of the time horizon if necessary
        tau = update_time_horizon(settings, tau)
        
	print('*** Major iteration with tau {:d}'.format(tau),
              file = output_stream)
        output_stream.flush()
        for num_pass in range(settings.num_passes):
            x, y, z = forward_pass(settings, n1, n2, k, m, p, q, tau, T_cols,
                                   b, d, w, scenarios, prob, inverse_cdf, 
                                   master, num_pass, cut_activity)
            # Store all LP solutions
            fwpass_sol.append((x, y, z))
            # Clean up LPs
            cut_activity = util.purge_cuts(settings, m, p, q, cut_activity,
                                           master, slave)
            # Computation of upper bounds
            cost_sol.append(pb.cost_primal_sol(x, y, x_obj, y_obj, 
                                               tau, delta))
            # Backward pass: collect Benders cuts
            cuts = backward_pass(settings, n1, n2, k, m, p, q, tau, T_cols, 
                                 scenarios, prob, x, y, z, slave, num_pass)
            util.add_cuts(master, cuts)
        # -- end time horizon tau

        # Extend existing sample paths to the desired length
        for (i, (sol_x, sol_y, sol_z)) in enumerate(fwpass_sol):
            while (len(sol_x) < tau):
                # Move forward in time starting from the last solution
                x, y, z = single_forward_step(settings, n1, n2, k, m, p, q, 
                                              T_cols, sol_y[-1], scenarios, 
                                              prob, inverse_cdf, master,
                                              cut_activity)
                sol_x.append(x)
                sol_y.append(y)
                sol_z.append(z)
                
                # Update upper bound
                cost_sol[i] += pb.cost_primal_sol([x], [y], x_obj, y_obj, 
                                                  tau, delta, tau - 1)
                # Backward pass: collect Benders cuts
                cuts = backward_pass(settings, n1, n2, k, m, p, q, 1,
                                     T_cols, scenarios, prob, [x], [y], [z],
                                     slave, 0)
                util.add_cuts(master, cuts)

        # Primal bound: compute cost for the entire path until infinity
        cost_path = list()
        
        for (i, (sol_x, sol_y, sol_z)) in enumerate(fwpass_sol):
            if (settings.primal_bound_method == 'no_y'):
                cost_path.append(pb.cost_tail_no_y(settings, n1, n2, k, m, p, 
                                                   q, T_cols, sol_y[-1], tau,
                                                   delta, scenarios, prob, 
                                                   ub_tail, pb_prob) +
                                 cost_sol[i])
            elif (settings.primal_bound_method == 'fixed_y'):
                cost_path.append(pb.cost_tail_fixed_y(settings, n1, n2, k, m,
                                                      p, q, T_cols, sol_y[-1],
                                                      tau, delta, scenarios, 
                                                      prob, ub_tail, pb_prob) +
                                 cost_sol[i])
            elif (settings.primal_bound_method == 'p_heur'):
                cost_path.append(pb.cost_tail_p_heur(settings, n1, n2, k, m,
                                                     p, q, T_cols, sol_y[-1],
                                                     tau, delta, scenarios, 
                                                     prob, pb_prob) +
                                 cost_sol[i])
            elif (settings.primal_bound_method == 'p_rebalance'):
                cost_path.append(pb.cost_tail_rebalance(settings, n1, n2, k, m,
                                                        p, q, x_obj, T_cols, 
                                                        sol_y[-1], tau, delta,
                                                        scenarios, prob,
                                                        ub_tail) +
                                 cost_sol[i])
            
	ub = pb.primal_bound(settings, cost_path)
	
        # Get current dual bound and compute optimality gap
        master.linear_constraints.set_rhs([(i, b[i]) for i in range(m)])
        master.linear_constraints.set_rhs([(m + i, d[i]) for i in range(p)])
        master.linear_constraints.set_rhs([(m + p + i, w[i]) 
                                           for i in range(q)])
        master.solve()
        
        lb = master.solution.get_objective_value()

        if (ub > 0):
            gap =  (ub - lb) / (ub + 1.0e-10)
        else:
            gap =  (ub - lb) / (-ub - 1.0e-10)
        # Report progress status
        #print('Primal bound: {:f}  '.format(ub) +
             #'Dual bound: {:f}  '.format(lb) +
             #'gap {:.2f} %'.format(100*gap))
        
        print('Primal bound: {:f}  '.format(ub) +
             'Dual bound: {:f}  '.format(lb) +
             'gap {:.2f} %'.format(100*gap),
             file = output_stream)
        output_stream.flush()
        cpu_time = time.clock() - start_time
        
        #Stop when the optimality gap is small enough
        if (tau > 1):
	  if (tau >= settings.max_tau or 
	      cpu_time >= settings.max_cpu_time or
	      gap <= settings.gap_relative or
	      (ub - lb) <= settings.gap_absolute):
	      stop = True

    # -- end main loop

    # Print last solution and exit
    util.print_LP_solution(settings, n1, n2, k, master,
                           'Final solution of the master:', 
                           output_stream = output_stream)

    total_time = time.clock() - start_time
    print(file = output_stream)
    print('Summary:', end = ' ', file = output_stream)
    print('ub {:.3f} lb {:.3f} '.format(ub, lb) + 
          'gap {:.2f} % tau {:d} '.format(100 * gap, tau) + 
          'time {:.4f}'.format(total_time),
          file = output_stream)
    output_stream.flush()

    return (master, ub, lb, gap, tau, total_time)

# -- end function

if (__name__ == "__main__"):
    # Test the algorithm on a small problem
    if (sys.version_info[0] >= 3):
        print('Error: Python 3.* is currently not supported.')
        print('Please use Python 2.7')
        sys.exit()

    # Settings for the problem
    settings = B2Settings(add_cuts_immediately = True,
                          aggregate_cuts = False,
                          rounds_before_cut_purge = 10000000,
                          primal_bound_method = 'fixed_y')

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
    points= 10
    k = 2
    delta = 0.95
    saturation = 0.9
     
    res = b2_sddp(settings, n1, n2, k, m, p, q, x_obj, y_obj, A_rows,
                  G_rows, T_rows, D_rows, W_rows, b, d, w, scenarios,
                  prob, z_lower, delta)
