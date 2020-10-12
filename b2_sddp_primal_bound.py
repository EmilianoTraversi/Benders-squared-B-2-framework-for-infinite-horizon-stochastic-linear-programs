"""Compute Primal Bound.

This module contains functions to compute a valid primal bound for a
stochastic infinite-horizon linear program.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import cplex
import numpy as np
import scipy.stats as ss
import b2_sddp_utility as util
import b2_sddp_forward as forward
from cplex import SparsePair
from b2_sddp_settings import B2Settings


def cost_primal_sol(x, y, x_obj, y_obj, tau, delta, first_stage = 0):
    """Compute the discounted cost of a primal solution.

    Parameters
    ----------
    x : List[float]
        Values of the x variables, one for each stage.

    y : List[float]
        Values of the y variables, one for each stage.

    x_obj : List[float]
        Cost coefficient for the x variables.

    y_obj : List[float]
         Cost coefficient for the y variables.

    tau : int
        Length of the time horizon.

    delta : float
        Discount factor, 0 <= delta < 1.

    first_stage : int
        The primal solution starts from this stage.

    Returns
    -------
    float
        The discounted cost of the solution.
    """
    assert(len(x) == tau - first_stage)
    assert(len(y) == tau - first_stage)
    assert(0 <= delta < 1)
    assert(first_stage < tau)
    x_cost = np.dot(x, x_obj)
    y_cost = np.dot(y, y_obj)
    return sum(delta**t * (x_cost[t - first_stage] + y_cost[t - first_stage]) 
               for t in range(first_stage, tau))

# -- end function


def primal_bound(settings, cost_path):
    """Compute an upper bound on the value of an optimal policy.

    Given the cost of all the sample paths simulated so far, compute
    an upper bound on the value of an optimal policy, using the
    confidence interval specified in the settings.

    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.
    
    cost_path : List[float]
        The costs of the sample paths.

    alpha: float
        0 <= delta < 1.

    Returns
    -------
    U_val : float
        total value of this part of the bound
    """
    assert(isinstance(settings, B2Settings))
    assert(cost_path)
    mu = np.mean(cost_path)
    # Unbiased estimator of variance
    var = np.var(cost_path, ddof = 1)
    # Compute size of confidence interval
    factor = ss.norm.ppf(1 - settings.alpha/2)
    
    return mu + factor * math.sqrt(var / len(cost_path))

# -- end function

def create_p_no_y_problem(settings, n1, k, m, p, q, x_obj,
                          A_rows, D_rows, b, d):
    """Create master problem without the y variables.

    Create the master problem for the B^2 algorithm in Cplex without y
    variables.  It is assumed that this is a minimization problem and
    the constraints are in >= form. The problem is written as:

    .. math ::

    \min \{c^T x : A x  \ge b , D x \ge d \}


    Parameters
    ----------
    settings : b2_sddp_settings.B2Settings
        Algorithmic settings.

    n1 : int
        Number of x variables of the problem.

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

    A_rows : List[List[int], List[float]]
        The coefficients of the rows of the A matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    D_rows : List[List[int], List[float]]
        The coefficients of the rows of the D matrix in D x >= d.
        These are in sparse form: indices and elements.

    b : List[float]
        The rhs of the first set of constraints in the master.

    d : List[float]
        The rhs of the second set of constraints in the master.

    Returns
    -------
    cplex.Cplex
        A cplex problem describing the master without y variables.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(x_obj) == n1)
    assert(len(A_rows) == m)
    assert(len(D_rows) == p)
    assert(len(b) == m)
    assert(len(d) == p)
    
    # Create empty problem
    master_no_y = cplex.Cplex()
    master_no_y.objective.set_sense(master_no_y.objective.sense.minimize)

    # Add original variables
    master_no_y.variables.add(obj = x_obj, lb = [-cplex.infinity] * n1, 
                              ub = [cplex.infinity] * n1, 
                              names=['x' + str(i) for i in range(n1)])
    
    # Add rows Ax + Gy >= b
    rows = [SparsePair(ind = A_rows[i][0],
                       val = A_rows[i][1])
            for i in range(m)]
    master_no_y.linear_constraints.add(lin_expr = rows, senses=['G'] * m, 
                                       rhs = b, names = ['R1_' + str(i)
                                                         for i in range(m)])

    # Add rows D x >= d
    rows = [SparsePair(ind = D_rows[i][0], val = D_rows[i][1])
            for i in range(p)]
    master_no_y.linear_constraints.add(lin_expr = rows, senses=['G'] * p,
                                       rhs = d, names = ['R2_' + str(i)
                                                         for i in range(p)])

    # Save problem for debugging
    if (settings.debug_save_lp):
        master_no_y.write('master_no_y.lp', 'lp')

    return (master_no_y)

# -- end function



def create_p_rebalance():
    """Create master problem without the y variables.

    do nothing

    Parameters
    ----------
    Returns
    -------
    cplex.Cplex
        A cplex problem describing the master without y variables.
    """
    # Create empty problem
    master_no_y = cplex.Cplex()

    return (master_no_y)

# -- end function

def cost_tail_no_y(settings, n1, n2, k, m, p, q, T_cols, carry, tau, delta,
                   scenarios, prob, cost_tail, no_y_problem):
    """Compute the cost of the tail from a given starting state.

    Compute an upper bound on the cost from the given state to
    infinity, using a policy in which all y (inventory) are fixed to
    zero.

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
        Number of rows of the problem.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    T_cols : List[List[int], List[float]]
        The coefficients of the columns of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    carry : List[float]
        Carried over inventory from previous time period, i.e. y_{t-1}.

    tau : int
        Current length of the time horizon, from which the tail starts.

    delta : float
        Discount factor, 0 <= delta < 1.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    cost_tail : float
        Cost of a solution to the no_y_problem with y_{t-1} = 0.

    no_y_problem : cplex.Cplex
        The LP describing the no_y master problem.

    Returns
    -------
    float
        An upper bound on the cost from a given state to infinity.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(carry) == n2)
    assert(0 <= delta < 1)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)

    # Cost for each scenario
    ub = list()

    # Carried over resources
    carry_over = util.col_matrix_vector_product(m, n2, T_cols, carry)
    
    # Numbering system for LPs written to file, for debugging
    lp_id = 0

    for j in range(k):
        # Get rhs from the corresponding scenario, adding carried over
        # resources
        rhs = ([scenarios[j][i] + carry_over[i] for i in range(m)] +
               scenarios[j][m:])
        no_y_problem.linear_constraints.set_rhs([(i, rhs[i])
                                                 for i in range(m + p + q)])
        #no_y_problem.solve()

        # Save problem for debugging
        if (settings.debug_save_lp):
            print('Writing ub_calc_tail_no_y_' + str(lp_id) + '.lp')
            no_y_problem.write('ub_calc_tail_no_y_' + str(lp_id) + '.lp', 'lp')
            lp_id += 1

        no_y_problem.solve()

        if (util.is_problem_optimal(settings, no_y_problem)):
            # Save optimum
            ub.append(np.dot(no_y_problem.objective.get_linear(0, n1 - 1),
                             no_y_problem.solution.get_values(0, n1 - 1)) *
                      prob[j])
        else:
            print('PRIMAL HEURISTIC : Cannot solve no_y problem; abort.')
            sys.exit()

    return (delta ** tau * sum(ub) + 
            delta ** (tau + 1) / (1 - delta) * cost_tail)

# -- end function

def create_p_heur_problem(settings, n1, n2, k, m, p, q, x_obj, y_obj,
                          A_rows, G_rows, T_rows, D_rows, W_rows, 
                          d, w, scenarios):
    """Create LP for primal heuristic that allows linear growth.

    Create a problem to determine a steady-state policy that can be
    implemented indefinitely, where the x part of the solution is
    fixed, and the y part is affine in the stage t. The problem that
    we solve is as follows:

    .. math ::

    \min \{c^T x + h^T (y + f): Ax + Gy \ge b_M + Ty_{t-1}, Ax + (G -
    T)y - Tf \ge b_M, Dx \ge d, Wy \ge w, Gf \ge 0, Wf \ge 0, (G -
    T)f \ge 0\}.


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
        b_M + T y_{t-1}. These are in sparse form: indices and elements.

    G_rows : List[List[int], List[float]]
        The coefficients of the rows of the G matrix in A x + G y >=
        b_M + T y_{t-1}. These are in sparse form: indices and elements.

    T_rows : List[List[int], List[float]]
        The coefficients of the rows of the T matrix in A x + G y >=
        b_M + T y_{t-1}. These are in sparse form: indices and elements.

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

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    Returns
    -------
    cplex.Cplex
        A cplex problem describing the LP.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(x_obj) == n1)
    assert(len(y_obj) == n2)
    assert(len(A_rows) == m)
    assert(len(G_rows) == m)
    assert(len(T_rows) == m)
    assert(len(D_rows) == p)
    assert(len(W_rows) == q)
    assert(len(d) == p)
    assert(len(w) == q)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)

    # Compute right-hand sides looping over scenarios
    b_M = [max(s[i] for s in scenarios) for i in range(m + p + q)]

    # Difference (G - T)
    GmT_rows = util.sparse_matrix_diff(G_rows, T_rows)

    # Create empty problem
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Add original variables
    prob.variables.add(obj = x_obj, lb = [-cplex.infinity] * n1,
                       ub = [cplex.infinity] * n1,
                       names=['x' + str(i) for i in range(n1)])

    # Add the state variables.
    prob.variables.add(obj = y_obj, lb = [-cplex.infinity] * n2,
                       ub = [cplex.infinity] * n2,
                       names=['y' + str(i) for i in range(n2)])

    # Add the steady-state policy variables.
    prob.variables.add(obj = y_obj, lb = [-cplex.infinity] * n2,
                       ub = [cplex.infinity] * n2,
                       names=['f' + str(i) for i in range(n2)])

    # Add rows Ax + Gy >= b_M + T y_{t-1}
    rows = [SparsePair(ind = A_rows[i][0] + [n1 + j for j in G_rows[i][0]],
                       val = A_rows[i][1] + G_rows[i][1])
            for i in range(m)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * m, 
                                rhs = b_M[:m],
                                names = ['R1_' + str(i) for i in range(m)])

    # Add rows D x >= d
    rows = [SparsePair(ind = D_rows[i][0], val = D_rows[i][1])
            for i in range(p)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * p, 
                                rhs = b_M[m:(m+p)],
                                names = ['R2_' + str(i) for i in range(p)])

    # Add rows W y >= w
    rows = [SparsePair(ind = [n1 + j for j in W_rows[i][0]],
                       val = W_rows[i][1]) for i in range(q)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * q, 
                                rhs = b_M[(m+p):(m+p+q)],
                                names = ['R3_' + str(i) for i in range(q)])

    # Add rows G f >= 0
    rows = [SparsePair(ind = [n1 + n2 + j for j in G_rows[i][0]],
                       val = G_rows[i][1]) for i in range(m)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * m, 
                                rhs = [0] * m,
                                names = ['R4_' + str(i) for i in range(m)])

    # Add rows W f >= 0
    rows = [SparsePair(ind = [n1 + n2 + j for j in W_rows[i][0]],
                       val = W_rows[i][1]) for i in range(q)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * q, 
                                rhs = [0] * q,
                                names = ['R5_' + str(i) for i in range(q)])

    # Add rows Ax + (G - T)y - Tf >= b_M
    rows = [SparsePair(ind = A_rows[i][0] + [n1 + j for j in GmT_rows[i][0]] + 
                       [n1 + n2 + j for j in T_rows[i][0]],
                       val = A_rows[i][1] + GmT_rows[i][1] + 
                       [-elem for elem in T_rows[i][1]])
            for i in range(m)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * m, 
                                rhs = b_M[:m],
                                names = ['R6_' + str(i) for i in range(m)])

    # Add rows (G - T)f >= 0
    rows = [SparsePair(ind = [n1 + n2 + j for j in GmT_rows[i][0]],
                       val = GmT_rows[i][1]) for i in range(m)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * m, 
                                rhs = [0] * m,
                                names = ['R7_' + str(i) for i in range(m)])

    # Save problem for debugging
    if (settings.debug_save_lp):
        prob.write('p_heur.lp', 'lp')

    return (prob)

# -- end function

def cost_tail_p_heur(settings, n1, n2, k, m, p, q, T_cols, carry, tau, 
                     delta, scenarios, prob, problem):
    """Compute the cost of the tail from a given starting state.

    Compute an upper bound on the cost from the given state to
    infinity, using a policy in which all y (inventory) are fixed to a
    possibly nonzero value, and there is a variable part for the
    inventory that grows linearly over time.

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
        Number of rows of the problem.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    T_cols : List[List[int], List[float]]
        The coefficients of the columns of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    carry : List[float]
        Carried over inventory from previous time period, i.e. y_{t-1}.

    tau : int
        Current length of the time horizon, from which the tail starts.

    delta : float
        Discount factor, 0 <= delta < 1.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    problem : cplex.Cplex
        The LP describing P_heur.

    Returns
    -------
    float
        An upper bound on the cost from a given state to infinity.
    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(carry) == n2)
    assert(0 <= delta < 1)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)

    # Compute right-hand sides looping over scenarios
    b_M = [max(s[i] for s in scenarios) for i in range(m + p + q)]

    # Carried over resources
    carry_over = util.col_matrix_vector_product(m, n2, T_cols, carry)
    rhs = [b_M[i] + carry_over[i] for i in range(m)] + b_M[m:]
    problem.linear_constraints.set_rhs([(i, rhs[i]) for i in range(m + p + q)])

    # Save problem for debugging
    if (settings.debug_save_lp):
        print('Writing ub_calc_tail_p_heur.lp')
        problem.write('ub_calc_tail_p_heur.lp', 'lp')

    problem.solve()

    if (util.is_problem_optimal(settings, problem)):
        # Save optimum
        fixed_cost = (np.dot(problem.objective.get_linear(0, n1 - 1),
                             problem.solution.get_values(0, n1 - 1)) +
                      np.dot(problem.objective.get_linear(n1, n1 + n2 - 1),
                             problem.solution.get_values(n1, n1 + n2 - 1)) - 
                      np.dot(problem.objective.get_linear(n1, n1 + n2 - 1),
                             problem.solution.get_values(n1 + n2, 
                                                         n1 + 2*n2 - 1)))
        var_cost = np.dot(problem.objective.get_linear(n1, n1 + n2 - 1),
                          problem.solution.get_values(n1 + n2, n1 + 2*n2 - 1))
    else:
        print('PRIMAL HEURISTIC : Cannot solve P_heur problem; abort.')
        sys.exit()
        
    return (delta ** tau / (1 - delta) * fixed_cost + 
            (tau * delta ** tau - (tau - 1) * delta ** (tau + 1)) / 
            (1 - delta) ** 2 * var_cost)

# -- end function

def create_p_fixed_y_problem(settings, n1, n2, k, m, p, q, x_obj, y_obj,
                             A_rows, G_rows, T_rows, D_rows, W_rows, d, w,
                             scenarios):
    """Create master problem with fixed y variables.

    Compute a good value of the y variables using a heuristic, and
    create the master problem for the B^2 algorithm in Cplex with y
    variables fixed to the value found. It is assumed that this is a
    minimization problem and the constraints are in >= form. The
    problem is written as (\bar{y} is fixed):

    .. math ::

    \min \{c^T x + \h^T bar{y}: A x  + (G - T) \bar{y} \ge b , D x \ge d \}

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

    d : List[float]
        The rhs of the second set of constraints in the master.

    w : List[float]
        The rhs of the third set of constraints in the master.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.
    
    Returns
    -------
    cplex.Cplex
        A cplex problem describing the master without y variables.

    """
    assert(isinstance(settings, B2Settings))
    assert(len(x_obj) == n1)
    assert(len(y_obj) == n2)
    assert(len(A_rows) == m)
    assert(len(G_rows) == m)
    assert(len(T_rows) == m)
    assert(len(D_rows) == p)
    assert(len(W_rows) == q)
    assert(len(d) == p)
    assert(len(w) == q)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)

    # Compute right-hand sides looping over scenarios
    b_M = [max(s[i] for s in scenarios) for i in range(m + p + q)]

    # Difference (G - T)
    GmT_rows = util.sparse_matrix_diff(G_rows, T_rows)
    
    # Create empty problem
    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.minimize)

    # Add original variables
    #prob.variables.add(obj = [0]*n1, lb = [-cplex.infinity] * n1,
    prob.variables.add(obj = x_obj, lb = [-cplex.infinity] * n1,
                       ub = [cplex.infinity] * n1,
                       names=['x' + str(i) for i in range(n1)])

    # Now add the state variables.
    prob.variables.add(obj = y_obj, lb = [-cplex.infinity] * n2,
                       ub = [cplex.infinity] * n2,
                       names=['y' + str(i) for i in range(n2)])

    # Add rows Ax + (G - T)y - T>= b_M
    rows = [SparsePair(ind = A_rows[i][0] + [n1 + j for j in GmT_rows[i][0]],
                       val = A_rows[i][1] + GmT_rows[i][1])
            for i in range(m)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * m,
                                rhs = b_M[:m],
                                names = ['R1_' + str(i) for i in range(m)])

    # Add rows D x >= d
    rows = [SparsePair(ind = D_rows[i][0], val = D_rows[i][1])
            for i in range(p)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * p,
                                rhs = b_M[m:(m+p)],
                                names = ['R2_' + str(i) for i in range(p)])

    # Add rows W y >= w
    rows = [SparsePair(ind = [n1 + j for j in W_rows[i][0]],
                       val = W_rows[i][1]) for i in range(q)]
    prob.linear_constraints.add(lin_expr = rows, senses=['G'] * q, 
                                rhs = b_M[(m+p):(m+p+q)],
                                names = ['R3_' + str(i) for i in range(q)])
    
    # Save problem for debugging
    if (settings.debug_save_lp):
        prob.write('fixed_y.lp', 'lp')

    if (not settings.print_cplex_output):
        prob.set_results_stream(None)

    # Obtain value of y
    prob.solve()
    if (util.is_problem_optimal(settings, prob)):
        # Save optimum
        fixed_y = prob.solution.get_values(n1, n1 + n2 - 1)
    else:
        print('PRIMAL HEURISTIC : Cannot obtain fixed_y; abort.')
        sys.exit()

    # Now modify the problem to fix y at the above value
    for i in range(n1):
        prob.objective.set_linear(i, x_obj[i])
    prob.linear_constraints.add(lin_expr = [SparsePair(ind = [n1 + i], 
                                                       val = [1])
                                            for i in range(n2)],
                                senses = ['E'] * n2, rhs = fixed_y,
                                names = ['R4_' + str(i) for i in range(n2)])
    return (prob)

# -- end function

def cost_tail_fixed_y(settings, n1, n2, k, m, p, q, T_cols, carry, tau, 
                      delta, scenarios, prob, cost_tail, fixed_y_problem):
    """Compute the cost of the tail from a given starting state.

    Compute an upper bound on the cost from the given state to
    infinity, using a policy in which all y (inventory) are fixed to
    a predetermined value (embedded in the fixed y problem).

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
        Number of rows of the problem.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    T_cols : List[List[int], List[float]]
        The coefficients of the columns of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    carry : List[float]
        Carried over inventory from previous time period, i.e. y_{t-1}.

    tau : int
        Current length of the time horizon, from which the tail starts.

    delta : float
        Discount factor, 0 <= delta < 1.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    cost_tail : float
        Cost of a solution to the fixed_y_problem with y_{t-1} equal
        to the pre-determined fixed value of y.

    fixed_y_problem : cplex.Cplex
        The LP describing the no_y master problem.

    Returns
    -------
    float
        An upper bound on the cost from a given state to infinity.

    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(carry) == n2)
    assert(0 <= delta < 1)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)

    # Cost for each scenario
    ub = list()

    # Carried over resources
    carry_over = util.col_matrix_vector_product(m, n2, T_cols, carry)
    
    # Numbering system for LPs written to file, for debugging
    lp_id = 0

    for j in range(k):
        # Get rhs from the corresponding scenario, adding carried over
        # resources
        rhs = ([scenarios[j][i] + carry_over[i] for i in range(m)] +
               scenarios[j][m:])
        fixed_y_problem.linear_constraints.set_rhs([(i, rhs[i])
                                                    for i in range(m + p + q)])
        fixed_y_problem.solve()

        # Save problem for debugging
        if (settings.debug_save_lp):
            print('Writing ub_calc_tail_fixed_y_' + str(lp_id) + '.lp')
            fixed_y_problem.write('ub_calc_tail_fixed_y_' + str(lp_id) + 
                                  '.lp', 'lp')
            lp_id += 1

        if (util.is_problem_optimal(settings, fixed_y_problem)):
            # Save optimum
            ub.append(np.dot(fixed_y_problem.objective.get_linear(0, n1 - 1),
                             fixed_y_problem.solution.get_values(0, n1 - 1)) *
                      prob[j])
        else:
            print('PRIMAL HEURISTIC : Cannot solve fixed_y problem; abort.')
            sys.exit()
    return (delta ** tau * sum(ub) + 
            delta ** (tau + 1) / (1 - delta) * cost_tail)

# -- end function

def cost_tail_rebalance(settings, n1, n2, k, m, p, q, x_obj, T_cols, carry, 
                        tau, delta, scenarios, prob, cost_tail):
    """Compute tail cost, tailored for rebalancing problem instances.

    Compute an upper bound on the cost from the given state to
    infinity, using a policy in which for each node of the rebalancing
    problem (location) we compute the worse delta of incoming and
    outgoing items (bycicles).

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
        Number of rows of the problem.

    p : int
        Number of rows in the second set of constraints D x >= d.

    q : int
        Number of rows in the third set of constraints W x >= w.

    x_obj : List[float]
        Cost coefficient for the x variables.

    T_cols : List[List[int], List[float]]
        The coefficients of the columns of the T matrix in A x + G y >=
        b + T y_{t-1}. These are in sparse form: indices and elements.

    carry : List[float]
        Carried over inventory from previous time period, i.e. y_{t-1}.

    tau : int
        Current length of the time horizon, from which the tail starts.

    delta : float
        Discount factor, 0 <= delta < 1.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    cost_tail : float
        Cost of a solution to the fixed_y_problem with y_{t-1} equal
        to the pre-determined fixed value of y.

    Returns
    -------
    float
        An upper bound on the cost from a given state to infinity.

    """
    assert(isinstance(settings, B2Settings))
    assert(len(T_cols) == n2)
    assert(len(carry) == n2)
    assert(0 <= delta < 1)
    assert(len(scenarios) == k)
    for i in range(k):
        assert(len(scenarios[i]) == m + p + q)
    assert(len(prob) == k)

    # Carried over resources
    carry_over = util.col_matrix_vector_product(m, n2, T_cols, carry)
    
    # Accumulate the upper bound here.
    bound_rebalancing = 0

    # Remember that m is the number of nodes/locations for this problem!
    
    # For every location, check the different between the largest
    # number of items entering the node, and the smallest number of
    # nodes leaving the node.
    for j in range(m):
        # Minimum number leaving the node
        min_out = min(sum(scenarios[i][m + n1 + j*2*m + h*2]
                          for h in range(m) if h != j) 
                      for i in range(k))
        # Maximum number entering the node
        max_in = max(sum(scenarios[i][m + n1 + h*2*m + j*2]
                         for h in range(m) if h != j)
                     for i in range(j))
        # Maximum cost over an arc
        max_cost_arc = max(x_obj[j*m + h] for h in range(m) if h != j)

        # Accumulate
        bound_rebalancing += (max_in - min_out) * max_cost_arc
    
    return (delta ** tau * bound_rebalancing + 
            delta ** (tau + 1) / (1 - delta) * cost_tail)

# -- end function
