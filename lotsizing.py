"""Lotsizing problems.

This module contains a class constructing random instances of the
lotsizing problem with backlog, with given parameters.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math
import random
import cplex
from cplex import SparsePair

class RandomLotsizing:
    """An instance of the lotsizing problem.

    This class constructs a random instance of the lotsizing problem
    with backlog, using given parameters. The instances are
    constructed so as to be nontivial.

    Parameters
    ----------
    m : int
        Number of rows of the problem, i.e. resources.

    n : int
        Number of variables of the problem.

    k : int
        Number of scenarios.

    delta : float
        Discount factor, 0 <= delta < 1.

    dens : float
        Density of the w terms, 0 <= dens <= 1.

    Attributes
    ----------
    x_obj : List[float]
        Profit for the original variables.

    obj_h : List[float]
        Holding cost for each row, i.e. resource.

    rows : List[any]
        The rows of the problem. It is assumed that these are in any
        of the formats supported by Cplex to add rows with the
        "linear_constraints.add" function.

    b_0 : List[float]
        The rhs of the master.

    scenarios : List[List[float]]
        The rhs vector (b, d, w) for each scenario.

    prob : List[float]
        The probability of each scenario.

    z_upper : List[float]
        Upper bounds on the objective value of each scenario.

    var_lower : List[int]
        Vector of lower bounds for the variables.

    var_upper : List[int]
        Vector of upper bounds for the variables.

    """

    def __init__(self, m, n, k, delta, dens):
        assert(0 <= delta < 1)
        assert(0 <= dens <= 1)
        self.m = m
        self.n1 = n
        self.n2 = m
        self.k = k
        self.p = n
        self.q = m
        self.delta = delta
        self.dens = dens

        base_const = 100

        # define x obj and bounds
        self.x_obj = []
        self.D_rows = []
        self.d = []
        for i in range(self.n1):
            self.x_obj.append(random.randint(1, base_const))
            self.D_rows.append([[i], [1]])
            self.d.append(0)

        # define y obj and bounds
        self.y_obj = []
        self.W_rows = []
        self.w = []
        for i in range(self.n2):
            self.y_obj.append((random.randint(int(0.75 * base_const),
                                              int(1.25 * base_const))) /
                              base_const / (dens * m))
            self.W_rows.append([[i], [1]])
            self.w.append(0)

        # define G and T
        self.G_rows = []
        self.T_rows = []
        for i in range(self.m):
            self.G_rows.append([[i], [1]])
        for i in range(self.m):
            self.T_rows.append([[i], [1]])

        # average weight of a row
        avg_weight = []

        # define rows
        self.A_rows = []
        for r in range(m):
            avg_weight.append(0.0)
            ind_row_tep = []
            val_row_tep = []
            for c in range(n):
                r_n = random.random()
                if (r_n < dens):
                    val_coef = random.randint(max(0, self.x_obj[c] -
                                                  int(base_const / 10)),
                                              self.x_obj[c] + 
                                              base_const / 10)
                    ind_row_tep.append(c)
                    val_row_tep.append(val_coef)
                    avg_weight[r] += val_coef
            if (len(ind_row_tep) > 0):
                self.A_rows.append([ind_row_tep, val_row_tep])

        # I am lazy: I want to avoid the case when a row is empty
        # (possible for low values of density and n)
        assert(m == len(self.A_rows))

        # define b_0 and scenarios
        self.b = []
        for r in range(m):
            center = (avg_weight[r]) / 2
            self.b.append(random.randint(int(0.75 * center), 
                                         int(1.25 * center)))

        self.scenarios = []
        self.prob = []
        self.z_lower = []
        for s in range(k):
            new_s = []
            for r in range(m):
                center = (avg_weight[r]) / 2
                new_s.append(random.randint(int(0.75 * center),
                                            int(1.25 * center)))
            self.scenarios.append(new_s + self.d + self.w)
            self.prob.append(1 / k)
            self.z_lower.append(0.0)


    def print_all(self):
        print ('*****')
        print (self.m)
        print (self.n1)
        print (self.n2)
        print (self.k)
        print (self.p)
        print (self.q)
        print (self.delta)
        print (self.x_obj)
        print (self.y_obj)
        print (self.A_rows)
        print (self.G_rows)
        print (self.T_rows)
        print (self.D_rows)
        print (self.W_rows)
        print (self.b)
        print (self.d)
        print (self.w)
        print (self.scenarios)
        print (self.prob)
        print (self.z_lower)

# -- end class RandomLotsizing

