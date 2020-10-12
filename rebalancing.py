"""Continuous rebalancing problems.

This module contains a class constructing random instances of the
continuous rebalancing problem with given parameters.
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import math

import random
import cplex
from cplex import SparsePair

class RandomRebalancing:
    """An instance of the rebalancing problem.

    This class constructs a random instance of the continuous
    rebalancing problem, using given parameters. The instances are
    constructed so as to be nontivial.

    Parameters
    ----------
    points : int
        Number of points.

    k : int
        Number of scenarios.

    delta : float
        Discount factor, 0 <= delta < 1.

    saturation : float
        Fraction of the total ammount of bicycles used in the base
        scenario, 0 <= saturation <= 1.

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
        The rhs vector for each scenario.

    prob : List[float]
        The probability of each scenario.

    z_upper : List[float]
        Upper bounds on the objective value of each scenario.

    var_lower : List[int]
        Vector of lower bounds for the variables.

    var_upper : List[int]
        Vector of upper bounds for the variables.

    negative_inventory : bool
        True if the inventory levels should be negative, False if not.

    """

    def __init__(self,points, k, delta, saturation):
        assert(0 <= delta < 1)
        assert(0 <= saturation <= 1)
        
        n_arcs = points*points
        
        self.m = points * 2
        self.n1 = n_arcs * 3
        self.n2 = points
        self.k = k
        self.p = n_arcs * 2 + 2 + n_arcs * 2
        self.q = points + 1
        self.delta = delta
        self.saturation = saturation

        base_const = 100
        
        # create base scenario
        tot_dem = 0
	base_scen = []
	
	for i in range(points):
            for j in range(points):
                if (i!=j):
                    new_dem = random.randint(0,100)
                    base_scen.append(new_dem)
                    tot_dem = tot_dem+new_dem
                else:
                    base_scen.append(0)

        
        
        
        # Define x obj and bounds.
        # The first list is the x part of the objective.
        # The second list is the \hat{x} part of the objective.
        # The last part is the \phi part of the objective.
        self.x_obj = ([(0 if i == j else 1) for i in range(points) 
                       for j in range(points)] +
                      [(0 if i == j else 100) for i in range(points) 
                       for j in range(points)] +
                      [0 for i in range(points) for j in range(points)])

        # Define y obj and bounds

        self.y_obj = [0] * points

        self.D_rows = []
        self.d = []
        
        for i in range(points):
            for j in range(points):
                ind_row_tep = []
                val_row_tep = []
                ind_row_tep.append(n_arcs + i*points+j)
                ind_row_tep.append(n_arcs * 2 + i*points+j)
                val_row_tep.append(1.0)
                val_row_tep.append(1.0)
                self.D_rows.append([ind_row_tep, val_row_tep])
                self.d.append(0)
	for i in range(self.n1):
		self.D_rows.append([[i], [1]])
		self.d.append(0)
	self.D_rows.append([[(n_arcs * 2+i) for i in range(n_arcs)], 
                            [1] * n_arcs])
	self.D_rows.append([[(n_arcs * 2+i) for i in range(n_arcs)], 
                            [-1] * n_arcs])
	self.d.append(tot_dem)
	self.d.append(-tot_dem)

        self.w = []
	self.W_rows = []
        for i in range(self.n2):
		self.W_rows.append([[i], [1]])
		self.w.append(0)
	self.W_rows.append([[i for i in range(points)], [1] * points])
	self.w.append(tot_dem)


        # define G and T
        self.G_rows = []
       	for i in range(points):
            self.G_rows.append([[i], [0]])
	for i in range(points):
            self.G_rows.append([[i], [-1]])
	
	
	self.T_rows = []
	for i in range(points):
            self.T_rows.append([[i], [-1]])
	for i in range(points):
            self.T_rows.append([[i], [0]])
	
	# define A 
        self.A_rows = []
        self.b = []
        for i in range(points):
            ind_row_tep = []
            val_row_tep = []
            for j in range(points):
                if (i !=j):
                    ind_row_tep.append(i*points+j)
                    val_row_tep.append(-1.0)
                    ind_row_tep.append(j*points+i)
                    val_row_tep.append(1.0)
            for j in range(points):
                ind_row_tep.append(2 * n_arcs + i*points+j)
                val_row_tep.append(-1.0)
            self.A_rows.append([ind_row_tep, val_row_tep])
            self.b.append(-tot_dem/points)

	for i in range(points):
            ind_row_tep = []
            val_row_tep = []
            for j in range(points):
                ind_row_tep.append(2 * n_arcs + j*points+i)
                val_row_tep.append(1.0)
            self.A_rows.append([ind_row_tep, val_row_tep])
            self.b.append(0)
	
	# create an array of demands with a given total demand

	self.scenarios = []
        self.prob = []
        self.z_lower = []
        for s in range(k):
            new_s = []
            list_scenario_coef = []
            # Construct b
            for i in range(2*points):
                new_s.append(0)
		# Construct d
            for i in range(n_arcs):
                scenario_coef = random.randint(
                    int(0.75 * base_scen[i] * saturation), 
                    int(1.25 * base_scen[i] * saturation))
                list_scenario_coef.append(scenario_coef)
                node1 = i%points
                node2 = int((i-node1)//points )
                if (node1 == node2):
                    new_s.append(0)			  
                else:
                    new_s.append(scenario_coef)
		
            for i in range(self.n1):
                new_s.append(0)
            new_s.append(tot_dem)
            new_s.append(-tot_dem)
            # Construct w
            for i in range(self.n2):
                new_s.append(0)
            new_s.append(tot_dem)

            self.scenarios.append(new_s)
            self.prob.append(1 / k)
            self.z_lower.append(0)
            
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

# -- end class RandomRebalancing

