"""Settings for the B^2 algorithm.

This module contains a class describing all the relevant settings for
the B^2 algorithm.
"""

import sys

class B2Settings:
    """Settings for the B^2 algorithm.

    This class describes algorithmic, debug, display and numerical
    settings for B^2.

    Parameters
    ----------
    eps_zero : float
        Zero tolerance. Default 1.0e-16.

    eps_opt : float
        Optimality tolerance for the Benders slave problem. Default
        1.0e-6.

    eps_activity : float
        Dual tolerance for cut activity: any cut with a dual variable
        smaller than this tolerance (in absolute value) is considered
        inactive. Default 1.0e-5.

    gap_relative : float
        Maximum allowed relative gap between lower and upper bound for
        early termination of the algoritm, computed as
        (upper-lower)/(upper + 1.0e-10). Must be between 0 and
        1. Default 1.0e-2.

    gap_absolute : float
        Maximum allowed absolute gap between lower and upper bound for
        early termination of the algoritm. Default 1.0.

    alpha : float
        Level of risk, i.e. the solution will be optimal up to the
        specified gap with a probability of 1 - alpha. Default 0.05.

    num_passes : int
        Number of forward/backward passes before the length of the
        time horizon is increased. Default 10.

    rounds_before_cut_purge : int
        Number of rounds a cut has to be inactive before it is deleted
        from the LPs. Default 2.

    aggregate_cuts : bool
        If True, cuts generated from multiple scenarios are aggregated
        into a single row before being added to the master and the
        slave. Default False.

    add_cuts_immediately : bool
        If True, disaggregate cuts are added immediately as soon as
        they are created, instead of waiting for the end of the
        scenario loop. Default True.

    primal_bound_method : string
        Method to compute primal bound. Possible choices are 'no_y',
        'fixed_y', 'p_heur', 'p_rebalance'. 'no_y' fixes the inventory
        values to zero in the tail, 'fixed_y' computes a fixed value
        for the inventory values (may be different than zero),
        'p_heur' tries to compute a feasible policy with a y component
        that grows at most linearly over time, 'p_rebalance' is a
        tailored heuristic for the rebalancing problem that looks at
        worst-case deficits. Default 'fixed_y'.

    debug_save_lp : bool
        If True, LP problems are saved on file and more information is
        printed. Default False.

    debug_check_violation : bool
        If True, the violation of each cut is checked after
        generation, and the algorithm exits if a non-violated cut is
        generated. Default False.

    print_cplex_output : bool
        If True, prints Cplex output to screen. Default False.

    print_lp_solution_forward : bool
        If True, prints each LP solution of the forward pass to screen.
        Default False.

    print_lp_solution_backward : bool
        If True, prints each LP solution of the backward pass to screen.
        Default False.

    max_tau : int
        Maximum length of the time horizon. Default 1000.

    max_cpu_time : float
        Maximum allowed CPU time in seconds. Default 36000.

    """
    def __init__(self,
                 eps_zero = 1.0e-16,
                 eps_opt = 1.0e-6,
                 eps_activity = 1.0e-5,
                 gap_relative = 1.0e-2,
                 gap_absolute = 1.0,
                 alpha = 0.05,
                 num_passes = 10,
                 rounds_before_cut_purge = 2,
                 aggregate_cuts = False,
                 add_cuts_immediately = True,
                 primal_bound_method = 'fixed_y',
                 debug_save_lp = False,
                 debug_check_violation = False,
                 print_cplex_output = False,
                 print_lp_solution_forward = False,
                 print_lp_solution_backward = False,
                 max_tau = 1000,
                 max_cpu_time = 36000):
        """Constructor.
        """
        assert(eps_zero >= 0)
        assert(eps_opt >= 0)
        assert(eps_activity >= 0)
        assert(0 <= gap_relative <= 1)
        assert(gap_absolute >= 0)
        assert(max_tau >= 0)
        assert(max_cpu_time >= 0)
        self.eps_zero = eps_zero
        self.eps_opt = eps_opt
        self.eps_activity = eps_activity
        self.gap_relative = gap_relative
        self.gap_absolute = gap_absolute
        self.alpha = alpha
        self.num_passes = num_passes
        self.rounds_before_cut_purge = rounds_before_cut_purge
        self.aggregate_cuts = aggregate_cuts
        self.add_cuts_immediately = add_cuts_immediately
        self.primal_bound_method = primal_bound_method
        self.debug_save_lp = debug_save_lp
        self.debug_check_violation = debug_check_violation
        self.print_cplex_output = print_cplex_output
        self.print_lp_solution_forward = print_lp_solution_forward
        self.print_lp_solution_backward = print_lp_solution_backward
        self.max_tau = max_tau
        self.max_cpu_time = max_cpu_time        
    # -- end function

    @classmethod
    def from_dictionary(cls, args):
        """Construct settings from dictionary containing parameter values.
    
        Construct an instance of B2Settings by looking up the value of
        the parameters from a given dictionary. The dictionary must
        contain parameter values in the form args['name'] =
        value. Anything else present in the dictionary will be
        ignored.

        Parameters
        ----------

        args : Dict[string]
            A dictionary containing the values of the parameters in a
            format args['name'] = value. 

        Returns
        -------
        B2Settings
            An instance of the object of the class.

        """
        copy = args.copy()
        valid_params = vars(cls())
        for param in copy.keys():
            if param not in valid_params:
                del copy[param]

        return cls(**copy)
    # -- end function

# -- end class
