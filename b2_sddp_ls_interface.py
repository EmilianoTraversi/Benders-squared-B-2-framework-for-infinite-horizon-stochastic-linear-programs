"""Solve random instances of the lotsizing problem.

This module provides a command line interface to create and solve
random instances of the lotsizing problem with backlog, using the B^2
algorithm or the Pereira-Pinto SDDP algorithm.

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import argparse
import ast
import random
import numpy as np
import lotsizing as ls
from b2_sddp import b2_sddp
from pp_sddp import pp_sddp
from b2_sddp_settings import B2Settings

def register_options(parser):
    """Add options to the command line parser.

    Register all the options for the optimization algorithm.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser.
    """
    # Get default values from here
    default = B2Settings()
    attrs = vars(default)
    docstring = default.__doc__
    param_docstring = docstring[docstring.find('Parameters'):
                                docstring.find('Attributes')].split(' : ')
    param_name = [val.split(' ')[-1].strip() for val in param_docstring[:-1]]
    param_type = [val.split('\n')[0].strip() for val in param_docstring[1:]]
    param_help = [' '.join(line.strip() for line in val.split('\n')[1:-2])
                  for val in param_docstring[1:]]
    # We extract the default from the docstring in case it is
    # necessary, but we use the actual default from the object above.
    param_default = [val.split(' ')[-1].rstrip('.').strip('\'') 
                     for val in param_help]
    for i in range(len(param_name)):
        if (param_type[i] == 'float'):
            type_fun = float
        elif (param_type[i] == 'int'):
            type_fun = int
        elif (param_type[i] == 'bool'):
            type_fun = ast.literal_eval
        else:
            type_fun = str
        parser.add_argument('--' + param_name[i], action = 'store',
                            dest = param_name[i],
                            type = type_fun,
                            help = param_help[i],
                            default = getattr(default, param_name[i]))
    parser.add_argument('--algorithm', '-a', action = 'store',
                        dest = 'algorithm', type = str, default = 'B2', 
                        help = 'Algorithm; choice of B2 and PP. Default B2.')
    parser.add_argument('--delta', '-df', action = 'store',
                        dest = 'delta', type = float, default = 0.95,
                        help = 'Discount factor (between 0 and 1). ' +
                        'Default 0.95.')
    parser.add_argument('--density', '-de', action = 'store',
                        dest = 'density', type = float, default = 0.75,
                        help = 'Density of the constraint matrix. ' +
                        'Default 0.75.')
    parser.add_argument('--log', '-l', action = 'store', dest = 'log', 
                        default = None, help = 'Name of file for ' +
                        'detailed output log. Default None.')
    parser.add_argument('--num_columns', '-nc', action = 'store',
                        dest = 'num_columns', type = int, default = 10,
                        help = 'Number of columns of the problem. ' +
                        'Default 10.')
    parser.add_argument('--num_instances', '-ni', action = 'store',
                        dest = 'num_instances', type = int, default = 10,
                        help = 'Number of instances to be generated. ' +
                        'Default 10.')
    parser.add_argument('--num_rows', '-nr', action = 'store',
                        dest = 'num_rows', type = int, default = 10,
                        help = 'Number of rows of the problem. ' +
                        'Default 10.')
    parser.add_argument('--num_scenarios', '-ns', action = 'store',
                        dest = 'num_scenarios', type = int, default = 10,
                        help = 'Number of scenarios of the problem. ' +
                        'Default 10.')
    parser.add_argument('--rand_seed', '-r', action = 'store',
                        dest = 'rand_seed', type = int, 
                        default = 51410971231, 
                        help = 'Random seed. Default 51410971231.')

# -- end function

def run_random_instances(algorithm, num_instances, num_columns,
                         num_rows, num_scenarios, density, delta,
                         detailed_output_stream):
    """Solve random lotsizing instances and collect statistics.

    Parameters
    ----------

    algorithm : string
        Either 'B2' for B^2 or 'PP' for Pereira-Pinto.

    num_instances : int
        Number of random instances to generate.

    num_columns : int
        Number of columns of the problem.

    num_rows : int
        Number of rows (i.e. resources) of the problem.

    num_scenarios : int
        Number of scenarios of the problem.

    density : float
        Density of the constraint matrix, between 0 and 1.

    delta : float
        Discount factor, between 0 and 1.

    detailed_output_stream : file
        A stream for the detailed output of all runs.
    """
    assert(algorithm == 'B2' or algorithm == 'PP')
    assert(num_instances >= 1)
    assert(num_columns >= 1)
    assert(num_rows >= 1)
    assert(num_scenarios >= 1)
    assert(0 <= density <= 1)
    assert(0 <= delta < 1)

    print('** n : {:>6d}'.format(num_columns),
          '  m : {:>6d}'.format(num_rows),
          '  k : {:>6d}'.format(num_scenarios),
          '  dens : {:>5.4f}'.format(density),
          '  delta : {:>5.4f}'.format(delta))
    print('')

    print('{:>4s}'.format('** #'),
          '{:>20s}'.format('ub'),
          '{:>20s}'.format('lb'),
          '{:>8s}'.format('gap %'),
          '{:>6s}'.format('tau'),
          '{:>9s}'.format('time'))
    print()

    # Accumulate data here
    list_gap = list()
    list_time = list()
    list_ub = list()
    list_lb = list()
    list_tau = list()    
    for i in range(num_instances):
      	rand_state = random.getstate()
	print(' ************ ********** ****************')
	print(' ************ rand state ****************')
	print(' ************ ********** ****************')
	print(format(rand_state))
	print(' ************ ********** ****************')
	print(' ************ ********** ****************')
	print(' ************ ********** ****************')

        instance = ls.RandomLotsizing(num_rows, num_columns, num_scenarios,
                                      delta, density)
        # Save state of random number generator to restore after
        # running the algorithm
	instance.print_all()
        print('Instance {:d}'.format(i), file = detailed_output_stream)
        print('', file = detailed_output_stream)
        
        if (algorithm == 'B2'):
            res = b2_sddp(settings, instance.n1, instance.n2, instance.k,
                          instance.m, instance.p, instance.q,
                          instance.x_obj, instance.y_obj, instance.A_rows,
                          instance.G_rows, instance.T_rows,
                          instance.D_rows, instance.W_rows, instance.b,
                          instance.d, instance.w, instance.scenarios,
                          instance.prob, instance.z_lower, instance.delta,
                          detailed_output_stream)
        elif (algorithm == 'PP'):
            res = pp_sddp(settings, instance.n1, instance.n2, instance.k,
                          instance.m, instance.p, instance.q,
                          instance.x_obj, instance.y_obj, instance.A_rows,
                          instance.G_rows, instance.T_rows,
                          instance.D_rows, instance.W_rows, instance.b,
                          instance.d, instance.w, instance.scenarios,
                          instance.prob, instance.z_lower, instance.delta,
                          detailed_output_stream)
        else:
            raise ValueError('Algorithm ' + algorithm + ' unknown.')
        
        print('', file = detailed_output_stream)
        detailed_output_stream.flush()
        print('{:>4d}'.format(i),
              '{:>20.6f}'.format(res[1]),
              '{:>20.6f}'.format(res[2]),
              '{:>8.3f}'.format(res[3]*100),
              '{:>6d}'.format(res[4]),
              '{:>9.3f}'.format(res[5]))        
        list_ub.append(res[1])
        list_lb.append(res[2])
        list_gap.append(res[3]*100)
        list_tau.append(res[4])
        list_time.append(res[5])
        
        random.setstate(rand_state)
        
    print('')
    print('** Avg/stdev gap : {:>10.3f}'.format(np.mean(list_gap)),
          '{:>10.3f}'.format(np.std(list_gap)),
          '   time : {:>10.3f}'.format(np.mean(list_time)),
          '{:>10.3f}'.format(np.std(list_time)),
          '   ub : {:>10.3f}'.format(np.mean(list_ub)),
          '{:>10.3f}'.format(np.std(list_ub)),
          '   lb : {:>10.3f}'.format(np.mean(list_lb)),
          '{:>10.3f}'.format(np.std(list_lb)),
          '   tau : {:>10.3f}'.format(np.mean(list_tau)),
          '{:>10.3f}'.format(np.std(list_tau)))

# -- end function

if (__name__ == "__main__"):
    if (sys.version_info[0] >= 3):
        print('Error: Python 3 is currently not tested.')
        print('Please use Python 2.7')
        sys.exit()
    # Create command line parsers
    desc = ('Apply SDDP to randomly generated problems.')
    parser = argparse.ArgumentParser(description = desc)
    # Add options to parser and parse arguments
    register_options(parser)
    args = parser.parse_args()
    # Create settings for the algorithm
    settings = B2Settings.from_dictionary(vars(args))
    random.seed(args.rand_seed)
    outstream = open(args.log if args.log is not None else os.devnull, 'w')
    run_random_instances(args.algorithm, args.num_instances,
                         args.num_columns, args.num_rows,
                         args.num_scenarios, args.density, args.delta,
                         outstream)
    outstream.flush()
    outstream.close()

