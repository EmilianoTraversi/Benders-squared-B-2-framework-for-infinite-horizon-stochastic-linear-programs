=========================
=========================



The computational experiments presented in the paper are related to two types of problem:
- Production Planning with Backlog (PPB)
- Continuous Rebalancing Problem (CRP)

******************************
******************************
******************************
******************************

To execute the experiments concerning the PPB, it is necessary to use the python file named "b2_sddp_ls_interface.py".

In the file "b2_sddp_settings.py", a class named B2Settings is defined, in the comments of the class a list the possible parameters is given.


Table 1 is obtained by running b2_sddp_ls_interface.py  with the following combinations of parameters 


num_instances = 10
num_columns = 10
num_rows = 5
num_scenarios =  10, 20, 30, 40, 50
gap_relative = 0.01
gap_absolute = 1, 100
algorithm = PP (Pereira_Pinto), B2 (our Algorithm)
rounds_before_cut_purge = 10
num_passes = 10 


an example of run is the following:

python b2_sddp_ls_interface.py --num_instances 10 --num_columns 10 --num_rows 5 --num_scenarios 20 --gap_relative 0.01 --gap_absolute 1 --algorithm B2 --rounds_before_cut_purge 10 --num_passes 10

and the output produced is the following:

** n :     10   m :      5   k :     20   dens : 0.7500   delta : 0.9500

** #                   ub                   lb    gap %    tau      time

   0          4403.980543          4394.929210    0.206      3     1.250
   1          6020.681813          5974.644161    0.765      3     1.240
   2          3433.818294          3400.347151    0.975      3     1.220
   3          2821.888098          2807.274784    0.518      3     1.090
   4          3775.379967          3766.815857    0.227      3     1.250
   5          6185.476542          6127.244053    0.941      6     7.300
   6          5509.836031          5455.270323    0.990      8    13.580
   7          1352.592740          1348.354892    0.313      3     1.110
   8          4437.312761          4395.819219    0.935      8     9.360
   9          5655.566347          5614.027755    0.734      3     1.190

** Avg/stdev gap :      0.660      0.302    time :      3.859      4.317    ub :   4359.653   1473.385    lb :   4328.473   1457.695    tau :      4.300      2.052

showing in each line the results concerning a specific randomly generated instance.
In the last line, the average and the standard deviation concerning the relative gap, the computing time, the maximum value of tau and the upper and lower bounds are reported.



Table 2 is obtained by running b2_sddp_ls_interface.py with the following combinations of parameters 


num_instances = 10
(num_columns,num_rows) = (10,5), (10,10), (10,20), (20,10), (20,20), (20,40) 
num_scenarios =  10, 20, 30, 40, 50
gap_relative = 0.01
gap_absolute = 1
algorithm = B2
rounds_before_cut_purge = 2, 5, 10
num_passes = 2, 5, 10


******************************
******************************
******************************
******************************

To execute the experiments concerning the CRP, it is necessary to use the python file named "b2_sddp_re_interface.py".


Table 3 is obtained by running b2_sddp_re_interface.py with the following combinations of parameters 


num_instances = 10
num_points = 10 
sat = 0.97, 0.98, 0.99, 1.00 
num_scenarios =  10, 20, 30, 40, 50
gap_relative = 0.45, 0.46, 0.47, 0.48, 0.49, 0.50 
algorithm = B2
rounds_before_cut_purge = 5, 10
num_passes = 2

Table 4 is obtained by running b2_sddp_re_interface.py with the following combinations of parameters 

num_instances = 10
num_points = 10 
sat = 0.97, 0.98, 0.99, 1.00 
num_scenarios =  10, 20, 30, 40, 50
gap_relative = 0.45
algorithm = B2
rounds_before_cut_purge = 10
num_passes = 2


and after adding the following modifications to the file  b2_sddp.py

1- before line 586 (i.e. before the start of the algorithm) the following line should be added:
  last_lb = -1e8

2- before line 665 (i.e. before the standard stopping criterion) the following lines should be added:

#new stopping criterion
        eps_newstop =1e-3
        if (last_lb >0):
          if ( ( (abs( lb -last_lb )) / abs(last_lb) ) <eps_newstop):
              stop = True
        last_lb = lb