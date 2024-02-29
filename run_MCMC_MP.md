## Instructions to run MontePython (MP) MCMCs

Starting from the KPJC6 base directory, cd into `montepython_public` :

    cd montepython_public

Create the fiducial 

    python montepython/MontePython.py run -p ../photometric_fofR/inputs/fofr_6Params_HS6_WL_MCMC.param -o ../results/fofr_6Params_HS6_WL_MCMC -f 0 

The fiducial synthetic data vector will be stored according to the settings in `euclid_photometric_z_fofr.data` and a results folder will be created in the base directory.

Now make an MCMC run of 1000 steps, using a previously computed proposal covmat by doing:

    python montepython/MontePython.py run -o ../results/fofr_6Params_HS6_WL_MCMC -N 1000 -c ../photometric_fofR/results/covmats/3x2pt_opt_6cosmo+freebias_HS6.covmat
