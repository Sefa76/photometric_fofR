## Instructions to run MontePython (MP) MCMCs

Create the fiducial 

    python montepython_public/montepython/MontePython.py run -p photometric_fofR/inputs/fofr_6Params_HS6_WL_MCMC.param -o results/fofr_6Params_HS6_WL_MCMC -f 0 

The fiducial synthetic data vector will be stored according to the settings in `euclid_photometric_z_fofr.data`

Now make an MCMC run of 1000 steps by doing:

    python montepython_public/montepython/MontePython.py run -o results/fofr_6Params_HS6_WL_MCMC -N 1000 -c photometric_fofR/results/covmats/3x2pt_opt_6cosmo+freebias_HS6.covmat
