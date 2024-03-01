#! /bin/bash

### USAGE:  launch this script from the current dir

MONTEPYTHON_DIR=../../montepython_public/ 
PARAM_INPUT=../photometric_fofR/inputs/wl_fofr_baryons_priors.param
DATA_INPUT=../photometric_fofR/inputs/wl_pess-model_winther_bary-data_forge_bary-theo_err.data
CHAINS=../photometric_fofR/results/wl_pess-model_winther_bary-data_forge_bary-theo_err/
Covmat=../GDrive-Kazuya/HS6_WL_pesbar_forge_winther/HS6_WL_pesbar_forge_winther.covmat

echo "Move to MontePython main dir"
cd $MONTEPYTHON_DIR

echo "delete chains folder"
rm -rv $CHAINS

echo "Copying data file to likelihood folder"
cp -v $DATA_INPUT  montepython/likelihoods/euclid_photometric_z_fofr/euclid_photometric_z_fofr.data

echo "Creating fiducial"
python montepython/MontePython.py run -p $PARAM_INPUT -o $CHAINS -f 0 -N 1
echo "Testing chi-squared"
python montepython/MontePython.py run -p $PARAM_INPUT -o $CHAINS -f 0 -N 1 --display-each-chi2

export OMP_NUM_THREADS=4
echo "Running chains"
mpirun -n 4 python montepython/MontePython.py run -o $CHAINS -f 1.9 -N 100000 --update 100 --superupdate 20 -c $Covmat
#python montepython/MontePython.py run -o $CHAINS -f 1.9 -N 100000 --update 100 --superupdate 20 -c $Covmat

echo "Chains run successfully"

