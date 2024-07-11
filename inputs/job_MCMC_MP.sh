#!/usr/local_rwth/bin/bash
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --account=rwth1304
#SBATCH --cpus-per-task=8
#SBATCH --partition=c23mm
#SBATCH --time=48:00:00
#SBATCH --output="WLpesbar_MeDf.out"
#SBATCH --error="WLpesbar_MeDf.err"
#SBATCH --mail-user=casas@physik.rwth-aachen.de
#SBATCH --mail-type=ALL
#SBATCH --job-name="WLpbef"
#SBATCH --mem-per-cpu=4G   
module purge
module restore intelpy23
source /home/qh043052/pyenv/intelpy23/bin/activate

echo "Running MP script"

### USAGE:  launch this script from the current dir

action=${1:-"none"}   #options: [run, info]
echo "***-->script:  Starting MontePython script with action: $action "

if [[ "$action" = "none" ]]; then
    echo "Please specify an action for the script [run, info]"
    echo "Exiting..."
fi
	
MPPy="montepython/MontePython.py"

MONTEPYTHON_DIR=../../montepython_public/ 
PARAM_INPUT=../photometric_fofR/inputs/wl_pes_fofr_eMantis-fitting_baryons.param
DATA_INPUT_1=../photometric_fofR/inputs/wl_pes_fofr_eMantis-fitting_baryons.data
DATA_INPUT_2=../photometric_fofR/inputs/bbn_gaussianprior.data
CHAINS=../photometric_fofR/results/chains/wl_pes_baryons_fofr_Winther-emantis/
Covmat=../photometric_fofR/results/covmats/HS6_WL_pesbar_emantis.covmat



keepnonmarkov=true
## MP run config params
MP_Nc=500000
MP_update=50
MP_superupdate=40
MP_jump=0.7


if [[ "$action" = "info" ]]; then
   echo "Move to MontePython main dir"
   cd $MONTEPYTHON_DIR
   echo "***--> script: Running MP info on chains: $CHAINS "
   if $keepnonmarkov;  then kpnmk="--keep-non-markovian"; else kpnmk=""; fi
   python $MPPy info --want-covmat "$kpnmk" "$CHAINS"*"$MP_Nchains"*
   echo "***--> script: Info ran successfully. "
fi


if [[ "$action" = "localrun" ]]; then
        echo "Move to MontePython main dir"
        cd $MONTEPYTHON_DIR

	echo "Copying data files to likelihood folders"
	cp -v $DATA_INPUT_1  montepython/likelihoods/euclid_photometric_z_fofr/euclid_photometric_z_fofr.data
	cp -v $DATA_INPUT_2  montepython/likelihoods/gaussianprior/gaussianprior.data

	echo "Creating fiducial"
	python montepython/MontePython.py run -p $PARAM_INPUT -o $CHAINS -f 0 -N 1
	echo "Testing chi-squared"
	python montepython/MontePython.py run -p $PARAM_INPUT -o $CHAINS -f 0 -N 1 --display-each-chi2
fi


if [[ "$action" = "run" ]]; then
        echo "Move to MontePython main dir"
        cd $MONTEPYTHON_DIR
	echo "delete chains folder"
	rm -rv $CHAINS

	echo "Copying data files to likelihood folders"
	cp -v $DATA_INPUT_1  montepython/likelihoods/euclid_photometric_z_fofr/euclid_photometric_z_fofr.data
	cp -v $DATA_INPUT_2  montepython/likelihoods/gaussianprior/gaussianprior.data

	echo "Creating fiducial"
	$MPIEXEC -n 1 python montepython/MontePython.py run -p $PARAM_INPUT -o $CHAINS -f 0 -N 1
	echo "Testing chi-squared"
	$MPIEXEC -n 1 python montepython/MontePython.py run -p $PARAM_INPUT -o $CHAINS -f 0 -N 1 --display-each-chi2

	$MPIEXEC $FLAGS_MPI_BATCH python montepython/MontePython.py run -o $CHAINS -f $MP_jump -N $MP_Nc --update $MP_update --superupdate $MP_superupdate -c $Covmat
fi

echo "Script ran successfully"


