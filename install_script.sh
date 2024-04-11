#!/bin/bash

echo "Current directory has to be the photometric_fofR directory"
curr=$(pwd)
echo "Current directory: $curr "
read -p "Are you currently inside the correct directory? (Y/y/Yes/yes) " response
# Check if the response is anything other than Y, y, Yes, yes
if [[ ! $response =~ ^(Y|y|Yes|yes)$ ]]
then
    # The user did not confirm, abort the operation
    echo "Installation script exited. Please install from correct directory."
    return 0
fi

read -p "Do you want to use uv as a faster wrapper for pip installations? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
pip install uv
pipi="uv pip"
else
pipi="pip"
fi

echo "This will create and activate the conda environment named photometric_fofR. Otherwise it will only activate the environment."
echo "** Please run this script as 'source install_script.sh' to allow for correct environment installation from within the terminal. **"
read -p ">> Create or activate conda environment? Type: (Y/y/Yes/yes) for creation; Type: (activate/act) if you only want to activate an existing environment. If you type anything else, the installation script will continue in your current environment. Enter your response: " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
    conda env create -f environment-pip.yml  --force
    sleep 10

    conda init bash

    conda activate photometric_fofR
fi
if [[ $response =~ ^(activate|act)$ ]]
then
    conda init bash
    conda activate photometric_fofR
    sleep 10
else
    # The user did not confirm, abort the operation
    echo "Creation of conda environment aborted. Install the following packages within your preferred python environment."
    echo "Attempting pip installation"
    read -p ">> Do you want to install the requirements.txt file with pip? Yes/No?" response
    if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
    then
    $pipi install -U -r requirements.txt
    fi
fi


read -p "Are you sure you want to proceed with the installation of packages and codes? (Y/y/Yes/yes) " response
# Check if the response is anything other than Y, y, Yes, yes
if [[ ! $response =~ ^(Y|y|Yes|yes)$ ]]
then
    # The user did not confirm, abort the operation
    echo "Operation aborted."
    return 0
fi


cd ..
read -p ">> Do you want to git clone class? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
#pip install classy
git clone git@github.com:lesgourg/class_public.git
fi

read -p ">> Do you want to install class/classy? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
echo "****Installing CLASS"
cd class_public
make clean
make -j
cd ..
echo "Installation successful ****"
fi

read -p ">> Do you want to git clone MontePython? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
git clone git@github.com:brinckmann/montepython_public.git
fi

read -p ">> Do you want to copy the needed files from this fofR project into the corresponding MontePython directories? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
echo "*** Installing Montepython****"
cp -v photometric_fofR/codes/MP_default.conf montepython_public/default.conf
echo "** Copying Winther formula file"
cp -v photometric_fofR/codes/MGfit_Winther.py montepython_public/montepython/
echo "** Copying luminosity file for WL IA:"
cp -v photometric_fofR/codes/scaledmeanlum_E2Sa.dat montepython_public/data/
echo "** Copying euclid_photometric_z_fofr likelihood and gaussianprior likelihood into montepython/likelihoods folder: "
cp -vr photometric_fofR/codes/euclid_photometric_z_fofr montepython_public/montepython/likelihoods/
cp -vr photometric_fofR/codes/gaussianprior montepython_public/montepython/likelihoods/
echo "** Copying react code:"
cp -vr photometric_fofR/codes/react montepython_public/montepython/
fi

read -p ">> Do you want to clone the FORGE emulator and copy its code to the montepython/ directory? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
echo "*** Cloning and copying the FORGE emulator"
git clone https://bitbucket.org/arnoldcn/forge_emulator.git photometric_fofR/codes/forge_emulator
cp -vr photometric_fofR/codes/forge_emulator montepython_public/montepython/
fi

read -p ">> Do you want to pip install the requirements for the fofR emulators? Yes/No? " response
if [[ $response =~ ^(Y|y|Yes|yes)$ ]]
then
echo "***Installing libraries in correct order to avoid conflicts, this is machine dependent"
$pipi install "flatbuffers==23.5.26"
$pipi install "jaxlib==0.4.12"
echo "**** Installing cosmopower emulator"
$pipi install cosmopower
echo "**** Installing emantis emulator"
$pipi install emantis
echo "**** Install BCemu baryonic emulator"
$pipi install BCemu
fi

echo ">>>> Installations successful<<<<<"
