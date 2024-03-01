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
    pip install -U -r requirements.txt
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
echo "****Installing CLASS"

#pip install classy
git clone git@github.com:lesgourg/class_public.git
cd class_public
make clean
make -j
cd ..
echo "Installation successful ****"

echo "*** Installing Montepython****"
git clone git@github.com:brinckmann/montepython_public.git
cp -v photometric_fofR/codes/MP_default.conf montepython_public/default.conf
cp -v photometric_fofR/codes/MGfit_Winther.py montepython_public/montepython/
cp -v photometric_fofR/codes/scaledmeanlum_E2Sa.dat montepython_public/data/
cp -vr photometric_fofR/codes/euclid_photometric_z_fofr montepython_public/montepython/likelihoods/
cp -vr photometric_fofR/codes/gaussianprior montepython_public/montepython/likelihoods/

echo "**** Installing cosmopower and ReACT"

echo "***Installing libraries in correct order to avoid conflicts, this is machine dependent"
pip install "flatbuffers==23.5.26"
pip install "jaxlib==0.4.12"
pip install cosmopower
echo "** Copying react folder"
cp -vr photometric_fofR/codes/react montepython_public/montepython/

echo "*** Installing FORGE emulator"

git clone https://bitbucket.org/arnoldcn/forge_emulator.git photometric_fofR/codes/forge_emulator
cp -vr photometric_fofR/codes/forge_emulator montepython_public/montepython/


echo "**** Installing emantis emulator"
pip install emantis

echo "**** Install BCemu baryonic emulator"
pip install BCemu


echo ">>>> Installations successful<<<<<"
