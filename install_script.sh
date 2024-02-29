#!/bin/bash

echo "Current directory has to be the photometric_fofR directory"
pwd

echo "This will create and update the conda environment"

read -p "Are you sure you want to proceed? (Y/y/Yes/yes) " response

# Check if the response is anything other than Y, y, Yes, yes
if [[ ! $response =~ ^(Y|y|Yes|yes)$ ]]
then
    # The user did not confirm, abort the operation
    echo "Operation aborted."
    exit 1
fi

echo "Create conda environment"
conda create -f environment-pip.yml -q -y


cd ..
echo "****Installing CLASS"

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
exit 0
