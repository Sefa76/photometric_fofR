# photometric_fofR
Repository containing the Montepython f(R) likelihood for photometric probes. 

## Contents

1. The folder

    inputs/

contains the `.param`, `.data` and `.ini` files to launch and execute MontePython MCMC runs for the f(R) models.

2. The folder

    results/

contains the summary of MCMC results, for example CovMats, info files, etc.
These files are ignored by default in the github repo. Add only small files that are useful by using --force.

3. The folder

    plots/

contains plots almost ready for publication.

4. The folder

    codes/

contains snippets of code for MP and for analysis.

## Installation

Create a parent folder for this repo to contain all needed dependencies, that after a correct installation will look like this:

     Parent_Folder/
         photometric_fofR/
         montepython_public/
         class_public/

Read each of the individual `install.md` files for instructions.

Or execute the `install_script.sh`  that will create optionally a conda environment for you and then install all dependencies.
If you don't want to use conda, then move into your preferred environment and use the installation script as well. Follow the instructions of the script.
Launch the script with:

    `source install_script.sh`

To enable the correct activation of conda environments.

## Running an MCMC

Read the run_MCMC_MP.md file for instructions

