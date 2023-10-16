This is a the FORGE f(R) gravity cosmic emulator package written by
Christian Arnold, ICC, Durham University, August 2021                                              
base on an older code version by                                                              
B. M. Giblin, Edinburgh University   

See Arnold et al. (2021) for details.

The package contains a python module (GPR_Emulator.py) which is designed to emulate the matter power spectrum for f(R) gravity. The package also contains the trainign data for the matter power spectrum response for 50 nodes (parametersets), varying f_R0, Omega_m, h and sigma_8. Included are two example python scripts one to perform a cross validation of the emulator training data and one to train the emulator on the response data provided and then make a prediction for a cosmological parameter combination. Note that the internal state of the emulator is stored once it has been trained on a certain dataset once, so predictions can be made without re-training.

Below, a detailed overview of the emulator methodss and classes is given.

Init signature: GPR_Emulator.PCA_Class(n_components)
Docstring:     
Class for performing pincipal component analysis up to a specified number of basis functions.
Options are to find the basis functions for some input data, 
or to use pre-specified basis functions to some data to find corresponding weights
Init docstring:
Initialise a class object
n_compoments: the number of basis functions to use
File:           /cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/powerspec_emulator/FORGE_Emulator/GPR_Emulator.py
Type:           type

Init signature: GPR_Emulator.GPR_Emu(save_file_base, n_x_bins, use_train_err=True, do_PCA=True, pca_components=4, normalise_nodes=True)
Docstring:     
Class to perform the power spectrum emulation via Gaussian process regression. Contains methods to 
perform a PCA via the PCA class, to normalise the nodes, to train the emulator and store the internal 
state, to make predictions for given cosmological parameters and to validate the emulator using 
cross-validation of the training set.
Init docstring:
Initialise an emulator object.
save_file_base: the file basis to save the internal emulator state after training and to reload it 
to make predictions without training again (string)
n_x_bins: the number of x-bins in the training/prediction data (int)
use_train_err: use the provided error for the training data [recommended] (bool)
do_PCA: perform a principal component analysis on the input date before training [recommended] (bool)
pca_components: the number of basis functions to use for the PCA (int)
normalise_nodes: wether to re-scale the parameter range of the training data to [0, 1] (bool)
File:           /cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/powerspec_emulator/FORGE_Emulator/GPR_Emulator.py
Type:           type


Signature: GPR_Emulator.GPR_Emu.train_GPRsk(self, HPs, nodes_train, y_train, y_err_train, n_restarts_optimizer)
Docstring:
Train the gaussian process emulator and store it's internal state.
HPs: initial guess for hyper parameters, can be set to 1 if unknown, must 
contain #parameters+1 values (array)
nodes train: training node parameters (2D array)
y_train: training data, e.g. the power spec response measured from simulations 
for the training nodes (2D array)
y_err_train: the error of y (2D array)
n_restarts_optimiser: how many independent training runs for the optimiser are used
(int, recommended is ~20)
File:      /cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/powerspec_emulator/FORGE_Emulator/GPR_Emulator.py
Type:      function


Signature: GPR_Emulator.GPR_Emu.predict_GPRsk(self, nodes_predict)
Docstring:
Make an emulator prediction for node parameters given using the saved internal state of the emulator
nodes_predict: the node parameters to make a prediction for (2D array)
returns: the predicted data, e.g. powerspectrum response, the estimated error of the prediction,
the HPs determined by the emulator.
File:      /cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/powerspec_emulator/FORGE_Emulator/GPR_Emulator.py
Type:      function

Signature: GPR_Emulator.GPR_Emu.Cross_Validation(self, HPs, nodes_train, y_train, y_err_train, n_restarts_optimizer)
Docstring:
Perform a cross validation of the emulator on the training sample (leave one out test)
HPs: initial guess for hyper parameters, can be set to 1 if unknown, must 
contain #parameters+1 values (array)
nodes train: training node parameters (2D array)
y_train: training data, e.g. the power spec response measured from simulations 
for the training nodes (2D array)
y_err_train: the error of y (2D array)
n_restarts_optimiser: how many independent training runs for the optimiser are used
(int, recommended is ~20)
returns: the predictions for all nodes in the CV test, the estimated error of the prediction
File:      /cosma6/data/dp004/dc-arno1/CosmicEmulatorNodes/powerspec_emulator/FORGE_Emulator/GPR_Emulator.py
Type:      function
