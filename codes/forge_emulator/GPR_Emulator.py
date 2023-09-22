'''
FoRGE cosmic emulator package
Christian Arnold, ICC, Durham University, August 2021
base on an older code version by 
B. M. Giblin, Edinburgh University
'''
import numpy as np
import os
import pickle

# For doing Gaussian processes with SK-learn
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
# For doing PCA
from sklearn.decomposition import PCA


class PCA_Class:
    '''
    Class for performing pincipal component analysis up to a specified number of basis functions.
    Options are to find the basis functions for some input data, 
    or to use pre-specified basis functions to some data to find corresponding weights
    '''
    def __init__(self, n_components):
        '''Initialise a class object
        n_compoments: the number of basis functions to use
        '''
        self.n_components = n_components
    
    def PCA_BySKL(self, Data):      # use scikit-learn
        '''Perform a principal component analysis on Data
        Data: a 2D numpy array contaning the y-values of the quantity to 
        perform the PCA on for each dataset (node) and x-bin.
        returns: the basis functions determined (2D array), 
        the weights for each dataset (2D array), 
        the recons, 
        the skl pca class object.
        '''
        pca = PCA(n_components=self.n_components)
        Weights = pca.fit_transform(Data)               # Weights of the PCA
        Recons = pca.inverse_transform(Weights) # The reconstructions
        BFs = pca.components_               # Derived basis functions
        return BFs, Weights, Recons, pca

    # Accept some basis functions, BFs, Data to perform PCA on,
    # + the mean (in each bin) of the data for which the BFs were identified,
    # and manually do the PC Reconstruction
    def PCA_ByHand(self, BFs, Data, data_Mean):
        '''Express data in given basis functions
        BFs: the basis functions to use (2D array)
        Data: the data to decompose into the BFs (2D array)
        data_Mean: the mean of the data used to initially determine 
        the BFs with PCA_bySKL
        returns: the weights for the data decomposed into the basis functions,
        the recons
        '''
        Data_MinusMean = np.empty_like( Data )
        for i in range(len(Data[0,:])):
            Data_MinusMean[:,i] = Data[:,i] - data_Mean[i]
        Weights = np.dot(Data_MinusMean,np.transpose(BFs))
    
        Recons = np.zeros([ Data.shape[0], BFs.shape[1] ]) 
        for j in range(len(Weights[:,0])):          # Scroll through the the Data
            for i in range(len(Weights[0,:])):      # scroll through the basis functions
                Recons[j,:] += Weights[j,i]*BFs[i,:]
            Recons[j,:] += data_Mean
        return Weights, Recons


    # Read in Weights, BFS, and data_Mean to recover original statistic
    def Convert_PCAWeights_2_Predictions(self, Weights, BFs, Mean):
        '''Transform given weights back into data (do the PCA backwards)
        Weights: the weights to transfor back into the initial frame of reference (2D array)
        BFs: the basis functions used (2D array)
        Mean: the mean of the data used to initially determine                                                                     
        the BFs with PCA_bySKL
        returns: the data in the initial frame of reference (2D array)
        '''
        if len(Weights.shape) == 1:
            Weights = Weights.reshape(1, len(Weights))
        Predictions = np.zeros([ Weights.shape[0], BFs.shape[1] ]) 
        for j in range(Weights.shape[0]):           # Scroll through number of predictions one needs to make.
            Predictions[j,:] += Mean
            for i in range(BFs.shape[0]):       # Scroll through BFs adding the correct linear combination to the Mean
                Predictions[j,:] += Weights[j,i] * BFs[i,:]
        return Predictions


# Class for doing emulation via Gaussian process regression
class GPR_Emu:
    '''
    Class to perform the power spectrum emulation via Gaussian process regression. Contains methods to 
    perform a PCA via the PCA class, to normalise the nodes, to train the emulator and store the internal 
    state, to make predictions for given cosmological parameters and to validate the emulator using 
    cross-validation of the training set.
    '''
    def __init__(self, save_file_base, n_x_bins, use_train_err = True, do_PCA = True, pca_components = 4, normalise_nodes = True):
        '''Initialise an emulator object.
        save_file_base: the file basis to save the internal emulator state after training and to reload it 
        to make predictions without training again (string)
        n_x_bins: the number of x-bins in the training/prediction data (int)
        use_train_err: use the provided error for the training data [recommended] (bool)
        do_PCA: perform a principal component analysis on the input date before training [recommended] (bool)
        pca_components: the number of basis functions to use for the PCA (int)
        normalise_nodes: wether to re-scale the parameter range of the training data to [0, 1] (bool)
        '''
        self.save_file_base = save_file_base
        self.use_train_err = use_train_err
        self.n_x_bins = n_x_bins
        self.do_PCA = do_PCA
        self.normalise_nodes = normalise_nodes
        self.train_nodes_max = np.array([None])
        self.train_nodes_min = np.array([None])

        if self.do_PCA:
            self.PCA = PCA_Class(pca_components)
            self.n_x_bins = pca_components


    def train_PCA(self, y, y_err = False):
        '''Train the pricipal component analysis and convert the data and errors. This is done 
        automatically by the training and prediction functions if the do_PCA option is selected.
        y: the data for training the PCA, will be converted to weights (2D array)
        y_err: the error of y, will be converted to an error of the weights (optional, 2D array)
        returns: weights and their error
        '''

        Train_BFs, Train_Weights, Train_Recons, Train_pca = self.PCA.PCA_BySKL(y)
        
        if type(y_err != bool):
            Train_Pred_Mean = np.mean(y, axis = 0)
            Train_Weights_Upper, Train_Recons_Upper = self.PCA.PCA_ByHand(Train_BFs, y + y_err, Train_Pred_Mean)
            Train_Weights_Lower, Train_Recons_Lower = self.PCA.PCA_ByHand(Train_BFs, y - y_err, Train_Pred_Mean)
            Train_ErrWeights = abs(Train_Weights_Upper - Train_Weights_Lower)

        else:
            Train_ErrWeights = np.zeros_like(Train_Weights)

        train_BFs_file = self.save_file_base + '_BFs.npy'
        np.save(train_BFs_file, (Train_BFs, Train_Pred_Mean), allow_pickle = True)

        return Train_Weights, Train_ErrWeights

    def perform_inverse_PCA(self, weights, weights_err = False):
        '''Convert the weights and their error obtained from the emulator back to 
        data in the initial frame of reference. This is done automatically by the training 
        and prediction functions if the do_PCA option is selected.
        weights: the weights to convert (2D array)
        weights_err: the error of the weights (optional, 2D array)
        returns: the data, the error of the data
        '''
        train_BFs_file = self.save_file_base + '_BFs.npy'
        BFs, Train_Pred_Mean = np.load(train_BFs_file, allow_pickle = True)

        pred = self.PCA.Convert_PCAWeights_2_Predictions(weights, BFs, Train_Pred_Mean)

        if type(weights_err != bool):
            Upper = self.PCA.Convert_PCAWeights_2_Predictions(weights + weights_err, BFs, Train_Pred_Mean)
            Lower = self.PCA.Convert_PCAWeights_2_Predictions(weights - weights_err, BFs, Train_Pred_Mean)
            pred_err = abs(Upper - Lower) / 2.
        else:
            pred_err = np.zeros_like(pred)

        return pred, pred_err

    def train_GPRsk(self, HPs, nodes_train, y_train, y_err_train, n_restarts_optimizer):
        '''Train the gaussian process emulator and store it's internal state.
        HPs: initial guess for hyper parameters, can be set to 1 if unknown, must 
        contain #parameters+1 values (array)
        nodes train: training node parameters (2D array)
        y_train: training data, e.g. the power spec response measured from simulations 
        for the training nodes (2D array)
        y_err_train: the error of y (2D array)
        n_restarts_optimiser: how many independent training runs for the optimiser are used
        (int, recommended is ~20)
        '''
        if self.do_PCA:
            y_train, y_err_train = self.train_PCA(y_train, y_err_train)

        if self.use_train_err:

            # If providing an error, SKL requires you run each x-bin separately:
            for i in range(y_train.shape[1]):
                savefile = self.save_file_base + '%i.npy'%i
                print( "Now on x bin %s of %s" %(i, y_train.shape[1]))

                if len(HPs.shape) == 1: 
                    # We do not have individual HPs per bin
                    self.train_GPRsk_x_bin(HPs, nodes_train, y_train[:, i], y_err_train[:, i], n_restarts_optimizer, savefile)

                else:
                    # We DO have individual HPs per bin!
                    self.train_GPRsk_x_bin(HPs[i], nodes_train, y_train[:, i], y_err_train[:, i], n_restarts_optimizer, savefile)

        else:
            savefile = self.save_file_base + '.npy'
            self.train_GPRsk_x_bin(HPs, nodes_train, y_train, y_err_train, n_restarts_optimizer, savefile)

    def predict_GPRsk(self, nodes_predict):
        '''Make an emulator prediction for node parameters given using the saved internal state of the emulator
        nodes_predict: the node parameters to make a prediction for (2D array)
        returns: the predicted data, e.g. powerspectrum response, the estimated error of the prediction,
        the HPs determined by the emulator.
        '''
        if self.use_train_err:
            # If providing an error, SKL requires you run each x-bin separately:
            GP_AVOUT = np.zeros([nodes_predict.shape[0], self.n_x_bins ])
            GP_STDOUT = np.zeros([nodes_predict.shape[0], self.n_x_bins ])   
            GP_HPs = np.zeros([self.n_x_bins, nodes_predict.shape[1]+1 ])
            for i in range(GP_AVOUT.shape[1]):
                savefile = self.save_file_base + '%i.npy'%i
                GP_AVOUT[:,i], GP_STDOUT[:,i], GP_HPs[i,:] = self.predict_GPRsk_x_bin(nodes_predict, savefile)


        else:
            savefile = self.save_file_base + '.npy'
            GP_AVOUT, GP_STDOUT, GP_HPs = self.predict_GPRsk_x_bin(nodes_predict, savefile)                 # with Scikit-Learn  
            GP_STDOUT = np.repeat(np.reshape(GP_STDOUT, (-1,1)), GP_AVOUT.shape[1], axis=1)         # SKL only returns 1 error bar per trial here

        if self.do_PCA:
            pred, pred_err = self.perform_inverse_PCA(GP_AVOUT, GP_STDOUT)
        else:
            pred = GP_AVOUT
            pred_err = GP_STDOUT
            
        return pred, pred_err, GP_HPs

    #-------------------------------------------
    def scale_nodes(self, nodes, train_nodes_max = np.array([None]), train_nodes_min = np.array([None])):
        '''internal function to rescale the node parameters''' 
        if train_nodes_max[0] == None:
            train_nodes_max = np.max(nodes, axis = 0)
            train_nodes_min = np.min(nodes, axis = 0)

        scaled_nodes = (nodes - train_nodes_min) / abs(train_nodes_max - train_nodes_min)

        return scaled_nodes, train_nodes_max, train_nodes_min

    def train_GPRsk_x_bin(self, hp, nodes_train, y_train, y_err_train, n_restarts_optimizer, savefile):
        '''internal function to train the emulator for a single x-bin'''
        if self.normalise_nodes:
            nodes_train, self.train_nodes_max, self.train_nodes_min = self.scale_nodes(nodes_train)

        kernel = hp[0] * RBF(hp[1:])
        if n_restarts_optimizer is not None:
            print("Optimising the emulator with %s restarts..." % n_restarts_optimizer)

        if y_err_train is not None:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, alpha=y_err_train, normalize_y = False)
        else:
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, normalize_y =False)
        gp.fit(nodes_train, y_train)
        f = open(savefile, 'wb')
        pickle.dump((gp, self.train_nodes_max, self.train_nodes_min), f)
        f.close()

    def predict_GPRsk_x_bin(self, nodes_predict, savefile):
        '''internal function to make a prediction for a single x-bin'''
        f = open(savefile, 'rb')
        gp, self.train_nodes_max, self.train_nodes_min = pickle.load(f)
        f.close()

        if self.normalise_nodes:
            nodes_predict, dummy, dummy = self.scale_nodes(nodes_predict, self.train_nodes_max, self.train_nodes_min)

        GP_AVOUT, GP_STDOUT = gp.predict(nodes_predict, return_std=True)

        return GP_AVOUT, GP_STDOUT, self.Process_gpkernel(gp.kernel_)


    def Process_gpkernel(self, gpkernel):
        '''internal function to process the GP kernel'''
        hp_amp = eval( str(gpkernel).split()[0] )
        hp_rest = eval( str(gpkernel).split('length_scale=')[-1].split(')')[0] ) 
        return np.append( hp_amp, hp_rest ) 

    def Cross_Validation(self, HPs, nodes_train, y_train, y_err_train, n_restarts_optimizer):
        '''Perform a cross validation of the emulator on the training sample (leave one out test)
        HPs: initial guess for hyper parameters, can be set to 1 if unknown, must 
        contain #parameters+1 values (array)
        nodes train: training node parameters (2D array)
        y_train: training data, e.g. the power spec response measured from simulations 
        for the training nodes (2D array)
        y_err_train: the error of y (2D array)
        n_restarts_optimiser: how many independent training runs for the optimiser are used
        (int, recommended is ~20)
        returns: the predictions for all nodes in the CV test, the estimated error of the prediction
        '''
        import time
        # Cycle through training set, omitting one, training on the rest, and testing accuracy with the omitted.
        t1 = time.time()
        NumNodes = y_train.shape[0]

        if self.do_PCA:
            weights, weights_err = self.train_PCA(y_train, y_err_train)
        else:
            weights = y_train.copy()
            weights_err = y_err_train.copy()

        pred = np.empty([NumNodes, y_train.shape[1]])
        pred_error = np.empty([NumNodes, y_train.shape[1]])

        #GP_HPs_AllNodes = np.zeros([ NumNodes, Predictions.shape[1], self.nodes_train.shape[1]+1 ])
        for i in range(NumNodes):
            print( "Performing cross-val. for node %s of %s..." %(i, NumNodes) )

            new_GPR = GPR_Emu(self.save_file_base + '_CV' + str(i).zfill(3) + ' ', self.n_x_bins, use_train_err = self.use_train_err, do_PCA = False, pca_components = 4)

            new_nodes_trial  = nodes_train[i,:].reshape((1,len(nodes_train[i,:])))

            new_nodes_train = np.delete(nodes_train, i, axis=0)
            new_y_train = np.delete(weights, i, axis=0)
            new_y_err_train = np.delete(weights_err, i, axis=0)

            # Train the Emulator and save the internal state
            new_GPR.train_GPRsk(HPs, new_nodes_train, new_y_train, new_y_err_train, n_restarts_optimizer)
            # make a prediction
            GP_AVOUT, GP_STDOUT, GP_HPs = new_GPR.predict_GPRsk(new_nodes_trial) 

            if self.do_PCA:     # If Perform_PCA is True, need to convert weights returned from emulator to predictions
                pred[i, :], pred_error[i, :] = self.perform_inverse_PCA(GP_AVOUT, GP_STDOUT)
            else:
                pred[i, :] = GP_AVOUT
                pred_error[i, :] = GP_STDOUT

        t2 = time.time()
        print( "Whole cross-val. took %.1f s for %i nodes..." %((t2-t1), NumNodes) )

        return pred, pred_error




