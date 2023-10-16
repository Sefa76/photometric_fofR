# Christian Arnold, ICC, Durham University, 08/2021,
# based on an older version of 23/04/2018, B. M. Giblin, PhD Student, Edinburgh
import numpy as np
import matplotlib
import sys
import os
from pathlib import Path
# For doing Gaussian processes
#from GPR_Emulator import GPR_Emu
try:
    import camb
except:
    print("FORGE warning: CAMB is not correctly installed. So you can't predict P(k), only boost")

import imp

parent_path = str(Path(Path(__file__).resolve().parents[0]))
sys.path.insert(0,parent_path)

gpr_emu = imp.load_source('GPR_Emulator', parent_path+'/GPR_Emulator.py')

class FORGE:
    def __init__(self, save_file_base = parent_path+'/EmulatorState/EmulatorTrainState'):
        self.save_file_base_Bk = save_file_base + '_Bk'
        self.save_file_base_Wiggles = save_file_base + '_Wiggles'

        self.GPR_Bk = {}
        self.GPR_Wiggles = {}

        self.output_redshifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0])
        self.k = np.array([])

    def train_emulator_single(self, GPR_Class, train_node_file, train_data_file, train_error_file):
        Train_Nodes = np.loadtxt(train_node_file)
        tmp = np.load(train_data_file)
        Train_x = tmp[0, 0, :] #k-vector

        if len(self.k) == 0:
            self.k = Train_x
        else:
            assert(all(self.k == Train_x))

        Train_Pred = tmp[1] #B(k), has dimension N_nodes x N_k_bins
        tmp = np.load(train_error_file)
        Train_ErrPred = tmp[1] # error of B(k) training data, same dimension as B(k)

        # Train the Emulator and save the internal state
        n_restarts_optimizer = 20 # number of independent training run restarts
        HPs = np.ones(len(Train_Nodes[0]) + 1) # initial guess for the hyper parameters - can be set to 1 if no good values are known
        GPR_Class.train_GPRsk(HPs, Train_Nodes, Train_Pred, Train_ErrPred, n_restarts_optimizer)

    def train(self, redshift):
        if redshift in self.GPR_Bk or redshift in self.GPR_Wiggles:
            raise ValueError("Emulator already trained for this redshift")

        print("Training Emulator for z = %.2f"%redshift)

        train_node_file, train_data_file_Bk, train_error_file_Bk, train_data_file_Ws, train_error_file_Ws = self.get_filenames(redshift)

        nx = np.load(train_data_file_Bk)[1].shape[1]
        self.GPR_Bk[redshift] = gpr_emu.GPR_Emu(self.save_file_base_Bk + '_z%.1f'%redshift, nx, use_train_err = True, normalise_nodes = True)
        self.GPR_Wiggles[redshift] = gpr_emu.GPR_Emu(self.save_file_base_Wiggles + '_z%.1f'%redshift, nx, use_train_err = True, normalise_nodes = True)

        self.train_emulator_single(self.GPR_Bk[redshift], train_node_file, train_data_file_Bk, train_error_file_Bk)
        self.train_emulator_single(self.GPR_Wiggles[redshift], train_node_file, train_data_file_Ws, train_error_file_Ws)

    def predict_for_output_redshift(self, redshift, predict_params):
        if redshift not in self.GPR_Bk:
            self.train(redshift)

        predict_params = np.array([predict_params])
        Bk_Pred, Bk_ErrPred, Bk_HPs = self.GPR_Bk[redshift].predict_GPRsk(predict_params)
        Wiggle_Pred, Wiggle_ErrPred, Wiggle_HPs = self.GPR_Wiggles[redshift].predict_GPRsk(predict_params)

        return Bk_Pred[0], Bk_ErrPred[0], Wiggle_Pred[0], Wiggle_ErrPred[0]


    def predict_Bk(self, redshift, omega_m, h, fR0, sigma8):
        '''Make a prediction for B(k) = P(k) / P(k)_Halofit^LCDM for a given
        cosmology and redshift. This will make a prediction for the bracketing
        redshifts the emulator was trained on and then return a linear
        interpolation between these.
        input: redshift, cosmological parameters, fR0=6 corresponds to f_R0=-10^(-6)
        returns: B(k), error on B(k)
        '''
        predict_params = np.array([omega_m, h, fR0, sigma8])
        zs, zl = self.get_bracket_output_redshifts(redshift)

        if zs == zl:
            Bk_Pred, Bk_ErrPred, Wiggle_Pred, Wiggle_ErrPred = self.predict_for_output_redshift(redshift, predict_params)
            Bk = Bk_Pred + Wiggle_Pred - 1.
            Bk_err = np.sqrt((Wiggle_ErrPred)**2 + (Bk_ErrPred)**2)

        else:
            result_s = self.predict_for_output_redshift(zs, predict_params)
            result_l = self.predict_for_output_redshift(zl, predict_params)

            Bk_s = result_s[0] + result_s[2] - 1.
            Bk_l = result_l[0] + result_l[2] - 1.
            Bk = self.interpolate(redshift, zs, zl, Bk_s, Bk_l)
            Bk_err = np.sqrt(result_s[1]**2 + result_s[3]**2)

        return Bk, Bk_err

    def predict_Pk(self, redshift, omega_m, h, fR0, sigma8):
        '''Make a prediction for P(k) using the CAMB/HALOFIT LCDM baseline and
        the emulated (and possibly interpolated) B(k).
        input: redshift and cosmological parameters, fR0=6 corresponds to f_R0=-10^(-6)
        returns: P(k), P(k) error.
        '''
        Bk, Bk_err = self.predict_Bk(redshift, omega_m, h, fR0, sigma8)
        Pk_camb = self.get_camb_prediction(redshift, self.k, omega_m, h, sigma8, omega_b = 0.049199, omega_k = 0.0, n_s = 0.9667)

        return Pk_camb * Bk, Pk_camb * Bk_err

    def interpolate(self, target_redshift, smaller_redshift, larger_redshift, smaller_values, larger_values):
        assert(smaller_redshift <= target_redshift <= larger_redshift)

        # interpolate in a rather than redshift
        at = 1 / (1 + target_redshift)
        asr = 1 / (1 + smaller_redshift)
        alr = 1 / (1 + larger_redshift)

        return smaller_values + (larger_values - smaller_values) / (alr - asr) * (at -asr)

    def get_bracket_output_redshifts(self, redshift):
        if redshift < 0:
            raise ValueError("Emulator not trained for z<0")
        if redshift > 3.0:
            raise ValueError("Emulator not trained for z>3")

        smaller = self.output_redshifts[self.output_redshifts <= redshift][-1]
        larger = self.output_redshifts[self.output_redshifts >= redshift][0]

        return smaller, larger

    def get_filenames(self, redshift):
        DATA_DIR =  Path(__file__).parent.resolve()
        if redshift == 0.0:
            node_file = DATA_DIR / 'FoR_MatterPowerSpec_TrainingData/nodes_learn_file_boxes.txt'
            data_file = DATA_DIR / 'FoR_MatterPowerSpec_TrainingData/powerspec_learn_data_boxes.npy'
            error_file = DATA_DIR /'FoR_MatterPowerSpec_TrainingData/powerspec_learn_error_boxes.npy'
            data_wiggle_file = DATA_DIR / 'FoR_MatterPowerSpec_TrainingData/powerspec_learn_data_wiggles.npy'
            error_wiggle_file = DATA_DIR / 'FoR_MatterPowerSpec_TrainingData/powerspec_learn_error_wiggles.npy'

        else:
            node_file = DATA_DIR / f'FoR_MatterPowerSpec_TrainingData/nodes_learn_file_boxes.txt'
            data_file = DATA_DIR / f'FoR_MatterPowerSpec_TrainingData/powerspec_learn_data_boxes_z{redshift:.1f}.npy'
            error_file = DATA_DIR / f'FoR_MatterPowerSpec_TrainingData/powerspec_learn_error_boxes_z{redshift:.1f}.npy'
            data_wiggle_file = DATA_DIR / f'FoR_MatterPowerSpec_TrainingData/powerspec_learn_data_wiggles_z{redshift:.1f}.npy'
            error_wiggle_file = DATA_DIR / f'FoR_MatterPowerSpec_TrainingData/powerspec_learn_error_wiggles_z{redshift:.1f}.npy'

        return node_file, data_file, error_file, data_wiggle_file, error_wiggle_file

    def calculate_sigma_8(self, n_s, h, obh2, och2, As):
        cp = camb.set_params(ns=n_s, H0=h*100, ombh2=obh2, omch2=och2, w=-1, lmax=2000)
        cp.InitPower.set_params(As=As, ns=n_s, nrun=0, nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, pivot_scalar=0.05, pivot_tensor=0.05, parameterization=2)
        cp.WantTransfer = True
        cp.DoLensing = False
        res = camb.get_results(cp)
        s8 = res.get_sigma8()
        return s8[0]

    def get_As(self, omega_m, h, sigma8, omega_b = 0.049199, omega_k = 0.0, n_s = 0.9667, As_max = 6e-8, As_min = 1e-10):
        s8 = 100.0

        obh2 = omega_b * h**2
        och2 = (omega_m - omega_b) * h**2
        As_top = As_max
        As_bottom = As_min
        s8_top = self.calculate_sigma_8(n_s, h, obh2, och2, As_top)
        s8_bottom = self.calculate_sigma_8(n_s, h, obh2, och2, As_bottom)

        while abs(s8 / sigma8 - 1) > 5e-3:
            assert(s8_top > sigma8)
            assert(s8_bottom < sigma8)

            As = 10**((np.log10(As_top) + np.log10(As_bottom)) / 2.)
            s8 = self.calculate_sigma_8(n_s, h, obh2, och2, As)

            if s8 < sigma8:
                As_bottom = As
                s8_bottom = s8
            else:
                As_top = As
                s8_top = s8

        return As


    def get_camb_prediction(self, redshift, k, omega_m, h, sigma8, omega_b = 0.049199, omega_k = 0.0, n_s = 0.9667):
        #--- get non-linear estimate for power spectrum ---
        omega_l = 1 - omega_m
        obh2 = omega_b * h**2
        och2 = (omega_m - omega_b) * h**2
        As = self.get_As(omega_m, h, sigma8)

        cp = camb.set_params(ns=n_s, H0=h*100, ombh2=obh2, omch2=och2, w=-1, lmax=2000)
        cp.InitPower.set_params(As=As, ns=n_s, nrun=0, nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, pivot_scalar=0.05, pivot_tensor=0.05, parameterization=2)
        cp.WantTransfer = True
        cp.DoLensing = False
        res = camb.get_results(cp)

        PK = camb.get_matter_power_interpolator(cp, zmin=0, zmax=130, nz_step=100, zs=None, kmax=10, nonlinear=True, var1=None, var2=None, hubble_units=True, k_hunit=True, return_z_k=False, k_per_logint=None, log_interp=True, extrap_kmax=None)
        return PK.P(redshift, k)


#below some doctest
if __name__ == "__main__":
    #create an emulator instance;
    #it's recommended to do this once at the start as multiple
    #instances might lead to duplications in training work
    forge_emulator = FORGE()

    #now we want to make predictions for the following cosmology
    omega_m = 0.3
    h = 0.72
    fR0 = 5 #this corresponds to f_R0 = -1e-5
    sigma8 = 0.8

    #and we have a list of redshifts we want to make predictions for
    redshifts = [0.0, 0.1, 0.12, 0.25]

    #plot a figure for the results
    fig, ax = matplotlib.pyplot.subplots(1,1)

    #create a dict to store the results for B(k)
    Bk = {}
    Pk = {}

    for redshift in redshifts:
        Bk[redshift], error = forge_emulator.predict_Bk(redshift, omega_m, h, fR0, sigma8)
        Pk[redshift], error = forge_emulator.predict_Pk(redshift, omega_m, h, fR0, sigma8)
        ax.plot(forge_emulator.k, Bk[redshift], ls = '-', label = 'z=%.2f'%redshift)


    ax.set_xscale('log')
    ax.set_xlabel(r'$k [Mpc/h]$')
    ax.set_ylabel(r'$B(k)$')
    ax.legend()
    fig.show()
