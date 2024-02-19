########################################################
# Euclid photometric likelihood
########################################################
# written by Maike Doerenkamp in 2020
# following the recipe of 1910.09237 (Euclid preparation: VII. Forecast validation for Euclid
# cosmological probes)
# (adapted from euclid_lensing likelihood)
# edited by Lena Rathmann in 2021

from montepython.likelihood_class import Likelihood

from scipy.integrate import trapz, simpson
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import curve_fit, minimize

import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from montepython.MGfit_Winther import pofk_enhancement, pofk_enhancement_linear, kazuya_correktion
from montepython.forge_emulator.FORGE_emulator import FORGE
from time import time

try:
    import cosmopower as cp
    from cosmopower import cosmopower_NN
except:
    print("Please install the cosmopower package from https://github.com/alessiospuriomancini/cosmopower !")
    pass


try:
    from emantis import FofrBoost
except:
    print("Please install the emantis package from https://gitlab.obspm.fr/e-mantis/e-mantis !")
    pass


try:
    import BCemu
except:
    raise Exception ("Please install the BCemu package from https://github.com/sambit-giri/BCemu !")

import numpy as np
import warnings
from scipy.special import erf

# Found I needed to manually set this, otherwise cp is not recognised in the ReACT boost call
#os.environ["OMP_NUM_THREADS"] = '10'


class euclid_photometric_z_fofr(Likelihood):

    def __init__(self, path, data, command_line):
        self.debug_save  = False
        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z) and for k up to k_max
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': 1.5*self.k_max_h_by_Mpc})

        # Compute non-linear power spectrum if requested
        if (self.use_halofit):
            self.need_cosmo_arguments(data, {'non linear':'halofit'})
            #self.need_cosmo_arguments(data, {'non linear':'HMcode'})


        # Define array of l values, evenly spaced in logscale
        if self.lmax_WL > self.lmax_GC:
            self.l_WL = np.logspace(np.log10(self.lmin), np.log10(self.lmax_WL), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_WL >= self.lmax_GC)[0])
            self.l_GC = self.l_WL[:self.idx_lmax+1]
            self.l_XC = self.l_WL[:self.idx_lmax+1]
            self.l_array = 'WL'

            self.ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            self.ell_jump = self.lmax_GC - self.lmin +1
            self.ells_GC = self.ells_WL[:self.ell_jump]
            self.ells_XC = self.ells_GC
        else:
            self.l_GC = np.logspace(np.log10(self.lmin), np.log10(self.lmax_GC), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_GC >= self.lmax_WL)[0])
            self.l_WL = self.l_GC[:self.idx_lmax+1]
            self.l_XC = self.l_GC[:self.idx_lmax+1]
            self.l_array = 'GC'

            self.ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            self.ell_jump = self.lmax_WL - self.lmin +1
            self.ells_WL = self.ells_GC[:self.ell_jump]
            self.ells_XC = self.ells_WL

        if self.debug_save :
            np.savetxt('ls.txt',self.l_GC)

        #########################################
        # Find distribution of n(z) in each bin #
        #########################################

        # Create the array that will contain the z boundaries for each bin.

        self.z_bin_edge = np.array([self.zmin, 0.418, 0.560, 0.678, 0.789, 0.900, 1.019, 1.155, 1.324, 1.576, self.zmax])
        self.z_bin_center = np.array([(self.z_bin_edge[i]+self.z_bin_edge[i+1])/2 for i in range(self.nbin)])

        # Fill array of discrete z values
        self.z = np.linspace(self.zmin, self.zmax, num=self.nzmax)

        # Fill distribution for each bin (convolving with photo_z distribution)
        # n_i = int n(z) dz over bin
        self.eta_z = np.zeros((self.nzmax, self.nbin), 'float64')
        self.photoerror_z = np.zeros((self.nzmax, self.nbin), 'float64')
        for Bin in range(self.nbin):
            for nz in range(self.nzmax):
                z = self.z[nz]
                self.photoerror_z[nz,Bin] = self.photo_z_distribution(z,Bin+1)
                self.eta_z[nz, Bin] = self.photoerror_z[nz,Bin] * self.galaxy_distribution(z)
        if self.debug_save : np.savetxt('./photoz.txt',self.photoerror_z) ## agrees
        if self.debug_save : np.savetxt('./unnorm_nofz.txt',self.eta_z) ## agrees
        # integrate eta(z) over z (in view of normalizing it to one)
        self.eta_norm = np.zeros(self.nbin, 'float64')
        #norm = np.array([trapz([self.photo_z_distribution(z1, i+1) for z1 in zint],dx=dz) for i in range(self.nbin)])
        for Bin in range(self.nbin):
            #self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],dx=self.zmax/self.nzmax)
            self.eta_z[:,Bin] /= trapz(self.eta_z[:,Bin],self.z[:])

        if self.debug_save : np.savetxt('./n.txt',self.eta_z)
        # the normalised galaxy distribution per bin (dimensionless)
        #print('eta_z: ', self.eta_z)
        # the number density of galaxies per bin in inv sr
        self.n_bar = self.gal_per_sqarcmn * (60.*180./np.pi)**2
        self.n_bar /= self.nbin

        ###########################
        # Add nuisance parameters #
        ###########################

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            self.bias = np.zeros(self.nbin)
            self.bias_names = []
            for ibin in range(self.nbin):
                self.bias_names.append('bias_'+str(ibin+1))
            self.nuisance += self.bias_names

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            self.nuisance += ['aIA', 'etaIA', 'betaIA']

            # Read the file for the IA- contribution
            lum_file = open(os.path.join(self.data_directory,'scaledmeanlum_E2Sa.dat'), 'r')
            content = lum_file.readlines()
            zlum = np.zeros((len(content)))
            lum = np.zeros((len(content)))
            for index in range(len(content)):
                line = content[index]
                zlum[index] = line.split()[0]
                lum[index] = line.split()[1]
            self.lum_func = interp1d(zlum, lum,kind='linear')

        self.forge = None

        if self.use_BCemu or self.data_use_BCemu:
            self.nuisance += ['log10Mc']
            self.nuisance += ['thej']
            self.nuisance += ['deta']
            self.bfcemu = BCemu.BCM_7param(verbose=False)

        #############
        # Read data #
        #############

        # If the file exists, read the fiducial values
        self.fid_values_exist = False
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file+'.npz')
        if os.path.exists(fid_file_path):
            self.fid_values_exist = True
            fid_file = np.load(fid_file_path)
            if fid_file['probe'] != self.probe:
                warnings.warn("Probes in fiducial file does not match the probes asked for.\n The fiducial Probe is {} and the probe asked for is {}.\n Please procede with caution".format(fid_file['probe'],self.probe))
            try:
                if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                    l_WL = fid_file['ells_LL']
                    if not np.isclose(l_WL,self.l_WL).all():
                        raise Exception("Maximum multipole of WL has changed between fiducial and now.\n Fiducial lmax = {}, new lmax = {}. \n Please remove old fiducial and generate a new one".format(max(l_WL),max(self.l_WL)))
                    Cl_LL = fid_file['Cl_LL']
                    inter_LL = interp1d(l_WL,Cl_LL,axis=0, kind='cubic')(self.ells_WL)
                    self.Cov_observ = inter_LL

                if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                    l_GC = fid_file['ells_GG']
                    if not np.isclose(l_GC,self.l_GC).all():
                        raise Exception("Maximum multipole of GC has changed between fiducial and now.\n Fiducial lmax = {}, new lmax = {}. \n Please remove old fiducial and generate a new one".format(max(l_GC),max(self.l_GC)))
                    Cl_GG = fid_file['Cl_GG']
                    inter_GG = interp1d(l_GC,Cl_GG,axis=0, kind='cubic')(self.ells_GC)
                    self.Cov_observ = inter_GG

                if 'WL_GCph_XC' in self.probe:
                    l_XC = fid_file['ells_GL']
                    Cl_GL = fid_file['Cl_GL']
                    inter_GL = interp1d(l_XC,Cl_GL,axis=0, kind='cubic')(self.ells_XC)
                    inter_LG = np.transpose(inter_GL,(0,2,1))
                    if self.lmax_WL > self.lmax_GC:
                        self.Cov_observ = np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                        self.Cov_observ_high = inter_LL[self.ell_jump:,:,:]
                    else:
                        self.Cov_observ = np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                        self.Cov_observ_high = inter_GG[self.ell_jump:,:,:]

            except KeyError:
                raise KeyError("The probe asked for in the survey specifications is not in the fiducial file. \n Please remove old fiducial and generate a new one")
        else:
            if self.fit_different_data:
                self.use_fofR = self.data_use_fofR
                self.use_BCemu= self.data_use_BCemu

        ## different non linear models
        if self.use_fofR != False:
            self.nuisance += ['lgfR0']

            if self.use_fofR in ['Forge','Forge_corr']:
                self.forge = FORGE()
                self.forge_norm_Bk = None

            if self.use_fofR == 'ReACT':
                self.cp_nn = cosmopower_NN(restore=True,restore_filename='./montepython/react/react_boost_spph_nn_wide_100k_mt')

            if self.use_fofR == 'emantis':
                # extrapolate_aexp: extrapolates linearly in B-aexp for z > 2, making sure that B>=1 always.
                # This should be disabled and the extrapolation done outside of emantis in the same way as for the other predictions.

                # extrapolate_k: linear extrapolation in B-log10(k) for k < kmin=0.03, since sometimes the boost is not exactly equal to one for k=kmin.
                # The extrapolation is done until the boost reaches B=1.
                # self.emantis = FofrBoost(verbose=True, extrapolate_aexp=True, extrapolate_low_k=True)

                # the high z-extrapolation is now done in the likelihood code as a power law
                self.emantis = FofrBoost(verbose=True, extrapolate_aexp=False, extrapolate_low_k=True)

        return



    def galaxy_distribution(self, z):
        """
        Galaxy distribution returns the function D(z) from the notes

        Modified by S. Clesse in March 2016 to add an optional form of n(z) motivated by ground based exp. (Van Waerbeke et al., 2013)
        See google doc document prepared by the Euclid IST - Splinter 2
        """
        zmean = 0.9
        z0 = zmean/np.sqrt(2)

        galaxy_dist = (z/z0)**2*np.exp(-(z/z0)**(1.5))

        return galaxy_dist


    def photo_z_distribution(self, z, bin):
        """
        Photo z distribution

        z:      physical galaxy redshift
        zph:    measured galaxy redshift
        """
        c0, z0, sigma_0 = 1.0, 0.1, 0.05
        cb, zb, sigma_b = 1.0, 0.0, 0.05
        f_out = 0.1

        if bin == 0 or bin >= 11:
            return None

        term1 = cb*f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin - 1]))/(sigma_0*(1+z)))
        term2 =-cb*f_out*    erf((0.707107*(z-z0-c0*self.z_bin_edge[bin    ]))/(sigma_0*(1+z)))
        term3 = c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin - 1]))/(sigma_b*(1+z)))
        term4 =-c0*(1-f_out)*erf((0.707107*(z-zb-cb*self.z_bin_edge[bin    ]))/(sigma_b*(1+z)))
        return (term1+term2+term3+term4)/(2*c0*cb)

    def get_sigma8_fofR(self,k,Pk,h,lgfR0):
        """ Obtain the MG eqivalent of sigma8 calculated with the winther fitting formula and corrected for by matching MGCAMB

        Parameters
        ----------
        k : numpy.ndarray
            list of k values used in the interal calculations in 1/Mpc
        Pk: numpy.ndarray
            linear LCDM power spectrum at z=0 on the grid
        h : float
            Vaule of the reduced hubble constant
        lgfR0: float
            Absolute value of the log10 of the absolute value of f_R0

        Returns
        -------
        float
            MG eqivalent of sigma8
        """

        x = k*8/h
        #Get Sigma windowfunktion
        W = 3 /np.power(x,3)*(np.sin(x)-x*np.cos(x))
        for ix, xi in enumerate(x):
            if xi<0.01:
                W[ix]=1-xi**2/10

        #Get fofR boost to linear spectrum
        fR0 = np.power(10,-lgfR0)
        Boost = np.array([pofk_enhancement_linear(0,fR0,ki/h) for ki in k])

        Integr= np.power(k*W,2)*Pk*Boost/(2*np.pi**2)
        sigma8= np.sqrt(simpson( Integr, k)) * kazuya_correktion(lgfR0)

        return sigma8


    def loglkl(self, cosmo, data):

        printtimes = False
        if printtimes:
            t_start = time()

        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        self.r = np.zeros(self.nzmax, 'float64')
        self.dzdr = np.zeros(self.nzmax, 'float64')

        self.r, self.dzdr = cosmo.z_of_r(self.z)

        # H(z)/c in 1/Mpc
        H_z = self.dzdr
        # H_0/c in 1/Mpc
        H0 = cosmo.h()/2997.92458

        kmin_in_inv_Mpc = self.k_min_h_by_Mpc * cosmo.h()
        kmax_in_inv_Mpc = self.k_max_h_by_Mpc * cosmo.h()

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            # constant bias in each zbin, marginalise
            self.bias = np.zeros((self.nbin),'float64')
            if self.bias_model == 'binned_constant' :
                for ibin in range(self.nbin):
                    self.bias[ibin] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']

            elif self.bias_model == 'binned' :
                biaspars = dict()
                for ibin in range(self.nbin):
                    biaspars['b'+str(ibin+1)] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']
                brang = range(1,len(self.z_bin_edge))
                last_bin_num = brang[-1]
                def binbis(zz):
                    lowi = np.where( self.z_bin_edge <= zz )[0][-1]
                    if zz >= self.z_bin_edge[-1] and lowi == last_bin_num:
                        bii = biaspars['b'+str(last_bin_num)]
                    else:
                        bii = biaspars['b'+str(lowi+1)]
                    return bii
                vbinbis = np.vectorize(binbis)
                self.biasfunc = vbinbis

            elif self.bias_model == 'interpld' :
                for ibin in range(self.nbin):
                    self.bias[ibin] = data.mcmc_parameters[self.bias_names[ibin]]['current']*data.mcmc_parameters[self.bias_names[ibin]]['scale']
                self.biasfunc = interp1d(self.z_bin_center, self.bias, bounds_error=False, fill_value="extrapolate")

        if printtimes:
            t_init = time()
            print("Initalisation Time:", t_init-t_start)

        ######################
        # Get power spectrum #
        ######################
        # [P(k)] = Mpc^3
        # Get power spectrum P(k=(l+1/2)/r,z) from cosmological module

        #Get Power-spectrum on class grid
        Pk_m_nl_grid, k_grid, z= cosmo.get_pk_and_k_and_z()
        Pk_m_l_grid, _, _=       cosmo.get_pk_and_k_and_z(nonlinear=False)

        #Get Power spectrum interpolators, re-order z to be striktly rising
        Pk_nl = RectBivariateSpline(z[::-1],k_grid,(np.flip(Pk_m_nl_grid,axis=1)).transpose())
        Pk_l  = RectBivariateSpline(z[::-1],k_grid,(np.flip(Pk_m_l_grid,axis=1)).transpose())

        if self.l_array == 'WL':
            l = self.l_WL
        if self.l_array == 'GC':
            l = self.l_GC
        k =(l[:,None]+0.5)/self.r

        #Find Values of k for which the powerspectrum is not null
        pk_m_nl = np.zeros((self.lbin, self.nzmax), 'float64')
        pk_m_l  = np.zeros((self.lbin, self.nzmax), 'float64')
        index_pknn = np.array(np.where((k> kmin_in_inv_Mpc) & (k<kmax_in_inv_Mpc))).transpose()

        for index_l, index_z in index_pknn:
            pk_m_nl[index_l, index_z] = Pk_nl(self.z[index_z],k[index_l,index_z])
            pk_m_l [index_l, index_z] = Pk_l (self.z[index_z],k[index_l,index_z])

        Pk = pk_m_nl

        if printtimes:
            t_power = time()
            print("Power Spectrum obtained in:", t_power-t_init)

        ########################
        # Boosts and Emulators #
        ########################

        if self.use_fofR == 'Winther':
            print("f(R) active with Winther fitting function")
            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)

            fofR_zmax = 2
            fofR_kmax = 10*cosmo.h() #In 1/Mpc
            fofR_boost = lambda k_l, z_l: pofk_enhancement(z_l, f_R0, k_l / cosmo.h(), hasBug = self.use_bug)

        elif self.use_fofR in ['Forge','Forge_corr']:
            print("f(R) active with Forge emulator")
            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)

            z_max = 2.0
            redshifts = self.z[self.z <= z_max]

            Bk = []

            Omc = cosmo.Omega0_cdm()
            Omb = cosmo.Omega_b()
            #Adding massive neutrinos for concistancy
            Omnu = cosmo.Omega_nu

            hubble = cosmo.h()
            pars_dict={'sigma8': cosmo.sigma8(),
                       'h': hubble,
                       'Omega_m': Omc+Omb+Omnu}

            forge_bounds={'Omega_m': [0.18, 0.55],
                          'sigma8': [0.6, 1.0],
                          'h': [0.6, 0.8]}

            if (lgfR0 > 6.2 or lgfR0 < 4.5):

                raise Exception("Outside of FORGE range")

            else:
                # Replace cosmo params by the values at the end of the range,
                # if necessary to extrapolate.
                for key in pars_dict.keys():
                    if (pars_dict[key] > forge_bounds[key][1]):
                        pars_dict[key] = forge_bounds[key][1]
                    elif (pars_dict[key] < forge_bounds[key][0]):
                        pars_dict[key] = forge_bounds[key][0]

                for redshift in redshifts:
                    B_f, _ = self.forge.predict_Bk(redshift,
                                                    pars_dict['Omega_m'],
                                                    pars_dict['h'],
                                                    lgfR0,
                                                    pars_dict['sigma8'])
                    Bk.append(B_f)

            Bk = np.asarray(Bk)
            k_forge = self.forge.k

            if self.use_fofR == 'Forge_corr':
                if self.forge_norm_Bk is None:

                    self.forge_norm_Bk = self.forge_norm()

                Bk = Bk / self.forge_norm_Bk

            Bk_interp = RectBivariateSpline(redshifts,k_forge,Bk)

            fofR_zmax = 2
            fofR_kmax = 10*cosmo.h() #In 1/Mpc
            fofR_boost = lambda k_l, z_l: Bk_interp(z_l, k_l / cosmo.h())


        if self.use_fofR == 'ReACT':
            # Halo model reaction based boost. Emulator based on output from ReACT and HMCode2020 (for pseudo and LCDM)
            # Emulator range given below. It outputs in [0.01,3] h/Mpc
            # Includes massive neutrinos which are here set to 0 manually

            # TODO:
            # Optimise clipping and setting boost = 1 for z>zmax
            # Extrapolate to small k? Perhaps not necessary as boost should be 1 at kmin = 0.01h/Mpc ....

            print("f(R) active with ReACT")

            # Set parameters for the emulator
            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)

            hubble = cosmo.h()
            Omc = cosmo.Omega0_cdm()
            Omb = cosmo.Omega_b()
            Omnu = cosmo.Omega_nu

            # Emulator parameters
            Om = (Omc + Omb + Omnu)  # Choosing total matter to include neutrino fraction
            Ob = Omb
            Onu = 0.0 # set omega_nu = 0 in the reaction

            primordial = cosmo.get_current_derived_parameters(['A_s','n_s'])
            myAs = primordial['A_s']
            myns = primordial['n_s']
            myH0 = hubble*100

            # Emulator max redshift
            zmax = 2;

            # Emulator parameter ranges
            react_bounds = {'Omega_m' : [0.24 , 0.35],
                            'Omega_b' : [0.04 , 0.06],
                            'H0' : [63.0 , 75.0],
                            'ns' : [0.9 , 1.01],
                            'Omega_nu' : [0.0, 0.00317],
                            'As' : [1.7e-9 , 2.5e-9],
                            'fR0' : [1e-10 , 1e-4],
                            'z' : [0 , zmax]}

            # Get boost, interpolate and extrapolate:

            # Set up dictionary with input parameters for boost over all redshifts
            nz_Pk = len(self.z)
            params_cp = {'Omega_m': Om*np.ones(nz_Pk),
                         'Omega_b': Ob*np.ones(nz_Pk),
                         'H0': myH0*np.ones(nz_Pk),
                         'ns': myns*np.ones(nz_Pk),
                         'Omega_nu': Onu*np.ones(nz_Pk),
                         'As': myAs*np.ones(nz_Pk),
                         'fR0': f_R0*np.ones(nz_Pk),
                         'z': self.z }


            # Replace cosmo params by values at end of range if outside range
            # OPTIMISE: clip proceeds elementwise, so this is 8 x nzPk comparisons .... ?
            for key in params_cp.keys():
                params_cp[key] = np.clip(params_cp[key], react_bounds[key][0], react_bounds[key][1])


            # Set all z>2 to have fr0 = 0 (i.e boost =1)
            # OPTIMISE: There is probably a better way to do this ...
            for index_z in range(nz_Pk):
                if(self.z[index_z]>=zmax):
                    params_cp['fR0'][index_z] = 1e-10

            # Calculate the boost at all redshifts create 2d spline in (k,z)

            # Ensure boost is always greater or equal to 1
            Boost = np.maximum(1.0,self.cp_nn.predictions_np(params_cp))
            kvals = self.cp_nn.modes

            ReACT_interp = RectBivariateSpline(self.z,kvals,Boost)
            fofR_zmax = 2
            fofR_kmax = 3*cosmo.h() #In 1/Mpc
            fofR_boost = lambda k_l, z_l: ReACT_interp(z_l, k_l / cosmo.h())

        elif self.use_fofR == 'emantis':
            print("f(R) active with e-MANTIS emulator")

            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)

            Omc = cosmo.Omega0_cdm()
            Omb = cosmo.Omega_b()
            #Adding Massive neutrinos for consistency
            Omnu = cosmo.Omega_nu
            Omm = Omc + Omb + Omnu
            hubble = cosmo.h()

            # Doing constant extrapolation outside of emmulaor bounds
            emantis_bounds = {'Omegam' : [0.24,0.39],
                              'sigma8' : [0.60,1.00]}
            emantis_Omm = np.clip(Omm,emantis_bounds['Omegam'][0],emantis_bounds['Omegam'][1])
            emantis_s8  = np.clip(cosmo.sigma8(),emantis_bounds['sigma8'][0],emantis_bounds['sigma8'][1])

            # Get e-mantis predictions for all z and k within emulator bounds
            fofR_kmax =  self.emantis.kbins[-1]*hubble
            fofR_zmax = 2

            k_grid_emantis = np.geomspace(0.01,fofR_kmax,100)
            z_grid_emantis = self.z[self.z <= fofR_zmax]

            # k indices within emantis range (k<kmax)
            B_grid_emantis = self.emantis.predict_boost(emantis_Omm, emantis_s8, lgfR0, 1/(1+z_grid_emantis), k_grid_emantis/hubble)
            interp_Boost = RectBivariateSpline(z_grid_emantis,k_grid_emantis,B_grid_emantis[:,0,:])

            fofR_boost = lambda k, z: interp_Boost(z,k)

        #Obtain derived quantity
        if self.use_fofR != False:
            if 'sigma8_fofR' in data.get_mcmc_parameters(['derived_lkl']):
                data.derived_lkl={'sigma8_fofR':self.get_sigma8_fofR(k_grid,Pk_m_l_grid[:,-1],cosmo.h(),lgfR0)}

        if printtimes:
            t_boost = time()
            print("Boost Initialized in:", t_boost-t_power)

        ##########################################
        # Extrapolation for the Different boosts #
        ##########################################
        """
        The idea here is to fit a power law curve to the edges of the generic function giving us the boost.
        This is done by calculateing the boost on a grid close to the edges and then fitting a line.
        Interpolating the line gives us the power law everywhere outside the boost edges
        To be safe the boost is caped at 2
        """

        if self.use_fofR != False:
            # Setup the grid on which we will do the fitting
            k_cut = np.geomspace(0.8*fofR_kmax,fofR_kmax,5)
            z_cut = np.linspace(0.8*fofR_zmax,fofR_zmax ,5)
            k_long= np.geomspace(0.01,fofR_kmax,5*self.fofR_interpolation_k_boost)
            z_long= np.linspace(0,fofR_zmax,5*self.fofR_interpolation_z_boost)

            # linear extrapolation in log log
            logk_cut= np.log(k_cut)
            logz_cut= np.log(z_cut)
            # k extrapolation -> grid : k_cut, z_long
            logBoostk = np.ones((5,5*self.fofR_interpolation_z_boost))
            k_power = []
            for iz, zi in enumerate(z_long):
                for ik, ki in enumerate(k_cut):
                    logBoostk[ik,iz] = np.log(fofR_boost(ki,zi))
                #fix boost at highest k to emulator value
                popt, _= curve_fit(lambda x,gamma : logBoostk[-1,iz]+gamma*(x-logk_cut[-1]),logk_cut,logBoostk[:,iz])
                k_power.append(popt[0])
            gammak_z =interp1d(z_long,k_power)

            # z extrapolation -> grid : k_long, z_cut
            logBoostz = np.ones((5*self.fofR_interpolation_k_boost,5))
            z_power = []
            for ik, ki in enumerate(k_long):
                for iz, zi in enumerate(z_cut):
                    logBoostz[ik,iz] = np.log(fofR_boost(ki,zi))
                #fix boost at highest z to emulator value
                popt, _= curve_fit(lambda x,gamma : logBoostz[ik,-1]+gamma*(x-logz_cut[-1]),logz_cut,logBoostz[ik,:])
                z_power.append(popt[0])
            gammaz_k =interp1d(k_long,z_power)

            # Of course I still love him
            def Boost_extrapolation(k,z):
                    return np.clip(fofR_boost(np.clip(k,0.01,fofR_kmax),np.minimum(z,fofR_zmax))*(np.maximum(z,fofR_zmax)/fofR_zmax)**gammaz_k(np.clip(k,0.01,fofR_kmax))*(np.maximum(k,fofR_kmax)/fofR_kmax)**gammak_z(np.minimum(z,fofR_zmax)),1,2)

            boost_m_nl_fofR = np.ones_like(k)
            for index_z, z_value in enumerate(self.z):
                pknn_mask = np.where((k[:,index_z]>kmin_in_inv_Mpc) & (k[:,index_z]<kmax_in_inv_Mpc))
                boost_m_nl_fofR[pknn_mask,index_z] = Boost_extrapolation(k[pknn_mask,index_z],z_value)
            Pk *= boost_m_nl_fofR

        if printtimes:
            t_extra = time()
            print("extrapolation done in:", t_extra - t_boost)

        if self.use_BCemu:
            # baryonic feedback modifications are only applied to k>kmin_bfc
            # it is very computationally expensive to call BCemu at every z in self.z, and it is a very smooth function with z,
            # so it is only called at self.BCemu_k_bins points in k and self.BCemu_z_bins points in z and then the result is
            # splined over all z in self.z. For k>kmax_bfc = 12.5 h/Mpc, the maximum k the emulator is trained on, a constant
            # suppression in k is assumed: BFC(k,z) = BFC(12.5 h/Mpc, z).

            log10Mc = data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale']
            thej = data.mcmc_parameters['thej']['current'] * data.mcmc_parameters['thej']['scale']
            deta = data.mcmc_parameters['deta']['current'] * data.mcmc_parameters['deta']['scale']
            nu_log10Mc = 0
            nu_thej = 0
            nu_deta = 0

            bcemu_dict ={
            'log10Mc' : log10Mc,
            'nu_Mc'   : nu_log10Mc,
            'mu'      : 1.0,
            'nu_mu'   : 0.0,
            'thej'    : thej,
            'nu_thej' : nu_thej,
            'gamma'   : 2.5,
            'nu_gamma': 0.0,
            'delta'   : 7.0,
            'nu_delta': 0.0,
            'eta'     : 0.2,
            'nu_eta'  : 0.0,
            'deta'    : deta,
            'nu_deta' : nu_deta
            }

            Ob = cosmo.Omega_b()
            Om = cosmo.Omega_m()

            fb = Ob/Om
            if fb < 0.1 or fb > 0.25:
                if self.verbose: print(" /!\ Skipping point because the baryon fraction is out of bounds!")
                return -1e10

            if log10Mc / 3**nu_log10Mc < 11 or log10Mc / 3**nu_log10Mc > 15 :
                if self.verbose: print(" /!\ Skipping point because BF parameters are out of bounds!")
                return -1e10

            if thej / 3**nu_thej < 2 or thej / 3**nu_thej > 8 :
                if self.verbose: print(" /!\ Skipping point because BF parameters are out of bounds!")
                return -1e10

            if deta / 3**nu_deta < 0.05 or deta / 3**nu_deta > 0.4 :
                if self.verbose: print(" /!\ Skipping point because BF parameters are out of bounds!")
                return -1e10

            kmin_in_inv_Mpc = self.k_min_h_by_Mpc * cosmo.h()
            kmin_bfc = 0.035
            kmax_bfc = 12.5
            k_bfc = np.logspace(np.log10(max(kmin_bfc, self.k_min_h_by_Mpc)), np.log10(min(kmax_bfc, self.k_max_h_by_Mpc)), self.BCemu_k_bins)
            # ^ all have units h/Mpc

            z_bfc = np.linspace(self.z[0], min(2, self.z[-1]), self.BCemu_z_bins)
            BFC = np.zeros((self.BCemu_k_bins, self.BCemu_z_bins))

            for index_z, z in enumerate(z_bfc):
                BFC[:,index_z] = self.bfcemu.get_boost(z,bcemu_dict,k_bfc,fb)

            BFC_interpolator = RectBivariateSpline(k_bfc*cosmo.h(), z_bfc, BFC)

            boost_m_nl_BCemu = np.zeros((self.lbin, self.nzmax), 'float64')

            for index_z, z in enumerate(self.z):
                boost_m_nl_BCemu[:, index_z] = BFC_interpolator(np.minimum(k[:,index_z],12.5*cosmo.h()),min(z, 2))[:,0]

            Pk *= boost_m_nl_BCemu

        if printtimes:
            t_baryon = time()
            print("Baryonic Feedback calculated in:", t_baryon-t_extra)

        ####################
        # Get Growthfactor #
        ####################
        if self.scale_dependent_f == False:
            D_z= np.zeros((self.nzmax), 'float64')
            for index_z, zz in enumerate(self.z):
                D_z[index_z] = cosmo.scale_independent_growth_factor(zz)
            D_z= D_z[None,:]

        elif self.scale_dependent_f ==True:
            D_z =np.ones((self.lbin,self.nzmax), 'float64')
            for index_l, index_z in index_pknn:
                D_z[index_l,index_z] = np.sqrt(Pk_l(self.z[index_z],k[index_l,index_z])/Pk_l(0,k[index_l,index_z]))

        if self.use_fofR and self.use_MGGrowth:
            D_z_boost = np.ones((self.lbin,self.nzmax), 'float64')
            for index_l, index_z in index_pknn:
                D_z_boost[index_l,index_z] = np.sqrt(pofk_enhancement_linear(self.z[index_z],f_R0,k[index_l,index_z]/cosmo.h()) /\
                                                     pofk_enhancement_linear(              0,f_R0,k[index_l,index_z]/cosmo.h()))
            D_z  *=  D_z_boost

        if printtimes:
            t_growth = time()
            print("growthfactor obtained in", t_growth-t_baryon)

        ################################################
        # Window functions W_L(l,z,bin) and W_G(z,bin) #
        ################################################
        # in units of [W] = 1/Mpc

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:

            integral = 3./2.*H0**2. *cosmo.Omega_m()*self.r[None,:,None]*(1.+self.z[None,:,None])*self.eta_z.T[:,None,:]*(1-self.r[None,:,None]/self.r[None,None,:])
            W_gamma  = np.trapz(np.triu(integral),self.z,axis=-1).T

            # Compute contribution from IA (Intrinsic Alignement)

            # - compute window function W_IA
            W_IA = self.eta_z *H_z[:,None]

            # - IA contribution depends on a few parameters assigned here
            C_IA = 0.0134
            A_IA = data.mcmc_parameters['aIA']['current']*(data.mcmc_parameters['aIA']['scale'])
            eta_IA = data.mcmc_parameters['etaIA']['current']*(data.mcmc_parameters['etaIA']['scale'])
            beta_IA = data.mcmc_parameters['betaIA']['current']*(data.mcmc_parameters['betaIA']['scale'])

            # - compute functions F_IA(z) and D(z)
            F_IA = (1.+self.z)**eta_IA * (self.lum_func(self.z))**beta_IA

            W_L = W_gamma[None,:,:] - A_IA*C_IA*cosmo.Omega_m()*F_IA[None,:,None]/D_z[:,:,None] *W_IA[None,:,:]

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            # Compute window function W_G(z) of galaxy clustering for each bin:

            # - case where there is one constant bias value b_i for each bin i
            if self.bias_model == 'binned_constant' :
                W_G = np.zeros((self.nzmax, self.nbin), 'float64')
                W_G = self.bias[None,:] * H_z[:,None] * self.eta_z
            # - case where the bias is a single function b(z) for all bins
            if self.bias_model == 'binned' or self.bias_model == 'interpld':
                W_G = np.zeros((self.nzmax, self.nbin), 'float64')
                W_G =  (H_z * self.biasfunc(self.z))[:,None] * self.eta_z

        if printtimes:
            t_window = time()
            print("window function obtained in:", t_window-t_growth)

        ###########
        # Calc Cl #
        ###########
        # dimensionless

        nell_WL = len(self.l_WL)
        nell_GC = len(self.l_GC)
        nell_XC = len(self.l_XC)

        # the indices are ell, z, bin_i, bin_j in the int and ell, bin_i, bin_j in the Ceeell
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None]
            Cl_LL     = trapz(Cl_LL_int,self.z,axis=1)[:nell_WL,:,:]

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_GG_int = W_G[None,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None]
            Cl_GG     = trapz(Cl_GG_int,self.z,axis=1)[:nell_GC,:,:]

        if 'WL_GCph_XC' in self.probe:
            Cl_LG_int = W_L[:,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None]
            Cl_LG     = trapz(Cl_LG_int,self.z,axis=1)[:nell_XC,:,:]
            Cl_GL     = np.transpose(Cl_LG,(0,2,1))

        if printtimes:
            t_cell = time()
            print("Cell calculated in:", t_cell-t_window)

        ####################
        # Plot Pk and Cl's #
        ####################

        Plot_debug = False
        if Plot_debug == True:
            print("For debug, returning Cls")
            return k, self.z, Pk, Cl_LL, Cl_GG, Cl_LG, Cl_GL

        # do you want to save the powerspectrum?
        Plot_debug = False
        if Plot_debug == True:
            debug_file_path = os.path.join( self.data_directory, 'euclid_photometric_Pkz.npz')
            # loading the file yields P(k,z), k(ell,z), z
            np.savez(debug_file_path,Pkz=Pk,k=k,z=self.z)
            print("Printed P(k,z)")

        # do you want to save the Ceeell?
        Plot_debug = False
        if Plot_debug == True:
            debug_file_path = os.path.join(self.data_directory, 'euclid_photometric_Cls.npz')
            # loading the file yields 3-D matrix Cl_XY[ell,bin i,bin j] and ells_XY
            if 'WL_GCph_XC' in self.probe:
                np.savez(debug_file_path, ells_LL=self.l_WL, ells_GG=self.l_GC, ells_GL=self.l_XC, Cl_LL = Cl_LL, Cl_GG = Cl_GG, Cl_GL = Cl_GL)
            if 'WL' in self.probe:
                np.savez(debug_file_path, ells_LL=self.l_WL, Cl_LL = Cl_LL)
            if 'GCph' in self.probe:
                np.savez(debug_file_path, ells_GG=self.l_GC, Cl_GG = Cl_GG)

        if printtimes:
            t_debug = time()
            print("Debug options obtained in:" , t_debug-t_cell)

        #########
        # Noise #
        #########
        # dimensionless

        self.noise = {
           'LL': self.rms_shear**2./self.n_bar,
           'LG': 0.,
           'GL': 0.,
           'GG': 1./self.n_bar}

        # add noise to Ceeells after saving to better compare
        for i in range(self.nbin):
            if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                Cl_LL[:,i,i] += self.noise['LL']
            if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                Cl_GG[:,i,i] += self.noise['GG']
            if 'WL_GCph_XC' in self.probe:
                Cl_GL[:,i,i] += self.noise['GL']
                Cl_LG[:,i,i] += self.noise['LG']

        #####################
        # Theoretical Error #
        #####################
        """If the inclusion of theoretical errors for modified gravity are asked for will follow the description found in 1210.2194.
        The recepie has been adjusted to also work with the full 3x2pt probe.
        Any other source of theorerical error must be added seperately in the computation of the relative theoretical error.
        """
        if self.theoretical_error != False:

            #calculate the relative theoretical error
            alpha = np.zeros_like(k)
            if self.use_fofR != False and self.theoretical_error == 'simple_fR':
                # In the simple case we set the relative theoretical error to be 2% of the boost
                alpha = boost_m_nl_fofR*self.fR_error

            elif self.use_fofR != False and self.theoretical_error == 'react_cast':
                # obtained fits from ratio of ReACT with Forge at HS6.
                def A_plateau(z):
                    Delta = 0.0356204
                    Base = 0.029017
                    Temperature = 0.06154762
                    potential = 0.56233084
                    return Delta / (np.exp((z-potential) / Temperature)+1)+Base
                
                def k_plateau(z):
                    kinf = 0.20253502 
                    pos = 1.55716999 
                    sigma = 0.55169064
                    kinf*np.exp(np.tanh((z-pos)/sigma))

                def smoothish_step(x):
                    return (np.power(x,2)+x)/(np.power(x,2)+x+1)
                
                for index_z, z_value in enumerate(self.z):
                    pknn_mask = np.where((k[:,index_z]>kmin_in_inv_Mpc) & (k[:,index_z]<kmax_in_inv_Mpc))
                    alpha[pknn_mask,index_z] = A_plateau(z_value) * smoothish_step(k[pknn_mask,index_z]/k_plateau(z_value))                

            else:
                raise Exception("Theorerical error model not recognized. Please choose from 'simple_fR' or ...")

            # calculate the covariance matrix error
            if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                El_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None] * alpha[:,:,None,None]
                El_LL     = trapz(El_LL_int,self.z,axis=1)[:nell_WL,:,:]

            if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                El_GG_int = W_G[None,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None] * alpha[:,:,None,None]
                El_GG     = trapz(El_GG_int,self.z,axis=1)[:nell_GC,:,:]

            if 'WL_GCph_XC' in self.probe:
                El_LG_int = W_L[:,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None] * alpha[:,:,None,None]
                El_LG     = trapz(El_LG_int,self.z,axis=1)[:nell_XC,:,:]
                El_GL     = np.transpose(El_LG,(0,2,1))

        if printtimes:
            t_therr = time()
            print("Theoretical errors obtained in:" , t_therr-t_debug)

        #######################
        # Create fiducial file
        #######################

        if self.fid_values_exist is False:
            # Store the values now, and exit.
            fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
            fiducial_cosmo = dict()
            for key, value in data.mcmc_parameters.items():
                    fiducial_cosmo[key] = value['current']*value['scale']

            if 'WL_GCph_XC' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_LL=self.l_WL, ells_GG=self.l_GC, ells_GL=self.l_XC, Cl_LL = Cl_LL, Cl_GG = Cl_GG, Cl_GL = Cl_GL)
            if 'WL' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_LL=self.l_WL, Cl_LL = Cl_LL)
            if 'GCph' in self.probe:
                np.savez(fid_file_path,fid_cosmo=fiducial_cosmo, probe=self.probe, ells_GG=self.l_GC, Cl_GG = Cl_GG)

            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        #############
        # Spline Cl #
        #############
        # Find C(l) for every integer l
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            inter_LL = interp1d(self.l_WL,Cl_LL,axis=0, kind='cubic')(self.ells_WL)
            Cov_theory = inter_LL
            ells = self.ells_WL
        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            inter_GG = interp1d(self.l_GC,Cl_GG,axis=0, kind='cubic')(self.ells_GC)
            Cov_theory = inter_GG
            ells = self.ells_GC
        if 'WL_GCph_XC' in self.probe:
            inter_GL = interp1d(self.l_XC,Cl_GL,axis=0, kind='cubic')(self.ells_XC)
            inter_LG = np.transpose(inter_GL,(0,2,1))
            if self.lmax_WL > self.lmax_GC:
                Cov_theory = np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                Cov_theory_high = inter_LL[self.ell_jump:,:,:]
                ells = self.ells_WL
            else:
                Cov_theory = np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                Cov_theory_high = inter_GG[self.ell_jump:,:,:]
                ells = self.ells_GC

        T_Rerr = np.zeros_like(Cov_theory)
        T_Rerr_high = np.zeros_like(Cov_theory_high)
        if self.theoretical_error != False:
            # Find the theoretical error matrix for every interger l
            if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
                inter_LL = interp1d(self.l_WL,El_LL,axis=0, kind='cubic')(self.ells_WL)
                norm = np.sqrt(self.lmax_WL - self.lmin + 1)
                T_Rerr =  norm * inter_LL
            if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
                inter_GG = interp1d(self.l_GC,El_GG,axis=0, kind='cubic')(self.ells_GC)
                norm = np.sqrt(self.lmax_GC - self.lmin + 1)
                T_Rerr =  norm * inter_GG
            if 'WL_GCph_XC' in self.probe:
                inter_GL = interp1d(self.l_XC,El_GL,axis=0, kind='cubic')(self.ells_XC)
                inter_LG = np.transpose(inter_GL,(0,2,1))
                norm = np.sqrt(np.maximum(self.lmax_WL,self.lmax_GC) - self.lmin + 1)
                if self.lmax_WL > self.lmax_GC:
                    T_Rerr = norm * np.block([[inter_LL[:self.ell_jump,:,:],inter_LG],[inter_GL,inter_GG]])
                    T_Rerr_high = norm * inter_LL[self.ell_jump:,:,:]
                else:
                    T_Rerr = norm * np.block([[inter_LL,inter_LG],[inter_GL,inter_GG[:self.ell_jump,:,:]]])
                    T_Rerr_high = norm * inter_GG[self.ell_jump:,:,:]

        if printtimes:
            t_spline = time()
            print("Cell Splined in:", t_spline-t_therr)

        ######################
        # Compute likelihood
        ######################

        if 'WL' in self.probe or 'GCph' in self.probe:
            compute_chiq = lambda eps: self.compute_chiq(eps, ells, self.Cov_observ, Cov_theory, T_Rerr)
            jac = lambda eps: self.jac(eps, ells, self.Cov_observ, Cov_theory, T_Rerr)

        elif 'WL_GCph_XC' in self.probe:
            compute_chiq = lambda eps: self.compute_chiq(eps, ells, self.Cov_observ, Cov_theory, T_Rerr, self.Cov_observ_high, Cov_theory_high, T_Rerr_high)
            jac = lambda eps: self.jac(eps, ells, self.Cov_observ, Cov_theory, T_Rerr, self.Cov_observ_high, Cov_theory_high, T_Rerr_high)

        eps_l = np.zeros_like(ells)
        if self.theoretical_error != False:
            do_binned = True
            if do_binned:
                ells_binned = np.unique(np.geomspace(self.lmin,np.maximum(self.lmax_WL, self.lmax_GC), self.lbin, dtype = np.uint64))
                index_low = ells_binned[np.where(ells_binned < self.ell_jump)] - self.lmin
                index_high = ells_binned[np.where(ells_binned >= self.ell_jump)] - self.ell_jump - self.lmin
                eps_binned = np.zeros_like(ells_binned)
                if 'WL' in self.probe or 'GCph' in self.probe:
                    Cov_obs_binned = self.Cov_observ[index_low]
                    Cov_the_binned = Cov_theory[index_low]
                    T_Rerr_binned = T_Rerr[index_low]

                    chisq_binned = lambda eps : self.compute_chiq(eps, ells_binned, Cov_obs_binned, Cov_the_binned, T_Rerr_binned)
                    jac_binned = lambda eps : self.jac(eps, ells_binned, Cov_obs_binned, Cov_the_binned, T_Rerr_binned)
                elif 'WL_GCph_XC' in self.probe:
                    Cov_obs_binned = self.Cov_observ[index_low]
                    Cov_the_binned = Cov_theory[index_low]
                    T_Rerr_binned = T_Rerr[index_low]
                    Cov_obs_binned_high = self.Cov_observ_high[index_high]
                    Cov_the_binned_high = Cov_theory_high[index_high]
                    T_Rerr_binned_high = T_Rerr_high[index_high]

                    chisq_binned = lambda eps : self.compute_chiq(eps, ells_binned, Cov_obs_binned, Cov_the_binned, T_Rerr_binned, Cov_obs_binned_high, Cov_the_binned_high, T_Rerr_binned_high)
                    jac_binned = lambda eps : self.jac(eps, ells_binned, Cov_obs_binned, Cov_the_binned, T_Rerr_binned, Cov_obs_binned_high, Cov_the_binned_high, T_Rerr_binned_high)
                
                res = minimize(chisq_binned, eps_binned, tol=1e-2, method='Newton-CG',jac=jac_binned, hess='3-point')
                eps_binned = res.x
                eps_l = interp1d(ells_binned, eps_binned, kind='cubic')(ells)
            else:
                res = minimize(compute_chiq, eps_l, tol=1e-2, method='Newton-CG',jac=jac, hess='3-point')
                eps_l = res.x

        if printtimes:
            t_lkl = time()
            print("Likelihood calculated in:" ,t_lkl-t_spline)
            print("Total time taken:", t_lkl-t_start)

        print("euclid photometric: chi2 = ",chi2)
        return -chi2/2.

    # Comopute the log likelihood for a given set of multipoles
    def compute_chiq(self, eps_l, ells, Cov_observ, Cov_theory, T_Rerr, Cov_observ_high=None, Cov_theory_high=None, T_Rerr_high=None):

        # find the  #redshift bins X #probes from the covariance matrix
        nbin = Cov_observ.shape[1]
        # find the multipole with the jump
        ell_jump = Cov_observ.shape[0]

        # calculate the necesarry determinants
        shifted_Cov = Cov_theory + eps_l[:ell_jump, None, None] * T_Rerr
        dtilde_the = np.linalg.det(shifted_Cov)

        d_obs = np.linalg.det(Cov_observ)

        dtilde_mix = np.zeros_like(dtilde_the)
        for i in range(nbin):
            newCov = np.copy(shifted_Cov)
            newCov[:, i] = Cov_observ[:, :, i]
            dtilde_mix += np.linalg.det(newCov)

        N = np.ones_like(ells) * nbin

        # if the probe is 3x2pt calculate the part with no cross correlation
        if "WL_GCph_XC" in self.probe:

            nbin = Cov_observ_high.shape[1]

            shifted_Cov_high = Cov_theory_high + eps_l[ell_jump:, None, None] * T_Rerr_high
            dtilde_the_high = np.linalg.det(shifted_Cov_high)

            d_obs_high = np.linalg.det(Cov_observ_high)

            dtilde_mix_high = np.zeros_like(dtilde_the_high)
            for i in range(nbin):
                newCov = np.copy(shifted_Cov_high)
                newCov[:, i] = Cov_observ_high[:, :, i]
                dtilde_mix_high += np.linalg.det(newCov)

            N[ell_jump:] = nbin
            dtilde_the = np.concatenate([dtilde_the, dtilde_the_high])
            d_obs = np.concatenate([d_obs, d_obs_high])
            dtilde_mix = np.concatenate([dtilde_mix, dtilde_mix_high])


        return np.sum((2 * ells + 1) * self.fsky * ((dtilde_mix / dtilde_the) + np.log(dtilde_the / d_obs) - N) + np.power(eps_l, 2))

    # compute the jacobian with respect to the epsilons for the minimization
    def jac(self, eps_l, ells, Cov_observ, Cov_theory, T_Rerr, Cov_observ_high=None, Cov_theory_high=None, T_Rerr_high=None):

        # find the  #redshift bins X #probes from the covariance matrix
        nbin = Cov_observ.shape[1]
        # find the multipole with the jump
        ell_jump = Cov_observ.shape[0]

        shifted_Cov = Cov_theory + eps_l[:ell_jump, None, None] * T_Rerr
        dtilde_the = np.linalg.det(shifted_Cov)
        inv_shifted_Cov = np.linalg.inv(shifted_Cov)
        dprime_the = dtilde_the * np.trace(
            np.matmul(inv_shifted_Cov, T_Rerr), axis1=1, axis2=2
        )

        d_obs = np.linalg.det(Cov_observ)

        dtilde_mix = np.zeros_like(dtilde_the)
        dprime_mix = np.zeros_like(dtilde_the)
        for i in range(nbin):
            newCov = np.copy(shifted_Cov)
            newCov[:, i] = Cov_observ[:, :, i]
            dnewCov = np.linalg.det(newCov)
            dtilde_mix += dnewCov

            newCovprime = np.copy(T_Rerr)
            newCovprime[:, i] = 0
            inv_newCov = np.linalg.inv(newCov)
            dprime_mix += dnewCov * np.trace(
                np.matmul(inv_newCov, newCovprime), axis1=1, axis2=2
            )

        # if the probe is 3x2pt calculate the part with no cross correlation
        if "WL_GCph_XC" in self.probe:

            nbin = Cov_observ_high.shape[1]

            shifted_Cov_high = Cov_theory_high + eps_l[ell_jump:, None, None] * T_Rerr_high
            dtilde_the_high = np.linalg.det(shifted_Cov_high)
            inv_shifted_Cov_high = np.linalg.inv(shifted_Cov_high)
            dprime_the_high = dtilde_the_high * np.trace(
                np.matmul(inv_shifted_Cov_high, T_Rerr_high), axis1=1, axis2=2
            )

            d_obs_high = np.linalg.det(Cov_observ_high)

            dtilde_mix_high = np.zeros_like(dtilde_the_high)
            dprime_mix_high = np.zeros_like(dtilde_the_high)
            for i in range(nbin):
                newCov = np.copy(shifted_Cov_high)
                newCov[:, i] = Cov_observ_high[:, :, i]
                dnewCov_high = np.linalg.det(newCov)
                dtilde_mix_high += dnewCov_high

                newCovprime_high = np.copy(T_Rerr_high)
                newCovprime_high[:, i] = 0
                inv_newCov = np.linalg.inv(newCov)
                dprime_mix_high += dnewCov_high * np.trace(
                    np.matmul(inv_newCov, newCovprime_high), axis1=1, axis2=2
                )

            dtilde_the = np.concatenate([dtilde_the, dtilde_the_high])
            dprime_the = np.concatenate([dprime_the, dprime_the_high])
            d_obs = np.concatenate([d_obs, d_obs_high])
            dtilde_mix = np.concatenate([dtilde_mix, dtilde_mix_high])
            dprime_mix = np.concatenate([dprime_mix, dprime_mix_high])

        return (2 * ells + 1) * self.fsky * ((dprime_mix + dprime_the) / dtilde_the - (dtilde_mix * dprime_the) / np.power(dtilde_the, 2)) + 2 * eps_l

    def forge_norm(self):
        """forge normalization calculation

        Returns
        -------
        b_arr: 2D array
           Array with boost

        """

        z_max = 2.0
        redshifts = self.z[self.z <= z_max]

        Bk = []

        Omega_m = 0.31315
        sigma8 = 0.82172
        h = 0.6737
        fR0= 6.5

        for redshift in redshifts:
            B_f, _ = self.forge.predict_Bk(redshift,
                                            Omega_m,
                                            h,
                                            fR0,
                                            sigma8)
            Bk.append(B_f)


        Bk=np.asarray(Bk)

        return Bk
