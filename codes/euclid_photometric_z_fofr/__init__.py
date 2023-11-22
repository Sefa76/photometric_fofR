########################################################
# Euclid photometric likelihood
########################################################
# written by Maike Doerenkamp in 2020
# following the recipe of 1910.09237 (Euclid preparation: VII. Forecast validation for Euclid
# cosmological probes)
# (adapted from euclid_lensing likelihood)
# edited by Lena Rathmann in 2021

from montepython.likelihood_class import Likelihood

from scipy.integrate import trapz,simpson
from scipy import interpolate as itp
from scipy.interpolate import interp1d, RectBivariateSpline

import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from montepython.MGfit_Winther import pofk_enhancement ,pofk_enhancement_linear , kazuya_correktion 
from montepython.forge_emulator.FORGE_emulator import FORGE


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
        self.debug_plot = False
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
        else:
            self.l_GC = np.logspace(np.log10(self.lmin), np.log10(self.lmax_GC), num=self.lbin, endpoint=True)
            self.idx_lmax = int(np.argwhere(self.l_GC >= self.lmax_WL)[0])
            self.l_WL = self.l_GC[:self.idx_lmax+1]
            self.l_XC = self.l_GC[:self.idx_lmax+1]
            self.l_array = 'GC'
        #print('l array WL: ', self.l_WL)
        #print('l array GC: ', self.l_GC)
        if self.debug_save :
            np.savetxt('ls.txt',self.l_GC)
        ########################################################
        # Find distribution of n(z) in each bin
        ########################################################

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

        if self.use_BCemu or self.model_use_BCemu:
            self.nuisance += ['log10Mc']
            self.nuisance += ['thej']
            self.nuisance += ['deta']
            self.bfcemu = BCemu.BCM_7param(verbose=False)

        ###########
        # Read data
        ###########

        # If the file exists, initialize the fiducial values
        # It has been stored flat, so we use the reshape function to put it in
        # the right shape.

        if self.lmax_WL > self.lmax_GC:
            ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            l_jump = self.lmax_GC - self.lmin +1
            ells_GC = ells_WL[:l_jump]
        else:
            ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            l_jump = self.lmax_WL - self.lmin +1
            ells_WL = ells_GC[:l_jump]
        self.fid_values_exist = False
        fid_file_path = os.path.join(self.data_directory, self.fiducial_file)
        if os.path.exists(fid_file_path):
            self.fid_values_exist = True
            if 'WL' in self.probe:
                self.Cov_observ = np.zeros((len(ells_WL), self.nbin, self.nbin), 'float64')
            if 'GCph' in self.probe:
                self.Cov_observ = np.zeros((len(ells_GC), self.nbin, self.nbin), 'float64')
            if 'WL_GCph_XC' in self.probe:
                self.Cov_observ = np.zeros((l_jump, 2*self.nbin, 2*self.nbin), 'float64')
                if self.lmax_WL > self.lmax_GC:
                    l_high = len(ells_WL)-l_jump
                else:
                    l_high = len(ells_GC)-l_jump
                self.Cov_observ_high = np.zeros(((l_high), self.nbin, self.nbin), 'float64')
            with open(fid_file_path, 'r') as fid_file:
                line = fid_file.readline()
                while line.find('#') != -1:
                    line = fid_file.readline()
                while (line.find('\n') != -1 and len(line) == 1):
                    line = fid_file.readline()
                if 'WL' in self.probe:
                    for Bin1 in range(self.nbin):
                        for Bin2 in range(self.nbin):
                            for nl in range(len(ells_WL)):
                                self.Cov_observ[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()
                if 'GCsp' in self.probe:
                    for Bin1 in range(self.nbin):
                        for Bin2 in range(self.nbin):
                            for nl in range(len(ells_GC)):
                                self.Cov_observ[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()
                if 'WL_GCph_XC' in self.probe:
                    for Bin1 in range(2*self.nbin):
                        for Bin2 in range(2*self.nbin):
                            for nl in range(l_jump):
                                self.Cov_observ[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()
                    for Bin1 in range(self.nbin):
                        for Bin2 in range(self.nbin):
                            for nl in range(l_high):
                                self.Cov_observ_high[nl,Bin1,Bin2] = float(line)
                                line = fid_file.readline()

        else:
            if self.fit_diffrent_data:
                self.use_fofR = self.model_use_fofR
                self.use_BCemu= self.model_use_BCemu

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
                self.emantis = FofrBoost(verbose=True, extrapolate_aexp=True, extrapolate_low_k=True)

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

        ########################
        # Boosts and Emulators #
        ########################

        if self.use_fofR == 'Winther':
            print("f(R) active with Winther fitting function")
            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)
            boost_m_nl_fofR = np.zeros((self.lbin, self.nzmax), 'float64')
            boost_m_l_fofR  = np.zeros((self.lbin, self.nzmax), 'float64')

            for index_l, index_z in index_pknn:
                boost_m_l_fofR [index_l, index_z]= pofk_enhancement_linear(self.z[index_z],f_R0,k[index_l,index_z]/cosmo.h()) 
                boost_m_nl_fofR[index_l, index_z]= pofk_enhancement       (self.z[index_z],f_R0,k[index_l,index_z]/cosmo.h(),hasBug=self.use_bug)

            if 'sigma8_fofR' in data.get_mcmc_parameters(['derived_lkl']):
                data.derived_lkl={'sigma8_fofR':self.get_sigma8_fofR(k_grid,Pk_m_l_grid[:,-1],cosmo.h(),lgfR0)}

            Pk *= boost_m_nl_fofR


        elif self.use_fofR in ['Forge','Forge_corr']:
            print("f(R) active with Forge emulator")
            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)

            z_max = 2.0
            redshifts = self.z[self.z <= z_max]

            Bk = []

            Omc = cosmo.Omega0_cdm()
            Omb = cosmo.Omega_b()
            hubble = cosmo.h()
            pars_dict={'sigma8': cosmo.sigma8(),
                       'h': hubble,
                       'Omega_m': Omc+Omb}

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

            boost_m_nl_fofR = np.zeros((self.lbin, self.nzmax), 'float64')
            boost_m_l_fofR  = np.zeros((self.lbin, self.nzmax), 'float64')

            for index_l, index_z in index_pknn:
                boost_m_l_fofR [index_l, index_z]= pofk_enhancement_linear(self.z[index_z],f_R0,k[index_l,index_z]/hubble)
                boost_m_nl_fofR[index_l, index_z]= Bk_interp(self.z[index_z],k[index_l,index_z]/hubble)

            if 'sigma8_fofR' in data.get_mcmc_parameters(['derived_lkl']):
                data.derived_lkl={'sigma8_fofR':self.get_sigma8_fofR(k_grid,Pk_m_l_grid[:,-1],cosmo.h(),lgfR0)}

            Pk *= boost_m_nl_fofR


        if self.use_fofR == 'ReACT':
            # Halo model reaction based boost. Emulator based on output from ReACT and HMCode2020 (for pseudo and LCDM)
            # Emulator range given below. It outputs in [0.01,3] h/Mpc
            # Includes massive neutrinos which are here set to 0 manually

            # TODO:
            # Extract omnuh2 from data vector (trivial ... )
            # Optimise clipping and setting boost = 1 for z>zmax
            # Extrapolate to small k? Perhaps not necessary as boost should be 1 at kmin = 0.01h/Mpc ....

            print("f(R) active with ReACT")

            # Empty arrays to store nonlinear and linear boosts
            boost_m_nl_fofR = np.zeros((self.lbin, self.nzmax), 'float64')
            boost_m_l_fofR  = np.zeros((self.lbin, self.nzmax), 'float64')

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

            # Extrapolate to high values of k
            # Values to extrapolate to
            mykmax = 50
            mykmin = kvals[-1]+0.01
            N_grid = 300

            # Create k bins at which to extrapolate to
            k_extrapolate = np.linspace(mykmin, mykmax , N_grid)

            # Array to store extrapolated values
            Bk_ext=[]

            #Extrapolate only for redshifts where boost is non-trivial, otherwise it is 1
            for index_z in range(nz_Pk):
                if(self.z[index_z]<=zmax):
                    Bk_ext.append(interp1d(kvals,Boost[index_z],fill_value='extrapolate')(k_extrapolate))
                else:
                    Bk_ext.append(np.ones(N_grid))

            # Attach extended k-grid and boosts to emulator's grid and boosts
            concatenated_boost = np.hstack((Boost,Bk_ext))
            concatenated_k = np.hstack((kvals, k_extrapolate))

            # Spline final function which now extends to mykmax using a power law extrapolation
            Bk_interp = RectBivariateSpline(self.z,concatenated_k,concatenated_boost)

            # Fill up boost array
            for index_l, index_z in index_pknn:
                boost_m_l_fofR [index_l, index_z]= pofk_enhancement_linear(self.z[index_z],f_R0,k[index_l,index_z]/cosmo.h())
                boost_m_nl_fofR[index_l, index_z] = Bk_interp(self.z[index_z],k[index_l,index_z]/hubble)

            if 'sigma8_fofR' in data.get_mcmc_parameters(['derived_lkl']):
                data.derived_lkl={'sigma8_fofR':self.get_sigma8_fofR(k_grid,Pk_m_l_grid[:,-1],cosmo.h(),lgfR0)}

            # Apply the boost
            Pk *= boost_m_nl_fofR


        elif self.use_fofR == 'emantis':
            print("f(R) active with e-MANTIS emulator")

            lgfR0 = data.mcmc_parameters['lgfR0']['current']*data.mcmc_parameters['lgfR0']['scale']
            f_R0=np.power(10,-1*lgfR0)
            boost_m_nl_fofR = np.zeros((self.lbin, self.nzmax), 'float64')
            boost_m_l_fofR  = np.zeros((self.lbin, self.nzmax), 'float64')

            Omc = cosmo.Omega0_cdm()
            Omb = cosmo.Omega_b()
            Omm = Omc + Omb
            hubble = cosmo.h()

            # Get e-mantis predictions for all z and k.

            # Flatten k array.
            k_flat = np.ravel(k)

            # Init. emantis prediction array.
            emantis_boost = np.ones((self.z.shape[0], 1, k_flat.shape[0]))

            # k indices outside emantis range (k>kmax).
            kmax = self.emantis.kbins[-1]*hubble
            k_extrap_idx = k_flat > kmax

            # Get emantis predictions for k<=kmax.
            pred = self.emantis.predict_boost(Omm, cosmo.sigma8(), lgfR0, 1/(1+self.z), k_flat[~k_extrap_idx]/hubble)
            emantis_boost[:,:,~k_extrap_idx] = pred

            # Constant extrapolation for k>kmax.
            # This seems like a messy way to do it, but in any case it should be replaced by a common extrapolation
            # for all types of predictions.
            # Get emantis predictions for k=kmax.
            pred_kmax = self.emantis.predict_boost(Omm, cosmo.sigma8(), lgfR0, 1/(1+self.z), kmax/hubble)
            for i in range(k_flat.shape[0]):
                if k_extrap_idx[i]:
                    emantis_boost[:,:,i] = pred_kmax[:,:,0]

            for index_l, index_z in index_pknn:
                boost_m_l_fofR[index_l, index_z] = pofk_enhancement_linear(self.z[index_z],f_R0,k[index_l,index_z]/hubble)

                # Select the required z and k.
                idx_k_flat = np.ravel_multi_index((index_l, index_z), k.shape)
                boost_m_nl_fofR[index_l, index_z] = emantis_boost[index_z, 0, idx_k_flat]

            if 'sigma8_fofR' in data.get_mcmc_parameters(['derived_lkl']):
                data.derived_lkl={'sigma8_fofR':self.get_sigma8_fofR(k_grid,Pk_m_l_grid[:,-1],cosmo.h(),lgfR0)}

            # Apply the boost
            Pk *= boost_m_nl_fofR


        if self.use_BCemu:
            # baryonic feedback modifications are only applied to k>kmin_bfc
            # it is very computationally expensive to call BCemu at every z in self.z, and it is a very smooth function with z,
            # so it is only called at self.BCemu_k_bins points in k and self.BCemu_z_bins points in z and then the result is
            # splined over all z in self.z. For k>kmax_bfc = 12.5 h/Mpc, the maximum k the emulator is trained on, a constant
            # suppression in k is assumed: BFC(k,z) = BFC(12.5 h/Mpc, z).

            log10Mc = data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale']
            thej = data.mcmc_parameters['thej']['current'] * data.mcmc_parameters['thej']['scale']
            deta = data.mcmc_parameters['deta']['current'] * data.mcmc_parameters['deta']['scale']


            bcemu_dict ={
            'log10Mc' : log10Mc,
            'nu_Mc'   : 0.0,
            'mu'      : 1.0,
            'nu_mu'   : 0.0,
            'thej'    : thej,
            'nu_thej' : 0.0,
            'gamma'   : 2.5,
            'nu_gamma': 0.0,
            'delta'   : 7.0,
            'nu_delta': 0.0,
            'eta'     : 0.2,
            'nu_eta'  : 0.0,
            'deta'    : deta,
            'nu_deta' : 0.0
            }

            Ob = cosmo.Omega_b()
            Om = cosmo.Omega_m()

            fb = Ob/Om
            if fb < 0.1 or fb > 0.25:
                if self.verbose: print(" /!\ Skipping point because the baryon fraction is out of bounds!")
                return -1e10

            if log10Mc / 3**nu_Mc < 11 or log10Mc / 3**nu_Mc > 15 :
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

        ####################
        # Plot Pk and Cl's #
        ####################

        if self.debug_plot == True:
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

        ##########
        # Noise
        ##########
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

        #############
        # Spline Cl #
        #############
        # Find C(l) for every integer l

        # Spline the Cls along l
        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            spline_LL = np.empty((self.nbin, self.nbin),dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_LL[Bin1,Bin2] = list(itp.splrep(
                        self.l_WL[:], Cl_LL[:,Bin1,Bin2]))

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            spline_GG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_GG[Bin1,Bin2] = list(itp.splrep(
                        self.l_GC[:], Cl_GG[:,Bin1,Bin2]))

        if 'WL_GCph_XC' in self.probe:
            spline_LG = np.empty((self.nbin, self.nbin), dtype=(list,3))
            spline_GL = np.empty((self.nbin, self.nbin), dtype=(list,3))
            for Bin1 in range(self.nbin):
                for Bin2 in range(self.nbin):
                    spline_LG[Bin1,Bin2] = list(itp.splrep(
                        self.l_XC[:], Cl_LG[:,Bin1,Bin2]))
                    spline_GL[Bin1,Bin2] = list(itp.splrep(
                        self.l_XC[:], Cl_GL[:,Bin1,Bin2]))

        # Create array of all integers of l
        if self.lmax_WL > self.lmax_GC:
            ells_WL = np.array(range(self.lmin,self.lmax_WL+1))
            l_jump = self.lmax_GC - self.lmin +1
            ells_GC = ells_WL[:l_jump]
        else:
            ells_GC = np.array(range(self.lmin,self.lmax_GC+1))
            l_jump = self.lmax_WL - self.lmin +1
            ells_WL = ells_GC[:l_jump]

        if 'WL_GCph_XC' in self.probe:

            Cov_theory = np.zeros((l_jump, 2*self.nbin, 2*self.nbin), 'float64')
            if self.lmax_WL > self.lmax_GC:
                Cov_theory_high = np.zeros(((len(ells_WL)-l_jump), self.nbin, self.nbin), 'float64')
            else:
                Cov_theory_high = np.zeros(((len(ells_GC)-l_jump), self.nbin, self.nbin), 'float64')
        elif 'WL' in self.probe:
            Cov_theory = np.zeros((len(ells_WL), self.nbin, self.nbin), 'float64')
        elif 'GCph' in self.probe:
            Cov_theory = np.zeros((len(ells_GC), self.nbin, self.nbin), 'float64')

        for Bin1 in range(self.nbin):
            for Bin2 in range(self.nbin):
                if 'WL_GCph_XC' in self.probe:
                    if self.lmax_WL > self.lmax_GC:
                        Cov_theory[:,Bin1,Bin2] = itp.splev(
                            ells_GC[:], spline_LL[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,Bin2] = itp.splev(
                            ells_GC[:], spline_GL[Bin1,Bin2])
                        Cov_theory[:,Bin1,self.nbin+Bin2] = itp.splev(
                            ells_GC[:], spline_LG[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,self.nbin+Bin2] = itp.splev(
                            ells_GC[:], spline_GG[Bin1,Bin2])

                        Cov_theory_high[:,Bin1,Bin2] = itp.splev(
                            ells_WL[l_jump:], spline_LL[Bin1,Bin2])
                    else:
                        Cov_theory[:,Bin1,Bin2] = itp.splev(
                            ells_WL[:], spline_LL[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,Bin2] = itp.splev(
                            ells_WL[:], spline_GL[Bin1,Bin2])
                        Cov_theory[:,Bin1,self.nbin+Bin2] = itp.splev(
                            ells_WL[:], spline_LG[Bin1,Bin2])
                        Cov_theory[:,self.nbin+Bin1,self.nbin+Bin2] = itp.splev(
                            ells_WL[:], spline_GG[Bin1,Bin2])

                        Cov_theory_high[:,Bin1,Bin2] = itp.splev(
                            ells_GC[l_jump:], spline_LL[Bin1,Bin2])

                elif 'WL' in self.probe:
                    Cov_theory[:,Bin1,Bin2] = itp.splev(
                        ells_WL[:], spline_LL[Bin1,Bin2])

                elif 'GCph' in self.probe:
                    Cov_theory[:,Bin1,Bin2] = itp.splev(
                        ells_GC[:], spline_GG[Bin1,Bin2])

        #print('Cov_theory: ', Cov_theory[1,2,3])

        #######################
        # Create fiducial file
        #######################

        if self.fid_values_exist is False:
            # Store the values now, and exit.
            fid_file_path = os.path.join(
                self.data_directory, self.fiducial_file)
            with open(fid_file_path, 'w') as fid_file:
                fid_file.write('# Fiducial parameters')
                for key, value in data.mcmc_parameters.items():
                    fid_file.write(
                        ', %s = %.5g' % (key, value['current']*value['scale']))
                fid_file.write('\n')
                if 'WL' in self.probe or 'GCph' in self.probe:
                    for Bin1 in range(self.nbin):
                        for Bin2 in range(self.nbin):
                            for nl in range(len(Cov_theory[:,0,0])):
                                fid_file.write("%.55g\n" % Cov_theory[nl, Bin1, Bin2])
                if 'WL_GCph_XC' in self.probe:
                    for Bin1 in range(2*self.nbin):
                        for Bin2 in range(2*self.nbin):
                            for nl in range(len(Cov_theory[:,0,0])):
                                fid_file.write("%.55g\n" % Cov_theory[nl, Bin1, Bin2])
                    for Bin1 in range(self.nbin):
                        for Bin2 in range(self.nbin):
                            for nl in range(len(Cov_theory_high[:,0,0])):
                                fid_file.write("%.55g\n" % Cov_theory_high[nl, Bin1, Bin2])
            print('\n')
            warnings.warn(
                "Writing fiducial model in %s, for %s likelihood\n" % (
                    self.data_directory+'/'+self.fiducial_file, self.name))
            return 1j

        ######################
        # Compute likelihood
        ######################
        # Define cov theory and observ on the whole integer range of ell values

        chi2 = 0.

        if 'WL_GCph_XC' in self.probe:
            if self.lmax_WL > self.lmax_GC:
                ells = ells_WL
            else:
                ells = ells_GC

            d_the = np.linalg.det(Cov_theory)
            d_obs = np.linalg.det(self.Cov_observ)
            d_mix = np.zeros_like(d_the)
            for i in range(2*self.nbin):
                newCov = Cov_theory.copy()
                newCov[:, i] = self.Cov_observ[:, :, i]
                d_mix += np.linalg.det(newCov)

            d_the_high = np.linalg.det(Cov_theory_high)
            d_obs_high = np.linalg.det(self.Cov_observ_high)
            d_mix_high = np.zeros_like(d_the_high)
            for i in range(self.nbin):
                newCov = Cov_theory_high.copy()
                newCov[:, i] = self.Cov_observ_high[:, :, i]
                d_mix_high += np.linalg.det(newCov)

            N =np.ones_like(ells)*2*self.nbin
            N[np.where(ells>self.lmax_XC)] = self.nbin

            d_the = np.concatenate([d_the,d_the_high])
            d_obs = np.concatenate([d_obs,d_obs_high])
            d_mix = np.concatenate([d_mix,d_mix_high])

            chi2 += np.sum((2*ells+1)*self.fsky*((d_mix/d_the)+np.log(d_the/d_obs)-N))

        elif 'WL' in self.probe:
            d_the = np.linalg.det(Cov_theory)
            d_obs = np.linalg.det(self.Cov_observ)
            d_mix = np.zeros_like(d_the)
            for i in range(self.nbin):
                newCov = np.copy(Cov_theory)
                newCov[:, i] = self.Cov_observ[:, :, i]
                d_mix += np.linalg.det(newCov)

            N =np.ones_like(ells_WL)*self.nbin

            chi2 += np.sum((2*ells_WL+1)*self.fsky*((d_mix/d_the)+np.log(d_the/d_obs)-N))

        elif 'GCph' in self.probe:
            d_the = np.linalg.det(Cov_theory)
            d_obs = np.linalg.det(self.Cov_observ)
            d_mix = np.zeros_like(d_the)
            for i in range(self.nbin):
                newCov = np.copy(Cov_theory)
                newCov[:, i] = self.Cov_observ[:, :, i]
                d_mix += np.linalg.det(newCov)

            N =np.ones_like(ells_GC)*self.nbin

            chi2 += np.sum((2*ells_GC+1)*self.fsky*((d_mix/d_the)+np.log(d_the/d_obs)-N))

        print("euclid photometric: chi2 = ",chi2)
        return -chi2/2.

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
