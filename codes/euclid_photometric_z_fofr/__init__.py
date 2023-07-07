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

try:
    import BCemu
except:
    raise Exception ("Please install the BCemu package from https://github.com/sambit-giri/BCemu !")

import numpy as np
import warnings
from scipy.special import erf

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

        if self.use_fofR == True:
            self.nuisance += ['lgfR0']
        
        if self.use_BCemu:
            self.nuisance += ['log10Mc']
            self.nuisance += ['nu_Mc']           
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

        if self.use_fofR:
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
 
        if self.use_BCemu:
            # baryonic feedback modifications are only applied to k>kmin_bfc
            # it is very computationally expensive to call BCemu at every z in self.z, and it is a very smooth function with z,
            # so it is only called at self.BCemu_k_bins points in k and self.BCemu_z_bins points in z and then the result is
            # splined over all z in self.z. For k>kmax_bfc = 12.5 h/Mpc, the maximum k the emulator is trained on, a constant
            # suppression in k is assumed: BFC(k,z) = BFC(12.5 h/Mpc, z).

            log10Mc = data.mcmc_parameters['log10Mc']['current'] * data.mcmc_parameters['log10Mc']['scale']
            nu_Mc   = data.mcmc_parameters['nu_Mc']['current']   * data.mcmc_parameters['nu_Mc']['scale']

            bcemu_dict ={
            'log10Mc' : log10Mc,
            'nu_Mc'   : nu_Mc,
            'mu'      : 0.93,
            'nu_mu'   : 0.0,
            'thej'    : 2.6, 
            'nu_thej' : 0.0,
            'gamma'   : 2.25,
            'nu_gamma': 0.0,
            'delta'   : 6.4,
            'nu_delta': 0.0,
            'eta'     : 0.15,
            'nu_eta'  : 0.0,
            'deta'    : 0.14,
            'nu_deta' : 0.06
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
            if self.bias_model == 'binned' :
                W_G = np.zeros((self.nzmax, self.nbin), 'float64')
                W_G =  (H_z * self.biasfunc(self.z))[:,None] * self.eta_z

        ##########
        # Noise
        ##########
        # dimensionless

        self.noise = {
           'LL': self.rms_shear**2./self.n_bar,
           'LG': 0.,
           'GL': 0.,
           'GG': 1./self.n_bar}

        ###########
        # Calc Cl #
        ###########
        # dimensionless

        nell_WL = len(self.l_WL)
        nell_GC = len(self.l_GC)
        nell_XC = len(self.l_XC)

        if 'WL' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_LL_int = W_L[:,:,:,None] * W_L[:,:,None,:] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None]
            Cl_LL     = trapz(Cl_LL_int,self.z,axis=1)[:nell_WL,:,:]
            for i in range(self.nbin):
                Cl_LL[:,i,i] += self.noise['LL']

        if 'GCph' in self.probe or 'WL_GCph_XC' in self.probe:
            Cl_GG_int = W_G[None,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None]
            Cl_GG     = trapz(Cl_GG_int,self.z,axis=1)[:nell_GC,:,:]
            for i in range(self.nbin):
                Cl_GG[:,i,i] += self.noise['GG']

        if 'WL_GCph_XC' in self.probe:
            Cl_LG_int = W_L[:,:,:,None] * W_G[None,: , None, :] * Pk[:,:,None,None] / H_z[None,:,None,None] / self.r[None,:,None,None] / self.r[None,:,None,None]
            Cl_LG     = trapz(Cl_LG_int,self.z,axis=1)[:nell_XC,:,:]
            Cl_GL     = np.transpose(Cl_LG,(0,2,1))
            for i in range(self.nbin):
                Cl_LG[:,i,i] += self.noise['LG']
                Cl_GL[:,i,i] += self.noise['GL']


        #############
        # Plot Cl's #
        #############

        Plot_debug = False
        if Plot_debug == True:
            Bin = 9
            debug_file_path = os.path.join(
                self.data_directory, 'z_Cl_'+str(Bin)+'.dat')
            with open(debug_file_path, 'w') as debug_file:
                for nl in range(len(self.l_XC)):
                    debug_file.write("%g  %.16g  %.16g  %.16g  %.16g\n" % (l[nl],Cl_LL[nl,Bin,Bin],Cl_GG[nl,Bin,Bin],Cl_LG[nl,Bin,Bin],Cl_GL[nl,Bin,Bin]))
            print("Printed Cl's")
            exit()

        Plot_debug = False
        if Plot_debug == True:
            if 'WL' in self.probe:
                debug_file_path = os.path.join(self.data_directory, 'euclid_WLz_Cl_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_LL)
                debug_file_path = os.path.join(self.data_directory, 'euclid_WLz_ells.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, self.l_WL)
            if 'WL_GCph_XC' in self.probe:
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_Cl_LL_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_LL)
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_Cl_GG_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_GG)
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_Cl_LG_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_LG)
                debug_file_path = os.path.join(self.data_directory, 'euclid_XCz_ells.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, self.l_XC)
            if 'GCph' in self.probe:
                debug_file_path = os.path.join(self.data_directory, 'euclid_GCz_Cl_'+str(self.nzmax)+'.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, Cl_GG)
                debug_file_path = os.path.join(self.data_directory, 'euclid_GCz_ells.npy')
                with open(debug_file_path, 'w') as debug_file:
                    np.save(debug_file, self.l_GC)
            exit()


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
