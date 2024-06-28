import os
from montepython.likelihood_class import Likelihood_prior


class gaussianprior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):
        ns = cosmo.n_s()
        omegab = cosmo.omega_b()
        log10Mc = cosmo.log10Mc()
        loglkl = self.ns_flag * (-0.5 * (ns - self.ns) ** 2 / (self.sigma_ns ** 2)) + self.omegab_flag * (-0.5 * (omegab - self.omegab) ** 2 / (self.sigma_omegab ** 2)) + self.log10Mc_flag * (-0.5 * (log10Mc - self.log10Mc) ** 2 / (self.sigma_log10Mc ** 2))
        return loglkl
        return loglkl
