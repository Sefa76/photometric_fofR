import os
from montepython.likelihood_class import Likelihood_prior


class gaussianprior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):
        ns = cosmo.n_s()
        omegab = cosmo.omega_b()
        loglkl = -0.5 * (ns - self.ns) ** 2 / (self.sigma_ns ** 2) -0.5 * (omegab - self.omegab) ** 2 / (self.sigma_omegab ** 2)
        return loglkl
        return loglkl