"""
"""

import os

from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from egedal_f.egedal_f_obj import egedal_f

class AxialProfiles():

    b_sample_len = 40

    def __init__(self, R_m, theta_NBI, E_NBI, cache_dir, cache_fn='egedal_profiles_cache.pickle', ndigits=6):
        """
        """
        self.egedal_model = egedal_f(R_m=R_m, E_NBI=E_NBI, theta_NBI=theta_NBI, mu_i=2.5, T_e=0.1*E_NBI)
        self.b_sample = np.logspace(0, 1, num=AxialProfiles.b_sample_len, base=R_m)
        self.R_m = R_m
        self.theta_NBI = theta_NBI
        self.E_NBI = E_NBI
        self.ndigits=ndigits
        # Load cached data from previous profile calculations
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.cache_file = Path(os.path.join(cache_dir, cache_fn))
        self.cache = {}
        if self.cache_file.exists():
            with self.cache_file.open("rb") as f:
                self.cache = pickle.load(f)

    def _key(self, *params):
        """
        uantize floats to avoid tiny differences killing cache hits
        """
        return tuple(round(float(x), self.ndigits) for x in params)
    
    def get_vel_avg_sigma_v(self):
        """Gets velocity averaged fusion reactivity <sigma v> """
        pass

    def get_profiles(self):
        """
        Returns the density and anistoropic pressure profiles as a function 
        of b=B/B0, as a pandas dictionary. Uses a cached dictionary for efficiency
        """
        key = self._key(self.R_m, self.theta_NBI, self.E_NBI)
        if key in self.cache:
            return self.cache[key]
        
        # Cache miss -> compute and store profiles
        n_sample = np.zeros_like(self.b_sample)
        p_par_sample = np.zeros_like(self.b_sample)
        p_perp_sample = np.zeros_like(self.b_sample)
        for i, b in enumerate(self.b_sample):
            n_sample[i] = self.egedal_model.n(b=b)
            p_par_sample[i], p_perp_sample[i] = self.egedal_model.p(b=b)
        profiles = pd.DataFrame({'b': self.b_sample, 
                                 'n': n_sample, 
                                 'p_par': p_par_sample,
                                 'p_perp': p_perp_sample
                                 })
        self.cache[key] = profiles
        return profiles
    
    def save_cache(self):
        """Save the cache file to disk for future use"""
        # First save to temporary file in case of crash during write
        tmp = self.cache_file.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(self.cache, f)
        tmp.replace(self.cache_file)
