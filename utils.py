import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import multiprocessing as mp
from functools import partial
import yaml
from tqdm import tqdm


class Experiment():

    def __init__(self, config, seed=0):

        self.read_config(config)
        self.set_seed(seed)
        self.data = None
        self.q_distributions = {}

    def read_config(self, config):
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        self.config = config
        self.mu = config['mu']
        self.bkg = np.array(config['bkg'])
        self.bkg_sigma = config['bkg_sigma']
        self.sig = np.array(config['sig'])
        self.alpha = config['alpha']
        self.limit_type = config['limit_type']

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        print(f"Seed: {seed}")

    def generate_data(self, mu_inj, n=1):
        x = np.random.poisson(self.bkg + mu_inj*self.sig, size=(n, len(self.bkg)))
        return x
    
    def nll(self, mu, bu, data=None, verbose=False):
        if data is None:
            data = self.data
        nll = -np.sum(data * np.log(bu*self.bkg + mu*self.sig) - (bu*self.bkg + mu*self.sig), axis=1, keepdims=True)
        if self.bkg_sigma > 0:
            nll += 0.5*((bu - 1) / (self.bkg_sigma))**2

        return nll.reshape(-1)

    def fit_scalar(self, fn, bounds=None):

        if bounds is None:
            bounds = (0, 10)

        fit_res = optimize.minimize_scalar(fn, bounds=bounds, method='bounded')

        if fit_res.success:
            if fit_res.x in bounds:
                print(f"Fit at boundary: {fit_res.x}")
            return fit_res.x
        else:
            raise RuntimeError("Fit failed")

    def q_mu(self):

        ### Null hypothesis
        if self.bkg_sigma > 0:
            bu_hat = []
            for data in self.data: ### Unfortunately, not vectorized :(
                nll_fn = lambda bu: self.nll(self.mu, bu, data.reshape(1, -1))[0]
                bu_hat.append(self.fit_scalar(nll_fn))
            bu_hat = np.array(bu_hat).reshape(len(self.data), -1)
        else:
            bu_hat = 1
        nll_null = self.nll(self.mu, bu_hat, verbose=True)

        ### Alternative hypothesis
        mu_hat = ((self.data - 1*self.bkg) / (self.sig + 1e-8))
        nll_alt = self.nll(mu_hat, 1)

        ### q_mu
        q_mu = 2 * (nll_null - nll_alt) * ((mu_hat <= self.mu).reshape(-1))

        return_dict = {
            'q_mu': q_mu,
            'bu_hat': bu_hat,
            'nll_null': nll_null,
            'nll_alt': nll_alt
        }

        return return_dict

    def get_q_distributions(self, mu_scan, n=10000):
        self.q_distributions = {}
        for mu_inj in tqdm(mu_scan):
            self.data = self.generate_data(mu_inj, n=n)
            self.q_distributions[mu_inj] = self.q_mu()['q_mu']