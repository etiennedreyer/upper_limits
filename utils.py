import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize_scalar
import yaml
from tqdm import tqdm


class Experiment():

    def __init__(self, config, seed):

        self.read_config(config)
        self.set_seed(seed)
        self.set_pdfs()
        self.data = None
        self.q_distributions = {}

    def read_config(self, config):
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        self.config = config
        self.mu = config['mu']
        self.bkg = config['bkg']
        self.bkg_sigma = config['bkg_sigma']
        self.sig = config['sig']
        self.alpha = config['alpha']
        self.limit_type = config['limit_type']

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
        print(f"Seed: {seed}")

    def set_pdfs(self):
        pass

    def generate_data(self, mu, bkg=None):
        pass

    def get_nll(self, data, mu, bkg=None, do_syst=True):
        pass

    def get_mu_hat(self, data, bkg=None):
        pass

    def get_q_mu(self, mu_hat, mu=None):
        pass

    def get_q_distributions(self, mu_scan):
        pass

class Experiment_0D(Experiment):

    def __init__(self, config, seed=0):
        super().__init__(config, seed)

    def generate_data(self, mu_inj, n=1):
        x = np.random.poisson(self.bkg + mu_inj*self.sig, n)
        return x
    
    def nll(self, mu, bu):
        nll = -np.sum(stats.poisson.pmf(self.data, bu*self.bkg + mu*self.sig))
        if self.bkg_sigma > 0:
            nll += 0.5*((bu - 1) / (self.bkg_sigma))**2
        return nll

    def fit_scalar(self, fn, bounds=None):

        if bounds is None:
            bounds = (0, 10)

        fit_res = minimize_scalar(fn, bounds=bounds, method='bounded')

        if fit_res.success:
            if fit_res.x in bounds:
                print(f"Fit at boundary: {fit_res.x}")
            return fit_res.x
        else:
            raise RuntimeError("Fit failed")

    def q_mu(self):

        ### Null hypothesis
        if self.bkg_sigma > 0:
            nll_fn = lambda bu: self.nll(self.mu, bu)
            bu_hat = self.fit_scalar(nll_fn)
        else:
            bu_hat = 1
        nll_null = self.nll(self.mu, bu_hat)

        ### Alternative hypothesis
        mu_hat = self.data - self.bkg
        nll_alt = self.nll(mu_hat, 1)

        ### q_mu
        q_mu = 2 * (nll_null - nll_alt) * (mu_hat <= self.mu)

        return q_mu

    def get_q_distributions(self, mu_scan, n=10000):
        self.q_distributions = {}
        for mu_inj in tqdm(mu_scan):
            q_mus = []
            dataset = self.generate_data(mu_inj, n=n)
            for data in dataset:
                self.data = data
                q_mus.append(self.q_mu())
            self.q_distributions[mu_inj] = np.array(q_mus)