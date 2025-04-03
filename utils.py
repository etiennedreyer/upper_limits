import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import matplotlib.pyplot as plt
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
        self.mu = config.get('mu', 0.0)
        self.bkg = np.array(config['bkg'])
        self.bkg_sigma = config['bkg_sigma']
        self.sig = np.array(config['sig'])
        self.alpha = config.get('alpha', 0.05)
        self.limit_type = config.get('limit_type', 'CLs')
        self.discovery = config.get('discovery', True)

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
        nexp = np.clip(self.bkg + mu*self.sig, 1e-12, None) # avoid log(0)

        nll = -np.sum(data * np.log(nexp) - nexp, axis=1, keepdims=True)

        ### Gaussian penalty on background norm factor
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

    def q_mu(self, data=None):

        if data is None:
            data = self.data

        ### Null hypothesis
        if self.bkg_sigma > 0:
            bu_hat = []
            for data_i in data: ### Unfortunately, not vectorized :(
                nll_fn = lambda bu: self.nll(self.mu, bu, data=data_i.reshape(1, -1))[0]
                bu_hat.append(self.fit_scalar(nll_fn))
                # if np.abs(bu_hat[-1] - 1) < 0.5:
                #     print(f"WARNING: bu_hat is close to 1: {bu_hat[-1]}")
            bu_hat = np.array(bu_hat).reshape(len(data), -1)
        else:
            bu_hat = 1
        nll_null = self.nll(self.mu, bu_hat, data=data, verbose=True)

        ### Alternative hypothesis
        mu_hat = ((data - 1*self.bkg) / (self.sig + 1e-8))
        nll_alt = self.nll(mu_hat, bu=1, data=data)

        ### q_mu
        q_mu = 2 * (nll_null - nll_alt)

        if self.discovery:
            if self.mu != 0:
                raise RuntimeError("Discovery assumes mu=0")
            ### Eq. 12 of 1007.1727
            q_mu[mu_hat.flatten() < self.mu] = 0
            # q_mu *= ((mu_hat >= self.mu).reshape(-1))
        else:
            ### Eq. 14 of 1007.1727
            q_mu[mu_hat.flatten() > self.mu] = 0

        return_dict = {
            'q_mu': q_mu,
            'bu_hat': bu_hat,
            'mu_hat': mu_hat,
            'nll_null': nll_null,
            'nll_alt': nll_alt
        }

        return return_dict

    def get_q_distributions(self, mu_scan=None, n=10000):
        if mu_scan is None:
            mu_scan = np.linspace(self.mu, (self.mu + 1)*10, 20)
        ### make sure we have the most interesting mu values
        if 0.0 not in mu_scan:
            mu_scan = np.concatenate(([0.0], mu_scan))
        if self.mu not in mu_scan:
            mu_scan = np.concatenate(([self.mu], mu_scan))
        mu_scan = np.sort(mu_scan)

        self.q_distributions = {}
        for mu_inj in tqdm(mu_scan):
            toy_data = self.generate_data(mu_inj, n=n)
            self.q_distributions[mu_inj] = self.q_mu(data=toy_data)['q_mu']
    
    def get_upper_limit(self, q_hat=None):

        if q_hat is None:
            fit_result = self.q_mu(self.data)
            mu_hat, q_hat = fit_result['mu_hat'], fit_result['q_mu']

        if self.q_distributions == {}:
            raise RuntimeError("No q distributions found. Run get_q_distributions() first.")

        def get_p_values(dist, q_hat):
            p_left  = np.sum(dist <= q_hat, axis=-1) / dist.shape[-1]
            p_right = 1 - p_left
            return p_left, p_right

        # sort dictionary by key (q_inj)
        self.q_distributions = dict(sorted(self.q_distributions.items()))

        # create a 2D array of q distributions
        mu_inj_arr = np.array(list(self.q_distributions.keys()))
        q_dist_arr = np.stack(list(self.q_distributions.values()), axis=0)

        # get the p-value for each q_inj in vectorized way
        p_left, p_right = get_p_values(q_dist_arr, q_hat)

        # get the right-tail p-value for null (mu_inj = mu)
        mask = mu_inj_arr == self.mu ; assert mask.sum() == 1, f"More than one mu_inj = mu found: {mu_inj_arr[mask]}"
        p_mu_null = p_right[mask][0]

        # get the left-tail p-value for alternate (mu_inj != mu)
        mask = mu_inj_arr != self.mu
        p_mu_alt= p_left[mask]
        mu_inj_alt = mu_inj_arr[mask]

        CLs = p_mu_alt / (1 - p_mu_null)

        # Find where CLs crosses alpha
        # assert np.all(np.diff(CLs) < 0), "CLs is not monotonic"
        if (CLs < self.alpha).sum() == 0:
            print(f"WARNING: CLs never crosses alpha={self.alpha}! Minimum CLs: {CLs.min()}")
        mu_inj_where_CLs_at_alpha = np.interp(self.alpha, CLs[::-1], mu_inj_alt[::-1])

        return_dict = {
            'mu_upper_limit': mu_inj_where_CLs_at_alpha,
            'p_mu_null': p_mu_null,
            'p_mu_alt': p_mu_alt,
            'mu_inj_alt': mu_inj_alt,
            'CLs': CLs,
        }

        return return_dict
    

    def plot_single_limit(self, result):

        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=150)

        ax.axhline(y=result['p_mu_null'], label='$p_\\mu, \\mu\'=\\mu$', color='b', ls='-.')
        ax.plot(result['mu_inj_alt'], result['p_mu_alt'], label='$p_\\mu, \\mu\'\\neq \\mu$', color='r', ls='--')
        ax.plot(result['mu_inj_alt'], result['CLs'], label='$CL_s$', ls='-', color='purple')
        ax.axhline(y=self.alpha, color='k', ls=':', label='$\\alpha$')
        ax.scatter(result['mu_upper_limit'], self.alpha, color='k', marker='o', label='$\\mu_{UL}$', facecolors='none', edgecolors='k')
        ax.set_xlabel('$q_{inj}$')
        ax.set_ylabel('$p_{\\mu=' + f'{self.mu:.2f}' + '}$')
        ax.legend()
        
        return fig, ax