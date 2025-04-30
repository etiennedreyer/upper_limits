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
        self.two_sided = config.get('two_sided', False)

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

    def q_mu(self, data=None, mu=None):

        if data is None:
            data = self.data

        if mu is None:
            mu = self.mu

        ### Null hypothesis
        if self.bkg_sigma > 0:
            bu_hat = []
            for data_i in data: ### Unfortunately, not vectorized :(
                nll_fn = lambda bu: self.nll(mu, bu, data=data_i.reshape(1, -1))[0]
                bu_hat.append(self.fit_scalar(nll_fn))
                # if np.abs(bu_hat[-1] - 1) < 0.5:
                #     print(f"WARNING: bu_hat is close to 1: {bu_hat[-1]}")
            bu_hat = np.array(bu_hat).reshape(len(data), -1)
        else:
            bu_hat = 1
        nll_null = self.nll(mu, bu_hat, data=data, verbose=True)

        ### Alternative hypothesis
        mu_hat = ((data - 1*self.bkg) / (self.sig + 1e-8))
        nll_alt = self.nll(mu_hat, bu=1, data=data)

        ### q_mu
        q_mu = 2 * (nll_null - nll_alt)

        factor = -1 if self.two_sided else 0

        mu_hat = mu_hat.flatten()
        if self.discovery:
            if mu != 0:
                raise RuntimeError("Discovery assumes mu=0")
            ### Eq. 12 of 1007.1727
            q_mu[mu_hat < mu] *= factor
        else:
            ### Eq. 14 of 1007.1727
            if len(mu_hat) == 1:
                if mu_hat[0] > mu:
                    q_mu *= factor
            else:
                q_mu[mu_hat > mu] *= factor

        return_dict = {
            'mu': mu,
            'q_mu': q_mu,
            'bu_hat': bu_hat,
            'mu_hat': mu_hat,
            'nll_null': nll_null,
            'nll_alt': nll_alt
        }

        return return_dict

    def get_q_distribution(self, mu_inj, mu=None, n=100000):

        if mu is None:
            mu = self.mu
        toy_data = self.generate_data(mu_inj, n=n)
        results = self.q_mu(data=toy_data, mu=mu)
        return results['q_mu'], results
    
    def get_upper_limit(self, mu_hat=None, mu_scan=None, n=100000):

        if mu_hat is None:
            fit_result = self.q_mu(self.data, mu=0)
            mu_hat, q_0 = fit_result['mu_hat'], fit_result['q_mu']

        if mu_scan is None:
            mu_scan = np.linspace(mu_hat[0], 10*(mu_hat[0]+1), 20)
        else:
            assert np.diff(mu_scan).min() > 0, "mu_scan must be monotonically increasing"

        # get q hat values for each mu
        q_hats = []
        for mu in mu_scan:
            q_hats.append(self.q_mu(data=self.data, mu=mu)['q_mu'])
        q_hats = np.array(q_hats)

        # get q distributions for each mu=mu_inj
        q_distributions_splusb = {}
        q_distributions_bonly  = {}
        for mu in tqdm(mu_scan, desc="Getting q distributions"):
            q_distributions_splusb[mu], _ = self.get_q_distribution(mu_inj=mu, mu=mu, n=n)

            if self.limit_type in ['CLs']:
                for mu in tqdm(mu_scan, desc="Getting q distributions"):
                    q_distributions_bonly[mu], _ = self.get_q_distribution(mu_inj=0, mu=mu, n=n)
            else:
                q_distributions_bonly[mu] = np.zeros_like(q_distributions_splusb[mu])

        # create a 2D array of q distributions
        q_array_splusb = np.stack(list(q_distributions_splusb.values()), axis=0)
        q_array_bonly  = np.stack(list(q_distributions_bonly.values()), axis=0)

        def get_p_value(dist, q_hat, right=True):
            if right:
                p = np.sum(dist > q_hat, axis=-1) / dist.shape[-1]
            else:
                p = np.sum(dist < q_hat, axis=-1) / dist.shape[-1]
            return p

        # get the p-value for each q_inj in vectorized way
        p_splusb = get_p_value(q_array_splusb, q_hats)
        if self.limit_type in ['CLs']:
            p_bonly = get_p_value(q_array_bonly, q_hats, right=False)
        else:
            p_bonly = np.zeros_like(p_splusb)

        CLs = p_splusb / (1 - p_bonly)

        # Find where CLs crosses alpha
        # assert np.all(np.diff(CLs) < 0), "CLs is not monotonic"
        if (CLs < self.alpha).sum() == 0:
            print(f"WARNING: CLs never crosses alpha={self.alpha}! Minimum CLs: {CLs.min()}")
        mu_inj_where_CLs_at_alpha = np.interp(self.alpha, CLs[::-1], mu_scan[::-1])

        return_dict = {
            'mu_hat': mu_hat,
            'mu_upper_limit': mu_inj_where_CLs_at_alpha,
            'p_splusb': p_splusb,
            'p_bonly': p_bonly,
            'q_hats': q_hats,
            'mu_scan': mu_scan,
            'CLs': CLs,
            'q_distributions_splusb': q_distributions_splusb,
            'q_distributions_bonly': q_distributions_bonly,
        }

        return return_dict
    

    def plot_single_limit(self, result):

        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=150)

        ax.plot(result['mu_scan'], result['p_bonly'], label='$p_{b}$', color='b', ls='-.')
        ax.plot(result['mu_scan'], result['p_splusb'], label='$p_{s+b}$', color='r', ls='--')
        ax.plot(result['mu_scan'], result['CLs'], label='$CL_s$', ls='-', color='purple')
        ax.axhline(y=self.alpha, color='k', ls=':', label='$\\alpha$')
        ax.scatter(result['mu_upper_limit'], self.alpha, color='k', marker='o', label='$\\mu_{UL}$', facecolors='none', edgecolors='k')
        ax.set_xlabel('$\\mu_{inj}$')
        ax.set_ylabel('$p_{\\mu=' + f'{self.mu:.2f}' + '}$')
        ax.legend()
        
        return fig, ax