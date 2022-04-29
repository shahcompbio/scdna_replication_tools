import argparse
import logging
import sys

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.contrib.examples.polyphonic_data_loader as poly
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, infer_discrete, config_enumerate
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings, optional

import numpy as np
import pandas as pd

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.DEBUG)

# Add another handler for logging debugging events (e.g. for profiling)
# in a separate stream that can be captured.
log = logging.getLogger()
debug_handler = logging.StreamHandler(sys.stdout)
debug_handler.setLevel(logging.DEBUG)
debug_handler.addFilter(filter=lambda record: record.levelno <= logging.DEBUG)
log.addHandler(debug_handler)


class pyro_infer_scRT():
    def __init__(self, cn_s, cn_g1, input_col='reads', gc_col='gc', rt_prior_col='mcf7rt',
                 clone_col='clone_id', cell_col='cell_id', library_col='library_id',
                 chr_col='chr', start_col='start', cn_state_col='state',
                 rs_col='rt_state', frac_rt_col='frac_rt',
                 learning_rate=0.05, max_iter=1000, min_iter=100, rel_tol=5e-5,
                 cuda=False, seed=0, num_states=13, gc0=0, gc1=1.3, A=1, sigma1=0.1):
        '''
        initialise the pyro_infer_scRT object
        :param cn_s: long-form dataframe containing copy number and read count information from S-phase cells. (pandas.DataFrame)
        :param cn_g1: long-form dataframe containing copy number and read count information from G1-phase cells. (pandas.DataFrame)
        :param input_col: column containing read count input. (str)
        :param gc_col: column for gc content of each bin. (str)
        :param rt_prior_col: column RepliSeq-determined replication timing values to be used as a prior. (str)
        :param clone_col: column for clone ID of each cell. (str)
        :param cell_col: column for cell ID of each cell. (str)
        :param library_col: column for library ID of each cell. (str)
        :param chr_col: column for chromosome of each bin. (str)
        :param start_col: column for start position of each bin. (str)
        :param cn_state_col: column for the HMMcopy-determined somatic copy number state of each bin; only needs to be present in cn_g1. (str)
        :param rs_col: output column added containing the replication state of each bin for S-phase cells. (str)
        :param frac_rt_col: column added containing the fraction of replciated bins for each S-phase cell. (str)
        :param learning_rate: learning rate of Adam optimizer. (float)
        :param max_iter: max number of iterations of elbo optimization during inference. (int)
        :param rel_tol: when the relative change in elbo drops to rel_tol, stop inference. (float)
        :param cuda: use cuda tensor type. (bool)
        :param seed: random number generator seed. (int)
        :param num_states: number of integer copy number states to include in the model, values range from 0 to num_states-1. (int)
        :param gc0: prior for gc bias y-intercept parameter. (float)
        :param gc1: prior for gc bias slope parameter. (float)
        :param A: prior for per-cell replication stochasticity. (float)
        '''
        self.cn_s = cn_s
        self.cn_g1 = cn_g1

        self.input_col = input_col
        self.gc_col = gc_col
        self.rt_prior_col = rt_prior_col
        self.clone_col = clone_col
        self.cell_col = cell_col
        self.library_col = library_col
        self.chr_col = chr_col
        self.start_col = start_col
        self.cn_state_col = cn_state_col

        self.rs_col = rs_col
        self.frac_rt_col = frac_rt_col

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.rel_tol = rel_tol
        self.cuda = cuda
        self.seed = seed

        self.num_states = num_states
        self.gc0_prior = gc0
        self.gc1_prior = gc1
        self.A_prior = A
        self.sigma1_prior = sigma1

        self.map_estimates_s = None
        self.map_estimates_g1 = None


    def process_input_data(self):
        # sort rows by correct genomic ordering
        self.cn_g1 = self.sort_by_cell_and_loci(self.cn_g1)
        self.cn_s = self.sort_by_cell_and_loci(self.cn_s)

        # drop any row where read count input is NaN
        self.cn_g1 = self.cn_g1[self.cn_g1[self.input_col].notna()]
        self.cn_s = self.cn_s[self.cn_s[self.input_col].notna()]

        # pivot to 2D matrix where each row is a unique cell, columns are loci
        cn_g1_reads_df = self.cn_g1.pivot(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.input_col)
        cn_g1_states_df = self.cn_g1.pivot(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.cn_state_col)
        cn_s_reads_df = self.cn_s.pivot(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.input_col)

        cn_g1_reads_df = cn_g1_reads_df.dropna()
        cn_g1_states_df = cn_g1_states_df.dropna()
        cn_s_reads_df = cn_s_reads_df.dropna()

        assert cn_g1_states_df.shape == cn_g1_reads_df.shape
        assert cn_s_reads_df.shape[0] == cn_g1_reads_df.shape[0]

        # convert to tensor and unsqueeze the data dimension
        cn_g1_reads = torch.tensor(cn_g1_reads_df.values).to(torch.float32)
        cn_g1_states = torch.tensor(cn_g1_states_df.values).to(torch.float32)
        cn_s_reads = torch.tensor(cn_s_reads_df.values).to(torch.float32)

        # TODO: get tensors for GC and RT profiles
        gc_profile = self.cn_s[[self.chr_col, self.start_col, self.gc_col]].drop_duplicates()
        rt_prior_profile = self.cn_s[[self.chr_col, self.start_col, self.rt_prior_col]].drop_duplicates()

        gc_profile = gc_profile.dropna()
        rt_prior_profile = rt_prior_profile.dropna()

        assert cn_s_reads.shape[1] == gc_profile.shape[0] == rt_prior_profile.shape[0]

        gc_profile = torch.tensor(gc_profile[self.gc_col].values).unsqueeze(-1).to(torch.float32)
        rt_prior_profile = torch.tensor(rt_prior_profile[self.rt_prior_col].values).unsqueeze(-1).to(torch.float32)

        rt_prior_profile = self.convert_rt_prior_units(rt_prior_profile)

        return cn_g1_reads_df, cn_g1_states_df, cn_s_reads_df, cn_g1_reads, cn_g1_states, cn_s_reads, gc_profile, rt_prior_profile


    def sort_by_cell_and_loci(self, cn):
        """ Sort long-form dataframe so each cell follows correct genomic ordering """
        cn[self.chr_col] = cn[self.chr_col].astype('category')
        chr_order = [str(i+1) for i in range(22)]
        chr_order.append('X')
        chr_order.append('Y')
        cn[self.chr_col] = cn[self.chr_col].cat.set_categories(chr_order)
        cn = cn.sort_values(by=[self.cell_col, self.chr_col, self.start_col])
        return cn


    def convert_rt_prior_units(self, rt_prior_profile):
        """ Make sure that early RT regions are close to 1, late RT regions are close to 0 """
        rt_prior_profile = rt_prior_profile / max(rt_prior_profile)
        return rt_prior_profile



    @config_enumerate
    def model_S(self, read_profiles, gc_profile, rt_profile, gc_values, beta, A, num_states, include_prior=True, prior_CN_prob=0.99):
        with ignore_jit_warnings():
            num_cells, num_loci = map(int, read_profiles.shape)
            assert num_loci == gc_profile.shape[0]

        # scale each cell's read count so that it sums to 1 million reads
        F = int(1e6)
        read_profiles = read_profiles * F / torch.sum(read_profiles, 1).reshape(-1, 1)
        gc_profile = gc_profile.reshape(1, -1)
        epsilon = torch.finfo(torch.float32).eps
        
        # establish prior distributions
        with poutine.mask(mask=include_prior):
            # TODO: set values for init_somatic_CN that reflect each S-phase cell's closest G1-phase cell
            # the diploid (CN=2) state is currently set to 90% for all bins
            init_somatic_CN = torch.ones(num_cells, num_loci, num_states) * ((1 - prior_CN_prob) / (num_states - 1))
            init_somatic_CN[:, :, 2] = prior_CN_prob

            probs_A = pyro.sample('expose_A', dist.LogNormal(torch.ones(num_cells).fill_(A),
                                                             torch.ones(num_cells).fill_(0.1)).to_event(1))
            
            # time in S-phase should be drawn from a Beta distribution with a bell-shaped
            # curve centered around 0.5
            probs_s_times = pyro.sample("expose_s_times",
                                        dist.Beta(torch.ones(num_cells).fill_(2.0),
                                                  torch.ones(num_cells).fill_(2.0)).to_event(1))

            probs_bulk_RT = pyro.sample('expose_bulk_RT', dist.Normal(rt_profile, 0.1).to_event(2))
        
        somatic_CN_prob = pyro.sample('expose_somatic_CN_prob', dist.Dirichlet(init_somatic_CN).to_event(2))
        # draw somatic CN state from categorical
        somatic_CN = pyro.sample("somatic_CN", dist.Categorical(somatic_CN_prob).to_event(2))
        
        p_rep = 1 / (1 + torch.exp(-probs_A.reshape(-1, 1) * (probs_bulk_RT.reshape(1, -1) - (1 - torch.abs(probs_s_times.reshape(-1, 1))))))
        alphas = p_rep * 5 + 1
        betas = (1 - p_rep) * 5 + 1
        rt_state_probs = pyro.sample("expose_rt_state_prob", dist.Beta(alphas, betas).to_event(2))
        replicated = pyro.sample("rt_state", dist.Bernoulli(rt_state_probs).to_event(2))
        total_CN = somatic_CN * (1 + replicated)
        
        # add gc bias to the total CN
        # Is a simple linear model sufficient here?
        biased_CN = total_CN * ((gc_profile * gc_values[1]) + gc_values[0])
        biased_CN = biased_CN.reshape(num_cells, num_loci)

        # add gamma-distributed noise
        noisy_CN = pyro.sample("noisy_CN", dist.Gamma(biased_CN * beta + epsilon, beta).to_event(2))

        # draw true read count from multinomial distribution
        pyro.sample("read_count", dist.Multinomial(total_count=F, probs=noisy_CN, validate_args=False).to_event(1), obs=read_profiles)



    @config_enumerate
    def model_G1(self, read_profiles, gc_profile, gc0, gc1, sigma1, num_states, include_prior=True, prior_CN_prob=0.99):
        with ignore_jit_warnings():
            num_cells, num_loci = map(int, read_profiles.shape)
            assert num_loci == gc_profile.shape[0]

        # scale each cell's read count so that it sums to 1 million reads
        # this is necessary because multinomial sampling with different number of samples per cell
        # is not currently supported by pyro https://github.com/pytorch/pytorch/issues/42407
        F = int(1e6)
        read_profiles = read_profiles * F / torch.sum(read_profiles, 1).reshape(-1, 1)
        # print('read_profiles shape', read_profiles.shape)
        gc_profile = gc_profile.reshape(1, -1)
        beta_prior = torch.log(torch.tensor([1 / sigma1]))
        epsilon = torch.finfo(torch.float32).eps
        
        # establish prior distributions
        with poutine.mask(mask=include_prior):
            # priors on what the gc bias coefficients should look like        
            beta = pyro.sample('expose_beta', dist.LogNormal(torch.tensor([beta_prior]), torch.tensor([0.2])).to_event(1))

            # TODO: set values for init_somatic_CN that reflect each S-phase cell's closest G1-phase cell
            # the diploid (CN=2) state is currently set to 90% for all bins
            init_somatic_CN = torch.ones(num_cells, num_loci, num_states) * ((1 - prior_CN_prob) / (num_states - 1))
            init_somatic_CN[:, :, 2] = prior_CN_prob
        
        
        somatic_CN_prob = pyro.sample('expose_somatic_CN_prob', dist.Dirichlet(init_somatic_CN).to_event(2))
        # satisfy simplex constraint and remove any NaN values manually
        somatic_CN_prob = somatic_CN_prob / torch.sum(somatic_CN_prob, axis=2).reshape(num_cells, num_loci, 1)
        somatic_CN_prob = torch.nan_to_num(somatic_CN_prob)
        # draw somatic CN state from categorical
        somatic_CN = pyro.sample("somatic_CN", dist.Categorical(probs=somatic_CN_prob).to_event(2))
        
        # add gc bias to the total CN
        # Is a simple linear model sufficient here?
        gc_values = pyro.sample("expose_gc_values", dist.HalfNormal(torch.tensor([gc0+epsilon, gc1+epsilon]), 0.1).to_event(1))
        biased_CN = somatic_CN * ((gc_profile * gc_values[1]) + gc_values[0])
        biased_CN = biased_CN.reshape(num_cells, num_loci)
        
        # add gamma-distributed noise
        noisy_CN = pyro.sample("noisy_CN", dist.Gamma(biased_CN * beta + epsilon, beta).to_event(2))

        # draw true read count from multinomial distribution
        pyro.sample("read_count", dist.Multinomial(total_count=F, probs=noisy_CN, validate_args=False).to_event(1), obs=read_profiles)


    def model_4(self, read_profiles, gc_profile, rt_profile, gc0, gc1, A, num_states, include_prior=True, prior_CN_prob=0.95):
        with ignore_jit_warnings():
            num_cells, num_loci = map(int, read_profiles.shape)
            assert num_loci == gc_profile.shape[0]

        # scale each cell's read count so that it sums to 1 million reads
        # this is necessary because multinomial sampling with different number of samples per cell
        # is not currently supported by pyro https://github.com/pytorch/pytorch/issues/42407
        F = int(1e6)
        read_profiles = read_profiles * F / torch.sum(read_profiles, 1).reshape(-1, 1)
        gc_profile = gc_profile.reshape(1, -1)

        # establish prior distributions
        with poutine.mask(mask=include_prior):
            # priors on what the gc bias coefficients should look like
            probs_gc = pyro.param('expose_gc', torch.tensor([gc0, gc1]))
        
            # TODO: set values for init_somatic_CN that reflect each S-phase cell's closest G1-phase cell
            # the diploid (CN=2) state is currently set to 90% for all bins
            init_somatic_CN = torch.ones(num_cells, num_loci, num_states) * ((1 - prior_CN_prob) / (num_states - 1))
            init_somatic_CN[:, :, 2] = prior_CN_prob


        somatic_CN_prob = pyro.sample('expose_somatic_CN_prob', dist.Dirichlet(init_somatic_CN).to_event(2))
        # draw somatic CN state from categorical
        somatic_CN = pyro.sample("somatic_CN", dist.Categorical(somatic_CN_prob).to_event(2))

        # add gc bias to the total CN
        biased_CN = somatic_CN * ((gc_profile * probs_gc[1]) + probs_gc[0])
        biased_CN = biased_CN.reshape(num_cells, num_loci)

        # draw true read count from multinomial distribution
        pyro.sample("read_count", dist.Multinomial(total_count=F, probs=biased_CN, validate_args=False).to_event(1), obs=read_profiles)




    def run_pyro_model(self):
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        logging.info('Loading data')

        logging.info('-' * 40)
        model_s = self.model_S
        model_g1 = self.model_G1

        optim = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)

        cn_g1_reads_df, cn_g1_states_df, cn_s_reads_df, cn_g1_reads, cn_g1_states, cn_s_reads, gc_profile, rt_prior_profile = self.process_input_data()

        # TODO: Assign S-phase cells to best G1-phase matching cell

        # TODO: fit GC params using G1-phase cells
        guide_g1 = AutoDelta(poutine.block(model_g1, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        svi = SVI(model_g1, guide_g1, optim, loss=elbo)

        # start inference
        logging.info('Start inference for G1-phase cells.')
        losses = []
        for i in range(self.max_iter):
            loss = svi.step(cn_g1_reads, gc_profile, self.gc0_prior, self.gc1_prior, self.sigma1_prior, self.num_states)

            if i >= self.min_iter:
                loss_diff = abs((losses[-1] - loss) / losses[-1])
                if loss_diff < self.rel_tol:
                    logging.info('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            logging.info('step: {}, loss: {}'.format(i, loss))

        map_estimates_g1 = guide_g1(cn_g1_reads, gc_profile, self.gc0_prior, self.gc1_prior, self.sigma1_prior, self.num_states)
        self.map_estimates_g1 = map_estimates_g1

        # extract GC and beta params from fitting to G1-phase cells
        gc0_fit, gc1_fit = map_estimates_g1['expose_gc_values'].detach().numpy()
        beta_fit = map_estimates_g1['expose_beta'].detach().numpy()[0]
        gc_values = [gc0_fit, gc1_fit]

        # extract CN states from G1-phase cells
        cn_g1_states_out_probs = np.array(map_estimates_g1['expose_somatic_CN_prob'].detach().numpy())
        cn_g1_states_out = np.argmax(cn_g1_states_out_probs, axis=2)
        cn_g1_states_out_df = pd.DataFrame(data=cn_g1_states_out, index=cn_g1_states_df.index, columns=cn_g1_states_df.columns)

        cn_g1_out = pd.merge(self.cn_g1, cn_g1_states_out_df.melt(ignore_index=False, value_name='assigned_state').reset_index())
        cn_g1_out['model_gc0'] = gc0_fit
        cn_g1_out['model_gc1'] = gc1_fit
        cn_g1_out['model_beta'] = beta_fit

        # # Now run the model for S-phase cells
        # logging.info('read_profiles after loading data: {}'.format(cn_s_reads.shape))
        # logging.info('read_profiles data type: {}'.format(cn_s_reads.dtype))

        num_observations = float(cn_s_reads.shape[0] * cn_s_reads.shape[1])
        pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()
        pyro.enable_validation(__debug__)

        guide_s = AutoDelta(poutine.block(model_s, expose_fn=lambda msg: msg["name"].startswith("expose_")))
        optim_s = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo_s = TraceEnum_ELBO(max_plate_nesting=1)
        svi_s = SVI(model_s, guide_s, optim_s, loss=elbo_s)

        # start inference
        logging.info('Start inference for S-phase cells.')
        losses = []
        for i in range(self.max_iter):
            loss = svi_s.step(cn_s_reads, gc_profile, rt_prior_profile, gc_values, beta_fit, self.A_prior, self.num_states)

            if i >= self.min_iter:
                loss_diff = abs((losses[-1] - loss) / losses[-1])
                if loss_diff < self.rel_tol:
                    logging.info('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            logging.info('step: {}, loss: {}'.format(i, loss))

        map_estimates_s = guide_s(cn_s_reads, gc_profile, rt_prior_profile, gc_values, beta_fit, self.A_prior, self.num_states)
        self.map_estimates_s = map_estimates_s

        # add inferred CN and RT states to the S-phase output df
        cn_s_state_out_probs = map_estimates_s['expose_somatic_CN_prob'].detach().numpy()
        rt_state_probs = map_estimates_s['expose_rt_state_prob'].detach().numpy()
        rt_state_probs_df = pd.DataFrame(rt_state_probs, index=cn_s_reads_df.index, columns=cn_s_reads_df.columns)
        cn_states_argmax_df = pd.DataFrame(np.argmax(cn_s_state_out_probs, axis=2), index=cn_s_reads_df.index, columns=cn_s_reads_df.columns)
        cn_states_argmax_df = cn_states_argmax_df.melt(ignore_index=False, value_name='model_maxprob_cn_state').reset_index()
        rt_state_probs_df = rt_state_probs_df.melt(ignore_index=False, value_name='model_prob_rt_state').reset_index()
        cn_s_out = pd.merge(cn_s, cn_states_argmax_df)
        cn_s_out = pd.merge(cn_s_out, rt_state_probs_df)
        cn_s_out['model_maxprob_rt_state'] = cn_s_out['model_prob_rt_state'].apply(lambda x: 1 if x>0.5 else 0)

        # add other inferred parameters to cn_s_out
        s_times_out = pd.DataFrame(
            map_estimates_s['expose_s_times'].detach().numpy(),
            index=cn_s_reads_df.index, 
            columns=['model_s_time']).reset_index()
        As_out = pd.DataFrame(
            map_estimates_s['expose_A'].detach().numpy(),
            index=cn_s_reads_df.index, 
            columns=['model_A']).reset_index()
        bulk_rt_out = pd.DataFrame(
            map_estimates_s['expose_bulk_RT'].detach().numpy(),
            index=cn_s_reads_df.columns,
            columns=['model_bulk_RT']).reset_index()
        cn_s_out = pd.merge(cn_s_out, s_times_out)
        cn_s_out = pd.merge(cn_s_out, As_out)
        cn_s_out = pd.merge(cn_s_out, bulk_rt_out)
        cn_s_out['model_gc0'] = gc0_fit
        cn_s_out['model_gc1'] = gc1_fit
        cn_s_out['model_beta'] = beta_fit
        
        return cn_g1_out, cn_s_out
