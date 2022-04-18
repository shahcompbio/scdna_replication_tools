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
                 learning_rate=0.05, max_iter=400, rel_tol=5e-5,
                 cuda=False, seed=0, num_states=13, gc0=0, gc1=1.3, A=5):
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
        self.rel_tol = rel_tol
        self.cuda = cuda
        self.seed = seed

        self.num_states = num_states
        self.gc0_prior = gc0
        self.gc1_prior = gc1
        self.A_prior = A

        self.map_estimates = None


    def process_input_data(self):
        # sort rows by correct genomic ordering
        self.cn_g1 = self.sort_by_cell_and_loci(self.cn_g1)
        self.cn_s = self.sort_by_cell_and_loci(self.cn_s)

        # drop any row where read count input is NaN
        self.cn_g1 = self.cn_g1[self.cn_g1[self.input_col].notna()]
        self.cn_s = self.cn_s[self.cn_s[self.input_col].notna()]

        # pivot to 2D matrix where each row is a unique cell, columns are loci
        cn_g1_reads = self.cn_g1.pivot(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.input_col)
        cn_g1_states = self.cn_g1.pivot(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.cn_state_col)
        cn_s_reads = self.cn_s.pivot(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.input_col)

        cn_g1_reads = cn_g1_reads.dropna()
        cn_g1_states = cn_g1_states.dropna()
        cn_s_reads = cn_s_reads.dropna()

        assert cn_g1_states.shape == cn_g1_reads.shape
        assert cn_s_reads.shape[1] == cn_g1_reads.shape[1]

        # convert to tensor and unsqueeze the data dimension
        cn_g1_reads = torch.tensor(cn_g1_reads.values).unsqueeze(-1).to(torch.float32)
        cn_g1_states = torch.tensor(cn_g1_states.values).unsqueeze(-1).to(torch.float32)
        cn_s_reads = torch.tensor(cn_s_reads.values).unsqueeze(-1).to(torch.float32)

        # TODO: get tensors for GC and RT profiles
        gc_profile = self.cn_s[[self.chr_col, self.start_col, self.gc_col]].drop_duplicates()
        rt_prior_profile = self.cn_s[[self.chr_col, self.start_col, self.rt_prior_col]].drop_duplicates()

        gc_profile = gc_profile.dropna()
        rt_prior_profile = rt_prior_profile.dropna()

        assert cn_s_reads.shape[1] == gc_profile.shape[0] == rt_prior_profile.shape[0]

        gc_profile = torch.tensor(gc_profile[self.gc_col].values).unsqueeze(-1).to(torch.float32)
        rt_prior_profile = torch.tensor(rt_prior_profile[self.rt_prior_col].values).unsqueeze(-1).to(torch.float32)

        rt_prior_profile = self.convert_rt_prior_units(rt_prior_profile)

        return cn_g1_reads, cn_g1_states, cn_s_reads, gc_profile, rt_prior_profile


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
    def model_0(self, read_profiles, gc_profile, rt_profile, gc0, gc1, A, num_states, include_prior=True, prior_CN_prob=0.95):

        with ignore_jit_warnings():
            num_cells, num_loci, data_dim = map(int, read_profiles.shape)
            assert num_loci == gc_profile.shape[0]

        # scale each cell's read count so that it sums to 1 million reads
        # this is necessary because multinomial sampling with different number of samples per cell
        # is not currently supported by pyro https://github.com/pytorch/pytorch/issues/42407
        F = 1E6
        read_profiles = torch.reshape(read_profiles.reshape(num_cells, num_loci) * F / torch.sum(read_profiles, 1), (num_cells, num_loci, data_dim))
        
        # establish prior distributions
        with poutine.mask(mask=include_prior):

            # time in S-phase should be drawn from a Beta distribution with a bell-shaped
            # curve centered around 0.5
            probs_s_times = pyro.sample("expose_s_times",
                                        dist.Beta(torch.ones(num_cells).fill_(2.0),
                                                  torch.ones(num_cells).fill_(2.0))
                                        .to_event(1))

            # A controls the "noise" of replication for each cell
            init_A = torch.ones(num_cells).fill_(A)
            probs_A = pyro.param('expose_A', init_A, constraint=constraints.greater_than(0))

            # bulk replication timing value of each bin (ranging from 0-1)
            probs_bulk_RT = pyro.param('expose_bulk_RT', rt_profile, constraint=constraints.interval(0, 1))

            # priors on what the gc bias coefficients should look like
            probs_gc = pyro.param('expose_gc', torch.tensor([gc0, gc1]))

            # prior for the cell-specific noise coefficient
            # init_sigma1 = torch.ones(num_cells).fill_(sigma1)
            # probs_sigma1 = pyro.param('expose_sigma1', init_sigma1, constraint=constraints.greater_than(0))

            # TODO: set values for init_somatic_CN that reflect each S-phase cell's closest G1-phase cell
            # the diploid (CN=2) state is currently set to 90% for all bins
            init_somatic_CN = torch.ones(num_cells, num_loci, num_states) * ((1 - prior_CN_prob) / (num_states - 1))
            init_somatic_CN[:, :, 2] = prior_CN_prob
            somatic_CN = pyro.sample("expose_somatic_CN", dist.Categorical(init_somatic_CN).to_event(1))


        with pyro.plate("cells", num_cells):
            with pyro.plate("loci", num_loci):
                # compute the probability of each bin being replicated for this cell given
                # probs_A, probs_bulk_RT, and probs_s_times
                p_rep = 1 / (1 + torch.exp(-probs_A.reshape(-1, 1) * (probs_bulk_RT - (1 - torch.abs(probs_s_times.reshape(-1, 1))))))

                replicated = pyro.sample("expose_rt_state", dist.Binomial(1, p_rep))

                total_CN = somatic_CN * (1 + replicated)

                # add gc bias to the total CN
                # Is a simple linear model sufficient here?
                biased_CN = total_CN * ((gc_profile * probs_gc[1]) + probs_gc[0])

                # add some random noise to the GC-biased copy number
                # This doesn't seem necessary for now but can add back later
                # noisy_CN = pyro.sample("noisy_CN", dist.Gamma(biased_CN * probs_sigma1.reshape(-1, 1), 1/probs_sigma1.reshape(-1, 1)))

                # draw true read count from multinomial distribution
                pyro.sample("read_count", dist.Multinomial(F, biased_CN), obs=read_profiles)


    def run_pyro_model(self):
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        logging.info('Loading data')

        logging.info('-' * 40)
        model = self.model_0

        optim = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=2)

        cn_g1_reads, cn_g1_states, cn_s_reads, gc_profile, rt_prior_profile = self.process_input_data()

        # TODO: Assign S-phase cells to best G1-phase matching cell


        # TODO: fit GC params using G1-phase cells


        logging.info('read_profiles after loading data: {}'.format(cn_s_reads.shape))
        logging.info('read_profiles data type: {}'.format(cn_s_reads.dtype))

        num_observations = float(cn_s_reads.shape[0] * cn_s_reads.shape[1])
        pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()
        pyro.enable_validation(__debug__)

        global_guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        svi = SVI(model, global_guide, optim, loss=elbo)

        # start inference
        print('Start Inference.')
        losses = []
        for i in range(self.max_iter):
            loss = svi.step(cn_s_reads, gc_profile, rt_prior_profile, self.gc0_prior, self.gc1_prior, self.A_prior, self.num_states)

            if i >= 1:
                loss_diff = abs((losses[-1] - loss) / losses[-1])
                if loss_diff < self.rel_tol:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            print('.' if i % 200 else '\n', end='')

        map_estimates = global_guide(cn_s_reads, gc_profile, rt_prior_profile, self.gc0_prior, self.gc1_prior, self.A_prior, self.num_states)

        # record parameters for CN and RT states --> store as DFs
        somatic_CN_prob = map_estimates['expose_somatic_CN']
        somatic_CN_prob_df = pd.DataFrame(somatic_CN_prob.data.numpy())

        RT_state_prob = map_estimates['expose_rt_state']
        RT_state_prob_df = pd.DataFrame(RT_state_prob.data.numpy())

        # store other parameters
        self.map_estimates = map_estimates
        print(map_estimates)

        print(somatic_CN_prob.head())
        print(somatic_CN_prob.shape)
        print(RT_state_prob.head())
        print(RT_state_prob.shape)
        
        return somatic_CN_prob_df, RT_state_prob_df
