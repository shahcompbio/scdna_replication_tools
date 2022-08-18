import argparse
import logging
import sys

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, infer_discrete, config_enumerate
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings, optional

import numpy as np
import pandas as pd

from scdna_replication_tools.compute_consensus_clone_profiles import compute_consensus_clone_profiles

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
                 rs_col='rt_state', frac_rt_col='frac_rt', cn_prior_method='hmmcopy',
                 learning_rate=0.05, max_iter=2000, min_iter=100, rel_tol=5e-5,
                 cuda=False, seed=0, num_states=13, poly_degree=4):
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
        :param cn_prior_method: Method for building the cn prior. Options are 'hmmcopy', 'g1_cells', 'g1_clones', 'diploid'. (str)
        :param learning_rate: learning rate of Adam optimizer. (float)
        :param max_iter: max number of iterations of elbo optimization during inference. (int)
        :param rel_tol: when the relative change in elbo drops to rel_tol, stop inference. (float)
        :param cuda: use cuda tensor type. (bool)
        :param seed: random number generator seed. (int)
        :param num_states: number of integer copy number states to include in the model, values range from 0 to num_states-1. (int)
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
        self.poly_degree = poly_degree

        self.cn_prior_method = cn_prior_method


    def process_input_data(self):
        # sort rows by correct genomic ordering
        self.cn_g1 = self.sort_by_cell_and_loci(self.cn_g1)
        self.cn_s = self.sort_by_cell_and_loci(self.cn_s)

        # drop any row where read count input is NaN
        self.cn_g1 = self.cn_g1[self.cn_g1[self.input_col].notna()]
        self.cn_s = self.cn_s[self.cn_s[self.input_col].notna()]

        # pivot to 2D matrix where each row is a unique cell, columns are loci
        cn_g1_reads_df = self.cn_g1.pivot_table(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.input_col)
        cn_g1_states_df = self.cn_g1.pivot_table(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.cn_state_col)
        cn_s_reads_df = self.cn_s.pivot_table(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.input_col)
        cn_s_states_df = self.cn_s.pivot_table(index=self.cell_col, columns=[self.chr_col, self.start_col], values=self.cn_state_col)

        cn_g1_reads_df = cn_g1_reads_df.dropna()
        cn_g1_states_df = cn_g1_states_df.dropna()
        cn_s_reads_df = cn_s_reads_df.dropna()
        cn_s_states_df = cn_s_states_df.dropna()

        assert cn_g1_states_df.shape == cn_g1_reads_df.shape
        assert cn_s_reads_df.shape[1] == cn_g1_reads_df.shape[1]

        cn_g1_reads_df = cn_g1_reads_df.T
        cn_g1_states_df = cn_g1_states_df.T
        cn_s_reads_df = cn_s_reads_df.T
        cn_s_states_df = cn_s_states_df.T

        # convert to tensor and unsqueeze the data dimension
        # convert to int64 before float32 to ensure that all values are rounded to the nearest int
        cn_g1_reads = torch.tensor(cn_g1_reads_df.values).to(torch.int64).to(torch.float32)
        cn_g1_states = torch.tensor(cn_g1_states_df.values).to(torch.int64).to(torch.float32)
        cn_s_reads = torch.tensor(cn_s_reads_df.values).to(torch.int64).to(torch.float32)
        cn_s_states = torch.tensor(cn_s_states_df.values).to(torch.int64).to(torch.float32)

        # TODO: get tensors for GC and RT profiles
        gc_profile = self.cn_s[[self.chr_col, self.start_col, self.gc_col]].drop_duplicates()
        gc_profile = gc_profile.dropna()
        gc_profile = torch.tensor(gc_profile[self.gc_col].values).to(torch.float32)

        if (self.rt_prior_col is not None) and (self.rt_prior_col in self.cn_s.columns):
            rt_prior_profile = self.cn_s[[self.chr_col, self.start_col, self.rt_prior_col]].drop_duplicates()
            rt_prior_profile = rt_prior_profile.dropna()
            rt_prior_profile = torch.tensor(rt_prior_profile[self.rt_prior_col].values).unsqueeze(-1).to(torch.float32)
            rt_prior_profile = self.convert_rt_prior_units(rt_prior_profile)
            assert cn_s_reads.shape[0] == gc_profile.shape[0] == rt_prior_profile.shape[0]
        else:
            rt_prior_profile = None

        return cn_g1_reads_df, cn_g1_states_df, cn_s_reads_df, cn_s_states_df, cn_g1_reads, cn_g1_states, cn_s_reads, cn_s_states, gc_profile, rt_prior_profile


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


    def build_trans_mat(self, cn):
        trans_mat = torch.eye(self.num_states, self.num_states) + 1
        num_loci, num_cells = cn.shape
        for i in range(num_cells):
            for j in range(1, num_loci):
                cur_state = int(cn[j, i])
                prev_state = int(cn[j-1, i])
                trans_mat[prev_state, cur_state] += 1
        return trans_mat


    def build_cn_prior(self, cn, weight=1e6):
        num_loci, num_cells = cn.shape
        cn_prior = torch.ones(num_loci, num_cells, self.num_states)
        for i in range(num_loci):
            for n in range(num_cells):
                state = int(cn[i, n].numpy())
                cn_prior[i, n, state] = weight
        return cn_prior


    def make_gc_features(self, x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in reversed(range(0, self.poly_degree+1))], 1)


    @config_enumerate
    def model_s(self, gc_profile, betas=None, cn0=None, rt0=None, num_cells=None, num_loci=None, data=None, trans_mat=None, cn_prior=None, u_guess=70., nb_r_guess=10000.):
        with ignore_jit_warnings():
            if data is not None:
                num_loci, num_cells = data.shape
            elif cn0 is not None:
                num_loci, num_cells = cn0.shape
            assert num_cells is not None
            assert num_loci is not None

        # controls the consistency of replicating on time
        a = pyro.sample('expose_a', dist.Gamma(torch.tensor([2.]), torch.tensor([0.2])))
        
        # negative binomial dispersion
        nb_r = pyro.param('expose_nb_r', torch.tensor([nb_r_guess]), constraint=constraints.positive)
        
        # define cell and loci plates
        loci_plate = pyro.plate('num_loci', num_loci, dim=-2)
        cell_plate = pyro.plate('num_cells', num_cells, dim=-1)

        # gc bias params
        if betas is None:
            betas = pyro.sample('expose_betas', dist.Normal(0., 1.).expand([self.poly_degree+1]).to_event(1))

        if rt0 is not None:
            # fix rt as constant when input into model
            rt = rt0
        else:
            with loci_plate:
                # bulk replication timing profile
                rt = pyro.sample('expose_rt', dist.Beta(torch.tensor([1.]), torch.tensor([1.])))

        with cell_plate:

            # per cell replication time
            time = pyro.sample('expose_time', dist.Beta(torch.tensor([1.]), torch.tensor([1.])))

            # per cell reads per copy per bin
            u = pyro.sample('expose_u', dist.Normal(torch.tensor([u_guess]), torch.tensor([u_guess/10.])))
            
            with loci_plate:

                if cn0 is None:
                    if cn_prior is None:
                        cn_prior = torch.ones(num_loci, num_cells, 13)
                    # sample cn probabilities of each bin from Dirichlet
                    cn_prob = pyro.sample('expose_cn_prob', dist.Dirichlet(cn_prior))
                    # sample cn state from categorical based on cn_prob
                    cn = pyro.sample('cn', dist.Categorical(cn_prob), infer={"enumerate": "parallel"})

                # per cell per bin late or early 
                t_diff = time.reshape(-1, num_cells) - rt.reshape(num_loci, -1)

                # probability of having been replicated
                p_rep = 1 / (1 + torch.exp(-a * t_diff))

                # binary replicated indicator
                rep = pyro.sample('rep', dist.Bernoulli(p_rep), infer={"enumerate": "parallel"})

                # copy number accounting for replication
                rep_cn = cn * (1. + rep)

                # copy number accounting for gc bias
                gc_features = self.make_gc_features(gc_profile)
                gc_rate = torch.exp(torch.sum(betas * gc_features, 1))
                biased_cn = rep_cn * gc_rate.reshape(-1, 1)

                # expected reads per bin per cell
                expected_reads = (u * biased_cn)

                nb_p = expected_reads / (expected_reads + nb_r)
                
                if data is not None:
                    obs = data
                else:
                    obs = None
                
                reads = pyro.sample('reads', dist.NegativeBinomial(nb_r, probs=nb_p), obs=obs)


    @config_enumerate
    def model_g1(self, gc_profile, cn=None, num_cells=None, num_loci=None, data=None, u_guess=70.):
        with ignore_jit_warnings():
            if data is not None:
                num_loci, num_cells = data.shape
            elif cn is not None:
                num_loci, num_cells = cn.shape
            assert num_cells is not None
            assert num_loci is not None
        
        # negative binomial dispersion
        nb_r = pyro.param('expose_nb_r', torch.tensor([10000.0]), constraint=constraints.positive)

        # gc bias params
        betas = pyro.sample('expose_betas', dist.Normal(0., 1.).expand([self.poly_degree+1]).to_event(1))

        with pyro.plate('num_cells', num_cells):

            # per cell reads per copy per bin
            u = pyro.sample('expose_u', dist.Normal(torch.tensor([u_guess]), torch.tensor([u_guess/10.])))

            with pyro.plate('num_loci', num_loci):

                # copy number accounting for gc bias
                gc_features = self.make_gc_features(gc_profile)
                gc_rate = torch.exp(torch.sum(betas * gc_features, 1))
                biased_cn = cn * gc_rate.reshape(-1, 1)

                # expected reads per bin per cell
                expected_reads = (u * biased_cn)

                nb_p = expected_reads / (expected_reads + nb_r)

                if data is not None:
                    obs = data
                else:
                    obs = None

                reads = pyro.sample('reads', dist.NegativeBinomial(nb_r, probs=nb_p), obs=obs)


    def run_pyro_model(self):
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        logging.info('Loading data')

        logging.info('-' * 40)
        model_s = self.model_s
        model_g1 = self.model_g1

        optim = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo = JitTraceEnum_ELBO(max_plate_nesting=2)

        cn_g1_reads_df, cn_g1_states_df, cn_s_reads_df, cn_s_states_df, \
            cn_g1_reads, cn_g1_states, cn_s_reads, cn_s_states, \
            gc_profile, rt_prior_profile = self.process_input_data()

        # build transition matrix and cn prior for S-phase cells
        trans_mat = self.build_trans_mat(cn_g1_states)

        if self.cn_prior_method == 'hmmcopy':
            # use hmmcopy states for the S-phase cells to build the prior
            cn_prior = self.build_cn_prior(cn_s_states)
        elif self.cn_prior_method == 'g1_cells':
            # use G1-phase cell that has highest correlation to each S-phase cell as prior
            raise ValueError("g1_cells method not implemented yet")
            cn_prior = ...
        elif self.cn_prior_method == 'g1_clones':
            # use G1-phase clone that has highest correlation to each S-phase cell as prior
            # compute consensuse clone profiles for cn state
            clone_cn_profiles = compute_consensus_clone_profiles(
                self.cn_g1, self.cn_state_col, clone_col=self.clone_col, cell_col=self.cell_col, chr_col=self.chr_col,
                start_col=self.start_col, cn_state_col=self.cn_state_col
            )

            print('clone_cn_profiles\n', clone_cn_profiles)
            print('cn_s prior to adding cn_prior\n', self.cn_s.head())
            print('cn_s_reads_df\n', cn_s_reads_df.head())
            print('cn_s_reads_df.shape', cn_s_reads_df.shape)
            print('cn_s_states.shape', cn_s_states.shape)

            cn_prior_input = torch.zeros(cn_s_states.shape)
            for i, cell_id in enumerate(cn_s_reads_df.columns):
                if i < 2:
                    print('cell_id', cell_id)
                cell_cn = self.cn_s.loc[self.cn_s[self.cell_col]==cell_id]  # get full cn data for this cell
                if i < 2:
                    print('cell_cn\n', cell_cn.head())
                cell_clone = cell_cn[self.clone_col].values[0]  # get clone id
                cn_prior_input[:, i] = torch.tensor(clone_cn_profiles[cell_clone].values).to(torch.int64).to(torch.float32)  # assign consensus clone cn profile for this cell
            
            print('cn_prior_input\n', cn_prior_input)

            # build a proper prior over num_states using the consensus clone cn calls for each cell
            cn_prior = self.build_cn_prior(cn_prior_input)
        elif self.cn_prior_method == 'diploid':
            # assume that every S-phase cell has a diploid prior
            num_loci, num_cells = cn_s_states.shape
            cn_s_diploid = torch.ones(num_loci, num_cells, self.num_states) * 2
            cn_prior = self.build_cn_prior(cn_s_diploid)
        else:
            # assume uniform prior otherwise
            num_loci, num_cells = cn_s_states.shape
            cn_prior = torch.ones(num_loci, num_cells, self.num_states) / self.num_states

        print('cn_prior_method', self.cn_prior_method)
        print('cn_prior\n', cn_prior)

        # TODO: Assign S-phase cells to best G1-phase matching cell

        # guess the initial mean for u assuming that no bins should be replicated
        u_guess_g1 = torch.mean(cn_g1_reads) / torch.mean(cn_g1_states)

        # TODO: fit GC params using G1-phase cells    
        guide_g1 = AutoDelta(poutine.block(model_g1, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        svi = SVI(model_g1, guide_g1, optim, loss=elbo)

        # start inference
        logging.info('Start inference for G1-phase cells.')
        losses = []
        for i in range(self.max_iter):
            loss = svi.step(gc_profile, cn=cn_g1_states, data=cn_g1_reads, u_guess=u_guess_g1)

            # fancy convergence check that sees if the past 10 iterations have plateaued
            if i >= self.min_iter:
                loss_diff = abs(max(losses[-10:-1]) - min(losses[-10:-1])) / abs(losses[-1])
                if loss_diff < 5e-5:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            logging.info('step: {}, loss: {}'.format(i, loss))


        # replay model
        guide_trace_g1 = poutine.trace(guide_g1).get_trace(gc_profile, cn=cn_g1_states, data=cn_g1_reads, u_guess=u_guess_g1)
        trained_model_g1 = poutine.replay(model_g1, trace=guide_trace_g1)

        # infer discrete sites and get model trace
        inferred_model_g1 = infer_discrete(
            trained_model_g1, temperature=0,
            first_available_dim=-3)
        trace_g1 = poutine.trace(inferred_model_g1).get_trace(gc_profile, cn=cn_g1_states, data=cn_g1_reads, u_guess=u_guess_g1)

        # extract fitted parameters
        nb_r_fit = trace_g1.nodes['expose_nb_r']['value'].detach()
        betas_fit = trace_g1.nodes['expose_betas']['value'].detach()
        u_fit = trace_g1.nodes['expose_u']['value'].detach()

        # # Now run the model for S-phase cells
        # logging.info('read_profiles after loading data: {}'.format(cn_s_reads.shape))
        # logging.info('read_profiles data type: {}'.format(cn_s_reads.dtype))


        num_observations = float(cn_s_reads.shape[0] * cn_s_reads.shape[1])
        pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()
        pyro.enable_validation(__debug__)

        guide_s = AutoDelta(poutine.block(model_s, expose_fn=lambda msg: msg["name"].startswith("expose_")))
        optim_s = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo_s = JitTraceEnum_ELBO(max_plate_nesting=2)
        svi_s = SVI(model_s, guide_s, optim_s, loss=elbo_s)

        # guess the initial mean for u assuming that half the bins should be replicated
        u_guess_s = torch.mean(cn_s_reads) / (1.5 * torch.mean(cn_g1_states))

        # start inference
        logging.info('Start inference for S-phase cells.')
        losses = []
        for i in range(self.max_iter):
            loss = svi_s.step(gc_profile, betas=betas_fit, data=cn_s_reads, cn_prior=cn_prior, u_guess=u_guess_s, nb_r_guess=nb_r_fit)

            # fancy convergence check that sees if the past 10 iterations have plateaued
            if i >= self.min_iter:
                loss_diff = abs(max(losses[-10:-1]) - min(losses[-10:-1])) / abs(losses[-1])
                if loss_diff < 5e-5:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            logging.info('step: {}, loss: {}'.format(i, loss))

        # replay model
        guide_trace_s = poutine.trace(guide_s).get_trace(gc_profile, betas=betas_fit, data=cn_s_reads, cn_prior=cn_prior, u_guess=u_guess_s, nb_r_guess=nb_r_fit)
        trained_model_s = poutine.replay(model_s, trace=guide_trace_s)

        # infer discrete sites and get model trace
        inferred_model_s = infer_discrete(
            trained_model_s, temperature=0,
            first_available_dim=-3)
        trace_s = poutine.trace(inferred_model_s).get_trace(gc_profile, betas=betas_fit, data=cn_s_reads, cn_prior=cn_prior, u_guess=u_guess_s, nb_r_guess=nb_r_fit)

        # extract fitted parameters
        nb_r_fit_s = trace_s.nodes['expose_nb_r']['value']
        u_fit_s = trace_s.nodes['expose_u']['value']
        rt_fit_s = trace_s.nodes['expose_rt']['value']
        a_fit_s = trace_s.nodes['expose_a']['value']
        time_fit_s = trace_s.nodes['expose_time']['value']
        model_rep = trace_s.nodes['rep']['value']
        model_cn = trace_s.nodes['cn']['value']

        # add inferred CN and RT states to the S-phase output df
        model_cn_df = pd.DataFrame(model_cn.detach().numpy(), index=cn_s_reads_df.index, columns=cn_s_reads_df.columns)
        model_rep_df = pd.DataFrame(model_rep.detach().numpy(), index=cn_s_reads_df.index, columns=cn_s_reads_df.columns)
        model_cn_df = model_cn_df.melt(ignore_index=False, value_name='model_cn_state').reset_index()
        model_rep_df = model_rep_df.melt(ignore_index=False, value_name='model_rep_state').reset_index()
        cn_s_out = pd.merge(self.cn_s, model_cn_df)
        cn_s_out = pd.merge(cn_s_out, model_rep_df)

        # add other inferred parameters to cn_s_out
        s_times_out = pd.DataFrame(
            time_fit_s.detach().numpy(),
            index=cn_s_reads_df.columns, 
            columns=['model_s_time']).reset_index()
        Us_out = pd.DataFrame(
            u_fit_s.detach().numpy(),
            index=cn_s_reads_df.columns, 
            columns=['model_u']).reset_index()
        bulk_rt_out = pd.DataFrame(
            rt_fit_s.detach().numpy(),
            index=cn_s_reads_df.index,
            columns=['model_bulk_RT']).reset_index()
        cn_s_out = pd.merge(cn_s_out, s_times_out)
        cn_s_out = pd.merge(cn_s_out, Us_out)
        cn_s_out = pd.merge(cn_s_out, bulk_rt_out)
        cn_s_out['model_nb_r_s'] = nb_r_fit_s.detach().numpy()[0]
        cn_s_out['model_a'] = a_fit_s.detach().numpy()[0]
        for i in range(self.poly_degree):
            cn_s_out['model_gc_beta{}'.format(i)] = betas_fit.detach().numpy()[i]
        
        return cn_s_out
