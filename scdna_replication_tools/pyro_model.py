import argparse
import logging
import sys

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, JitTrace_ELBO, TraceEnum_ELBO, TraceTMC_ELBO, infer_discrete, config_enumerate
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings, optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.stats import skew
from scipy.spatial.distance import cityblock

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
                 cuda=False, seed=0, num_states=13, poly_degree=4, gamma=6):
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
        :param gamma: value that alpha and beta should sum to when creating a beta distribution prior for time. (int)
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

        self.num_libraries = None

        self.gamma = gamma


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

        cn_g1_reads_df = cn_g1_reads_df.dropna(axis=1)
        cn_g1_states_df = cn_g1_states_df.dropna(axis=1)
        cn_s_reads_df = cn_s_reads_df.dropna(axis=1)
        cn_s_states_df = cn_s_states_df.dropna(axis=1)

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

        # get tensor of library_id index
        # need this because each library will have unique gc params
        libs_s, libs_g1 = self.get_libraries_tensor(self.cn_s, self.cn_g1)

        # make sure there's one library index per cell
        assert libs_s.shape[0] == cn_s_reads.shape[1]
        assert libs_g1.shape[0] == cn_g1_reads.shape[1]

        # get tensor for GC profile
        gc_profile = self.cn_s[[self.chr_col, self.start_col, self.gc_col]].drop_duplicates()
        gc_profile = gc_profile.dropna()
        gc_profile = torch.tensor(gc_profile[self.gc_col].values).to(torch.float32)

        # get tensor for rt prior if provided
        if (self.rt_prior_col is not None) and (self.rt_prior_col in self.cn_s.columns):
            rt_prior_profile = self.cn_s[[self.chr_col, self.start_col, self.rt_prior_col]].drop_duplicates()
            rt_prior_profile = rt_prior_profile.dropna()
            rt_prior_profile = torch.tensor(rt_prior_profile[self.rt_prior_col].values).unsqueeze(-1).to(torch.float32)
            rt_prior_profile = self.convert_rt_prior_units(rt_prior_profile)
            assert cn_s_reads.shape[0] == gc_profile.shape[0] == rt_prior_profile.shape[0]
        else:
            rt_prior_profile = None

        return cn_g1_reads_df, cn_g1_states_df, cn_s_reads_df, cn_s_states_df, cn_g1_reads, cn_g1_states, cn_s_reads, cn_s_states, gc_profile, rt_prior_profile, libs_g1, libs_s


    def sort_by_cell_and_loci(self, cn):
        """ Sort long-form dataframe so each cell follows correct genomic ordering """
        cn[self.chr_col] = cn[self.chr_col].astype('category')
        chr_order = [str(i+1) for i in range(22)]
        chr_order.append('X')
        chr_order.append('Y')
        cn[self.chr_col] = cn[self.chr_col].cat.set_categories(chr_order)
        cn = cn.sort_values(by=[self.cell_col, self.chr_col, self.start_col])
        return cn


    def get_libraries_tensor(self, cn_s, cn_g1):
        """ Create a tensor of integers representing the unique library_id of each cell. """
        libs_s = cn_s[[self.cell_col, self.library_col]].drop_duplicates()
        libs_g1 = cn_g1[[self.cell_col, self.library_col]].drop_duplicates()

        # get all unique library ids found across cells of both cell cycle phases
        all_library_ids = pd.concat([libs_s, libs_g1])[self.library_col].unique()

        self.num_libraries = int(len(all_library_ids))
        
        # replace library_id strings with integer values
        for i, library_id in enumerate(all_library_ids):
            libs_s[self.library_col].replace(library_id, i, inplace=True)
            libs_g1[self.library_col].replace(library_id, i, inplace=True)
        
        # convert to tensors of type int (ints needed to index other tensors)
        libs_s = torch.tensor(libs_s[self.library_col].values).to(torch.int64)
        libs_g1 = torch.tensor(libs_g1[self.library_col].values).to(torch.int64)

        return libs_s, libs_g1


    def convert_rt_prior_units(self, rt_prior_profile):
        """ Make sure that early RT regions are close to 1, late RT regions are close to 0 """
        rt_prior_profile = rt_prior_profile / max(rt_prior_profile)
        return rt_prior_profile


    def build_trans_mat(self, cn):
        """ Use the frequency of state transitions in cn to build a new transition matrix. """
        trans_mat = torch.eye(self.num_states, self.num_states) + 1
        num_loci, num_cells = cn.shape
        for i in range(num_cells):
            for j in range(1, num_loci):
                cur_state = int(cn[j, i])
                prev_state = int(cn[j-1, i])
                trans_mat[prev_state, cur_state] += 1
        return trans_mat


    def build_cn_prior(self, cn, weight=1e6):
        """ Build a prior for each bin's cn state based on its value in cn. """
        num_loci, num_cells = cn.shape
        cn_prior = torch.ones(num_loci, num_cells, self.num_states)
        for i in range(num_loci):
            for n in range(num_cells):
                state = int(cn[i, n].numpy())
                cn_prior[i, n, state] = weight
        return cn_prior


    def manhattan_binarization(self, X, MEAN_GAP_THRESH=0.7, EARLY_S_SKEW_THRESH=0.2, LATE_S_SKEW_THRESH=-0.2):
        """ Binarize X into binary replicated vs unreplicated states by drawing an optimal threshold through X that minimizes the manhattan distance of all points. """
        # center and scale the data
        X = (X - np.mean(X)) / np.std(X)
        
        # fit a 2-state GMM to the data
        gm = GaussianMixture(n_components=2, random_state=0)
        states = gm.fit_predict(X)
        
        # use GMM means to assign binary values for thresholding
        mean_0 = gm.means_[0][0]
        mean_1 = gm.means_[1][0]

        # find the distance between the two means for each state
        mean_gap = abs(mean_0 - mean_1)

        # assume means denote binary values
        binary_0 = min(mean_0, mean_1)
        binary_1 = max(mean_0, mean_1)
        
        X = X.flatten()
        
        # use skew to define the binary values if means are close together
        if mean_gap < MEAN_GAP_THRESH:
            cell_skew = skew(X)
            # positive skew indicates early S-phase
            if cell_skew > EARLY_S_SKEW_THRESH:
                binary_0 = np.percentile(X, 50)
                binary_1 = np.percentile(X, 95)
            # negative skew indicates late S-phase
            elif cell_skew < LATE_S_SKEW_THRESH:
                binary_0 = np.percentile(X, 5)
                binary_1 = np.percentile(X, 50)
            # assume mid-S when skew is neutral
            else:
                binary_0 = np.percentile(X, 25)
                binary_1 = np.percentile(X, 75)

        # now that binary values are selected, I must compute the Manhattan distance
        # between binarized data and X for 100 different thresholds
        threshs = np.linspace(binary_0, binary_1, 100)
        lowest_dist = np.inf
        best_t = None
        manhattan_dists = []
        for t in threshs:
            # set values to binary_1 when above t, to binary_0 when below t
            B = np.where(X>t, binary_1, binary_0)
            # compute Manhattan distance between two vectors
            dist = cityblock(X, B)
            manhattan_dists.append(dist)
            if dist < lowest_dist:
                lowest_dist = dist
                best_t = t

        # binarize X based on the best threshold
        cell_rt = np.where(X>best_t, 1, 0)
        # compute fraction of replicated bins (cell's time within s-phase)
        frac_rt = sum(cell_rt) / len(cell_rt)
        
        return cell_rt, frac_rt


    def guess_times(self, cn_s_reads, cn_prior):
        """ 
        Come up with a guess for what each cell's time in S-phase should be by
        binarizing the cn-normalized read count.
        """
        num_loci, num_cells = cn_s_reads.shape
        t_init = torch.zeros(num_cells)
        t_alpha_prior = torch.zeros(num_cells)
        t_beta_prior = torch.zeros(num_cells)

        # normalize raw read count by whatever state has the highest probability in the cn prior
        reads_norm_by_cn = cn_s_reads / torch.argmax(cn_prior, dim=2)

        for i in range(num_cells):
            cell_profile = reads_norm_by_cn[:, i]
            
            X = cell_profile.numpy().reshape(-1, 1)
            y_pred2, t_guess = self.manhattan_binarization(X)
            
            t_init[i] = t_guess
            
            # use t_guess as the mean of a beta distribution parameterized by alpha and beta
            # where alpha and beta must sum to gamma
            alpha = t_guess * self.gamma
            beta = self.gamma - alpha
            t_alpha_prior[i] = alpha
            t_beta_prior[i] = beta

        return t_init, t_alpha_prior, t_beta_prior


    def make_gc_features(self, x):
        """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
        x = x.unsqueeze(1)
        return torch.cat([x ** i for i in reversed(range(0, self.poly_degree+1))], 1)


    @config_enumerate
    def model_s(self, gc_profile, libs, cn0=None, rt0=None, num_cells=None, num_loci=None, data=None, cn_prior=None, nb_r_guess=10000., t_alpha_prior=None, t_beta_prior=None, t_init=None):
        with ignore_jit_warnings():
            if data is not None:
                num_loci, num_cells = data.shape
            elif cn0 is not None:
                num_loci, num_cells = cn0.shape
            assert num_cells is not None
            assert num_loci is not None
            assert data is not None

        # controls the consistency of replicating on time
        a = pyro.sample('expose_a', dist.Gamma(torch.tensor([2.]), torch.tensor([0.2])))
        
        # negative binomial dispersion
        nb_r = pyro.param('expose_nb_r', torch.tensor([nb_r_guess]), constraint=constraints.positive)

        # gc bias params
        beta_means = pyro.sample('expose_beta_means', dist.Normal(0., 1.).expand([self.num_libraries, self.poly_degree+1]).to_event(2))
        beta_stds = pyro.param('expose_beta_stds', torch.logspace(start=0, end=-self.poly_degree, steps=(self.poly_degree+1)).reshape(1, -1).expand([self.num_libraries, self.poly_degree+1]),
                               constraint=constraints.positive)
        
        # define cell and loci plates
        loci_plate = pyro.plate('num_loci', num_loci, dim=-2)
        cell_plate = pyro.plate('num_cells', num_cells, dim=-1)

        if rt0 is not None:
            # fix rt as constant when input into model
            rt = rt0
        else:
            with loci_plate:
                # bulk replication timing profile
                rt = pyro.sample('expose_rt', dist.Beta(torch.tensor([1.]), torch.tensor([1.])))

        with cell_plate:

            # per cell replication time
            # draw from prior if provided
            if (t_alpha_prior is not None) and (t_beta_prior is not None):
                time = pyro.sample('expose_time', dist.Beta(t_alpha_prior, t_beta_prior))
            elif t_init is not None:
                time = pyro.param('expose_time', t_init, constraint=constraints.unit_interval)
            else:
                time = pyro.sample('expose_time', dist.Beta(torch.tensor([1.5]), torch.tensor([1.5])))
            
            # per cell reads per copy per bin
            # u should be inversely related to time and cn, positively related to reads
            if cn0 is not None:
                cell_ploidies = torch.mean(cn0.type(torch.float32), dim=0)
            elif cn_prior is not None:
                temp_cn0 = torch.argmax(cn_prior, dim=2).type(torch.float32)
                cell_ploidies = torch.mean(temp_cn0, dim=0)
            else:
                cell_ploidies = torch.ones(num_cells) * 2.
            u_guess = torch.mean(data.type(torch.float32), dim=0) / ((1 + time) * cell_ploidies)
            u_stdev = u_guess / 10.
        
            u = pyro.sample('expose_u', dist.Normal(u_guess, u_stdev))

            # sample beta params for each cell based on which library the cell belongs to
            betas = pyro.sample('expose_betas', dist.Normal(beta_means[libs], beta_stds[libs]).to_event(1))
            
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
                gc_features = self.make_gc_features(gc_profile).reshape(num_loci, 1, self.poly_degree+1)
                gc_rate = torch.exp(torch.sum(torch.mul(betas, gc_features), 2))
                biased_cn = rep_cn * gc_rate

                # expected reads per bin per cell
                expected_reads = (u * biased_cn)

                nb_p = expected_reads / (expected_reads + nb_r)
                
                reads = pyro.sample('reads', dist.NegativeBinomial(nb_r, probs=nb_p), obs=data)


    def model_g1(self, gc_profile, libs, cn=None, num_cells=None, num_loci=None, data=None):
        with ignore_jit_warnings():
            if data is not None:
                num_loci, num_cells = data.shape
            elif cn is not None:
                num_loci, num_cells = cn.shape
            assert num_cells is not None
            assert num_loci is not None
            assert (data is not None) and (cn is not None)
        
        # negative binomial dispersion
        nb_r = pyro.param('expose_nb_r', torch.tensor([10000.0]), constraint=constraints.positive)

        # gc bias params
        beta_means = pyro.sample('expose_beta_means', dist.Normal(0., 1.).expand([self.num_libraries, self.poly_degree+1]).to_event(2))
        beta_stds = pyro.param('expose_beta_stds', torch.logspace(start=0, end=-self.poly_degree, steps=(self.poly_degree+1)).reshape(1, -1).expand([self.num_libraries, self.poly_degree+1]),
                               constraint=constraints.positive)

        with pyro.plate('num_cells', num_cells):

            # per cell reads per copy per bin
            # u should be solved for when cn and read count are both observed
            cell_ploidies = torch.mean(cn.type(torch.float32), dim=0)
            u = torch.mean(data.type(torch.float32), dim=0) / cell_ploidies

            # sample beta params for each cell based on which library the cell belongs to
            betas = pyro.sample('expose_betas', dist.Normal(beta_means[libs], beta_stds[libs]).to_event(1))

            with pyro.plate('num_loci', num_loci):

                # copy number accounting for gc bias
                gc_features = self.make_gc_features(gc_profile).reshape(num_loci, 1, self.poly_degree+1)
                gc_rate = torch.exp(torch.sum(torch.mul(betas, gc_features), 2))
                biased_cn = cn * gc_rate

                # expected reads per bin per cell
                expected_reads = (u * biased_cn)

                nb_p = expected_reads / (expected_reads + nb_r)

                reads = pyro.sample('reads', dist.NegativeBinomial(nb_r, probs=nb_p), obs=data)


    def run_pyro_model(self):
        if self.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        logging.info('Loading data')

        logging.info('-' * 40)
        model_s = self.model_s
        model_g1 = self.model_g1

        cn_g1_reads_df, cn_g1_states_df, cn_s_reads_df, cn_s_states_df, \
            cn_g1_reads, cn_g1_states, cn_s_reads, cn_s_states, \
            gc_profile, rt_prior_profile, libs_g1, libs_s = self.process_input_data()

        print('libs_g1', libs_g1)
        print('libs_s', libs_s)

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

            cn_prior_input = torch.zeros(cn_s_states.shape)
            for i, cell_id in enumerate(cn_s_reads_df.columns):
                cell_cn = self.cn_s.loc[self.cn_s[self.cell_col]==cell_id]  # get full cn data for this cell
                cell_clone = cell_cn[self.clone_col].values[0]  # get clone id
                cn_prior_input[:, i] = torch.tensor(clone_cn_profiles[cell_clone].values).to(torch.int64).to(torch.float32)  # assign consensus clone cn profile for this cell
            
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

        # fit GC params using G1-phase cells    
        guide_g1 = AutoDelta(poutine.block(model_g1, expose_fn=lambda msg: msg["name"].startswith("expose_")))

        optim = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo = JitTrace_ELBO(max_plate_nesting=2)

        svi = SVI(model_g1, guide_g1, optim, loss=elbo)

        # start inference
        logging.info('Start inference for G1-phase cells.')
        losses = []
        for i in range(self.max_iter):
            loss = svi.step(gc_profile, libs_g1, cn=cn_g1_states, data=cn_g1_reads)

            # fancy convergence check that sees if the past 10 iterations have plateaued
            if i >= self.min_iter:
                loss_diff = abs(max(losses[-10:-1]) - min(losses[-10:-1])) / abs(losses[-1])
                if loss_diff < 5e-5:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            logging.info('step: {}, loss: {}'.format(i, loss))


        # replay model
        guide_trace_g1 = poutine.trace(guide_g1).get_trace(gc_profile, libs_g1, cn=cn_g1_states, data=cn_g1_reads)
        trained_model_g1 = poutine.replay(model_g1, trace=guide_trace_g1)

        # infer discrete sites and get model trace
        inferred_model_g1 = infer_discrete(
            trained_model_g1, temperature=0,
            first_available_dim=-3)
        trace_g1 = poutine.trace(inferred_model_g1).get_trace(gc_profile, libs_g1, cn=cn_g1_states, data=cn_g1_reads)

        # extract fitted parameters
        nb_r_fit = trace_g1.nodes['expose_nb_r']['value'].detach()
        betas_fit = trace_g1.nodes['expose_betas']['value'].detach()
        beta_means_fit = trace_g1.nodes['expose_beta_means']['value'].detach()
        beta_stds_fit = trace_g1.nodes['expose_beta_stds']['value'].detach()

        # # Now run the model for S-phase cells
        # logging.info('read_profiles after loading data: {}'.format(cn_s_reads.shape))
        # logging.info('read_profiles data type: {}'.format(cn_s_reads.dtype))


        num_observations = float(cn_s_reads.shape[0] * cn_s_reads.shape[1])
        pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()
        pyro.enable_validation(__debug__)

        # condition gc betas of S-phase model using fitted results from G1-phase model
        model_s = poutine.condition(
            model_s,
            data={
                'expose_beta_means': beta_means_fit,
                'expose_beta_stds': beta_stds_fit,
                'expose_nb_r': nb_r_fit
            })

        # use manhattan binarization method to come up with an initial guess for each cell's time in S-phase
        t_init, t_alpha_prior, t_beta_prior = self.guess_times(cn_s_reads, cn_prior)

        guide_s = AutoDelta(poutine.block(model_s, expose_fn=lambda msg: msg["name"].startswith("expose_")))
        optim_s = pyro.optim.Adam({'lr': self.learning_rate, 'betas': [0.8, 0.99]})
        elbo_s = JitTraceEnum_ELBO(max_plate_nesting=2)
        svi_s = SVI(model_s, guide_s, optim_s, loss=elbo_s)

        # start inference
        logging.info('Start inference for S-phase cells.')
        losses = []
        for i in range(self.max_iter):
            loss = svi_s.step(gc_profile, libs_s, data=cn_s_reads, cn_prior=cn_prior, t_init=t_init)

            # fancy convergence check that sees if the past 10 iterations have plateaued
            if i >= self.min_iter:
                loss_diff = abs(max(losses[-10:-1]) - min(losses[-10:-1])) / abs(losses[-1])
                if loss_diff < 5e-5:
                    print('ELBO converged at iteration ' + str(i))
                    break

            losses.append(loss)
            logging.info('step: {}, loss: {}'.format(i, loss))

        # replay model
        guide_trace_s = poutine.trace(guide_s).get_trace(gc_profile, libs_s, data=cn_s_reads, cn_prior=cn_prior, t_init=t_init)
        trained_model_s = poutine.replay(model_s, trace=guide_trace_s)

        # infer discrete sites and get model trace
        inferred_model_s = infer_discrete(
            trained_model_s, temperature=0,
            first_available_dim=-3)
        trace_s = poutine.trace(inferred_model_s).get_trace(gc_profile, libs_s, data=cn_s_reads, cn_prior=cn_prior, t_init=t_init)

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
        print('beta_means_fit.shape', beta_means_fit.shape)
        print('beta_stds_fit.shape', beta_stds_fit.shape)
        for i in range(self.poly_degree):
            for j in range(self.num_libraries):
                cn_s_out['model_gc_beta{}_mean_library{}'.format(i, j)] = beta_means_fit.numpy()[j, i]
                cn_s_out['model_gc_beta{}_stds_library{}'.format(i, j)] = beta_stds_fit.numpy()[j, i]
        
        return cn_s_out
