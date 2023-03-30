import numpy as np
import pandas as pd
from argparse import ArgumentParser

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.util import ignore_jit_warnings


def get_args():
    p = ArgumentParser()

    p.add_argument('-si', '--df_s', help='True somatic CN profiles for S-phase cells')
    p.add_argument('-gi', '--df_g', help='True somatic CN profiles for G1/2-phase cells')
    p.add_argument('-n', '--num_reads', type=int, help='number of reads per cell')
    p.add_argument('-l', '--lamb', type=float, help='negative binomial success probability term lambda (controls overdispersion)')
    p.add_argument('-a', '--a', type=int, help='amplitude of sigmoid curve when generating rt noise')
    p.add_argument('-b', '--betas', type=float, nargs='+', help='list of beta coefficients for gc bias')
    p.add_argument('-rt', '--rt_cols', type=str, nargs='+', help='list of column in cn input containing appropriate replication timing values (one rt column per clone)')
    p.add_argument('-gc', '--gc_col', type=str, help='column in cn input containing GC content of each bin')
    p.add_argument('-c', '--clones', type=str, nargs='+', help='list of unique clone ids')
    p.add_argument('-so', '--s_out', help='simulated S-phase cells')
    p.add_argument('-go', '--g_out', help='simulated G1/2-phase cells')

    return p.parse_args()


def make_gc_features(x, K):
    """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in reversed(range(0, K+1))], 1)


def model_s(gammas, libs, cn0=None, rho0=None, num_cells=None, num_loci=None, etas=None, u_guess=70., lambda_init=1e-1, t_alpha_prior=None, t_beta_prior=None, t_init=None,  K=4, L=1):
    with ignore_jit_warnings():
        if cn0 is not None:
            num_loci, num_cells = cn0.shape
        assert num_cells is not None
        assert num_loci is not None

    # controls the consistency of replicating on time
    a = pyro.sample('expose_a', dist.Gamma(torch.tensor([2.]), torch.tensor([0.2])))
    
    # variance of negative binomial distribution is governed by the success probability of each trial
    lamb = pyro.param('expose_lambda', torch.tensor([lambda_init]), constraint=constraints.unit_interval)

    # gc bias params
    beta_means = pyro.sample('expose_beta_means', dist.Normal(0., 1.).expand([L, K+1]).to_event(2))
    beta_stds = pyro.param('expose_beta_stds', torch.logspace(start=0, end=-K, steps=(K+1)).reshape(1, -1).expand([L, K+1]),
                            constraint=constraints.positive)
    
    # define cell and loci plates
    loci_plate = pyro.plate('num_loci', num_loci, dim=-2)
    cell_plate = pyro.plate('num_cells', num_cells, dim=-1)

    if rho0 is not None:
        # fix replication timing as constant when input into model
        rho = rho0
    else:
        with loci_plate:
            # bulk replication timing profile
            rho = pyro.sample('expose_rho', dist.Beta(torch.tensor([1.]), torch.tensor([1.])))

    with cell_plate:

        # per cell time in S-phase (tau)
        # draw from prior if provided
        if (t_alpha_prior is not None) and (t_beta_prior is not None):
            tau = pyro.sample('expose_tau', dist.Beta(t_alpha_prior, t_beta_prior))
        elif t_init is not None:
            tau = pyro.param('expose_tau', t_init, constraint=constraints.unit_interval)
        else:
            tau = pyro.sample('expose_tau', dist.Beta(torch.tensor([1.]), torch.tensor([1.])))
        
        # per cell reads per copy per bin
        u = pyro.sample('expose_u', dist.Normal(torch.tensor([u_guess]), torch.tensor([u_guess/10.])))
    
        # sample the gc bias params from a normal distribution for this cell
        betas = pyro.sample('expose_betas', dist.Normal(beta_means[libs], beta_stds[libs]).to_event(1))
        
        with loci_plate:

            if cn0 is None:
                if etas is None:
                    etas = torch.ones(num_loci, num_cells, 13)
                # sample cn probabilities of each bin from Dirichlet
                pi = pyro.sample('expose_pi', dist.Dirichlet(etas))
                # sample cn state from categorical based on cn_prob
                cn = pyro.sample('cn', dist.Categorical(pi), infer={"enumerate": "parallel"})
            else:
                cn = cn0

            # per cell per bin late or early 
            t_diff = tau.reshape(-1, num_cells) - rho.reshape(num_loci, -1)

            # probability of having been replicated
            phi = 1 / (1 + torch.exp(-a * t_diff))

            # binary replicated indicator
            rep = pyro.sample('rep', dist.Bernoulli(phi), infer={"enumerate": "parallel"})

            # total copy number accounting for replication
            chi = cn * (1. + rep)

            # copy number accounting for gc bias
            gc_features = make_gc_features(gammas, K).reshape(num_loci, 1, K+1)
            omega = torch.exp(torch.sum(torch.mul(betas, gc_features), 2))  # compute the gc 'rate' of each bin

            # expected reads per bin per cell
            theta = u * chi * omega

            # use lambda and the expected read count to define the number of trials (delta)
            # that should be drawn for each bin
            delta = theta * (1 - lamb) / lamb

            # replace all delta<1 values with 1 since delta should be >0
            # this avoids NaN errors when theta=0 at a given bin
            delta[delta<1] = 1
            
            reads = pyro.sample('reads', dist.NegativeBinomial(delta, probs=lamb), obs=None)



def model_g1(gammas, libs, cn=None, num_cells=None, num_loci=None, u_guess=70., lambda_init=1e-1, K=4, L=1):
    with ignore_jit_warnings():
        if cn is not None:
            num_loci, num_cells = cn.shape
        assert num_cells is not None
        assert num_loci is not None
    
    # negative binomial dispersion
    lamb = pyro.param('expose_lambda', torch.tensor([lambda_init]), constraint=constraints.positive)

    # gc bias params
    beta_means = pyro.sample('expose_beta_means', dist.Normal(0., 1.).expand([L, K+1]).to_event(2))
    beta_stds = pyro.param('expose_beta_stds', torch.logspace(start=0, end=-K, steps=(K+1)).reshape(1, -1).expand([L, K+1]),
                            constraint=constraints.positive)

    with pyro.plate('num_cells', num_cells):

        # per cell reads per copy per bin
        u = pyro.sample('expose_u', dist.Normal(torch.tensor([u_guess]), torch.tensor([u_guess/10.])))

        # sample the gc bias params from a normal distribution for this cell
        betas = pyro.sample('expose_betas', dist.Normal(beta_means[libs], beta_stds[libs]).to_event(1))

        with pyro.plate('num_loci', num_loci):

            # copy number accounting for gc bias
            gc_features = make_gc_features(gammas, K).reshape(num_loci, 1, K+1)
            omega = torch.exp(torch.sum(torch.mul(betas, gc_features), 2))  # compute the gc 'rate' of each bin
            
            # print('u.shape', u.shape)
            # print('cn.shape', cn.shape)
            # print('omega.shape', omega.shape)

            # expected reads per bin per cell
            theta = u * cn * omega

            # use lambda and the expected read count to define the number of trials (delta)
            # that should be drawn for each bin
            delta = theta * (1 - lamb) / lamb

            # replace all delta<1 values with 1 since delta should be >0
            # this avoids NaN errors when theta=0 at a given bin
            delta[delta<1] = 1
            
            reads = pyro.sample('reads', dist.NegativeBinomial(delta, probs=lamb), obs=None)

    return reads


def convert_rt_units(rt):
    # make sure rt units range from 0-1 with largest values being latest times
    return 1 - ((rt - rt.min()) / (rt.max() - rt.min()))


def get_libraries_tensor(cn):
    """ Create a tensor of integers representing the unique library_id of each cell. """
    libs = cn[['cell_id', 'library_id']].drop_duplicates()

    # get all unique library ids found across cells of both cell cycle phases
    all_library_ids =libs['library_id'].unique()

    L = int(len(all_library_ids))
    
    # replace library_id strings with integer values
    for i, library_id in enumerate(all_library_ids):
        libs['library_id'].replace(library_id, i, inplace=True)
    
    # convert to tensors of type int (ints needed to index other tensors)
    libs = torch.tensor(libs['library_id'].values).to(torch.int64)

    return libs, L


def simulate_s_cells(gc_profile, libs, cn, rt, L_val, num_reads, lamb, betas, a):
    pyro.clear_param_store()

    num_loci, num_cells = cn.shape
    gc_profile = torch.tensor(gc_profile.values)
    cn = torch.tensor(cn.values)
    rt_profile = torch.tensor(convert_rt_units(rt.values))

    u_guess = float(num_reads) / (1.5 * num_loci * torch.mean(cn))
    true_lambda = torch.tensor([lamb])

    # print('num_loci', num_loci)
    # print('num_cells', num_cells)
    # print('gc_profile', gc_profile.shape)
    # print('cn', cn.shape)
    # print('rt_profile', rt_profile.shape)
    # print('u_guess', u_guess)

    conditioned_model = poutine.condition(
        model_s,
        data={
            'expose_beta_means': torch.tensor(betas).reshape(1, -1).expand([L_val, len(betas)]),
            'expose_lambda': true_lambda,
            'expose_u': u_guess,
            'expose_a': torch.tensor([a]),
            'expose_rho': rt_profile
        })

    model_trace = pyro.poutine.trace(conditioned_model)

    samples = model_trace.get_trace(gc_profile, libs, cn0=cn, u_guess=u_guess, lambda_init=true_lambda, K=len(betas)-1, L=L_val)

    t = samples.nodes['expose_tau']['value']
    u = samples.nodes['expose_u']['value']

    t_diff = t.reshape(-1, num_cells) - rt_profile.reshape(num_loci, -1)
    p_rep = 1 / (1 + torch.exp(-a * t_diff))

    rep = samples.nodes['rep']['value']

    rep_cn = cn * (1. + rep)

    reads = samples.nodes['reads']['value']

    # normalize read count
    reads_norm = (reads / torch.sum(reads, 0)) * num_reads
    reads_norm = reads_norm.type(torch.int64)

    return reads_norm, reads, rep, p_rep, t


def simulate_g_cells(gc_profile, libs, cn, L_val, num_reads, lamb, betas):
    pyro.clear_param_store()

    num_loci, num_cells = cn.shape
    gc_profile = torch.tensor(gc_profile.values)
    cn = torch.tensor(cn.values)

    u_guess = float(num_reads) / (1. * num_loci * torch.mean(cn))
    true_lambda = torch.tensor([lamb])

    conditioned_model = poutine.condition(
        model_g1,
        data={
            'expose_beta_means': torch.tensor(betas).reshape(1, -1).expand([L_val, len(betas)]),
            'expose_lambda': true_lambda,
            'expose_u': u_guess,
        })

    model_trace = pyro.poutine.trace(conditioned_model)

    samples = model_trace.get_trace(gc_profile, libs, cn=cn, u_guess=u_guess, lambda_init=true_lambda, K=len(betas)-1, L=L_val)

    u = samples.nodes['expose_u']['value']

    reads = samples.nodes['reads']['value']

    # normalize read count
    reads_norm = (reads / torch.sum(reads, 0)) * num_reads
    reads_norm = reads_norm.type(torch.int64)

    return reads_norm, reads


def pert_simulator(df_s, df_g, num_reads, rt_cols, clones, lamb, betas, a, gc_col='gc', input_cn_col='true_somatic_cn'):
    """
    Simulate S-phase and G1-phase read count data given cells with known copy number states.
    
    Parameters
    ----------
    df_s : pandas.DataFrame
        S-phase cells with known copy number states.
        Required columns are as follows:
            - cell_id
            - library_id
            - clone_id
            - chr
            - start
            - end
    df_g : pandas.DataFrame
        G1/2-phase cells with known copy number states.
        Required columns are as follows:
            - cell_id
            - library_id
            - clone_id
            - chr
            - start
            - end
    num_reads : int
        Number of reads to simulate per cell.
    rt_cols : list of str
        List of column names in df_s containing replication timing values. 
        Each item corresponds to the RT profile of each clone.
    clones : list of str
        List of clone IDs.
    lamb : float
        Lambda parameter for negative binomial overdispersion.
    betas : list of float
        List of beta parameters for GC bias polynomial.
    a : float
        Parameter for sigmoidal replication timing function.
    gc_col : str, optional
        Name of column in df_s and df_g containing GC content values.
    input_cn_col : str, optional
        Name of column in df_s containing input copy number states.
    """
    df_s.chr = df_s.chr.astype(str)
    df_g.chr = df_g.chr.astype(str)

    # extract a global gc profile that applies to all G1/2 and S-phase cells
    gc_profile = df_s[['chr', 'start', gc_col]].drop_duplicates()[gc_col]

    # make sure there's exactly one rt column per clone before looping through clones
    assert len(rt_cols) == len(clones) 

    df_s_out = []
    for (rt_col, clone_id) in zip(rt_cols, clones):
        # subset df_s to just S-phase cells belonging to this clone
        clone_df_s = df_s.query('clone_id=="{}"'.format(clone_id))

        # convert library_id to integer tensor
        libs_s, L_val_s = get_libraries_tensor(clone_df_s)

        # pivot clone cn states into matrix
        clone_cn_s = pd.pivot_table(clone_df_s, index=['chr', 'start'], columns='cell_id', values=input_cn_col)

        # get rt profile that matches this clone
        rt_profile = clone_df_s[['chr', 'start', rt_col]].drop_duplicates()[rt_col]

        # S-phase: condition each model based in argv parameters and simulate read count
        clone_reads_norm, clone_reads, clone_rep, clone_p_rep, clone_t = simulate_s_cells(gc_profile, libs_s, clone_cn_s, rt_profile, L_val_s, num_reads, lamb, betas, a)

        # convert tensors to dataframes
        clone_reads_norm_df = pd.DataFrame(clone_reads_norm.numpy(), columns=clone_cn_s.columns, index=clone_cn_s.index)
        clone_reads_df = pd.DataFrame(clone_reads.numpy(), columns=clone_cn_s.columns, index=clone_cn_s.index)
        clone_rep_df = pd.DataFrame(clone_rep.numpy(), columns=clone_cn_s.columns, index=clone_cn_s.index)
        clone_p_rep_df = pd.DataFrame(clone_p_rep.numpy(), columns=clone_cn_s.columns, index=clone_cn_s.index)
        clone_t_df = pd.DataFrame(clone_t.numpy(), columns=['true_t'], index=clone_cn_s.columns)

        # print('clone_reads_norm_df\n', clone_reads_norm_df.head())
        # print('clone_reads_df\n', clone_reads_df.head())
        # print('clone_rep_df\n', clone_rep_df.head())
        # print('clone_p_rep_df\n', clone_p_rep_df.head())
        # print('clone_t_df\n', clone_t_df.head())

        # merge normalized read count
        clone_reads_norm_df = clone_reads_norm_df.reset_index().melt(id_vars=['chr', 'start'], var_name='cell_id', value_name='true_reads_norm')
        clone_reads_norm_df.chr = clone_reads_norm_df.chr.astype(str)
        clone_df_s = pd.merge(clone_df_s, clone_reads_norm_df)
        # merge raw reads before normalizing total read count
        clone_reads_df = clone_reads_df.reset_index().melt(id_vars=['chr', 'start'], var_name='cell_id', value_name='true_reads_raw')
        clone_reads_df.chr = clone_reads_df.chr.astype(str)
        clone_df_s = pd.merge(clone_df_s, clone_reads_df)
        # merge true replication states
        clone_rep_df = clone_rep_df.reset_index().melt(id_vars=['chr', 'start'], var_name='cell_id', value_name='true_rep')
        clone_rep_df.chr = clone_rep_df.chr.astype(str)
        clone_df_s = pd.merge(clone_df_s, clone_rep_df)
        # merge probability of each bin being replicated
        clone_p_rep_df = clone_p_rep_df.reset_index().melt(id_vars=['chr', 'start'], var_name='cell_id', value_name='true_p_rep')
        clone_p_rep_df.chr = clone_p_rep_df.chr.astype(str)
        clone_df_s = pd.merge(clone_df_s, clone_p_rep_df)
        # merge s-phase times
        clone_df_s = pd.merge(clone_df_s, clone_t_df.reset_index(), on='cell_id')

        # append clone_df_s to a list of df_s for all the output
        df_s_out.append(clone_df_s)

    # concatenate all the clones into one df
    df_s_out = pd.concat(df_s_out, ignore_index=True)
    df_s = df_s_out

    # convert library_id to integer tensor
    libs_g, L_val_g = get_libraries_tensor(df_g)

    # G1-phase: condition each model based in argv parameters and simulate read count
    cn_g = pd.pivot_table(df_g, index=['chr', 'start'], columns='cell_id', values=input_cn_col)
    reads_norm_g, reads_g = simulate_g_cells(gc_profile, libs_g, cn_g, L_val_g, num_reads, lamb, betas)
    
    reads_norm_g_df = pd.DataFrame(reads_norm_g.numpy(), columns=cn_g.columns, index=cn_g.index)
    reads_g_df = pd.DataFrame(reads_g.numpy(), columns=cn_g.columns, index=cn_g.index)
    
    # merge normalized read count
    reads_norm_g_df = reads_norm_g_df.reset_index().melt(id_vars=['chr', 'start'], var_name='cell_id', value_name='true_reads_norm')
    reads_norm_g_df.chr = reads_norm_g_df.chr.astype(str)
    df_g = pd.merge(df_g, reads_norm_g_df)
    # merge raw reads before normalizing total read count
    reads_g_df = reads_g_df.reset_index().melt(id_vars=['chr', 'start'], var_name='cell_id', value_name='true_reads_raw')
    reads_g_df.chr = reads_g_df.chr.astype(str)
    df_g = pd.merge(df_g, reads_g_df)
    df_g['true_t'] = 0.0
    df_g['true_rep'] = 0.0
    df_g['true_p_rep'] = 0.0

    # the true total copy number is the the sum of the true somaitc copy number and the true replication state
    df_s['true_total_cn'] = df_s[input_cn_col] * (df_s['true_rep'] + 1)
    df_g['true_total_cn'] = df_g[input_cn_col] * (df_g['true_rep'] + 1)

    return df_s, df_g


def main():
    argv = get_args()

    df_s = pd.read_csv(argv.df_s, sep='\t')
    df_g = pd.read_csv(argv.df_g, sep='\t')

    # add dummy column for library_id
    df_s['library_id'] = 'ABCD'
    df_g['library_id'] = 'ABCD'

    # simulate read count data using pert simulator
    df_s, df_g = pert_simulator(df_s, df_g, argv.num_reads, argv.rt_cols, argv.clones, argv.lamb, argv.betas, argv.a, gc_col=argv.gc_col, input_cn_col='true_somatic_cn')

    df_s.to_csv(argv.s_out, sep='\t', index=False)
    df_g.to_csv(argv.g_out, sep='\t', index=False)


if __name__ == '__main__':
    main()
