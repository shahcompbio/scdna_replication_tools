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


def model_0(read_profiles, gc_profile, rt_profile, sigma1, gc0, gc1, A, num_reads,
            num_states, batch_size=None, include_prior=True):
    with ignore_jit_warnings():
        num_sequences, sequence_length, data_dim = map(int, read_profiles.shape)
        assert lengths.shape == (num_sequences,) == num_reads.shape
        assert lengths.max() <= sequence_length
        assert read_profiles.shape == G1_state_profiles.shape == gc_profile.shape
    
    # establish prior distributions
    with poutine.mask(mask=include_prior):

        # time in S-phase should be drawn from a Beta distribution with a bell-shaped
        # curve centered around 0.5
        probs_s_times = pyro.sample("expose_s_times",
                                    dist.Beta(torch.ones(num_sequences).fill_(2.0),
                                              torch.ones(num_sequences).fill_(2.0))
                                    .to_event(1))

        # A controls the "noise" of replication for each cell
        init_A = torch.ones(num_sequences).fill_(A)
        probs_A = pyro.param('expose_A', init_A, constraint=constraints.greater_than(0))

        # bulk replication timing value of each bin (ranging from 0-1)
        probs_bulk_RT = pyro.param('expose_bulk_RT', rt_profile, constraint=constraints.interval(0, 1))

        # priors on what the gc bias coefficients should look like
        probs_gc = pyro.param('expose_gc', torch.tensor([gc0, gc1]))

        # prior for the cell-specific noise coefficient
        init_sigma1 = torch.ones(num_sequences).fill_(sigma1)
        probs_sigma1 = pyro.param('expose_sigma1', init_sigma1, constraint=constraints.greater_than(0))

        # TODO: set values for init_somatic_CN that reflect each S-phase cell's closest G1-phase cell
        # the diploid (CN=2) state is currently set to 90% for all bins
        prior_CN_prob = 0.9
        init_somatic_CN = torch.ones(num_sequences, sequence_length, num_states) * ((1 - prior_CN_prob) / (num_states - 1))
        init_somatic_CN[:, :, 2] = prior_CN_prob
        somatic_CN = pyro.sample("expose_somatic_CN", dist.Categorical(init_somatic_CN).to_event(1))


    with pyro.plate("cells", num_sequences):
        with pyro.plate("loci", sequence_length):
            # compute the probability of each bin being replicated for this cell given
            # probs_A, probs_bulk_RT, and probs_s_times
            p_rep = 1 / (1 + torch.exp(-probs_A.reshape(-1, 1) * (probs_bulk_RT - probs_s_times.reshape(-1, 1))))

            replicated = pyro.sample("rt_state", dist.Binomial(1, p_rep))

            total_CN = somatic_CN * (1 + replicated)

            # add gc bias to the total CN
            # Is a simple linear model sufficient here?
            biased_CN = total_CN * ((gc_profile * probs_gc[1]) + probs_gc[0])

            # add some random noise to the GC-biased copy number
            # TODO: verify that this step is correct
            noisy_CN = pyro.sample("noisy_CN", dist.Gamma(biased_CN * sigma1, 1/sigma1))

            # scale noisy_CN and then draw true read count from multinomial distribution
            noisy_CN_pval = noisy_CN / torch.sum(noisy_CN, axis=1).reshape(-1, 1)
            # TODO: sampling with a different number of reads per cell might not be supported
            # https://github.com/pytorch/pytorch/issues/42407
            pyro.sample("read_count", dist.Multinomial(num_reads, noisy_CN_pval), obs=read_profiles)


models = {name[len('model_'):]: model
          for name, model in globals().items()
          if name.startswith('model_')}



def load_data(args):
    # read input
    if args.input_path.endswith('.tsv'):
        df = pd.read_csv(args.input_path, sep='\t')
    else:
        df = pd.read_csv(args.input_path)

    # drop any row where copy is NaN
    df2 = df[df['copy'].notna()]

    # set a max number of cells if truncate_cells flag is used
    if args.truncate_cells:
        max_num_cells = args.truncate_cells
    else:
        max_num_cells = np.inf

    # add column that has the relative order of loci within each cell+chr
    pieces = []
    i = 0
    for cell_id, group1 in df2.groupby('cell_id'):
        for chrom, group2 in group1.groupby('chr'):
            group2.insert(0, 'relative_position', range(0, len(group2)))
            # truncate to a max relative position within the chr
            # if args.truncate_lengths is used
            if args.truncate_lengths:
                group2 = group2.iloc[:args.truncate_lengths]
            pieces.append(group2)
        # break out if we've hit the max number of cells
        if i >= max_num_cells:
            break
        i += 1

    df3 = pd.concat(pieces)
    df3 = df3.reset_index().drop(columns=['index'])

    # pivot to 2D matrix where each row is a unique cell+chr, columns are loci
    # within the cell+chr, and the values are copy
    cn = df3.pivot(index=['cell_id', 'chr'], columns='relative_position', values='copy')

    # find the number of non-NA values in each row
    lengths = torch.tensor(cn.notnull().sum(axis=1).values)

    # fill NaN values with -99 so error doesn't get triggered
    # these values will still be blocked by lengths
    cn = cn.fillna(-99)

    # convert to tensor and unsqueeze the data dimension
    cn_tensor = torch.tensor(cn.values).unsqueeze(-1)
    return df3, cn, cn_tensor.to(torch.float32), lengths


def load_simulated_data(input_path):
    cn = pd.read_csv(input_path, sep='\t')
    cn.chr = cn.chr.astype(str)
    cn.set_index(['chr', 'start', 'end'], inplace=True)
    cn = cn.T

    cn_tensor = torch.tensor(cn.values).unsqueeze(-1)
    return cn, cn_tensor.to(torch.float32)


# run pyro model to get replication timing states
def pyro_model(self.cn_s, self.cn_g1, input_col='reads', gc_col='gc',
               clone_col='clone_id', cell_col='cell_id', library_col='library_id',
               chr_col='chr', start_col='start', cn_state_col='state',
               rs_col='rt_state', frac_rt_col='frac_rt',
               transition_prob=0.995, kappa=0.7, nu=2.1, batch_size=10
               ):
    logging.info('Params')
    logging.info('e = {}'.format(transition_prob))
    logging.info('kappa = {}'.format(kappa))
    logging.info('nu = {}'.format(nu))

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    logging.info('Loading data')

    logging.info('-' * 40)
    model = model_0
    MAP_decoder = viterbi_decoder0_0
    forward_backward_decoder = viterbi_decoder1_0

    # Assign S-phase cells to best G1-phase matching cell


    # fit GC params using G1-phase cells


    # split cn_s and cn_g1 into sequences and lengths


    logging.info('sequences after loading data: {}'.format(sequences.shape))
    logging.info('sequences data type: {}'.format(sequences.dtype))

    num_observations = float(sequences.shape[0] * sequences.shape[1])
    pyro.set_rng_seed(args.seed)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)

    logging.info('sequences after truncating data: {}'.format(sequences.shape))

    # We'll train using MAP Baum-Welch, i.e. MAP estimation while marginalizing
    # out the hidden state x. This is accomplished via an automatic guide that
    # learns point estimates of all of our conditional probability tables,
    # named probs_*.
    # global guide
    guide = AutoDelta(poutine.block(model, expose_fn=lambda msg: msg["name"].startswith("expose_")))

    # To help debug our tensor shapes, let's print the shape of each site's
    # distribution, value, and log_prob tensor. Note this information is
    # automatically printed on most errors inside SVI.
    guide_trace = poutine.trace(guide).get_trace(
        sequences, lengths, transition_prob, hidden_dim, kappa, nu, batch_size=batch_size)
    first_available_dim = -3
    if args.print_shapes:
        model_trace = poutine.trace(
            poutine.replay(poutine.enum(model, first_available_dim), guide_trace)).get_trace(
            sequences, lengths, transition_prob, hidden_dim, kappa, nu, batch_size=batch_size)
        logging.info(model_trace.format_shapes())

    # Enumeration requires a TraceEnum elbo and declaring the max_plate_nesting.
    # All of our models have two plates: "data" and "tones".
    optim = Adam({'lr': args.learning_rate})
    if args.tmc:
        if args.jit:
            raise NotImplementedError("jit support not yet added for TraceTMC_ELBO")
        elbo = TraceTMC_ELBO(max_plate_nesting=2)
        tmc_model = poutine.infer_config(
            model,
            lambda msg: {"num_samples": args.tmc_num_samples, "expand": False} if msg["infer"].get("enumerate", None) == "parallel" else {})  # noqa: E501
        svi = SVI(tmc_model, guide, optim, elbo)
    else:
        Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
        elbo = Elbo(max_plate_nesting=2,
                    # strict_enumeration_warning=(model is not model_7),
                    strict_enumeration_warning=True,
                    jit_options={"time_compilation": args.time_compilation})
        # svi = SVI(model, guide, optim, elbo)
        svi = SVI(model, guide, optim, elbo)

    # We'll train on small minibatches.
    max_loss = -np.inf
    prev_loss = np.inf
    deltas = [np.inf] * 5
    continue_fit = True
    step = 0
    logging.info('Step\tLoss')
    while step < args.num_steps and continue_fit:
        loss = svi.step(sequences, lengths, transition_prob, hidden_dim, kappa, nu, batch_size=batch_size)
        step_loss = loss / num_observations
        logging.info('{: >5d}\t{}'.format(step, step_loss))
        if step_loss > max_loss:
            max_loss = step_loss
        temp_delta = prev_loss - step_loss
        deltas.append(temp_delta)

        # see if last 5 steps have formed a plateau
        converged = all(abs(i) < 0.001 * max_loss for i in deltas[-5:])

        if converged:
            continue_fit = False
        else:
            step += 1
            prev_loss = step_loss

    if args.jit and args.time_compilation:
        logging.debug('time to compile: {} s.'.format(elbo._differentiable_loss.compile_time))

    for key, value in pyro.get_param_store().items():
        logging.info(key)
        logging.info(value.shape)
        if key != 'assignment_probs':
            logging.info(value)
    
    logging.info('computing marginals on all sequences')
    marginals = elbo.compute_marginals(model, guide, sequences, lengths, transition_prob, hidden_dim, kappa, nu)

    logging.info('assigning marginals to a tensor')
    full_marginals = torch.ones(sequences.shape[0], sequences.shape[1], args.hidden_dim) * -1
    logging.info('full marginals shape {}'.format(full_marginals.shape))
    for key, value in marginals.items():
        j = int(key.split('_')[1])
        full_marginals[:, j, :] = value.probs[:, 0, :]

    logging.info('finding state with highest probability according to marginals')
    marginal_states = torch.argmax(full_marginals, dim=2)
    logging.info('marginal states shape {}'.format(marginal_states.shape))

    # create pandas matrix of assigned states that matches cn
    logging.info('merging marginal states with input df')
    marginal_states = marginal_states.cpu().detach().numpy()
    marginal_states = pd.DataFrame(marginal_states,
                          index=cn.index,
                          columns=cn.columns)
    # melt df and merge with input df
    assigned_df = marginal_states.melt(ignore_index=False, value_name='assigned_marginals_state').reset_index()
    out_df = pd.merge(df, assigned_df)

    # add columns for each marginal_state at each row in out_df
    # create df where rows are cell+chr+relative position and values are a marginal_probs array
    logging.info('adding marginal probs of each state to output df')
    dict_df = {}
    for i, item in enumerate(list(full_marginals.detach().numpy())):
        dict_df[i] = list(item)
    temp_df = pd.DataFrame(dict_df)
    temp_df = temp_df.T
    temp_df.index = cn.index
    melted_df = temp_df.melt(ignore_index=False, value_name='marginal_probs', var_name='relative_position').reset_index()
    # split array in marginal_probs column into separate columns for each state
    marginal_columns = []
    for i in range(args.hidden_dim):
        marginal_columns.append('prob_sate_{}'.format(i))
    melted_df[marginal_columns] = pd.DataFrame(melted_df.marginal_probs.tolist(), index=melted_df.index)
    melted_df.drop(columns=['marginal_probs'], inplace=True)
    # add marginal probs to out_df
    out_df = pd.merge(out_df, melted_df)

    # run viterbi with infer_discrete temp set to 0 for MAP
    if args.viterbi_temp0:
        logging.info('Running MAP Viterbi')
        out_df = run_viterbi(MAP_decoder, sequences, lengths, transition_prob, hidden_dim, kappa, nu, cn, out_df, 'assigned_viterbi0_state')
        print(out_df.columns)

    # run viterbi with infer_discrete temp set to 1 for forward-backward
    if args.viterbi_temp1:
        logging.info('Running forward-backward Viterbi')
        out_df = run_viterbi(forward_backward_decoder, sequences, lengths, transition_prob, hidden_dim, kappa, nu, cn, out_df, 'assigned_viterbi1_state')
        print(out_df.columns)

    out_df.to_csv(args.output_path, sep='\t', index=False)

    # save params learned during model training
    transition = pd.DataFrame(pyro.param('AutoDelta.probs_transition').detach().numpy())
    transition.to_csv(args.transition_output, sep='\t')
    logging.info("moving probs_nu into dataframe")
    probs_nu = pd.DataFrame(pyro.param('expose_nu').detach().numpy()).T
    logging.info("saving probs_nu")
    probs_nu.to_csv(args.variation_output, sep='\t', index=False)



def get_args():
    # assert pyro.__version__.startswith('1.5.0')
    parser = argparse.ArgumentParser(description="MAP Baum-Welch learning HMMCopy")
    parser.add_argument("-m", "--model", default="0", type=str,
                        help="one of: {}".format(", ".join(sorted(models.keys()))))
    parser.add_argument("-i", "--input-path", type=str)
    parser.add_argument("-o", "--output-path", type=str)
    parser.add_argument("-n", "--num-steps", default=200, type=int)
    parser.add_argument("-b", "--batch-size", default=10, type=int)
    parser.add_argument("-d", "--hidden-dim", default=12, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-e", "--transition-prob",
                        help="Probability of extending a segment, increase to lengthen"
                             "segments, decrease to shorten segments. Range: (0, 1)", default=0.995, type=float)
    parser.add_argument("-k", "--kappa", default=0.7, type=float)
    parser.add_argument("-nu", default=2.1, type=float, help="degrees of freedom")
    parser.add_argument("-tc", "--truncate_cells", type=int)
    parser.add_argument("-tl", "--truncate_lengths", type=int)
    parser.add_argument("-p", "--print-shapes", action="store_true")
    parser.add_argument('-to', '--transition_output', type=str)
    parser.add_argument('-vo', '--variation_output', type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--viterbi_temp0', action='store_true')
    parser.add_argument('--viterbi_temp1', action='store_true')
    parser.add_argument('--time-compilation', action='store_true')
    parser.add_argument('-rp', '--raftery-parameterization', action='store_true')
    parser.add_argument('--tmc', action='store_true',
                        help="Use Tensor Monte Carlo instead of exact enumeration "
                             "to estimate the marginal likelihood. You probably don't want to do this, "
                             "except to see that TMC makes Monte Carlo gradient estimation feasible "
                             "even with very large numbers of non-reparametrized variables.")
    parser.add_argument('--tmc-num-samples', default=10, type=int)
    
    return parser.parse_args()
    