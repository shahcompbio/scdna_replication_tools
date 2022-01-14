from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats import mode


def get_args():
    p = ArgumentParser()

    p.add_argument('input_cn', help='copynumber tsv file with clone_id, state, copy, etc')
    p.add_argument('col_name', help='column in input_cn to use for computing consensus clone profile')
    p.add_argument('clone_profiles', help='output clone profiles with rows as segments, columns as clones, values match col_name')

    return p.parse_args()


def filter_ploidies(cn):
    """ Only use cells from the majority ploidy of each clone. """
    ploidy_counts = cn.groupby(['clone_id', 'ploidy']).size()

    pieces = []
    for clone_id, group in cn.groupby('clone_id'):
        keep_ploidy = group.groupby('ploidy').size().idxmax()

        pieces.append(group[group['ploidy'] == keep_ploidy].copy())

    return pd.concat(pieces).reset_index(drop=True)


def add_cell_ploidies(cn):
    """ Add a ploidy value for each cell. """
    cn.set_index('cell_id', inplace=True)

    # use mode of state to assign a ploidy value for each cell
    for cell_id, group in cn.groupby('cell_id'):
        cn.loc[cell_id, 'ploidy'] = mode(group['state'])[0][0]

    cn.reset_index(inplace=True)
    return cn


def compute_consensus_clone_profiles(cn, col_name, aggfunc=np.median):
    '''
    Compute consensus copy number profiles for a given set of G1-phase cells

    Args:
        cn: long-form data frame of all G1/2-phase cell profiles; rows are segments & cells, columns are cell_id, clone_id, chr, start, end, reads, copy, etc
        col_name: column (i.e. reads, copy, state) to use for computing the consensus profile
        aggfunc: function for aggregating (median by default)
    Returns:
        data frame of consensus clone profiles with columns as clone ids, rows as segments, values match col_name
    '''

    # removing any clone belonging to None
    cn = cn[cn['clone_id'] != 'None']

    coord_cols = ['chr', 'start', 'end']
    bin_idx = cn.set_index(coord_cols).index

    # add ploidy values for each cell
    cn = add_cell_ploidies(cn)

    # get rid of cell_id column since we are aggregating for each clone
    del cn['cell_id']

    # remove cells from certain clones that don't belong to the majority ploidy
    # i.e. remove tetraploid cells if clone is 90% diploid
    cn = filter_ploidies(cn)

    # pivot long-form df to matrix and aggregate by clone_id
    clone_profiles = cn.pivot_table(
        index=coord_cols, columns='clone_id', values=col_name, aggfunc=aggfunc
    )

    return clone_profiles


def main():
    argv = get_args()
    cn = pd.read_csv(argv.input_cn, sep='\t', dtype={'chr': str})

    clone_profiles = compute_consensus_clone_profiles(cn, argv.col_name)

    clone_profiles.to_csv(argv.clone_profiles, sep='\t')


if __name__ == '__main__':
    main()

