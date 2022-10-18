from argparse import ArgumentParser
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn import preprocessing
from scipy.stats import mode, ttest_ind
from scdna_replication_tools.compute_consensus_clone_profiles import add_cell_ploidies
from scdna_replication_tools.normalize_by_cell import remove_cell_specific_CNAs


def get_args():
    p = ArgumentParser()

    p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
    p.add_argument('clone_profiles', help='consensus clone profiles in matrix format')
    p.add_argument('input_col', help='column within s_phase_cells to use as input')
    p.add_argument('output', help='same as s_phase_cells input but with new column for copy_norm')

    return p.parse_args()


def cell_clone_norm(clone_profiles, cell_cn, clone_id, input_col, output_col, chr_col='chr', start_col='start'):
    '''
    Find normalized copy or state values between an S-phase cell and its clone.
    '''
    cell_df = cell_cn[input_col]
    clone_df = clone_profiles[clone_id]

    # remove any rows with NaN
    cell_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    cell_df.dropna(inplace=True)

    # merge so that cell and clone have matching indices
    merged_df = pd.merge(cell_df, clone_df, left_index=True, right_index=True)
    merged_df.dropna(inplace=True)

    # divide cell profile by consensus clone profile
    merged_df[output_col] = merged_df[input_col] / (merged_df[clone_id] + np.finfo(float).eps)
    merged_df = merged_df.drop(columns=[input_col, clone_id])
    
    # merge this new output_col column back into the original cell_cn
    out = pd.merge(cell_cn, merged_df, left_index=True, right_index=True)

    # re-sort the rows since they get messed up while merging
    out.reset_index(inplace=True)
    out.sort_values(by=[chr_col, start_col])

    return out


def normalize_by_clone(cn_s, clone_profiles, input_col='rpm_gc_norm', clone_col='clone_id', cell_col='cell_id',
                       output_col='rt_value', chr_col='chr', start_col='start', cn_state_col='state', ploidy_col='ploidy'):
    # drop loci with nans
    cn_s.dropna(inplace=True)
    clone_profiles.dropna(inplace=True)

    clone_idx = [chr_col, start_col]
    clone_profiles = clone_profiles.reset_index().set_index(clone_idx)

    cn_s = add_cell_ploidies(cn_s, cell_col=cell_col, cn_state_col=cn_state_col, ploidy_col=ploidy_col)

    # loop through each cell and divide copy by its corresponding clone
    output_list = []
    for cell_id, cell_cn in cn_s.groupby(cell_col):
        # set index to match clone dfs
        temp_cell_cn = cell_cn.set_index(clone_idx).copy()
        # extract the assigned clone_id
        temp_clone_id = temp_cell_cn[clone_col].unique()[0]
        # add normalized copy column (with new indices)
        temp_out_cn = cell_clone_norm(clone_profiles, temp_cell_cn, temp_clone_id, input_col, output_col, chr_col=chr_col, start_col=start_col)

        output_list.append(temp_out_cn)

    # convert list to output df
    cn_s_output = pd.concat(output_list, ignore_index=True)

    return cn_s_output


def main():
    argv = get_args()
    cn_s = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})
    clone_profiles = pd.read_csv(argv.clone_profiles, sep='\t')

    output_df = normalize_by_clone(cn_s, clone_profiles, argv.input_col)

    output_df.to_csv(argv.output, sep='\t', index=False)



if __name__ == '__main__':
    main()
