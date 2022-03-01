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
    p.add_argument('clone_copy', help='output clone copynumber copy tsv file')
    p.add_argument('clone_reads', help='output clone copynumber reads tsv file')
    p.add_argument('clone_states', help='output clone copynumber reads tsv file')
    p.add_argument('clone_rpm_gc_norm', help='output clone copynumber gc corrected reads per million tsv file')
    p.add_argument('output', help='same as s_phase_cells input but with new column for copy_norm')

    return p.parse_args()


def cell_clone_norm(clone_profiles, cell_cn, clone_id, input_col, temp_col):
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
    merged_df[temp_col] = merged_df[input_col] / (merged_df[clone_id] + np.finfo(float).eps)
    merged_df = merged_df.drop(columns=[input_col, clone_id])
    
    # merge this new temp_col column back into the original cell_cn
    out = pd.merge(cell_cn, merged_df, left_index=True, right_index=True)

    # re-sort the rows since they get messed up while merging
    out.reset_index(inplace=True)
    out.sort_values(by=['chr', 'start', 'end'])

    return out



def normalize_by_clone(cn_s, clone_profiles, input_col='rpm_gc_norm', clone_col='clone_id',
                       temp_col='temp_rt', output_col='rt_value', seg_col='changepoint_segments'):
    # drop loci with nans
    cn_s.dropna(inplace=True)
    clone_profiles.dropna(inplace=True)

    clone_idx = ['chr', 'start', 'end']
    clone_profiles = clone_profiles.set_index(clone_idx)

    cn_s = add_cell_ploidies(cn_s)

    # loop through each cell and divide copy by its corresponding clone
    output_list = []
    for cell_id, cell_cn in cn_s.groupby('cell_id'):
        # set index to match clone dfs
        temp_cell_cn = cell_cn.set_index(clone_idx).copy()
        # extract the assigned clone_id
        temp_clone_id = temp_cell_cn.clone_id.unique()[0]
        # add normalized copy column (with new indices)
        temp_out_cn = cell_clone_norm(clone_profiles, temp_cell_cn, temp_clone_id, input_col, temp_col)
        # remove cell specific CNAs by nominating changepoints
        temp_out_cn = remove_cell_specific_CNAs(temp_out_cn, input_col=temp_col, output_col=output_col, seg_col=seg_col)

        output_list.append(temp_out_cn)

    # convert list to output df
    cn_s_output = pd.concat(output_list)

    return cn_s_output




def main():
    argv = get_args()
    df = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})

    # drop the Y chromosome
    df = df[df['chr'] != 'Y']

    clone_idx = ['chr', 'start', 'end']
    clone_copy = pd.read_csv(argv.clone_copy, sep='\t', index_col=clone_idx)
    clone_reads = pd.read_csv(argv.clone_reads, sep='\t', index_col=clone_idx)
    clone_states = pd.read_csv(argv.clone_states, sep='\t', index_col=clone_idx)
    clone_rpm_gc_norm = pd.read_csv(argv.clone_rpm_gc_norm, sep='\t', index_col=clone_idx)

    df = add_cell_ploidies(df)

    # loop through each cell and divide copy by its corresponding clone
    output_list = []
    for cell_id, cell_cn in df.groupby('cell_id'):
        # set index to match clone dfs
        temp_cell_cn = cell_cn.set_index(clone_idx).copy()
        # extract the assigned clone_id
        temp_clone_id = temp_cell_cn.clone_id.unique()[0]
        # add normalized copy column (with new indices)
        temp_out_cn = cell_clone_norm(clone_copy, clone_reads, clone_states, clone_rpm_gc_norm, temp_cell_cn, temp_clone_id)
        # remove cell specific CNAs by nominating changepoints
        temp_out_cn = remove_cell_specific_CNAs(temp_out_cn)

        output_list.append(temp_out_cn)

    # convert list to df and save as tsv
    output_df = pd.concat(output_list)
    output_df.to_csv(argv.output, sep='\t', index=False)



if __name__ == '__main__':
    main()
