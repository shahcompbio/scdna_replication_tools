from argparse import ArgumentParser
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn import preprocessing
from scipy.stats import mode, ttest_ind
from compute_consensus_clone_profiles import add_cell_ploidies
from normalize_by_cell2 import remove_cell_specific_CNAs


def get_args():
	p = ArgumentParser()

	p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
	p.add_argument('clone_copy', help='output clone copynumber copy tsv file')
	p.add_argument('clone_reads', help='output clone copynumber reads tsv file')
	p.add_argument('clone_states', help='output clone copynumber reads tsv file')
	p.add_argument('clone_rpm_gc_norm', help='output clone copynumber gc corrected reads per million tsv file')
	p.add_argument('output', help='same as s_phase_cells input but with new column for copy_norm')

	return p.parse_args()


def cell_clone_norm(clone_copy, clone_reads, clone_states, clone_rpm_gc_norm, cell_cn, clone_id):
	'''
	Find normalized copy or state values between an S-phase cell and its clone.
	'''
	copy_df = cell_cn[['copy', 'ploidy']]
	reads_df = cell_cn['reads']
	states_df = cell_cn[['state', 'ploidy']]
	rpm_gc_norm_df = cell_cn[['rpm_gc_norm']]
	clone_cn = clone_copy[clone_id]
	clone_rds = clone_reads[clone_id]
	clone_st = clone_states[clone_id]
	clone_rpm = clone_rpm_gc_norm[clone_id]

	# remove any rows with NaN
	copy_df.replace([np.inf, -np.inf], np.nan, inplace=True)
	copy_df.dropna(inplace=True)
	reads_df.dropna(inplace=True)
	states_df.dropna(inplace=True)
	rpm_gc_norm_df.dropna(inplace=True)

	# merge so that cell and clone have matching indices
	copy_merged_df = pd.merge(copy_df, clone_cn, left_index=True, right_index=True)
	copy_merged_df.dropna(inplace=True)
	reads_merged_df = pd.merge(reads_df, clone_rds, left_index=True, right_index=True)
	reads_merged_df.dropna(inplace=True)
	states_merged_df = pd.merge(states_df, clone_st, left_index=True, right_index=True)
	states_merged_df.dropna(inplace=True)
	rpm_merged_df = pd.merge(rpm_gc_norm_df, clone_rpm, left_index=True, right_index=True)
	rpm_merged_df.dropna(inplace=True)

	# divide cell copy by consensus clone profiles
	# normalize copy to ploidy before normalizing to clone
	copy_merged_df['copy_norm'] = (copy_merged_df['copy'] / copy_merged_df['ploidy']) / (copy_merged_df[clone_id] + np.finfo(float).eps)
	copy_norm_df = copy_merged_df.drop(columns=['copy', 'ploidy', clone_id])
	# reads are normalized by just division -- no ploidy adjustment needed
	reads_merged_df['reads_norm'] = reads_merged_df['reads'] / (reads_merged_df[clone_id] + np.finfo(float).eps)
	reads_norm_df = reads_merged_df.drop(columns=['reads', clone_id])
	# state should normalized just like copy
	states_merged_df['state_norm'] = (states_merged_df['state'] / states_merged_df['ploidy']) / (states_merged_df[clone_id] + np.finfo(float).eps)
	states_norm_df = states_merged_df.drop(columns=['state', 'ploidy', clone_id])
	# gc corrected reads per million are normalized by just division -- no ploidy adjustment needed
	rpm_merged_df['rpm_gc_norm_clone_norm'] = rpm_merged_df['rpm_gc_norm'] / (rpm_merged_df[clone_id] + np.finfo(float).eps)
	rpm_norm_df = rpm_merged_df.drop(columns=['rpm_gc_norm', clone_id])
	
	# merge this new copy_norm column back into the original cell_cn
	norm_df = pd.merge(reads_norm_df, copy_norm_df, left_index=True, right_index=True)
	norm_df = pd.merge(norm_df, states_norm_df, left_index=True, right_index=True)
	norm_df = pd.merge(norm_df, rpm_norm_df, left_index=True, right_index=True)
	out = pd.merge(cell_cn, norm_df, left_index=True, right_index=True)

	# re-sort the rows since they get messed up while merging
	out.reset_index(inplace=True)
	out.sort_values(by=['chr', 'start', 'end'])

	# center and scale all values for this cell
	out['reads_norm_2'] = preprocessing.scale(out['reads_norm'].values)
	out['copy_norm_2'] = preprocessing.scale(out['copy_norm'].values)

	return out



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
