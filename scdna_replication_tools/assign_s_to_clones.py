from argparse import ArgumentParser
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def get_args():
	p = ArgumentParser()

	p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, reads, etc for s-phase cells')
	p.add_argument('clone_df', help='consensus clone profiles with columns as clone ids, rows as segments, values match col_name')
	p.add_argument('col_name', help='column used for computing correlations (i.e. reads, copy, state, etc)')
	p.add_argument('s_with_clone_id', help='same as s_phase_cells input but with clone_id added for each cell')

	return p.parse_args()


def clone_correlations(clone_df, cell_cn, col_name='reads'):
	'''
	Find pearson correlations between cell_cn and all columns of clone_df.
	The clone with the highest correlation is returned.

	Args:
		clone_df: data frame of consensus clone profiles with columns as clone ids, rows as segments, values match col_name
		cell_cn: long-form data frame of one S-phase cell profile with rows are segments, columns are chr, start, end, reads, copy, etc
		col_name: column (i.e. reads, copy, state) to use for computing the correlation
	Returns:
		data frame mapping the cell's correlation to each clone
	'''
	df = cell_cn[col_name]

	# remove any rows with NaN
	df.replace([np.inf, -np.inf], np.nan, inplace=True)
	df.dropna(inplace=True)
	# merge dfs together to ensure same segments are used for computing correlation
	merged_df = pd.merge(df, clone_df, left_index=True, right_index=True)
	merged_df.dropna(inplace=True)
	clone_df = merged_df.drop(columns=[col_name])

	# find correlation between cell reads and clones
	clone_corrs = {}
	for clone_id, clone_cn in clone_df.iteritems():
		r, pval = pearsonr(merged_df[col_name], clone_cn)
		clone_corrs[clone_id] = [r, pval]

	return pd.DataFrame(clone_corrs)


def assign_s_to_clones(s_phase_cells, clone_df, col_name='reads', clone_col='clone_id'):
	'''
	Find the clone that belongs to each S-phase cell

	Args:
		clone_df: data frame of consensus clone profiles with columns as clone ids, rows as segments, values match col_name
		s_phase_cells: long-form data frame of all S-phase cells profile with rows are segments & cells, columns are cell_id, chr, start, end, reads, copy, etc
		col_name: column (i.e. reads, copy, state) to use for computing the correlation
	Returns:
		dataframe matching s_phase_cells with clone_id column added from best matching clone
	'''
	s_phase_cells['chr'] = s_phase_cells['chr'].astype(str)

	# ensure that clone indices are loci
	clone_idx = ['chr', 'start', 'end']
	clone_df.set_index(clone_idx, inplace=True)

	# find clone for every S-phase cell
	for cell_id, cell_cn in s_phase_cells.groupby('cell_id'):
		# set index to match clone dfs
		temp_cell_cn = cell_cn.set_index(clone_idx).copy()

		# find best clone_id using read profiles
		copy_corrs = clone_correlations(clone_reads, temp_cell_cn, col_name)
		temp_idx = copy_corrs.iloc[0].argmax()
		best_clone = clone_reads.columns[temp_idx]

		s_phase_cells.loc[cell_cn.index, clone_col] = best_clone

	return s_phase_cells


def main():
	argv = get_args()
	s_phase_cells = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})

	clone_df = pd.read_csv(argv.clone_df, sep='\t')

	df = assign_s_to_clones(s_phase_cells, clone_df, argv.col_name)

	df.to_csv(argv.s_with_clone_id, sep='\t', index=False)


if __name__ == '__main__':
	main()