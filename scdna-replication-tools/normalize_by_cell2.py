from argparse import ArgumentParser
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn import preprocessing
from scipy.stats import mode, ttest_ind, pearsonr
from compute_consensus_clone_profiles import add_cell_ploidies


def get_args():
	p = ArgumentParser()

	p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
	p.add_argument('g1_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
	p.add_argument('output', help='same as s_phase_cells input but with new column for copy_norm')

	return p.parse_args()


def sort_by_cell_and_loci(cn):
	""" Sort long-form dataframe so each cell follows correct genomic ordering """
	cn.chr = cn.chr.astype('category')
	chr_order = [str(i+1) for i in range(22)]
	chr_order.append('X')
	chr_order.append('Y')
	cn.chr.cat.set_categories(chr_order, inplace=True)
	cn = cn.sort_values(by=['cell_id', 'chr', 'start'])
	return cn


def identify_changepoint_segs(cell_cn, col):
	Y = cell_cn[col].copy().values
    
	# array to note the nominated CNA regions
	chng = np.zeros(len(Y))

	# iteratively remove CNAs by searching for 2 breakpoints and normalizing to rest of genome
	keep_going = True
	j = 1
	while keep_going:
		algo = rpt.KernelCPD(kernel='linear', min_size=2).fit(Y)
		result = algo.predict(n_bkps=2)
		temp_indices = np.arange(result[0], result[1])
		region = Y[temp_indices]
		background = Y[~temp_indices]

		# see if region median is <0.9 or >1.1x the background
		median_ratio = np.median(region) / np.median(background)

		# compute two-sided t-test between groups
		[statistic, pval] = ttest_ind(region, background)

		# the segment would fall within one chromosome if it were real
		left_chr = cell_cn['chr'].values[result[0]]
		right_chr = cell_cn['chr'].values[result[1]-1]

		if (median_ratio>1.1 or median_ratio<0.9) and pval<0.05 and left_chr==right_chr:
			# mark that this region was a CNA
			chng[temp_indices] = j
			j += 1
			# normalize this region to its median ratio
			Y[temp_indices] /= median_ratio
		else:
			keep_going = False

	# see if any CNAs are at beginning or end of genome by searching for 1 breakpoint
	keep_going = True
	while keep_going:
		algo = rpt.KernelCPD(kernel='linear', min_size=2).fit(Y)
		result = algo.predict(n_bkps=1)

		# find the chromosome where the breakpoint lies
		ind = result[0]
		left_chr = cell_cn['chr'].values[ind]
		right_chr = cell_cn['chr'].values[ind-1]

		# set indices based on whether changepoint occurs at chr1 or chrX
		if right_chr=='1':
			temp_indices = np.arange(0, ind)
		elif left_chr=='X':
			temp_indices = np.arange(ind, len(chng))

		if right_chr=='1' or left_chr=='X':
			region = Y[temp_indices]
			background = Y[~temp_indices]

			# see if region median is <0.9 or >1.1x the background
			median_ratio = np.median(region) / np.median(background)

			# compute two-sided t-test between groups
			[statistic, pval] = ttest_ind(region, background)

			# only look for losses on chr1 and gains on chrX
			# to avoid overcorrecting for early RT of chr1
			# and late RT of chrX
			if ((median_ratio>1.1 and left_chr=='X') or (median_ratio<0.9 and right_chr=='1')) and pval<0.05:
				# mark that this region was a CNA
				chng[temp_indices] = j
				j += 1
				# normalize this region to its median ratio
				Y[temp_indices] /= median_ratio
			else:
				# breakout because change isn't large enough
				keep_going = False
		else:
			# breakout because change isn't on chr1 or chrX
			keep_going = False

	return Y, chng


def remove_cell_specific_CNAs(cell_cn, input_col='copy_norm', seg_col='changepoint_segments'):
	# make sure genomic loci are in correct order
	cell_cn = sort_by_cell_and_loci(cell_cn)
    
	# trim tails of normal distribution before identifying any changepoints
	output_col2 = input_col + '_2'
	cell_cn[output_col2] = cell_cn[input_col].where(
								preprocessing.scale(cell_cn[input_col].values)<4,
								other=np.percentile(cell_cn[input_col].values, 95))
	cell_cn[output_col2] = cell_cn[output_col2].where(
								preprocessing.scale(cell_cn[output_col2].values)>-4,
								other=np.percentile(cell_cn[output_col2].values, 5))

	Y, chng = identify_changepoint_segs(cell_cn, output_col2)

	# save chng as our changepoint segments for this cell
	cell_cn[seg_col] = chng
	output_col3 = input_col + '_3'
	cell_cn[output_col3] = Y
    
	# scale within each CNA that was called via changepoints
	output_col4 = input_col + '_4'
	for chunk_id, chunk in cell_cn.groupby(seg_col):
		cell_cn.loc[chunk.index, output_col4] = preprocessing.scale(chunk[output_col3].values)

	return cell_cn


def compute_cell_corrs(s_cell_cn, clone_cn_g1, s_cell_id):
	# rename columns that appear in both cns
	s_cell_cn['copy_s'] = s_cell_cn['copy']
	s_cell_cn['reads_s'] = s_cell_cn['reads']
	s_cell_cn['rpm_gc_norm_s'] = s_cell_cn['rpm_gc_norm']
	s_cell_cn['ploidy_s'] = s_cell_cn['ploidy']
	s_cell_cn['multiplier_s'] = s_cell_cn['multiplier']
	s_cell_cn['state_s'] = s_cell_cn['state']
	s_cell_cn.drop(columns=['copy', 'state', 'reads', 'rpm_gc_norm', 'multiplier', 'ploidy'], inplace=True)

	cell_corrs = []
	for g1_cell_id, g1_cell_cn in clone_cn_g1.groupby('cell_id'):
		# rename columns that appear in both cns
		g1_cell_cn = g1_cell_cn[['chr', 'start', 'end', 'copy', 'reads', 'rpm_gc_norm', 'is_s_phase_prob_new', 'state']]
		g1_cell_cn['copy_g1'] = g1_cell_cn['copy']
		g1_cell_cn['reads_g1'] = g1_cell_cn['reads']
		g1_cell_cn['rpm_gc_norm_g1'] = g1_cell_cn['rpm_gc_norm']
		g1_cell_cn['state_g1'] = g1_cell_cn['state']
		g1_cell_cn.drop(columns=['copy', 'state', 'reads', 'rpm_gc_norm'], inplace=True)

		# merge g1_cell_cn and s_cell_cn into one df based on loci
		temp_merged_cn = pd.merge(s_cell_cn, g1_cell_cn)

		# compute pearson correlation
		r, pval = pearsonr(temp_merged_cn.reads_s.values, temp_merged_cn.reads_g1.values)
		temp_df = pd.DataFrame({'s_cell_id': [s_cell_id], 'g1_cell_id': [g1_cell_id],
								'pearson_r': [r], 'pearson_pval': [pval],
								'is_s_phase_prob_new': [g1_cell_cn.is_s_phase_prob_new.values[0]]})
		cell_corrs.append(temp_df)

	cell_corrs = pd.concat(cell_corrs)

	# sort by pearson r in descending order
	cell_corrs.sort_values(by=['pearson_r'], ascending=False, inplace=True)

	return cell_corrs


def g1_cell_norm(s_cell_cn, g1_cell_cn):
	''' Normalize the S-phase cell by the G1-phase cell '''
	# only change column names for G1-phase cells (S-phase columns already changed)
	g1_cell_cn = g1_cell_cn[['chr', 'start', 'end', 'copy', 'reads', 'rpm_gc_norm', 'is_s_phase_prob_new', 'state', 'ploidy', 'multiplier']]
	g1_cell_cn['copy_g1'] = g1_cell_cn['copy']
	g1_cell_cn['ploidy_g1'] = g1_cell_cn['ploidy']
	g1_cell_cn['multiplier_g1'] = g1_cell_cn['multiplier']
	g1_cell_cn['reads_g1'] = g1_cell_cn['reads']
	g1_cell_cn['rpm_gc_norm_g1'] = g1_cell_cn['rpm_gc_norm']
	g1_cell_cn['state_g1'] = g1_cell_cn['state']
	g1_cell_cn.drop(columns=['copy', 'state', 'reads', 'rpm_gc_norm', 'ploidy', 'multiplier'], inplace=True)

	# merge S and G1-phase cell cns to the same loci
	temp_merged_cn = pd.merge(s_cell_cn, g1_cell_cn)


	# account for inappropriate multiplier_s when normalizing copy
	temp_merged_cn['copy_norm'] = (temp_merged_cn['copy_s'] * temp_merged_cn['multiplier_g1']) / \
								(temp_merged_cn['copy_g1'] * temp_merged_cn['multiplier_s'] + np.finfo(float).eps)
	# reads are normalized by just division -- no ploidy adjustment needed
	temp_merged_cn['reads_norm'] = temp_merged_cn['reads_s'] / (temp_merged_cn['reads_g1'] + np.finfo(float).eps)
	# rpm of S-phase cell is normalized by the state of the G1-phase cell
	temp_merged_cn['state_norm'] = (temp_merged_cn['rpm_gc_norm_s'] * temp_merged_cn['multiplier_g1']) / \
								((temp_merged_cn['state_g1'] * temp_merged_cn['multiplier_s']) + np.finfo(float).eps)
	# rpm are normalized by just division -- no ploidy adjustment needed
	temp_merged_cn['rpm_gc_norm_cell_norm'] = temp_merged_cn['rpm_gc_norm_s'] / (temp_merged_cn['rpm_gc_norm_g1'] + np.finfo(float).eps)

	# center and scale all values for this cell
	temp_merged_cn['reads_norm_2'] = preprocessing.scale(temp_merged_cn['reads_norm'].values)
	temp_merged_cn['copy_norm_2'] = preprocessing.scale(temp_merged_cn['copy_norm'].values)
	temp_merged_cn['state_norm_2'] = preprocessing.scale(temp_merged_cn['state_norm'].values)
	temp_merged_cn['rpm_gc_norm_cell_norm_2'] = preprocessing.scale(temp_merged_cn['rpm_gc_norm_cell_norm'].values)

	return temp_merged_cn


def main():
	argv = get_args()
	cn_s = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})
	cn_g1 = pd.read_csv(argv.g1_phase_cells, sep='\t', dtype={'chr': str})

	# drop non-essential columns with NaNs
	cn_s.drop(columns=['cell_cycle_state'], inplace=True)
	cn_g1.drop(columns=['cell_cycle_state'], inplace=True)

	# drop the Y chromosome
	cn_s = cn_s[cn_s['chr'] != 'Y']
	cn_g1 = cn_g1[cn_g1['chr'] != 'Y']

	# drop loci with nans
	cn_s.dropna(inplace=True)
	cn_g1.dropna(inplace=True)

	# add cell ploidies
	cn_s = add_cell_ploidies(cn_s)
	cn_g1 = add_cell_ploidies(cn_g1)

	# loop through each S-phase cell, find it's matching G1-phase cell & normalize
	output_list = []
	for cell_id, cell_cn in cn_s.groupby('cell_id'):
		temp_cell_cn = cell_cn.copy()

		clone_id = temp_cell_cn.clone_id.values[0]
		clone_cn_g1 = cn_g1.loc[cn_g1['clone_id']==clone_id]
		temp_cell_cn = temp_cell_cn[['chr', 'start', 'end', 'cell_id', 'copy', 'reads', 'rpm_gc_norm', 'quality', 'state', 'ploidy', 'multiplier']]
		
		# compute pearson correlations between this S-phase cell and all G1-phase cells
		cell_corrs = compute_cell_corrs(temp_cell_cn, clone_cn_g1, cell_id)
		
		# get data from the G1 cell that matches best
		g1_cell_id = cell_corrs.iloc[0].g1_cell_id
		g1_cell_cn = clone_cn_g1.loc[clone_cn_g1['cell_id']==g1_cell_id]

		# normalise S-phase cell by the G1-phase cell
		temp_merged_cn = g1_cell_norm(temp_cell_cn, g1_cell_cn)

		temp_merged_cn = temp_merged_cn[['chr', 'start', 'end' , 'cell_id', 'copy_norm', 'reads_norm', 'state_norm', 'rpm_gc_norm_cell_norm']]
		temp_merged_cn['G1_match_cell_id'] = g1_cell_id
		temp_merged_cn['G1_match_pearsonr'] = cell_corrs.iloc[0].pearson_r

		# remove cell specific CNAs by nominating changepoints
		temp_merged_cn = remove_cell_specific_CNAs(temp_merged_cn, input_col='state_norm')

		# merge with original S-phase cell_cn
		temp_out_cn = pd.merge(temp_merged_cn, cell_cn)

		output_list.append(temp_out_cn)

	# convert list to df and save as tsv
	output_df = pd.concat(output_list)
	output_df.to_csv(argv.output, sep='\t', index=False)


if __name__ == '__main__':
	main()
