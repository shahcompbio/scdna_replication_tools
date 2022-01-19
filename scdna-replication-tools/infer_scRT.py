import numpy as np
import pandas as pd
from compute_consensus_clone_profiles import compute_consensus_clone_profiles
from assign_s_to_clones import assign_s_to_clones
from bulk_gc_correction import bulk_g1_gc_correction
from normalize_by_cell import normalize_by_cell
from binarize_profiles import binarize_profiles
from compute_pseudobulk_rt_profiles import compute_pseudobulk_rt_profiles
from calculate_twidth import compute_time_from_scheduled_column, calculate_twidth
from scgenome.cncluster import kmeans_cluster
from argparse import ArgumentParser


def get_args():
	p = ArgumentParser()

	p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
	p.add_argument('g1_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for g1-phase cells')
	p.add_argument('output', help='same as s_phase_cells input but with inferred scRT info added')


class scRT:
	def __init__(self, cn_s, cn_g1, input_col='reads', s_prob_col='is_s_phase_prob', 
				rv_col='rt_value', rs_col='rt_state', frac_rt_col='frac_rt', clone_col='clone_id',
				col2='rpm_gc_norm', col3='temp_rt', col4='changepoint_segments', col5='binary_thresh'):
		self.cn_s = cn_s
		self.cn_g1 = cn_g1

		# column for computing consensus clone profiles, assigning cells to clones, input for GC correction
		self.input_col = input_col

		# column representing clone IDs. If none, then we must perform clustering on our own during inference
		self.clone_col = clone_col

		# column representing S-phase probability of each cell
		self.s_prob_col = s_prob_col

		# column representing continuous replication timing value of each bin
		self.rv_col = rv_col

		# column representing binary replication state of each bin
		self.rs_col = rs_col

		# column representing fraction of replicated bins per cell
		self.frac_rt_col = frac_rt_col

		# some other columns that get added in infer()
		self.col2 = col2
		self.col3 = col3
		self.col4 = col4
		self.col5 = col5

		# class objects that get computed are initialized as None
		self.clone_profiles = None
		self.gc_curve = None
		self.bulk_cn = None


	def infer(self):
		# run clustering if no clones are included in G1 input
		if self.clone_col is None:
			clusters = kmeans_cluster(self.cn_g1)
			self.cn_g1 = pd.merge(self.cn_g1, clusters, on='cell_id')
			self.clone_col = 'cluster_id'

		# compute conesensus clone profiles
		self.clone_profiles = compute_consensus_clone_profiles(self.cn_g1, self.input_col, clone_col=self.clone_col)

		# assign S-phase cells to clones
		self.cn_s = assign_s_to_clones(self.cn_s, self.clone_profiles, col_name=self.input_col, clone_col=self.clone_col)

		# GC correction
		self.cn_s, self.cn_g1, self.gc_curve = bulk_g1_gc_correction(self.cn_s, self.cn_g1, input_col=self.input_col, output_col=self.col2)

		# normalize by cell
		self.cn_s = normalize_by_cell(self.cn_s, self.cn_g1, input_col=self.col2, s_prob_col=self.s_prob_col, clone_col=self.clone_col,
										temp_col=self.col3, output_col=self.rv_col, seg_col=self.col4)

		# binarize
		self.cn_s = binarize_profiles(self.cn_s, self.rv_col, rs_col=self.rs_col, frac_rt_col=self.frac_rt_col, thresh_col=self.col5,
										MEAN_GAP_THRESH=0.7, EARLY_S_SKEW_THRESH=0.2, LATE_S_SKEW_THRESH=-0.2)

		return self.cn_s


	def compute_pseudobulk_rt_profiles(self, output_col='pseduobulk', time_col='hours'):
		self.bulk_cn = compute_pseudobulk_rt_profiles(self.cn_s, self.rv_col, output_col=output_col, time_col=time_col, clone_col=self.clone_col)
		return self.bulk_cn


	def calculate_twidth(self, pseudobulk_col='pseudobulk_hours', tfs_col='time_from_scheduled_rt',
						per_cell=False, query2=None, curve='sigmoid'):
		cn = pd.merge(self.cn_s, self.bulk_cn)
		cn = compute_time_from_scheduled_column(cn, pseudobulk_col=pseudobulk_col,
												frac_rt_col=self.frac_rt_col, tfs_col=tfs_col)
		return calculate_twidth(cn, tfs_col=tfs_col, rs_col=self.rs_col, per_cell=per_cell, query2=query2, curve=curve)


# if run as a script, only infer scRT profiles
def main():
	argv = get_args()
	cn_s = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})
	cn_g1 = pd.read_csv(argv.g1_phase_cells, sep='\t', dtype={'chr': str})

	# create scRT object
	scrt = scRT(cn_s, cn_g1)

	# infer scRT profiles for S-phase cells
	out_df = scrt.infer()

	out_df.to_csv(argv.output, sep='\t', index=False)



if __name__ == '__main__':
	main()
