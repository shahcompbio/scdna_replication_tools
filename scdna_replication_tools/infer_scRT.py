import numpy as np
import pandas as pd
from scdna_replication_tools.compute_consensus_clone_profiles import compute_consensus_clone_profiles
from scdna_replication_tools.assign_s_to_clones import assign_s_to_clones
from scdna_replication_tools.bulk_gc_correction import bulk_g1_gc_correction
from scdna_replication_tools.normalize_by_cell import normalize_by_cell
from scdna_replication_tools.normalize_by_clone import normalize_by_clone
from scdna_replication_tools.binarize_rt_profiles import binarize_profiles
from scdna_replication_tools.compute_pseudobulk_rt_profiles import compute_pseudobulk_rt_profiles
from scdna_replication_tools.calculate_twidth import compute_time_from_scheduled_column, calculate_twidth
from scdna_replication_tools.cncluster import kmeans_cluster
from scdna_replication_tools.pyro_model import pyro_infer_scRT
from argparse import ArgumentParser


def get_args():
    p = ArgumentParser()

    p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
    p.add_argument('g1_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for g1-phase cells')
    p.add_argument('output', help='same as s_phase_cells input but with inferred scRT info added')


class scRT:
    def __init__(self, cn_s, cn_g1, input_col='reads', assign_col='copy', library_col='library_id', ploidy_col='ploidy',
                 cell_col='cell_id', cn_state_col='state', chr_col='chr', start_col='start', gc_col='gc',
                 rv_col='rt_value', rs_col='rt_state', frac_rt_col='frac_rt', clone_col='clone_id', rt_prior_col='mcf7rt',
                 col2='rpm_gc_norm', col3='temp_rt', col4='changepoint_segments', col5='binary_thresh'):
        self.cn_s = cn_s
        self.cn_g1 = cn_g1

        # input for GC correction --> inferring scRT states
        self.input_col = input_col

        # column for computing consensus clone profiles, assigning cells to clones
        self.assign_col = assign_col

        # column representing clone IDs. If none, then we must perform clustering on our own during inference
        self.clone_col = clone_col

        # column representing library IDs, cell IDs, etc.
        self.library_col = library_col
        self.cell_col = cell_col
        self.cn_state_col = cn_state_col
        self.chr_col = chr_col
        self.start_col = start_col
        self.gc_col = gc_col
        self.ploidy_col = ploidy_col
        self.rt_prior_col = rt_prior_col

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
        self.bulk_cn = None
        self.manhattan_df = None


    def infer(self, level='cell'):
        if level=='cell':
            self.cn_s = self.infer_cell_level()
        elif level=='clone':
            self.cn_s = self.infer_clone_level()
        elif level=='bulk':
            self.cn_s = self.infer_bulk_level()
        elif level=='pyro':
            self.cn_s = self.infer_pyro_model()

        return self.cn_s


    def infer_pyro_model(self):
        # run clustering if no clones are included in G1 input
        if self.clone_col is None:
            clusters = kmeans_cluster(self.cn_g1)
            self.cn_g1 = pd.merge(self.cn_g1, clusters, on=self.cell_col)
            self.clone_col = 'cluster_id'

        # compute conesensus clone profiles
        self.clone_profiles = compute_consensus_clone_profiles(
            self.cn_g1, self.assign_col, clone_col=self.clone_col, cell_col=self.cell_col, chr_col=self.chr_col,
            start_col=self.start_col, cn_state_col=self.cn_state_col
        )

        # assign S-phase cells to clones
        self.cn_s = assign_s_to_clones(self.cn_s, self.clone_profiles, col_name=self.assign_col, clone_col=self.clone_col,
                                       cell_col=self.cell_col, chr_col=self.chr_col, start_col=self.start_col)

        # run pyro model to get replication timing states
        pyro_model = pyro_infer_scRT(self.cn_s, self.cn_g1, input_col=self.input_col, gc_col=self.gc_col, rt_prior_col=self.rt_prior_col,
                                     clone_col=self.clone_col, cell_col=self.cell_col, library_col=self.library_col,
                                     chr_col=self.chr_col, start_col=self.start_col, cn_state_col=self.cn_state_col,
                                     rs_col=self.rs_col, frac_rt_col=self.frac_rt_col)
        self.cn_g1, self.cn_s  = pyro_model.run_pyro_model()

        return self.cn_s


    def infer_cell_level(self):
        # run clustering if no clones are included in G1 input
        if self.clone_col is None:
            clusters = kmeans_cluster(self.cn_g1)
            self.cn_g1 = pd.merge(self.cn_g1, clusters, on=self.cell_col)
            self.clone_col = 'cluster_id'

        # compute conesensus clone profiles
        self.clone_profiles = compute_consensus_clone_profiles(
            self.cn_g1, self.assign_col, clone_col=self.clone_col, cell_col=self.cell_col, chr_col=self.chr_col,
            start_col=self.start_col, cn_state_col=self.cn_state_col
        )

        # assign S-phase cells to clones
        self.cn_s = assign_s_to_clones(self.cn_s, self.clone_profiles, col_name=self.assign_col, clone_col=self.clone_col,
                                       cell_col=self.cell_col, chr_col=self.chr_col, start_col=self.start_col)

        # GC correction
        self.cn_s, self.cn_g1 = bulk_g1_gc_correction(self.cn_s, self.cn_g1, input_col=self.input_col, gc_col=self.gc_col,
                                                      cell_col=self.cell_col, library_col=self.library_col, output_col=self.col2)

        # normalize by cell
        self.cn_s = normalize_by_cell(self.cn_s, self.cn_g1, input_col=self.col2, clone_col=self.clone_col,
                                      temp_col=self.col3, output_col=self.rv_col, seg_col=self.col4,
                                      cell_col=self.cell_col, chr_col=self.chr_col, start_col=self.start_col,
                                      cn_state_col=self.cn_state_col, ploidy_col=self.ploidy_col)

        # binarize
        self.cn_s, self.manhattan_df = binarize_profiles(
            self.cn_s, self.rv_col, rs_col=self.rs_col, frac_rt_col=self.frac_rt_col, thresh_col=self.col5,
            MEAN_GAP_THRESH=0.7, EARLY_S_SKEW_THRESH=0.2, LATE_S_SKEW_THRESH=-0.2
        )

        return self.cn_s


    def infer_clone_level(self):
        # run clustering if no clones are included in G1 input
        if self.clone_col is None:
            clusters = kmeans_cluster(self.cn_g1)
            self.cn_g1 = pd.merge(self.cn_g1, clusters, on=self.cell_col)
            self.clone_col = 'cluster_id'

        # compute conesensus clone profiles
        self.clone_profiles = compute_consensus_clone_profiles(
            self.cn_g1, self.assign_col, clone_col=self.clone_col, cell_col=self.cell_col, chr_col=self.chr_col,
            start_col=self.start_col, cn_state_col=self.cn_state_col
        )

        # assign S-phase cells to clones
        self.cn_s = assign_s_to_clones(self.cn_s, self.clone_profiles, col_name=self.input_col, clone_col=self.clone_col,
                                       cell_col=self.cell_col, chr_col=self.chr_col, start_col=self.start_col)

        # GC correction
        self.cn_s, self.cn_g1 = bulk_g1_gc_correction(self.cn_s, self.cn_g1, input_col=self.input_col, gc_col=self.gc_col,
                                                      cell_col=self.cell_col, library_col=self.library_col, output_col=self.col2)

        # compute conesensus clone profiles for GC-normed read depth
        self.clone_profiles_gc_norm = compute_consensus_clone_profiles(self.cn_g1, self.col2, clone_col=self.clone_col)

        # normalize by clone
        self.cn_s = normalize_by_clone(self.cn_s, self.clone_profiles_gc_norm, input_col=self.col2, clone_col=self.clone_col,
                                       temp_col=self.col3, output_col=self.rv_col, seg_col=self.col4,
                                       cell_col=self.cell_col, chr_col=self.chr_col, start_col=self.start_col,
                                       cn_state_col=self.cn_state_col, ploidy_col=self.ploidy_col)

        # binarize
        self.cn_s, self.manhattan_df = binarize_profiles(
            self.cn_s, self.rv_col, rs_col=self.rs_col, frac_rt_col=self.frac_rt_col, thresh_col=self.col5,
            MEAN_GAP_THRESH=0.7, EARLY_S_SKEW_THRESH=0.2, LATE_S_SKEW_THRESH=-0.2
        )

        return self.cn_s


    def infer_bulk_level(self):
        # assign all cells to one pseudobulk dummy clone
        self.cn_g1.loc[self.cn_g1.index, self.clone_col] = 'A'
        self.cn_s.loc[self.cn_s.index, self.clone_col] = 'A'

        # GC correction
        self.cn_s, self.cn_g1 = bulk_g1_gc_correction(self.cn_s, self.cn_g1, input_col=self.input_col, gc_col=self.gc_col,
                                                      cell_col=self.cell_col, library_col=self.library_col, output_col=self.col2)

        # compute pseudobulk profile for GC-normed read depth
        self.bulk_profile_gc_norm = compute_consensus_clone_profiles(
            self.cn_g1, self.col2, clone_col=self.clone_col, cell_col=self.cell_col, chr_col=self.chr_col,
            start_col=self.start_col, cn_state_col=self.cn_state_col
        )

        # normalize by the pseudobulk profile
        self.cn_s = normalize_by_clone(self.cn_s, self.bulk_profile_gc_norm, input_col=self.col2, clone_col=self.clone_col,
                                       temp_col=self.col3, output_col=self.rv_col, seg_col=self.col4)

        # binarize
        self.cn_s, self.manhattan_df = binarize_profiles(
            self.cn_s, self.rv_col, rs_col=self.rs_col, frac_rt_col=self.frac_rt_col, thresh_col=self.col5,
            MEAN_GAP_THRESH=0.7, EARLY_S_SKEW_THRESH=0.2, LATE_S_SKEW_THRESH=-0.2
        )

        return self.cn_s


    def compute_pseudobulk_rt_profiles(self, output_col='pseduobulk', time_col='hours'):
        self.bulk_cn = compute_pseudobulk_rt_profiles(self.cn_s, self.rv_col, output_col=output_col, time_col=time_col, 
                                                      clone_col=self.clone_col, chr_col=self.chr_col, start_col=self.start_col)
        return self.bulk_cn


    def calculate_twidth(self, pseudobulk_col='pseudobulk_hours', tfs_col='time_from_scheduled_rt',
                        per_cell=False, query2=None, curve='sigmoid'):
        cn = pd.merge(self.cn_s, self.bulk_cn)
        cn = compute_time_from_scheduled_column(cn, pseudobulk_col=pseudobulk_col,
                                                frac_rt_col=self.frac_rt_col, tfs_col=tfs_col)
        return calculate_twidth(cn, tfs_col=tfs_col, rs_col=self.rs_col, cell_col=self.cell_col, per_cell=per_cell, query2=query2, curve=curve)


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
