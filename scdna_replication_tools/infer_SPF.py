import numpy as np
import pandas as pd
from scdna_replication_tools.compute_consensus_clone_profiles import compute_consensus_clone_profiles
from scdna_replication_tools.assign_s_to_clones import assign_s_to_clones
from scdna_replication_tools.cncluster import kmeans_cluster
from argparse import ArgumentParser


def get_args():
    p = ArgumentParser()

    p.add_argument('s_phase_cells', help='copynumber tsv file with state, copy, etc for s-phase cells')
    p.add_argument('g1_phase_cells', help='copynumber tsv file with state, copy, clone_id, etc for g1-phase cells')
    p.add_argument('output_s', help='same as s_phase_cells input but clone assignments added')
    p.add_argument('output_spf', help='table of S-phase fractions (with stdev) for each clone')


class SPF:
    def __init__(self, cn_s, cn_g1, input_col='reads', clone_col='clone_id'):
        self.cn_s = cn_s
        self.cn_g1 = cn_g1

        # column for computing consensus clone profiles, assigning cells to clones, input for GC correction
        self.input_col = input_col

        # column representing clone IDs. If none, then we must perform clustering on our own during inference
        self.clone_col = clone_col


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

        # downsample assignments in cn_s to compute SPF means and variances for each clone
        self.output_df = self.calculate_clone_fractions()

        return self.cn_s, self.output_df


    def calculate_clone_fractions(self, N_subsamples=500, frac_subsample=0.75):
        # reduce to just unique mappings of cells to clones
        s_df = self.cn_s[['cell_id', self.clone_col]].drop_duplicates()
        non_s_df = self.cn_g1[['cell_id', self.clone_col]].drop_duplicates()

        # count nubmer of cells in each clone
        s_counts = s_df[self.clone_col].value_counts()
        non_s_counts = non_s_df[self.clone_col].value_counts()

        # sort both dicts alphabetically so colors & order are the same
        s_counts = dict(sorted(s_counts.items()))
        non_s_counts = dict(sorted(non_s_counts.items()))

        # compute the fraction of S-phase cells for each clone
        s_fracs = {}
        for clone_id in s_counts.keys():
            s = s_counts[clone_id]
            g = non_s_counts[clone_id]
            s_fracs[clone_id] = s / (s + g)

        # merge s and non_s dfs
        s_df['is_s_phase'] = True
        non_s_df['is_s_phase'] = False
        df = pd.concat([s_df, non_s_df], ignore_index=True)

        # df for storing subsampled fractions
        subsampled_fracs = pd.DataFrame(columns=['n', self.clone_col, 'frac'])

        # calculate clone fractions based on subsamples of df
        i = 0
        for n in range(N_subsamples):
            temp = df.sample(frac=frac_subsample)
            temp_s_count = temp.query("is_s_phase == True").clone_id.value_counts()
            temp_non_s_count = temp.query("is_s_phase == False").clone_id.value_counts()

            for clone_id in s_counts.keys():
                try:
                    s = temp_s_count[clone_id]
                except:
                    s = 0  # 0 cells present when clone not in dict
                try:
                    g = temp_non_s_count[clone_id]
                except:
                    g = 0  # 0 cells present when clone not in dict
                frac = s / (s + g)
                subsampled_fracs.loc[i] = [n, clone_id, frac]
                i += 1

        # use stdev of subsampled fracs as error for s_fracs
        s_frac_std = {}
        for clone_id in s_fracs.keys():
            std = subsampled_fracs.query("{} == '{}'".format(self.clone_col, clone_id)).frac.std()
            s_frac_std[clone_id] = std

        # combine all info into one output_df
        frac_df = pd.DataFrame.from_dict(s_fracs, orient='index', columns=['SPF']).reset_index()
        std_df = pd.DataFrame.from_dict(s_frac_std, orient='index', columns=['SPF_std']).reset_index()
        s_count_df = pd.DataFrame.from_dict(s_counts, orient='index', columns=['num_s']).reset_index()
        non_s_count_df = pd.DataFrame.from_dict(non_s_counts, orient='index', columns=['num_g']).reset_index()
        output_df = frac_df.merge(std_df).merge(s_count_df).merge(non_s_count_df)
        output_df.rename(columns={'index': 'clone_id'}, inplace=True)

        return output_df


def main():
    argv = get_args()
    cn_s = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})
    cn_g1 = pd.read_csv(argv.g1_phase_cells, sep='\t', dtype={'chr': str})

    # create SPF object
    spf = SPF(cn_s, cn_g1)

    # infer S-phase fractions
    cn_s, out_df = spf.infer()

    cn_s.to_csv(argv.output_s, sep='\t', index=False)
    out_df.to_csv(argv.output_spf, sep='\t', index=False)


if __name__ == '__main__':
    main()
