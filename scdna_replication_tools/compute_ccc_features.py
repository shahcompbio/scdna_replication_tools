from argparse import ArgumentParser
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture


def get_args():
    p = ArgumentParser()

    p.add_argument('input', help='copynumber tsv file with columns used for computing cell cycle classifier features')
    p.add_argument('output', help='same as input but with features added as new columns')

    return p.parse_args()


def calculate_features(cn, cell_col='cell_id', rpm_norm_col='rpm_clone_norm', madn_col='madn', lrs_col='lrs', bk_col='breakpoints', cn_col='state'):
    for cell_id, cell_cn in cn.groupby(cell_col):
        # comput lrs
        # this is the log likelihood ratio statistic for a 1 component vs 2 component gaussian mixture model
        X = cell_cn[rpm_norm_col].values.reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1).fit(X)
        log_lik1 = gmm1.score(X)
        gmm2 = GaussianMixture(n_components=2).fit(X)
        log_lik2 = gmm2.score(X)    
        LR_statistic = -2*(log_lik1 - log_lik2)
        
        # compute madn
        # this represents the median absolute deviation of the normalized read counts beteen adjacent bins
        madn = np.median(np.abs(np.diff(cell_cn[rpm_norm_col].values)))
        
        cn.loc[cell_cn.index, madn_col] = madn
        cn.loc[cell_cn.index, lrs_col] = LR_statistic
    
    # compute breakpoints if not already present in the input
    if bk_col not in cn.columns:
        cn = calculate_breakpoints(cn, cn_col=cn_col, bk_col=bk_col)

    return cn


def calculate_breakpoints(cn, cell_col='cell_id', cn_col='state', bk_col='breakpoints'):
    cn[bk_col] = 0
    # loop through each cell
    for cell_id, cell_cn in cn.groupby(cell_col):
        cell_bk_count = 0
        # loop through each chromosome
        # this is necassary as adjacent breakpoints on different chromosomes are not counted
        for chrom_id, chrom_cn in cell_cn.groupby('chr'):
            # compute the number of breakpoints on this chromosome
            chrom_bk_count = breakpoints(chrom_cn[cn_col].values)
            cell_bk_count += chrom_bk_count
        # update the number of breakpoints for this cell
        cn.loc[cell_cn.index, bk_col] = cell_bk_count
    return cn


def correct_breakpoints(cell_features, bk_col='breakpoints', clone_col='clone_id', output_col='corrected_breakpoints'):
    # correct for the number of breakpoints in each clone
    # this is necessary as clones with more somatic CNAs will have more breakpoints
    cell_features[output_col] = 0
    for clone_id, clone_features in cell_features.groupby(clone_col):
        X = clone_features[bk_col].values
        avg_clone_bk = np.mean(X)
        cell_features.loc[clone_features.index, output_col] = X - avg_clone_bk
    return cell_features


def correct_madn(cell_features, madn_col='madn', num_reads_col='total_mapped_reads_hmmcopy', output_col='corrected_madn'):
    # use a linear regression model to correct for the number of reads in each cell as it influences its MADN score
    X = cell_features[num_reads_col].values.reshape(-1, 1)
    y = cell_features[madn_col].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    cell_features[output_col] = cell_features[madn_col] - y_pred
    
    return cell_features


def compute_clone_normalization(cn, rpm_col='rpm', rpm_norm_col='rpm_clone_norm', clone_col='clone_id', cell_col='cell_id'):
    temp_df = []

    for col_id, chunk in cn.groupby(clone_col):
        chunk_mat = chunk.pivot_table(values=rpm_col, index=['chr', 'start'], columns=cell_col)
        filtered_chunk_mat = chunk_mat.interpolate(method='linear', axis=0)
        mean_profile = filtered_chunk_mat.mean(axis=1)
        mat_norm = filtered_chunk_mat.divide(mean_profile, axis=0)
        mat_norm_df = mat_norm.reset_index().melt(id_vars=['chr', 'start'], value_name=rpm_norm_col)
        temp_df.append(mat_norm_df)
    
    # one last round of filtering to ensure all cells have the same loci
    temp_df = pd.concat(temp_df, ignore_index=True)
    temp2 = temp_df.pivot_table(values=rpm_norm_col, index=['chr', 'start'], columns=cell_col)
    temp3 = temp2.dropna(axis=0)
    temp4 = temp3.reset_index().melt(id_vars=['chr', 'start'], value_name=rpm_norm_col)
    
    out = pd.merge(cn, temp4)
    return out


def autocorr(data, min_lag=10, max_lag=50):
    acorr = sm.tsa.acf(data, nlags=max_lag)
    return np.mean(acorr[min_lag-1:])


def breakpoints(data):
    ''' Find the number of adjacent bins with different values in `data`. '''
    temp_diff = np.diff(data)
    return sum(np.where(temp_diff!=0, 1, 0))


def compute_read_count(cn, input_col='reads', output_col='total_mapped_reads_hmmcopy'):
    ''' Compute the total number of mapped reads for each cell in `cn`. '''
    cn[output_col] = 0
    for cell_id, cell_cn in cn.groupby('cell_id'):
        cn.loc[cell_cn.index, output_col] = cell_cn[input_col].sum()
    return cn

def compute_cell_frac(cn, frac_rt_col='cell_frac_rep', rep_state_col='model_rep_state'):
    ''' Compute the fraction of replicated bins for all cells in `cn`, noting which cells have extreme values. '''
    cn['extreme_cell_frac'] = False
    for cell_id, cell_cn in cn.groupby('cell_id'):
        temp_rep = cell_cn[rep_state_col].values
        temp_frac = sum(temp_rep) / len(temp_rep)
        cn.loc[cell_cn.index, frac_rt_col] = temp_frac

        if temp_frac > 0.95 or temp_frac < 0.05:
            cn.loc[cell_cn.index, 'extreme_cell_frac'] = True
    return cn


def compute_ccc_features(cn, cell_col='cell_id', rpm_col='rpm', cn_col='state', clone_col='clone_id', madn_col='madn', lrs_col='lrs', num_reads_col='total_mapped_reads_hmmcopy', bk_col='breakpoints'):
    ''' 
    Compute the per-cell features that correspond to cell cycle for all cells in `cn`. 
    Parameters
    ----------
    cn : pandas.DataFrame
        A dataframe containing the copy number data for all cells.
    cell_col : str
        The name of the column containing the cell IDs in `cn` input.
    rpm_col : str
        The name of the column containing the read depth in `cn` input.
    cn_col : str
        The name of the column containing the copy number state in `cn` input.
    clone_col : str
        The name of the column containing the clone IDs in `cn` input.
    madn_col : str
        The name of the column containing the MADN scores in `cn` output.
    lrs_col : str
        The name of the column containing the likelihood ratio statistics (LRS) in `cn` output.
    num_reads_col : str
        The name of the column containing the total number of reads per cell in `cn` input.
    bk_col : str
        The name of the column containing the number of breakpoints per cell in `cn`.
        If not included in the input, this will be computed and included in the output.
    
    '''
    # normalize the read depth of all cells based on the consensus clone profile
    # this is most necessary for obtaining accurate likelihood ratio statistics (lrs)
    rpm_norm_col = '{}_clone_norm'.format(rpm_col)
    cn = compute_clone_normalization(cn, rpm_col=rpm_col, rpm_norm_col=rpm_norm_col, clone_col=clone_col, cell_col=cell_col)

    # calculate madn and likelihood ratio statistic (bimodality score) for all cells
    cn = calculate_features(cn, rpm_norm_col=rpm_norm_col, madn_col=madn_col, lrs_col=lrs_col, cell_col=cell_col, bk_col=bk_col, cn_col=cn_col)

    # compute the total number of reads for each cell if not already provided
    if num_reads_col not in cn.columns:
        cn = compute_read_count(cn, input_col=rpm_col, output_col=num_reads_col)
    
    # condense to a dataframe of per-cell features
    cell_features = cn[[
        cell_col, clone_col, madn_col, lrs_col, num_reads_col, bk_col
    ]].drop_duplicates()

    # correct madn scores based on the number of reads for each cell using linear regression
    cell_features = correct_madn(cell_features, madn_col=madn_col, num_reads_col=num_reads_col, output_col='corrected_{}'.format(madn_col))

    # normalize the number of breakpoints within each clone so that normalized values are centered at 0
    cell_features = correct_breakpoints(cell_features, bk_col=bk_col, clone_col=clone_col, output_col='corrected_{}'.format(bk_col))

    # merge cn with cell features to get corrected madn and breakpoint values
    cn_out = pd.merge(cn, cell_features)

    return cn_out, cell_features


def main():
    argv = get_args()
    cn = pd.read_csv(argv.input, sep='\t')
    cn, cell_features = compute_ccc_features(cn)
    cn.to_csv(argv.output, sep='\t', index=False)


if __name__ == '__main__':
    main()
