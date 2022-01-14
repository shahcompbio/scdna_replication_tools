from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.stats import skew
from scipy.spatial.distance import cityblock


def get_args():
    p = ArgumentParser()

    p.add_argument('input', help='copynumber tsv file with state, copy, clone_id, copy_norm_4, etc')
    p.add_argument('column', help='column in input df that represents continuous value of replication to be used for binarization')
    p.add_argument('output', help='same as input but with new columns corresponding to binarization of RT')
    p.add_argument('thresh_dists', help='table of manhattan distances for each threshold attempted for each cell')

    return p.parse_args()


def binarize_profiles(cn, column, MEAN_GAP_THRESH=0.7, EARLY_S_SKEW_THRESH=0.2, LATE_S_SKEW_THRESH=-0.2):
    '''
    Use normalized read depth profiles to compute binary replication timing profiles for each cell.
    This binarization process is an implementation of what is described in Dileep & Gilbert, Nature Comms (2018).

    Args:
        cn: long-form data frame of all S-phase cells profile with rows are segments & cells,
            columns include cell_id, clone_id, chr, start, end, reads, copy, etc
        column: column in cn to be used for binarization
    Returns:
        cn: same as input cn with new columns added to reflect each bin's replication state (rt_state)
            as well as each cell's fraction of replicated bins (frac_rt) and chosen threshold (binary_thresh)
        manhattan_df: dataframe of manhattan distances for each threshold attempted for each cell
    '''
    manhattan_df = []
    for cell_id, cell_cn in cn.groupby('cell_id'):
        # run GMM
        X = cell_cn[column].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0)
        states = gm.fit_predict(X)

        # add means and covariances of this cell to cn 
        for j in range(2):
            cn.loc[cell_cn.index, 'mean_{}'.format(j)] = gm.means_[j][0]
            cn.loc[cell_cn.index, 'covariance_{}'.format(j)] = gm.covariances_[j][0][0]

        # add GMM states for this cell to cn
        cn.loc[cell_cn.index, 'gmm_state'] = states

        # use GMM means to assign binary values for thresholding
        mean_0 = gm.means_[0][0]
        mean_1 = gm.means_[1][0]
        X = cell_cn[column].values

        # find the distance between the two means for each state
        mean_gap = abs(mean_0 - mean_1)

        # assume means denote binary values
        binary_0 = min(mean_0, mean_1)
        binary_1 = max(mean_0, mean_1)

        # use skew to define the binary values if means are close together
        if mean_gap < MEAN_GAP_THRESH:
            cell_skew = skew(X)
            print('cell_skew', cell_skew)
            # positive skew indicates early S-phase
            if cell_skew > EARLY_S_SKEW_THRESH:
                binary_0 = np.percentile(X, 50)
                binary_1 = np.percentile(X, 95)
            # negative skew indicates late S-phase
            elif cell_skew < LATE_S_SKEW_THRESH:
                binary_0 = np.percentile(X, 5)
                binary_1 = np.percentile(X, 50)
            # assume mid-S when skew is neutral
            else:
                binary_0 = np.percentile(X, 25)
                binary_1 = np.percentile(X, 75)

        # now that binary values are selected, I must compute the Manhattan distance
        # between binarized data and X for 100 different thresholds
        threshs = np.linspace(-3, 3, 100)
        lowest_dist = np.inf
        best_t = None
        manhattan_dists = []
        for t in threshs:
            # set values to binary_1 when above t, to binary_0 when below t
            B = np.where(X>t, binary_1, binary_0)
            # compute Manhattan distance between two vectors
            dist = cityblock(X, B)
            manhattan_dists.append(dist)
            if dist < lowest_dist:
                lowest_dist = dist
                best_t = t

        # save table of threshs vs manhattan distance for this cell
        temp_manhattan_df = pd.DataFrame({'thresh': threshs, 'manhattan_dist': manhattan_dists, 
        								'cell_id': [cell_id]*100, 'best_thresh': [best_t]*100})
        manhattan_df.append(temp_manhattan_df)

        # compute binary RT values based on the best threshold
        cell_rt = np.where(X>best_t, 1, 0)
        # compute fraction of replicated bins (cell's time within s-phase)
        frac_rt = sum(cell_rt) / len(cell_rt)

        cn.loc[cell_cn.index, 'rt_state'] = cell_rt
        cn.loc[cell_cn.index, 'frac_rt'] = frac_rt
        cn.loc[cell_cn.index, 'binary_thresh'] = best_t

    manhattan_df = pd.concat(manhattan_df)

    return cn, manhattan_df


def main():
	argv = get_args()
	cn = pd.read_csv(argv.input, sep='\t', dtype={'chr': str})

	cn, manhattan_df = binarize_profiles(cn, argv.column)

	cn.to_csv(argv.output, sep='\t', index=False)
	manhattan_df.to_csv(argv.thresh_dists, sep='\t', index=False)



if __name__ == '__main__':
	main()
