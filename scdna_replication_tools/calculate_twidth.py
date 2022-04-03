from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit


def get_args():
    p = ArgumentParser()

    p.add_argument('s_phase_cells', help='copynumber tsv file with replication time info for s-phase cells')
    p.add_argument('pseudobulk', help='pseudobulk replication timing profile')
    p.add_argument('pseudobulk_col', help='column in pseduobulk that represents replication time of each bin (0-10 hrs)')
    p.add_argument('frac_rt_col', help='column in s_phase_cells that represents fraction of bins replicated for each cell')
    p.add_argument('rs_col', help='column in s_phase_cells that represents binary replication state of each bin')
    p.add_argument('output', help='t-width plot for S-phase cells of interest')
    p.add_argument('curve', help='type of curve (linear or sigmoid) to fit to data')

    return p.parse_args()


def compute_time_from_scheduled_column(cn, pseudobulk_col='pseudobulk_hours', frac_rt_col='frac_rt', tfs_col='time_from_scheduled_rt'):
    '''
    Args:
        cn: dataframe where rows represent unique segments from all cells, columns contain cell-specific and pseudobulk replication timing info
        pseudobulk_col: column that represents the time (0-10 hrs) in which the locus replicates in pseudobulk
        frac_rt_col: column that represents the fraction of replicated bins for each S-phase cell
        tfs_col: column name to store each unique segment's time (hrs) from scheduled replication
    Returns:
        copy of cn dataframe with tfs_col added
    '''
    cn[tfs_col] = cn[pseudobulk_col] - (cn[frac_rt_col] * 10.0)
    return cn


def calc_pct_replicated_per_time_bin(cn, tfs_col='time_from_scheduled_rt', rs_col='rt_state', per_cell=False, query2=None, cell_col='cell_id',):
    '''
    Compute the percent of replicated segments for a set of time from scheduled intervals

    Args:
        cn: dataframe where rows represent unique segments from all cells, columns contain cell-specific and pseudobulk replication timing info
        tfs_col: column in cn that represents each bin's time from scheduled replication
        rs_col: column in cn that represents each bin's binary replication state
        per_cell: if true, do not aggregate across cells within the same time from scheduled interval
        query2: addidional query on the input cn; can be used to subset calculation to certain cells or certain regions of genome
    Returns:
        time_bins: list of time from replication intervals
        pct_reps: list of percent replicated values for each interval in time_bins
    '''
    intervals = np.linspace(-10, 10, 201)
    time_bins = []
    pct_reps = []
    for i in range(200):
        a = intervals[i]
        b = intervals[i+1]
        temp_cn = cn.query("{col} < {b} & {col} >= {a}".format(a=a, b=b, col=tfs_col))
        if query2:
            temp_cn = temp_cn.query(query2)
        if temp_cn.shape[0] > 0:
            if per_cell:
                for cell_id, chunk_cn in temp_cn.groupby(cell_col):
                    if chunk_cn.shape[0] > 0:
                        percent_replicated = sum(chunk_cn[rs_col]) / len(chunk_cn[rs_col])
                        pct_reps.append(percent_replicated)
                        time_bins.append(a)
            else:
                percent_replicated = sum(temp_cn[rs_col]) / len(temp_cn[rs_col])
                pct_reps.append(percent_replicated)
                time_bins.append(a)
    return time_bins, pct_reps


# helper functions to use if you want to calculate T-width via sigmoid regression
def sigmoid(x, x0, k, b):
    y = 1 / (1 + np.exp(-k*(x-x0)))+b
    return y

def inv_sigmoid(y, x0, k, b):
    temp = (1 / (y-b)) - 1
    x = (np.log(temp) / -k) + x0
    return x

def fit_sigmoid(xdata, ydata):
    p0 = [np.median(xdata), 1, 0.0] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0, method='dogbox')
    return popt, pcov

def calc_t_width(popt, low=0.25, high=0.75):
    right_time = inv_sigmoid(low, *popt)
    left_time = inv_sigmoid(high, *popt)
    t_width = right_time - left_time
    return t_width, left_time, right_time

# helper functions to use if you want to calculate T-width via linear regression
def linear(x, m, b):
    y = m * x + b
    return y

def inv_linear(y, m, b):
    x = (y - b) / m
    return x

def fit_linear(xdata, ydata):
    p0 = [-1.0, -1.0] # this is an mandatory initial guess
    popt, pcov = curve_fit(linear, xdata, ydata, p0)
    return popt, pcov

def calc_linear_t_width(popt, low=0.25, high=0.75):
    right_time = inv_linear(low, *popt)
    left_time = inv_linear(high, *popt)
    t_width = right_time - left_time
    return t_width, left_time, right_time


# superimpose T-width lines onto sigmoid function & data
def plot_cell_variability(xdata, ydata, popt=None, left_time=None, right_time=None, t_width=None,
                          alpha=1, title='Cell-to-cell variabilty', curve='sigmoid', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(xdata, ydata, label='data', alpha=alpha)
    if popt is not None:
        x = np.linspace(-10, 10, 1000)
        if curve == 'sigmoid':
            y = sigmoid(x, *popt)
        elif curve == 'linear':
            y = linear(x, *popt)
        ax.plot(x, y, color='r', label='fit')
        ax.axhline(y=0.75, color='k', linestyle='--')
        ax.axhline(y=0.25, color='k', linestyle='--')
        ax.axvline(x=left_time, color='k', linestyle='--')
        ax.axvline(x=right_time, color='k', linestyle='--', label='T_width={}'.format(round(t_width, 3)))
    ax.set_xlabel('time from scheduled replication (h)')
    ax.set_ylabel('% replicated')
    ax.set_title(title)
    ax.legend(loc='best')

    return ax


def calculate_twidth(cn, tfs_col='time_from_scheduled_rt', rs_col='rt_state', per_cell=False, query2=None, curve='sigmoid', cell_col='cell_id'):
    '''
    Calculate T-width value for a set of S-phase cells

    Args:
        cn: dataframe where rows represent unique segments from all cells, columns contain cell-specific and pseudobulk replication timing info
        tfs_col: column in cn that represents each bin's time from scheduled replication
        rs_col: column in cn that represents each bin's binary replication state
        per_cell: if true, do not aggregate across cells within the same time from scheduled interval
        query2: addidional query on the input cn; can be used to subset calculation to certain cells or certain regions of genome
        curve: type of curve to fit data to ('sigmoid' or 'linear')
    Returns:
        t_width: width of time from scheduled replication window that ranges from bins that are 25% to 75% replicated;
                 large values indicate high cell-to-cell heterogeneity in replication timing
        right_time: time from scheduled replication value in which bins are 25% replicated
        left_time: time from scheduled replication value in which bins are 75% replicated
        popt: list of parameters optimized during curve fitting
        time_bins: list of time from replication intervals
        pct_reps: list of percent replicated values for each interval in time_bins
    '''
    time_bins, pct_reps = calc_pct_replicated_per_time_bin(cn, tfs_col=tfs_col, rs_col=rs_col, per_cell=per_cell, query2=query2, cell_col=cell_col)
    if curve == 'sigmoid':
        popt, pcov = fit_sigmoid(time_bins, pct_reps)
        t_width, right_time, left_time = calc_t_width(popt)
    elif curve == 'linear':
        popt, pcov = fit_linear(time_bins, pct_reps)
        t_width, right_time, left_time = calc_linear_t_width(popt)

    return t_width, right_time, left_time, popt, time_bins, pct_reps


def compute_and_plot_twidth(cn, tfs_col='time_from_scheduled_rt', rs_col='rt_state', per_cell=False, query2=None,
                            cell_col='cell_id', alpha=1, title='Cell-to-cell variabilty', curve='sigmoid', ax=None):
    '''
    Calculate T-width value and plot optimized curve alongside data

    Args:
        cn: dataframe where rows represent unique segments from all cells, columns contain cell-specific and pseudobulk replication timing info
        tfs_col: column in cn that represents each bin's time from scheduled replication
        rs_col: column in cn that represents each bin's binary replication state
        per_cell: if true, do not aggregate across cells within the same time from scheduled interval
        query2: addidional query on the input cn; can be used to subset calculation to certain cells or certain regions of genome
        curve: type of curve to fit data to ('sigmoid' or 'linear')
        alpha: alpha value for plot
        title: title for plot
        ax: axis object (optional)
    Returns:
        ax: axis object containing desired plot
        t_width: width of time from scheduled replication window that ranges from bins that are 25% to 75% replicated;
                large values indicate high cell-to-cell heterogeneity in replication timing
    '''
    t_width, right_time, left_time, popt, time_bins, pct_reps = calculate_twidth(
        cn, tfs_col=tfs_col, rs_col=rs_col, per_cell=per_cell, query2=query2, curve=curve, cell_col=cell_col
    )
    ax = plot_cell_variability(time_bins, pct_reps, popt,
                               left_time, right_time, t_width,
                               alpha=alpha, title=title, curve=curve, ax=ax)

    return ax, t_width


def main():
    argv = get_args()
    cn = pd.read_csv(argv.s_phase_cells, sep='\t', dtype={'chr': str})
    pseudobulk = pd.read_csv(argv.pseudobulk, sep='\t', dtype={'chr': str})

    cn = pd.merge(cn, pseudobulk)

    cn = compute_time_from_scheduled_column(cn, pseudobulk_col=argv.pseudobulk_col, 
                                            frac_rt_col=argv.frac_rt_col, tfs_col='time_from_scheduled_rt')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax, t_width = compute_and_plot_twidth(cn, tfs_col='time_from_scheduled_rt', rs_col=argv.rs_col, per_cell=False, query2=None,
                                          alpha=1, title='Cell-to-cell variabilty', curve=argv.curve, ax=ax)
    
    fig.savefig(argv.output, bbox_inches='tight')



if __name__ == '__main__':
    main()
