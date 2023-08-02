import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scgenome import refgenome
from sklearn import preprocessing
from scipy.stats import mode
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as dst
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch


def plot_cell_cn_profile2(ax, cn_data, value_field_name, cn_field_name=None, max_cn=13,
                          chromosome=None, s=5, squashy=False, color=None, alpha=1, rawy=False,
                          lines=False, label=None, scale_data=False, rasterized=True, cmap=None,
                          min_ci_field_name=None, max_ci_field_name=None, chrom_labels_to_remove=[]):
    """ Plot copy number profile on a genome axis

    Args:
        ax: matplotlib axis
        cn_data: copy number table
        value_field_name: column in cn_data to use for the y axis value
    
    Kwargs:
        cn_field_name: state column to color scatter points
        max_cn: max copy number for y axis
        chromosome: single chromosome plot
        s: size of scatter points
        cmap: colormap for cn_field_name
        rasterized: when true, raterize the scatter points in the figure to save space
        chrom_labels_to_remove: chromosome labels that should be removed from x-axis ticks

    The cn_data table should have the following columns (in addition to value_field_name and
    optionally cn_field_name):
        - chr
        - start
        - end
    """
    chromosome_info = refgenome.info.chromosome_info[['chr', 'chromosome_start', 'chromosome_end']].copy()
    chromosome_info['chr'] = pd.Categorical(chromosome_info['chr'], categories=cn_data['chr'].cat.categories)
    plot_data = cn_data.merge(chromosome_info)
    plot_data = plot_data[plot_data['chr'].isin(refgenome.info.chromosomes)]
    plot_data['start'] = plot_data['start'] + plot_data['chromosome_start']
    plot_data['end'] = plot_data['end'] + plot_data['chromosome_start']

    squash_coeff = 0.15
    squash_f = lambda a: np.tanh(squash_coeff * a)
    if squashy:
        plot_data[value_field_name] = squash_f(plot_data[value_field_name])
    
    if scale_data:
        plot_data[value_field_name] = preprocessing.scale(plot_data[value_field_name].values)
    
    if lines:
        chr_order = [str(i+1) for i in range(22)]
        chr_order.append('X')
        chr_order.append('Y')
        plot_data.chr.cat.set_categories(chr_order, inplace=True)
        plot_data = plot_data.sort_values(by=['chr', 'start'])
        if cn_field_name is not None:
            ax.plot(
                plot_data['start'], plot_data[value_field_name], alpha=0.3, c='k', label='', rasterized=rasterized
            )
            if min_ci_field_name and max_ci_field_name:
                ax.fill_between(
                    plot_data['start'], plot_data[min_ci_field_name], plot_data[max_ci_field_name],
                    alpha=0.2, color='k', label='', rasterized=rasterized
                )
        elif color is not None:
            ax.plot(
                plot_data['start'], plot_data[value_field_name], alpha=0.3, c=color, label='', rasterized=rasterized
            )
            if min_ci_field_name and max_ci_field_name:
                ax.fill_between(
                    plot_data['start'], plot_data[min_ci_field_name], plot_data[max_ci_field_name],
                    alpha=0.2, color=color, label='', rasterized=rasterized
                )
        else:
            ax.plot(
                plot_data['start'], plot_data[value_field_name], alpha=0.3, label='', rasterized=rasterized
            )
            if min_ci_field_name and max_ci_field_name:
                ax.fill_between(
                    plot_data['start'], plot_data[min_ci_field_name], plot_data[max_ci_field_name],
                    alpha=0.2, label='', rasterized=rasterized
                )
    
    if label is None:
        label = value_field_name
    
    if cn_field_name is not None:
        if cmap is not None:
            ax.scatter(
                plot_data['start'], plot_data[value_field_name],
                c=plot_data[cn_field_name], s=s, alpha=alpha, label=label,
                cmap=cmap, rasterized=rasterized
            )
        else:   
            ax.scatter(
                plot_data['start'], plot_data[value_field_name],
                c=plot_data[cn_field_name], s=s, alpha=alpha, label=label,
                cmap=get_cn_cmap(plot_data[cn_field_name].astype(int).values),
                rasterized=rasterized
            )
    elif color is not None:
         ax.scatter(
            plot_data['start'], plot_data[value_field_name],
            c=color, s=s, alpha=alpha, label=label,
            rasterized=rasterized
        )
    else:
        ax.scatter(
            plot_data['start'], plot_data[value_field_name], s=s, alpha=alpha, label=label,
            rasterized=rasterized
        )
    
    if chromosome is not None:
        chromosome_length = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_length']
        chromosome_start = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_start']
        chromosome_end = refgenome.info.chromosome_info.set_index('chr').loc[chromosome, 'chromosome_end']
        xticks = np.arange(0, chromosome_length, 2e7)
        xticklabels = ['{0:d}M'.format(int(x / 1e6)) for x in xticks]
        xminorticks = np.arange(0, chromosome_length, 1e6)
        ax.set_xlabel(f'chromosome {chromosome}')
        ax.set_xticks(xticks + chromosome_start)
        ax.set_xticklabels(xticklabels)
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(xminorticks + chromosome_start))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlim((chromosome_start, chromosome_end))

    else:
        ax.set_xlim((-0.5, refgenome.info.chromosome_end.max()))
        ax.set_xlabel('chromosome')
        ax.set_xticks([0] + list(refgenome.info.chromosome_end.values))
        ax.set_xticklabels([])
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(refgenome.info.chromosome_mid))
        chrom_labels = ['' if x in chrom_labels_to_remove else x for x in refgenome.info.chromosomes]
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(chrom_labels))

    if squashy and not rawy:
        yticks = np.array([0, 2, 4, 7, 20])
        yticks_squashed = squash_f(yticks)
        ytick_labels = [str(a) for a in yticks]
        ax.set_yticks(yticks_squashed)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim((-0.01, 1.01))
        ax.spines['left'].set_bounds(0, 1)
    elif not rawy:
        ax.set_ylim((-0.05*max_cn, max_cn))
        ax.set_yticks(range(0, int(max_cn) + 1))
        ax.spines['left'].set_bounds(0, max_cn)
    

    if chromosome is not None:
        sns.despine(ax=ax, offset=10, trim=False)
    else:
        sns.despine(ax=ax, offset=10, trim=True)

    return chromosome_info


def plot_clustered_cell_cn_matrix(
    ax, cn_data, cn_field_name, cluster_field_name='cluster_id', secondary_field_name=None, 
    raw=False, max_cn=13, cmap=None, chromosome=None, chrom_boundary_width=1, chrom_labels_to_remove=[]):
    
    if chromosome is not None:
        cn_data = cn_data.query('chr=="{}"'.format(chromosome))
    
    plot_data = cn_data.merge(refgenome.info.chrom_idxs)

    if secondary_field_name is not None:
        plot_data = plot_data.set_index(['chr_index', 'start', 'cell_id', cluster_field_name])
        plot_data = plot_data[[secondary_field_name, cn_field_name]]
        plot_data = plot_data.unstack(level=['cell_id', cluster_field_name])
        ordering_mat = plot_data[secondary_field_name].values
        ordering = mode(ordering_mat)[0]
        ordering = np.reshape(ordering, -1)

        plot_data = cn_data.merge(refgenome.info.chrom_idxs)
        plot_data = plot_data.set_index(['chr_index', 'start', 'cell_id', cluster_field_name])[cn_field_name].unstack(level=['cell_id', cluster_field_name]).fillna(0)
    else:
        plot_data = plot_data.set_index(['chr_index', 'start', 'cell_id', cluster_field_name])[cn_field_name].unstack(level=['cell_id', cluster_field_name]).fillna(0)
        ordering = _secondary_clustering(plot_data.values)
    
    ordering = pd.Series(ordering, index=plot_data.columns, name='cell_order')
    plot_data = plot_data.T.set_index(ordering, append=True).T

    plot_data = plot_data.sort_index(axis=1, level=[1, 2])

    if max_cn is not None:
        plot_data[plot_data > max_cn] = max_cn
    
    mat_chrom_idxs = plot_data.index.get_level_values(0).values
    chrom_boundaries = np.array([0] + list(np.where(mat_chrom_idxs[1:] != mat_chrom_idxs[:-1])[0]) + [plot_data.shape[0] - 1])
    chrom_sizes = chrom_boundaries[1:] - chrom_boundaries[:-1]
    chrom_mids = chrom_boundaries[:-1] + chrom_sizes / 2
    ordered_mat_chrom_idxs = mat_chrom_idxs[np.where(np.array([1] + list(np.diff(mat_chrom_idxs))) != 0)]
    chrom_names = np.array(refgenome.info.chromosomes)[ordered_mat_chrom_idxs]
    chrom_names = ['' if x in chrom_labels_to_remove else x for x in chrom_names]

    mat_cluster_ids = plot_data.columns.get_level_values(1).values
    cluster_boundaries = np.array([0] + list(np.where(mat_cluster_ids[1:] != mat_cluster_ids[:-1])[0]) + [plot_data.shape[1] - 1])
    cluster_sizes = cluster_boundaries[1:] - cluster_boundaries[:-1]
    cluster_mids = cluster_boundaries[:-1] + cluster_sizes / 2

    if not raw and cmap is None:
        cmap = get_cn_cmap(plot_data.values)

    im = ax.imshow(plot_data.astype(float).T, aspect='auto', cmap=cmap, interpolation='none')

    if chromosome is not None:
        ax.set_xlabel(f'chr{chromosome}')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    else:
        ax.set(xticks=chrom_mids)
        ax.set(xticklabels=chrom_names)
        for val in chrom_boundaries[:-1]:
            ax.axvline(x=val, linewidth=chrom_boundary_width, color='black', zorder=100)

    return plot_data

def _secondary_clustering(data):
    D = dst.squareform(dst.pdist(data.T, 'cityblock'))
    Y = sch.linkage(D, method='complete')
    Z = sch.dendrogram(Y, color_threshold=-1, no_plot=True)
    idx = np.array(Z['leaves'])
    ordering = np.zeros(idx.shape[0], dtype=int)
    ordering[idx] = np.arange(idx.shape[0])
    return ordering


# helper functions for plotting heatmaps
def plot_colorbar(ax, color_mat, title=None):
    ax.imshow(np.array(color_mat)[::-1, np.newaxis], aspect='auto', origin='lower')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title)


def plot_color_legend(ax, color_map, title=None):
    legend_elements = []
    for name, color in color_map.items():
        legend_elements.append(Patch(facecolor=color, label=name))
    ax.legend(handles=legend_elements, loc='center left', title=title)
    ax.grid(False)
    ax.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])


def make_color_mat_float(values, palette_color):
    """
    Make a color_mat for a 0-1 float array `values` and a
    corresponding color pallete.
    """
    pal = plt.get_cmap(palette_color)
    color_mat = []
    for val in values:
        color_mat.append(pal(val))
    color_dict = {0: pal(0.0), 1: pal(1.0)}
    return color_mat, color_dict


def format_embedding_frame(ax, xlabel='PC1', ylabel='PC2'):
    ''' Given a subplot that represents a PCA or UMAP embedding, despine to only show the x and y axis labels in the bottom left corner. '''
    # remove all axis ticks and tick labels
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    # remove the top and right frames from the subplot
    sns.despine(ax=ax)
    # only show the left and bottom spines for the first 0.25 of the plot
    # get the x and y limits of the subplot
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # set the second elements of xlim and ylim to be 25% of the difference between the first and second elements
    ax.spines['left'].set_bounds(ylim[0], ylim[0] + (ylim[1] - ylim[0]) * 0.25)
    ax.spines['bottom'].set_bounds(xlim[0], xlim[0] + (xlim[1] - xlim[0]) * 0.25)
    # set the x and y labels, centered in between the current spine limits
    ax.set_xlabel(xlabel, loc='left')
    ax.set_ylabel(ylabel, loc='bottom')


def get_cn_cmap(cn_data):
    color_reference = {0:'#3182BD', 1:'#9ECAE1', 2:'#CCCCCC', 3:'#FDCC8A', 4:'#FC8D59', 5:'#E34A33', 6:'#B30000', 7:'#980043', 8:'#DD1C77', 9:'#DF65B0', 10:'#C994C7', 11:'#D4B9DA'}
    min_cn = int(cn_data.min())
    max_cn = int(cn_data.max())
    assert min_cn - cn_data.min() == 0
    assert max_cn - cn_data.max() == 0
    color_list = []
    for cn in range(min_cn, max_cn+1):
        if cn > max(color_reference.keys()):
            cn = max(color_reference.keys())
        color_list.append(color_reference[cn])
    return ListedColormap(color_list)


def get_phase_cmap():
    ''' Global color map for cell cycle phases '''
    cmap = {
        'S': 'goldenrod',
        'G1/2': 'dodgerblue',
        'G1': 'dodgerblue',
        'G2': 'lightblue',
        'LQ': 'lightgrey',
        'G2M': 'yellowgreen'
    }
    return cmap


def get_signals_cmap(return_colors=False):
    phase_colors = {
        'A-Hom': '#56941E', -2: '#56941E',
        'A-Gained': '#94C773', -1: '#94C773',
        'Balanced': '#d5d5d4', 0: '#d5d5d4',
        'B-Gained': '#7B52AE', 1: '#7B52AE',
        'B-Hom': '#471871', 2: '#471871',
    }
    color_list = []
    for i in phase_colors.keys():
        color_list.append(phase_colors[i])
    if return_colors:
        return ListedColormap(color_list), phase_colors
    return ListedColormap(color_list)


def get_rt_cmap(return_colors=False):
    rt_colors = {0: '#552583', 1: '#FDB927'}
    color_list = []
    for i in [0, 1]:
        color_list.append(rt_colors[i])
    if return_colors:
        return ListedColormap(color_list), rt_colors
    return ListedColormap(color_list)


def get_acc_cmap(return_colors=False):
    """ Return a colormap for replication accuracy states. False positives are green, false negatives are purple, and correct calls are gray. """
    acc_colors = {0:'#CCCCCC', -1: '#532A44', 1: '#00685E'}
    color_list = []
    for i in [-1, 0, 1]:
        color_list.append(acc_colors[i])
    if return_colors:
        return ListedColormap(color_list), acc_colors
    return ListedColormap(color_list)


def get_cna_cmap():
    ''' Global color map for copy number alterations '''
    cmap = {
        'gain': 'red',  # red
        'loss': 'deepskyblue',  # dark blue
        'neutral': '#CCCCCC',  # grey
        'unaltered': '#CCCCCC'  # grey
    }
    return cmap   


def get_clone_cmap():
    cmap = {
        'A': 'cadetblue',
        1: 'cadetblue',
        'B': 'chocolate',
        2: 'chocolate',
        'C': 'olivedrab',
        3: 'olivedrab',
        'D': 'tan',
        4: 'tan',
        'E': 'plum',
        5: 'plum',
        'F': 'indianred',
        6: 'indianred',
        'G': 'lightpink',
        7: 'lightpink',
        'H': 'slategrey',
        8: 'slategrey',
        'I': 'darkseagreen',
        9: 'darkseagreen',
        'J': 'darkkhaki',
        10: 'darkkhaki',
        'K': 'lightsteelblue',
        11: 'lightsteelblue',
        'L': 'darksalmon',
        12: 'darksalmon',
        'M': 'lightgreen',
        13: 'lightgreen',
        'N': 'lightpink',
        14: 'lightpink',
        'O': 'lightgrey',
        15: 'lightgrey',
        'P': 'lightblue',
        16: 'lightblue',
        'Q': 'coral',
        17: 'coral',
        'R': 'lightcyan',
        18: 'lightcyan',
        'S': 'lightgoldenrodyellow',
        19: 'lightgoldenrodyellow',
        'T': 'darkseagreen',
        20: 'darkseagreen',
        'U': 'indigo',
        21: 'indigo'
    }
    return cmap


def get_htert_cmap():
    cmap = {
        'WT': 'C0',
        'SA039': 'C0',
        'TP53-/-': 'C1',
        'SA906a': 'C1',
        'SA906b': 'orange',
        'TP53-/-,BRCA1+/-' : 'C2',
        'SA1292': 'C2',
        'TP53-/-,BRCA1-/-': 'C3',
        'SA1056': 'C3',
        'TP53-/-,BRCA2+/-': 'C4',
        'SA1188': 'C4',
        'TP53-/-,BRCA2-/-': 'C5',
        'SA1054': 'C5',
        'SA1055': 'chocolate',
        'OV2295': 'lightgreen'
    }
    return cmap


def get_facs_cmap():
    ''' Global color map for FACS isolated cell lines '''
    cmap = {
        'GM18507': 'mediumpurple', 'SA928': 'mediumpurple', 1: 'mediumpurple',
        'T47D': 'khaki', 'SA1044': 'khaki', 2: 'khaki',
    }
    return cmap


def get_metacohort_feature_cmap():
    ''' Colormap for each feature used to predict RT in the metacohort. '''
    pal = sns.color_palette('cubehelix', 4)
    cmap = {'global': pal[0], 'ploidy': pal[1], 'type': pal[2], 'signature': pal[3]}
    return cmap


def get_metacohort_cmaps(return_cdicts=False):
    cell_type_cdict = {
        'hTERT': 'lightsteelblue', 0: 'lightsteelblue',
        'HGSOC': 'teal', 1: 'teal',
        'TNBC': 'salmon', 2: 'salmon',
        'OV2295': 'lightgreen', 3: 'lightgreen',
        'T47D': 'khaki', 4: 'khaki',
        'GM18507': 'mediumpurple', 5: 'mediumpurple',
    }
    cell_type_cmap = LinearSegmentedColormap.from_list('cell_type_cmap', list(cell_type_cdict.values()), N=len(cell_type_cdict))

    signature_cdict = {
        'FBI': 'plum', 0: 'plum',
        'HRD': 'cyan', 1: 'cyan',
        'TD': 'coral', 2: 'coral',
    }
    signature_cmap = LinearSegmentedColormap.from_list('signature_cmap', list(signature_cdict.values()), N=len(signature_cdict))

    condition_cdict = {
        'Line': 'tan', 0: 'tan',
        'PDX': 'lightskyblue', 1: 'lightskyblue',
    }
    condition_cmap = LinearSegmentedColormap.from_list('condition_cmap', list(condition_cdict.values()), N=len(condition_cdict))

    ploidy_cdict = {2:'#CCCCCC', 3:'#FDCC8A', 4:'#FC8D59', 5:'#E34A33'}
    ploidy_cmap = LinearSegmentedColormap.from_list('ploidy_cmap', list(ploidy_cdict.values()), N=len(ploidy_cdict))

    sample_cdict = {
        0: (0.6897625000000001, 0.38092750000000003, 0.26002749999999997),
        'SA039': (0.6897625000000001, 0.38092750000000003, 0.26002749999999997),
        1: (0.5157175, 0.22038749999999993, 0.1751124999999999),
        'SA906a': (0.5157175, 0.22038749999999993, 0.1751124999999999),
        2: (0.48497999999999997, 0.8148200000000001, 0.56322),
        'SA906b': (0.48497999999999997, 0.8148200000000001, 0.56322),
        3: (0.39483749999999995, 0.7669725000000001, 0.5814475000000001),
        'SA1188': (0.39483749999999995, 0.7669725000000001, 0.5814475000000001),
        4: (0.7276875, 0.43818749999999995, 0.2923024999999999),
        'SA1292': (0.7276875, 0.43818749999999995, 0.2923024999999999),
        5: (0.3537825000000001, 0.3585725000000001, 0.5539675),
        'SA1056': (0.3537825000000001, 0.3585725000000001, 0.5539675),
        6: (0.4277675000000001, 0.6180524999999999, 0.8058224999999999),
        'SA1055': (0.4277675000000001, 0.6180524999999999, 0.8058224999999999),
        7: (0.438735, 0.17170999999999997, 0.14948499999999998),
        'SA1054': (0.438735, 0.17170999999999997, 0.14948499999999998),
        8: (0.795725, 0.636195, 0.400895),
        'OV2295': (0.795725, 0.636195, 0.400895),
        9: (0.6097, 0.8151650000000001, 0.450715),
        'SA1091': (0.6097, 0.8151650000000001, 0.450715),
        10: (0.3180875000000001, 0.7139825, 0.6013175000000002),
        'SA1093': (0.3180875000000001, 0.7139825, 0.6013175000000002),
        11: (0.74617, 0.70018, 0.39843000000000006),
        'SA1049': (0.74617, 0.70018, 0.39843000000000006),
        12: (0.7040524999999999, 0.7179874999999999, 0.3808625),
        'SA1096': (0.7040524999999999, 0.7179874999999999, 0.3808625),
        13: (0.266125, 0.23539500000000002, 0.379105),
        'SA1162': (0.266125, 0.23539500000000002, 0.379105),
        14: (0.307585, 0.64131, 0.679735),
        'SA1050': (0.307585, 0.64131, 0.679735),
        15: (0.76332, 0.508735, 0.33322000000000007),
        'SA1051': (0.76332, 0.508735, 0.33322000000000007),
        16: (0.371325, 0.6353699999999998, 0.755515),
        'SA1052': (0.371325, 0.6353699999999998, 0.755515),
        17: (0.5687775, 0.8378224999999999, 0.5202075000000002),
        'SA1053': (0.5687775, 0.8378224999999999, 0.5202075000000002),
        18: (0.7874075, 0.5775675, 0.3718824999999999),
        'SA1181': (0.7874075, 0.5775675, 0.3718824999999999),
        19: (0.6413274999999999, 0.7885625, 0.40686749999999994),
        'SA1184': (0.6413274999999999, 0.7885625, 0.40686749999999994),
        20: (0.6740125, 0.7566225, 0.3876075000000001),
        'SA530': (0.6740125, 0.7566225, 0.3876075000000001),
        21: (0.43862500000000004, 0.5258800000000001, 0.761855),
        'SA604': (0.43862500000000004, 0.5258800000000001, 0.761855),
        22: (0.28434000000000004, 0.66728, 0.6241049999999999),
        'SA501': (0.28434000000000004, 0.66728, 0.6241049999999999),
        23: (0.4061625, 0.4502775, 0.6764275),
        'SA1035': (0.4061625, 0.4502775, 0.6764275),
        24: (0.6411825, 0.3249075, 0.2303075),
        'SA535': (0.6411825, 0.3249075, 0.2303075),
        25: (0.5831975, 0.2714525, 0.20205249999999997),
        'SA609': (0.5831975, 0.2714525, 0.20205249999999997),
        26: (0.7836400000000001, 0.6765800000000001, 0.4122999999999999),
        'T47D': (0.7836400000000001, 0.6765800000000001, 0.4122999999999999),
        27: (0.4506199999999998, 0.5851649999999997, 0.8097400000000001),
        'GM18507': (0.4506199999999998, 0.5851649999999997, 0.8097400000000001)
    }
    sample_cmap = LinearSegmentedColormap.from_list('sample_cmap', list(sample_cdict.values()), N=len(sample_cdict))

    if return_cdicts:
        return cell_type_cdict, signature_cdict, condition_cdict, ploidy_cdict, sample_cdict
    else:
        return cell_type_cmap, signature_cmap, condition_cmap, ploidy_cmap, sample_cmap
