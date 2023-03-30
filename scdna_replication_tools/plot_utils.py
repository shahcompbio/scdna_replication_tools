import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scgenome import refgenome
from sklearn import preprocessing
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def plot_cell_cn_profile2(ax, cn_data, value_field_name, cn_field_name=None, max_cn=13,
                          chromosome=None, s=5, squashy=False, color=None, alpha=1,
                          lines=False, label=None, scale_data=False):
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
                plot_data['start'], plot_data[value_field_name], alpha=0.3, c='k', label=''
            )
        elif color is not None:
            ax.plot(
                plot_data['start'], plot_data[value_field_name], alpha=0.3, c=color, label=''
            )
        else:
            ax.plot(
                plot_data['start'], plot_data[value_field_name], alpha=0.3, label=''
            )
    
    if label is None:
        label = value_field_name
    
    if cn_field_name is not None:
        ax.scatter(
            plot_data['start'], plot_data[value_field_name],
            c=plot_data[cn_field_name], s=s, alpha=alpha, label=label,
            cmap=get_cn_cmap(plot_data[cn_field_name].astype(int).values),
        )
    elif color is not None:
         ax.scatter(
            plot_data['start'], plot_data[value_field_name],
            c=color, s=s, alpha=alpha, label=label
        )
    else:
        ax.scatter(
            plot_data['start'], plot_data[value_field_name], s=s, alpha=alpha, label=label
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
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FixedFormatter(refgenome.info.chromosomes))

    if squashy:
        yticks = np.array([0, 2, 4, 7, 20])
        yticks_squashed = squash_f(yticks)
        ytick_labels = [str(a) for a in yticks]
        ax.set_yticks(yticks_squashed)
        ax.set_yticklabels(ytick_labels)
        ax.set_ylim((-0.01, 1.01))
        ax.spines['left'].set_bounds(0, 1)
    elif max_cn is not None:
        ax.set_ylim((-0.05*max_cn, max_cn))
        ax.set_yticks(range(0, int(max_cn) + 1))
        ax.spines['left'].set_bounds(0, max_cn)
    

    if chromosome is not None:
        sns.despine(ax=ax, offset=10, trim=False)
    else:
        sns.despine(ax=ax, offset=10, trim=True)

    return chromosome_info


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
        'LQ': 'lightgrey'
    }
    return cmap


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


def get_clone_cmap():
    cmap = {
        'A': 'C0',
        1: 'C0',
        'B': 'C1',
        2: 'C1',
        'C': 'C2',
        3: 'C2',
        'D': 'C3',
        4: 'C3',
        'E': 'C4',
        5: 'C4',
        'F': 'C5',
        6: 'C5',
        'G': 'C6',
        7: 'C6',
        'H': 'C7',
        8: 'C7',
        'I': 'C8',
        9: 'C8',
        'J': 'C9',
        10: 'C9',
        'K': 'lightsteelblue',
        11: 'lightsteelblue',
        'L': 'darksalmon',
        12: 'darksalmon',
        'M': 'lightgreen',
        13: 'lightgreen'
    }
    return cmap
