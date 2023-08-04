import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from scgenome import cncluster
from argparse import ArgumentParser
from scdna_replication_tools.plot_utils import get_rt_cmap, get_clone_cmap, plot_colorbar, make_color_mat_float, plot_clustered_cell_cn_matrix


def get_args():
    p = ArgumentParser()

    p.add_argument('cn_s', help='long-form dataframe of S-phase cells with pyro model results')
    p.add_argument('cn_g', help='long-form dataframe of G1/2-phase cells')
    p.add_argument('dataset')
    p.add_argument('plot1', help='heatmaps of all S-phase cells sorted the same')
    p.add_argument('plot2', help='heatmaps of G1- vs S-phase hmmcopy states')
    p.add_argument('plot3', help='heatmaps of G1- vs S-phase reads per million')

    return p.parse_args()


def plot_model_results(
    cn_s, cn_g, argv=None, clone_col='clone_id', second_sort_col='model_tau', 
    rpm_col='rpm', input_cn_col='state', output_cn_col='model_cn_state', output_rep_col='model_rep_state',
    top_title_prefix='S-phase cells', bottom_title_prefix='G1/2-phase cells',
    rpm_title='Reads per million', input_cn_title='Input CN states',
    output_cn_title='PERT CN states', rep_title='PERT replication states', 
    rt_cmap=get_rt_cmap(), clone_cmap=get_clone_cmap(), rpm_cmap='viridis',
    chromosome=None, chrom_boundary_width=1, chrom_labels_to_remove=[]
    ):
    ''' 
    Plot input and output PERT heatmaps for S-phase and G1/2-phase cells. 
    
    Parameters
    ----------
    cn_s : pandas.DataFrame
        long-form dataframe of S-phase cells with pert model results.
        this dataframe should have the following columns in addition to those specified by additional parameters:
            - chr (categorical)
            - start (int)
            - end (int)
            - cell_id (str)
    cn_g : pandas.DataFrame
        long-form dataframe of G1/2-phase cells with pert model results.
        this dataframe should have the following columns in addition to those specified by additional parameters:
            - chr (categorical)
            - start (int)
            - end (int)
            - cell_id (str)
    argv : list
        list of command line arguments (optional)
    clone_col : str
        column name for clone IDs
    second_sort_col : str
        column name for secondary sorting of cells (e.g. time in S-phase)
    rpm_col : str
        column name for reads per million (cell-normalized read count)
    input_cn_col : str
        column name for input CN states (i.e. hmmcopy states)
    output_cn_col : str
        column name for output CN states (from pert)
    output_rep_col : str
        column name for output replication states (from pert)
    top_title_prefix : str
        prefix for the title of the top row of plots (corresponding to cn_s)
    bottom_title_prefix : str
        prefix for the title of the bottom row of plots (corresponding to cn_g)
    rpm_title : str
        title for the rpm heatmap in the far-left column
    input_cn_title : str
        title for the input CN heatmap in the middle-left column
    output_cn_title : str
        title for the output CN heatmap in the middle-right column
    rep_title : str
        title for the replication state heatmap in the far-right column
    rt_cmap : matplotlib.colors.ListedColormap
        colormap for replication states
    clone_cmap : matplotlib.colors.ListedColormap
        colormap for clone IDs
    rpm_cmap : str
        colormap for reads per million
    chromosome : str
        chromosome to plot if we wish to only show one chr instead of the full genome (optional)
    chrom_boundary_width : int
        width of the chromosome boundary lines (optional)
    chrom_labels_to_remove : list
        list of chromosome labels to remove along the x-axis (optional)
    '''

    # create mapping of clone IDs
    cluster_col = 'cluster_id'
    clone_dict = dict([(y,x+1) for x,y in enumerate(sorted(cn_g[clone_col].unique()))])
    cn_g[cluster_col] = cn_g[clone_col]
    cn_g = cn_g.replace({cluster_col: clone_dict})
    cn_s[cluster_col] = cn_s[clone_col]
    cn_s = cn_s.replace({cluster_col: clone_dict})

    # plot the heatmaps
    fig = plt.figure(figsize=(28,14))

    # plot the S-phase cells in the top row
    # top left corner is the rpm
    ax0 = fig.add_axes([0.05,0.5,0.23,0.45])
    plot_data0 = plot_clustered_cell_cn_matrix(
        ax0, cn_s, rpm_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col, 
        max_cn=None, raw=True, cmap=rpm_cmap, chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, 
        chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax0.set_title('{}\n{}'.format(top_title_prefix, rpm_title))

    # top mid-left is the hmmcopy states
    ax1 = fig.add_axes([0.29,0.5,0.23,0.45])
    plot_data1 = plot_clustered_cell_cn_matrix(
        ax1, cn_s, input_cn_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col,
        chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax1.set_title('{}\n{}'.format(top_title_prefix, input_cn_title))

    # top mid-right is the model cn states
    ax2 = fig.add_axes([0.53,0.5,0.23,0.45])
    plot_data2 = plot_clustered_cell_cn_matrix(
        ax2, cn_s, output_cn_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col,
        chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax2.set_title('{}\n{}'.format(top_title_prefix, output_cn_title))

    # top right corner is the replication states
    ax3 = fig.add_axes([0.77,0.5,0.23,0.45])
    plot_data3 = plot_clustered_cell_cn_matrix(
        ax3, cn_s, output_rep_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col, cmap=rt_cmap,
        chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax3.set_title('{}\n{}'.format(top_title_prefix, rep_title))

    # plot the G1/2-phase cells in the bottom row
    # bottom left corner is the rpm
    ax4 = fig.add_axes([0.05,0.0,0.23,0.45])
    plot_data4 = plot_clustered_cell_cn_matrix(
        ax4, cn_g, rpm_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col, 
        max_cn=None, raw=True, cmap=rpm_cmap, chromosome=chromosome, chrom_boundary_width=chrom_boundary_width,
        chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax4.set_title('{}\n{}'.format(bottom_title_prefix, rpm_title))

    # bottom mid-left is the hmmcopy states
    ax5 = fig.add_axes([0.29,0.0,0.23,0.45])
    plot_data5 = plot_clustered_cell_cn_matrix(
        ax5, cn_g, input_cn_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col,
        chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax5.set_title('{}\n{}'.format(bottom_title_prefix, input_cn_title))

    # bottom mid-right is the model cn states
    ax6 = fig.add_axes([0.53,0.0,0.23,0.45])
    plot_data6 = plot_clustered_cell_cn_matrix(
        ax6, cn_g, output_cn_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col,
        chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax6.set_title('{}\n{}'.format(bottom_title_prefix, output_cn_title))

    # bottom right corner is the replication states
    ax7 = fig.add_axes([0.77,0.0,0.23,0.45])
    plot_data7 = plot_clustered_cell_cn_matrix(
        ax7, cn_g, output_rep_col, cluster_field_name=cluster_col, secondary_field_name=second_sort_col, cmap=rt_cmap,
        chromosome=chromosome, chrom_boundary_width=chrom_boundary_width, chrom_labels_to_remove=chrom_labels_to_remove
    )
    ax7.set_title('{}\n{}'.format(bottom_title_prefix, rep_title))

    # turn off the y-axis ticks in all subplots
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_yticks([])
        ax.set_ylabel('')

    # add the colorbars for clone_id and model_tau
    if len(clone_dict) > 1:
        cell_ids = plot_data0.columns.get_level_values(0).values
        cluster_ids0 = plot_data0.columns.get_level_values(1).values
        # use mcolors to change every element in the dict to rgba
        for key in clone_cmap.keys():
            clone_cmap[key] = mcolors.to_rgba(clone_cmap[key])
        color_mat0, color_map0 = cncluster.get_cluster_colors(cluster_ids0, color_map=clone_cmap, return_map=True)

        # get array of second_sort_col values that that match the cell_id order
        condensed_cn = cn_s[['cell_id', second_sort_col]].drop_duplicates()
        secondary_array = []
        for cell in cell_ids:
            s = condensed_cn[condensed_cn['cell_id'] == cell][second_sort_col].values[0]
            secondary_array.append(s)

        # make color mat according to secondary array
        secondary_color_mat, secondary_to_colors = make_color_mat_float(secondary_array, 'Blues')

        # create color bar that shows clone id for each row in heatmap
        ax = fig.add_axes([0.03,0.5,0.01,0.45])
        plot_colorbar(ax, color_mat0)

        # create color bar that shows secondary sort value for each row in heatmap
        ax = fig.add_axes([0.04,0.5,0.01,0.45])
        plot_colorbar(ax, secondary_color_mat)

        # repeat for the G1/2-phase cells in the bottom row
        cell_ids = plot_data4.columns.get_level_values(0).values
        cluster_ids4 = plot_data4.columns.get_level_values(1).values
        color_mat4, color_map4 = cncluster.get_cluster_colors(cluster_ids4, color_map=clone_cmap, return_map=True)

        # get array of second_sort_col values that that match the cell_id order
        condensed_cn = cn_g[['cell_id', second_sort_col]].drop_duplicates()
        secondary_array = []
        for cell in cell_ids:
            s = condensed_cn[condensed_cn['cell_id'] == cell][second_sort_col].values[0]
            secondary_array.append(s)
        
        # make color mat according to secondary array
        secondary_color_mat, secondary_to_colors = make_color_mat_float(secondary_array, 'Blues')

        # create color bar that shows clone id for each row in heatmap
        ax = fig.add_axes([0.03,0.0,0.01,0.45])
        plot_colorbar(ax, color_mat4)

        # create color bar that shows secondary sort value for each row in heatmap
        ax = fig.add_axes([0.04,0.0,0.01,0.45])
        plot_colorbar(ax, secondary_color_mat)

    # save the figure if an output file is specified (running script directly)
    if argv is not None:
        fig.savefig(argv.plot1, bbox_inches='tight', dpi=300)
    # otherwise return the figure (running from notebook)
    else:
        return fig


def plot_cn_states(cn_s, cn_g1, argv=None, clone_col='clone_id', cn_col='state', title0='HMMcopy states\nG1/2-phase', title1='HMMcopy states\nS-phase'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    ax = ax.flatten()

    plot_clustered_cell_cn_matrix(ax[0], cn_g1, cn_col, cluster_field_name=clone_col)
    plot_clustered_cell_cn_matrix(ax[1], cn_s, cn_col, cluster_field_name=clone_col)

    ax[0].set_title(title0)
    ax[1].set_title(title1)

    if argv is not None:
        fig.savefig(argv.plot2, bbox_inches='tight', dpi=300)
    else:
        return fig


def plot_rpm(cn_s, cn_g1, argv=None, clone_col='clone_id', rpm_col='rpm', title0='Reads per million\nG1/2-phase', title1='Reads per million\nS-phase', cmap='viridis'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    ax = ax.flatten()

    plot_clustered_cell_cn_matrix(ax[0], cn_g1, rpm_col, max_cn=None, raw=True, cmap=cmap, cluster_field_name=clone_col)
    plot_clustered_cell_cn_matrix(ax[1], cn_s, rpm_col, max_cn=None, raw=True, cmap=cmap, cluster_field_name=clone_col)

    ax[0].set_title(title0)
    ax[1].set_title(title1)

    if argv is not None:
        fig.savefig(argv.plot3, bbox_inches='tight', dpi=300)
    else:
        return fig


def main():
    argv = get_args()

    cn_s = pd.read_csv(argv.cn_s, sep='\t')
    cn_g = pd.read_csv(argv.cn_g, sep='\t')

    # show rpm, hmmcopy, inferred cn, inferred rep heatmaps for S-phase cells and G1/2-phase cells
    # where all the rows are sorted the same in all four heatmaps
    plot_model_results(cn_s, cn_g, argv=argv, top_title_prefix='{} S-phase cells'.format(argv.dataset), bottom_title_prefix='{} G1/2-phase cells'.format(argv.dataset))

    # show hmmcopy state heatmaps for both S-phase and G1-phase cells
    plot_cn_states(cn_s, cn_g, argv=argv, title0='{} HMMcopy states\nG1/2-phase cells'.format(argv.dataset), title1='{} HMMcopy states\nS-phase cells'.format(argv.dataset))

    # show reads per million heatmaps for both S-phase and G1-phase cells
    plot_rpm(cn_s, cn_g, argv=argv, title0='{} reads per million\nG1/2-phase cells'.format(argv.dataset), title1='{} reads per million\nS-phase cells'.format(argv.dataset))



if __name__=='__main__':
    main()
