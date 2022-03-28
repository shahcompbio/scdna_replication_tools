from argparse import ArgumentParser
import pandas as pd
import numpy as np
import statsmodels.api as sm


def get_args():
    p = ArgumentParser()

    p.add_argument('s_input', help='copynumber tsv file with state, copy, clone_id, etc for s-phase cells')
    p.add_argument('g1_input', help='copynumber tsv file with state, copy, clone_id, etc for g1-phase cells')
    p.add_argument('s_output', help='same as s_input but with gc norm columns added')
    p.add_argument('g1_output', help='same as g1_input but with gc norm columns added')
    p.add_argument('gc_curve', help='tsv file mapping gc to rpm values in the lowess curve of best fit')
    p.add_argument('input_col', help='reads column to use for GC correction (i.e. reads)')
    p.add_argument('output_col', help='column used to store GC-corrected reads (i.e. rpm_gc_norm)')

    return p.parse_args()


def compute_reads_per_million(cn, input_col='reads', rpm_col='rpm', cell_col='cell_id'):
    cn.loc[cn.index, rpm_col] = -1
    for cell_id, cell_cn in cn.groupby(cell_col):
        total_reads = sum(cell_cn[input_col].values)
        cn.loc[cell_cn.index, rpm_col] = (cell_cn[input_col] / total_reads) * 1E6
    return cn


def predict_gc(curve, gc, gc_col='gc', pred_rpm_col='pred_rpm'):
    df = curve.loc[curve[gc_col]==gc]
    return df[pred_rpm_col].values[0]


def bulk_g1_gc_correction(cn_s, cn_g1, input_col='reads', library_col='library_id', output_col='rpm_gc_norm', gc_col='gc', cell_col='cell_id'):
    '''
    Correcting the gc content of all cells by the lowess curve fit to all G1-phase cells

    Args:
        cn_s: long-form data frame of all S-phase cells profile with rows are segments & cells,
              columns are cell_id, chr, start, end, reads, copy, etc
        cn_g1: same as cn_g1 except for G1/2-phase cells
        input_col: reads column to use for GC correction (default: 'reads')
        output_col: column used to store GC-corrected reads (default: 'rpm_gc_norm')
    Returns:
        cn_s: same as cn_s input but with new column containing bulk-G1 GC corrected read counts
        cn_g1: same as cn_g1 input but with new column containing bulk-G1 GC corrected read counts
        gc_curve: dataframe containing mapping of gc values to predicted rpm values
    '''
    rpm_col = 'rpm'
    pred_rpm_col = 'pred_rpm'

    # use reads per million to normalize to total read count per cell
    cn_s = compute_reads_per_million(cn_s, input_col=input_col, rpm_col=rpm_col, cell_col=cell_col)
    cn_g1 = compute_reads_per_million(cn_g1, input_col=input_col, rpm_col=rpm_col, cell_col=cell_col)

    # set empty output columns
    cn_s[output_col] = -1
    cn_g1[output_col] = -1

    for lib_id, cn_s_chunk in cn_s.groupby(library_col):
        cn_g1_chunk = cn_g1.loc[cn_g1[library_col]==lib_id]

        # use lowess regression to predict rpm values of g1 phased cells from gc
        gc_vec = cn_s_chunk[gc_col].unique()
        lowess = sm.nonparametric.lowess
        z_g1 = lowess(cn_g1_chunk[rpm_col], cn_g1_chunk[gc_col], xvals=gc_vec)
        gc_curve = pd.DataFrame({gc_col: gc_vec, pred_rpm_col: z_g1})

        # create new column for rpm that has been gc normed
        # do this for both g1 and s phase cells
        cn_s.loc[cn_s[library_col]==lib_id, output_col] = cn_s_chunk.apply(lambda row: row[rpm_col] / predict_gc(gc_curve, row[gc_col], gc_col=gc_col, pred_rpm_col=pred_rpm_col), axis=1)
        cn_g1.loc[cn_g1[library_col]==lib_id, output_col] = cn_g1_chunk.apply(lambda row: row[rpm_col] / predict_gc(gc_curve, row[gc_col], gc_col=gc_col, pred_rpm_col=pred_rpm_col), axis=1)

    return cn_s, cn_g1


def main():
    argv = get_args()
    
    # load input data
    cn_s = pd.read_csv(argv.s_input, sep='\t')
    cn_g1 = pd.read_csv(argv.g1_input, sep='\t')

    # correct for GC content
    cn_s, cn_g1 = bulk_g1_gc_correction(cn_s, cn_g1, input_col=argv.input_col, output_col=argv.output_col)

    # save output
    cn_s.to_csv(argv.s_output, sep='\t', index=False)
    cn_g1.to_csv(argv.g1_output, sep='\t', index=False)


if __name__ == '__main__':
    main()
