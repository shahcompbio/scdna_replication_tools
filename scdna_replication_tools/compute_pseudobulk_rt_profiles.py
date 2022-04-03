from argparse import ArgumentParser
import numpy as np
import pandas as pd


def get_args():
    p = ArgumentParser()

    p.add_argument('input', help='cn tsv file with clone_id, state, copy, reads_norm_4, etc')
    p.add_argument('column', help='column to use when aggregating pseudobulk')
    p.add_argument('output', help='tsv file containing pseudobulk and gc profiles')

    return p.parse_args()


def calc_population_rt(cn, input_col, output_col, time_col='rt_hours', chr_col='chr', start_col='start'):
    pop_cn = []
    for (chrom, start), loci_cn in cn.groupby([chr_col, start_col]):
        averagee = np.mean(loci_cn[input_col].values)

        temp_df = pd.DataFrame({
            chr_col: [chrom], start_col: [start], output_col: [averagee]
        })
        pop_cn.append(temp_df)

    pop_cn = pd.concat(pop_cn, ignore_index=True)

    # compute time each loci replicates within S-phase by hours
    # push to all negative values and multiply by -1 so
    # latest loci (lowest avg_norm_reads_2) have the largest values
    a = pop_cn[output_col].values - max(pop_cn[output_col].values)
    a *= -1
    # normalize `a` from 0 to 10
    a = a / max(a)
    a *= 10
    pop_cn[time_col] = a

    return pop_cn


def compute_pseudobulk_rt_profiles(cn, input_col, output_col='pseduobulk', time_col='hours', clone_col='clone_id', chr_col='chr', start_col='start'):
    '''
    Compute population- and clone-level pseduobulk replication timing profiles

    Args:
        cn: dataframe where rows represent unique segments from all cells, columns contain genomic loci and input_col
        input_col: replication timing column in cn to use when computing pseudobulks
        output_col: prefix for all the pseudobulk column names
        time_col: suffix to use to represent replication time value in units of 0-10 hours
        clone_col: column in cn to use for computing clone-specific pseudobulks, can be set to None to ignore clones
    Returns:
        dataframe with rows as genomic loci, columns as all the desired pseudobulk profiles
    '''
    temp_output_col = str(output_col) + '_' + str(input_col)
    temp_time_col = str(output_col) + '_' + str(time_col)
    bulk_cn = calc_population_rt(cn, input_col=input_col, output_col=temp_output_col, time_col=temp_time_col, chr_col='chr', start_col='start')

    if clone_col is not None:
        for clone_id, clone_cn in cn.groupby('clone_id'):
            # compute pseudobulk for this clone
            temp_output_col = '{}_clone{}_{}'.format(output_col, clone_id, input_col)
            temp_time_col = '{}_clone{}_{}'.format(output_col, clone_id, time_col)
            temp_cn = calc_population_rt(clone_cn, input_col=input_col, output_col=temp_output_col, time_col=temp_time_col, chr_col='chr', start_col='start')
            
            # merge clone pseudobulk results with previous results
            temp_cn = temp_cn[['chr', 'start', temp_output_col, temp_time_col]]
            bulk_cn = pd.merge(bulk_cn, temp_cn)

    return bulk_cn


def main():
    argv = get_args()

    cn = pd.read_csv(argv.input, sep='\t')

    bulk_cn = compute_pseudobulk_rt_profiles(cn, argv.column, output_col='pseduobulk', time_col='hours', clone_col='clone_id')

    bulk_cn.to_csv(argv.output, sep='\t', index=False)


if __name__ == '__main__':
    main()
