from argparse import ArgumentParser
import numpy as np
import pandas as pd


def get_args():
	p = ArgumentParser()

	p.add_argument('input', help='cn tsv file with clone_id, state, copy, reads_norm_4, etc')
	p.add_argument('column', help='column to use when aggregating pseudobulk')
	p.add_argument('output', help='tsv file containing pseudobulk and gc profiles')

	return p.parse_args()


def calc_population_rt(cn, input_col, output_col, time_col='rt_hours'):
    pop_cn = []
    for (chrom, start, end), loci_cn in cn.groupby(['chr', 'start', 'end']):
        width = loci_cn.width.values[0]
        gc = loci_cn.gc.values[0]
        mapp = loci_cn.map.values[0]
        averagee = np.mean(loci_cn[input_col].values)

        temp_df = pd.DataFrame({'chr': [chrom], 'start': [start], 'end': [end],
                              'width': [width], output_col: [averagee],
                                'gc': [gc], 'map': [mapp]})
        pop_cn.append(temp_df)

    pop_cn = pd.concat(pop_cn)
    pop_cn = pop_cn.reset_index().drop(columns=['index'])
    
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


def main():
	argv = get_args()

	cn = pd.read_csv(argv.input, sep='\t')

	output_col = 'pseudobulk_' + str(argv.column)

	bulk_cn = calc_population_rt(cn, input_col=argv.column, output_col=output_col, time_col='pseudobulk_hours')
	print(bulk_cn.head())

	for clone_id, clone_cn in cn.groupby('clone_id'):
		# compute pseudobulk for this clone
		output_col = 'clone{}_{}'.format(clone_id, argv.column)
		time_col = 'clone{}_hours'.format(clone_id)
		temp_cn = calc_population_rt(clone_cn, input_col=argv.column, output_col=output_col, time_col=time_col)
		
		# merge clone pseudobulk results with previous results
		temp_cn = temp_cn[['chr', 'start', 'end', 'width', output_col, time_col]]
		print(temp_cn.head())
		bulk_cn = pd.merge(bulk_cn, temp_cn)
		print(bulk_cn.shape)

	bulk_cn.to_csv(argv.output, sep='\t', index=False)


if __name__ == '__main__':
	main()
