# test_with_pytest.py
import pandas as pd
import numpy as np
from scdna_replication_tools.pert_simulator import pert_simulator
from scdna_replication_tools.infer_scRT import scRT


def test_always_passes():
    assert True

def test_pert_simulator():
    # load a dataframe containing gc and MCF7 RepliSeq values for each 500kb bin in hg19
    df = pd.read_csv('notebooks/mcfrt.csv', dtype={'chr':str})

    # subset df to just chromosome 1 to speed up the simulation
    df = df.loc[df['chr']=='1']

    # assert that there are 6 columns in df
    assert df.shape[1] == 6

    # simulate true somatic copy number states for 100 G1/2-phase cells
    # 30 of which have a CN=3 amplification on the first 100 bins of chromosome 1
    num_cells = 100
    num_cells_with_CNA = 30
    CNA_size = 100
    CNA_start = 0
    CNA_end = CNA_start + CNA_size
    CNA_state = 3.0

    cn_g = []

    # loop through all cells we wish to simulate
    for i in range(num_cells):
        temp_cn = df.copy()
        temp_cn['cell_id'] = 'cell_{}_g'.format(i)
        temp_cn['library_id'] = 'ABCD'
        # set all bins as CN=2 by default
        temp_cn['true_somatic_cn'] = 2.0
        if i < num_cells_with_CNA:
            # take the first 100 bins of chromosome 1 and set their copy number to 3
            temp_cn.iloc[CNA_start:CNA_end, temp_cn.columns.get_loc('true_somatic_cn')] = CNA_state
            temp_cn['clone_id'] = 'A'
        else:
            temp_cn['clone_id'] = 'B'
        cn_g.append(temp_cn)

    cn_g = pd.concat(cn_g, ignore_index=True)

    # assert that there are 100 unique cell_id values in cn_g
    assert len(cn_g['cell_id'].unique()) == num_cells

    # assert that there are only two clones in cn_g
    assert len(cn_g['clone_id'].unique()) == 2

    # copy these G1/2 true somatic copy number states to a new dataframe for S-phase cells
    cn_s = cn_g.copy()
    # change the cell_id column to reflect that these are S-phase cells by replacing the '_g' suffix with '_s'
    cn_s['cell_id'] = cn_s['cell_id'].str.replace('_g', '_s')

    num_reads = int(1e6 / 20) ## assume 1 million reads per cell, and we have are only simulating ~5% of the genome in chr1
    clones = ['A', 'B']  # clone A has the CNA, clone B does not
    rt_cols = ['mcf7rt', 'mcf7rt']  # both clones A and B follow the same MCF7 RepliSeq profile (i.e. no clone-specific RT bias)
    lamb = 0.75  # negative binomial dispersion parameter
    betas = [0.5, 0.0]  # gc bias terms: slope=0.5, intercept=0.0
    a = 10.0  # replication stochasticity parameter alpha

    cn_s, cn_g = pert_simulator(cn_s, cn_g, num_reads, rt_cols, clones, lamb, betas, a, gc_col='gc', input_cn_col='true_somatic_cn')

    # check that certain columns exist in cn_g and cn_s
    assert 'true_reads_norm' in cn_s.columns
    assert 'true_t' in cn_s.columns
    assert 'true_rep' in cn_s.columns
    assert 'true_reads_norm' in cn_g.columns
    assert 'true_t' in cn_g.columns
    assert 'true_rep' in cn_g.columns

    # check that true_rep in cn_g are all 0s
    assert np.all(cn_g['true_rep'] == 0)


def test_scrt_class():
    cn_s = pd.read_csv('data/D1.0/s_phase_cells_hmmcopy_trimmed.csv.gz', dtype={'chr': str})
    cn_g1 = pd.read_csv('data/D1.0/g1_phase_cells_hmmcopy_trimmed.csv.gz', dtype={'chr': str})

    # add the replication columns for the G1-phase cells
    cn_g1['true_rep'] = 0.0
    cn_g1['true_p_rep'] = 0.0
    cn_g1['true_t'] = 1.0

    # temporarily remove columns that don't get used by PERT
    temp_cn_s = cn_s[['cell_id', 'chr', 'start', 'end', 'gc', 'state', 'library_id', 'true_reads_norm']]
    temp_cn_g1 = cn_g1[['cell_id', 'chr', 'start', 'end', 'gc', 'clone_id', 'state', 'library_id', 'true_reads_norm']]

    # asser that there are 400 cells and 271 loci in both cell cycle phases
    assert len(temp_cn_s['cell_id'].unique()) == 400
    assert len(temp_cn_g1['cell_id'].unique()) == 400
    assert len(temp_cn_s[['chr', 'start']].drop_duplicates()) == 271
    assert len(temp_cn_g1[['chr', 'start']].drop_duplicates()) == 271

    # create scRT object with input columns denoted
    scrt = scRT(temp_cn_s, temp_cn_g1, input_col='true_reads_norm', clone_col='clone_id', assign_col='state', rt_prior_col=None,
                cn_state_col='state', gc_col='gc', cn_prior_method='g1_clones', max_iter=3)
    
