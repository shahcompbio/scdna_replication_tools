# scdna_replication_tools (PERT)
Method for probabilistic estimation of replication timing (PERT) from single-cell whole genome sequencing data.

## Installation

It is recommended that you clone this repository and install all prerequisites with pip in a virtual environment:

```
git clone git@github.com:shahcompbio/scdna_replication_tools.git
cd scdna_replication_tools
conda create -n scdna_replication_tools python==3.7.4
conda activate scdna_replication_tools
python -m venv venv/
source venv/bin/activate
pip install numpy==1.21.4 cython==0.29.22
pip install -r requirements3.txt
python setup.py develop
```

Note that you will have to activate both the conda and venv environments in order to run this code.

## Usage

See the tutorials in the [notebooks directory](https://github.com/shahcompbio/scdna_replication_tools/tree/main/notebooks) for examples of how to use PERT.

`inference_tutorial.ipynb`: estimate per-bin copy number and replication states from a fully diploid sample with known cell cycle phases.

`inference_tutorial_pt2.ipynb`: estimate per-bin copy number and replication states from a polyclonal sample with unknown cell cyce phases.

`simulator_tutorial.ipynb` contains a tutorial for simulating data using PERT as a generative model.

## Input data

`PERT` was developed to work with single-cell WGS data downstream of copy number calling by HMMcopy. The input data should be a pandas dataframe with the following columns:
`chr`, `start`, `end`, `gc`, `library_id`, `cell_id`, `reads`, `state`.
The `reads` column should contain the number of reads mapped to each bin -- preferrably normalized such that all cells have the same total read count such as reads per million (int). The `state` column should contain the integer copy number state called by HMMcopy. The `cell_id` column should contain the cell ID for each bin (string). The `library_id` column should contain the library ID for each cell (string). The `gc` column should contain the GC content of each locus according to the reference genome (float from 0-1). The `chr` column should contain the chromosome of each bin (`'1', '2', ..., 'X'`). The `start` and `end` columns should contain the start and end positions of each bin (int). The bin size should be the same for all loci and the same loci (`chr`, `start`, `end`) should be present in all cells. 

We recommend 500kb bin size for DLP+ data (descried in [Laks et al](https://doi.org/10.1016/j.cell.2019.10.026)) given its coverage depth of 0.01-0.1x per cell; however, this can be adjusted depending on the coverage depth of your data. We have successfully run PERT on samples with bin sizes as small as 20kb but have found that too many bins with 0 reads can produce NaN errors during fitting when coverage is insufficient. Additionally, you must account for additional runtime when using smaller bin sizes as there will be more bins to fit.

When using copy number callers other than HMMcopy (such as the [10x CellRanger-DNA pipeline](https://support.10xgenomics.com/single-cell-dna/software/pipelines/latest/what-is-cell-ranger-dna)), you may need to convert some column names or use optional function arguments to avoid naming convention errors. Additionally, you may need to convert the copy number states to the same domain as HMMcopy (0-11) instead of allowing for many different >11 states. This is necessary as PERT samples somatic copy number from categorical distribution which requires enumation over all possible states for each bin.


## Output data

The main output when running PERT for scRT inference is a pandas dataframe with the following columns in addition to the input columns:

`model_rep_state`: the estimated replication state for each bin. This is a binary variable between 0 and 1, with 0 indicating the bin is unreplicated and 1 indicating the bin replicated.
`model_cn_state`: the estimated somatic copy number for each bin. These will be integer values ranging from 0-11 (same domain as input `state`). 

While there are other columns in the output dataframe, these are the most important for downstream analysis. These columns are used for downstream computation of pseudobulk replication timing profiles, each cell's fraction of replicated bins, cell cycle phase predictions, and a sample's time from scheduled replication (T-width). Other output columns from pert_model.py correspond to the name of different latent variables in the graphical model (see paper for details).


## Feedback

Please report any bugs or issues to the [issue tracker](https://github.com/shahcompbio/scdna_replication_tools/issues).


## Citation

If you use PERT in your work, please cite the following paper: [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.04.10.536250v1)
