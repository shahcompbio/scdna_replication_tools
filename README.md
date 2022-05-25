# scdna_replication_tools
Tools for assigning S-phase cells to clones and inferring single-cell replication timing (scRT) profiles

## Installation

It is recommended that you clone this repository and install all prerequisites with pip in a virtual environment:

```
git clone git@github.com:shahcompbio/scdna_replication_tools.git
cd scdna_replication_tools
virtualenv venv
source venv/bin/activate
pip install numpy cython
pip install -r requirements.txt
python setup.py develop
```

Note that you will have to install numpy and cython prior to other requirements.
