# dna-replication-tools
Tools for assigning S-phase cells to clones and inferring single-cell replication timing (scRT) profiles

## Installation

It is recommended that you install all prerequisites with pip in a virtual environment:

```
virtualenv venv
source venv/bin/activate
pip install numpy cython
pip install -r requirements.txt
python setup.py develop
```

Note that you will have to install numpy and cython prior to other requirements.