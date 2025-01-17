from setuptools import setup, find_packages

setup(name='scdna_replication_tools',
      version='0.0.1',
      description='Code for analyzing single-cell replication dynamics',
      author='Adam Weiner (Shah Lab)',
      url='https://github.com/shahcompbio/scdna_replication_tools',
      packages=find_packages(),
      entry_points={
        'console_scripts': [
            'infer_SPF = scdna_replication_tools.infer_SPF:main',
            'infer_scRT = scdna_replication_tools.infer_scRT:main'
        ]
      }
    )
