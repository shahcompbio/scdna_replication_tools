from setuptools import setup, find_packages

with open('requirements_setup.txt') as f:
    requirements = f.read().splitlines()

with open('dependency_links.txt') as f:
    dependency_links = f.read().splitlines()
  
# remove elements from requirements if the first character is a '#'
requirements = [x for x in requirements if x[0] != '#']
dependency_links = [x for x in dependency_links if x[0] != '#']

setup(name='scdna_replication_tools',
      version='0.0.1',
      description='Code for analyzing single-cell replication dynamics',
      author='Adam Weiner (Shah Lab)',
      url='https://github.com/shahcompbio/scdna_replication_tools',
      packages=find_packages(),
      install_requires=requirements,
      dependency_links=dependency_links,
      entry_points={
        'console_scripts': [
            'infer_SPF = scdna_replication_tools.infer_SPF:main',
            'infer_scRT = scdna_replication_tools.infer_scRT:main'
        ]
      }
    )
