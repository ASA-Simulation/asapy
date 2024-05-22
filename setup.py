from setuptools import find_packages, setup


docs_require = [
    'Sphinx>=2.2',
    'sphinx_rtd_theme',
    'sphinx-copybutton',
    'sphinx-autoapi',
    'sphinx_copybutton',
    
]

extras_require = {
    'docs': docs_require,
}


extras_require['all'] = [req for exts, reqs in extras_require.items()
                         for req in reqs]

setup(name='asapy',
      packages=find_packages(include=['asapy']),
      version='0.1.6',
      description='ASA data analysis library',
      author='ASA',
      license='MIT',
      setup_requires=['pytest-runner'],
      install_requires=[
            'asa-client @ git+https://gitlab.asa.dcta.mil.br/asa/asa-client.git@main#egg=asa-client', 
            'requests==2.28.1',
            'python-dateutil==2.8.2',
            'pyDOE==0.3.8',
            'tqdm==4.65.0',
            'scikit_posthocs==0.7.0',
            'pymoo==0.6.0.1',
            'protobuf==3.20.3',
            'tensorflow',
            'numpy==1.26.4',
            'pandas==1.5.3',
            'scipy==1.13.0',
            'tabulate==0.8.10',
            'regex==2022.7.9',
            'matplotlib==3.8.4',
            'scikit-learn==1.4.2',
            'seaborn==0.12.2',
            'joblib==1.4.2',
            'pingouin==0.5.4',
            'Sphinx>=2.2,<7',
            'sphinx_rtd_theme',
            'sphinx-copybutton',
            'sphinx-autoapi',
            'sphinx_copybutton',
            'bioinfokit==2.1.1',  
            'optuna==3.5.0',
            'gower==0.1.2',
            'xgboost==2.0.3'
      ],
      extras_require=extras_require,
      tests_require=['pytest==4.4.1'],
      test_suite='tests',
      )
