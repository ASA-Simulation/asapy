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
      version='0.1.0',
      description='ASA data analysis library',
      author='ASA',
      license='MIT',
      setup_requires=['pytest-runner'],
      install_requires=[
            'requests==2.28.1',
            'python-dateutil==2.8.2',
            'pandas==1.5.3',
            'pyDOE==0.3.8',
            'tqdm==4.65.0',
            'pingouin==0.5.3',
            'scikit_posthocs==0.7.0',
            'pymoo==0.6.0.1',
            'protobuf==3.20.3',
            'tensorflow',
            'numpy==1.22.*',
            'scipy',
            'tabulate',
            'scikit_posthocs',
            'regex',
            'matplotlib',
            'scikit-learn',
            'keras_tuner',
            'seaborn',
            'joblib',
            'pingouin',
            'Sphinx>=2.2',
            'sphinx_rtd_theme',
            'sphinx-copybutton',
            'sphinx-autoapi',
            'sphinx_copybutton',          
      ],
      extras_require=extras_require,
      tests_require=['pytest==4.4.1'],
      test_suite='tests',
      )