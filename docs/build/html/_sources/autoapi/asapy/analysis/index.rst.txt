:py:mod:`asapy.analysis`
========================

.. py:module:: asapy.analysis


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.analysis.Analysis




.. py:class:: Analysis

   The Analysis object.

   .. py:method:: hypothesis(df: pandas.DataFrame, alpha: float = 0.05, verbose: bool = False) -> pandas.DataFrame
      :staticmethod:

      Method that performs hypothesis testing

      :param df: (Pandas DataFrame)
                 Input data (must contain at least two distributions).
      :param alpha: (float)
                    Significance level. Represents a cutoff value, a criterion that we set to reject or not H0. Default 0.05.
      :param verbose: (bool, optional)
                      Variable that defines whether or not to display detailed messages. Defaults to False.

      :raises ValueError: Input variable is empty.
      :raises ValueError: Input data must match at least two distributions.

      :returns: Indicates which distributions are statistically similar.
      :rtype: (Pandas DataFrame)

      The figure below shows the flow of the hypothesis method:

      .. image:: /../../../../../image/Diagrama_hypothesis.png


      .. seealso::

          `pingouin.homoscedasticity <https://pingouin-stats.org/build/html/generated/pingouin.homoscedasticity.html#pingouin.homoscedasticity>`_: teste de igualdade de variância.

          `pingouin.normality <https://pingouin-stats.org/build/html/generated/pingouin.normality.html#pingouin.normality>`_: teste de normalidade.

          `scipy.stats.f_oneway <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html>`_: one-way ANOVA.

          `scipy.stats.tukey_hsd <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html>`_: teste HSD de Tukey para igualdade de médias.

          `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_: teste H de Kruskal-Wallis para amostras independentes.

          `scikit_posthocs.posthoc_conover <https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html>`_: teste de Conover.

      Example usage:

      .. code-block::

          >>> import pandas as pd
          >>> import asapy
          >>> import numpy as np
          >>> # Set random seed for reproducibility
          >>> np.random.seed(123)
          >>> # Create DataFrame with 5 columns and 100 rows
          >>> data = pd.DataFrame({
          >>>     'col0': np.random.gamma(1, size=100),
          >>>     'col1': np.random.uniform(size=100),
          >>>     'col2': np.random.exponential(size=100),
          >>>     'col3': np.random.logistic(size=100),
          >>>     'col4': np.random.pareto(1, size=100) + 1})
          >>> output = asapy.Analysis.hypothesis(data, verbose = True)

          Teste de normalidade
                      W         pval  normal
          ----  --------  -----------  --------
          col1   74.7177  5.96007e-17  False
          col2   31.6041  1.3717e-07   False
          col3   40.6985  1.45356e-09  False
          col4   10.2107  0.00606431   False
          col5  212.599   6.8361e-47   False
          Conclusão: Ao menos uma distribuição não se assemelha à gaussiana (normal).

          Teste de homocedasticidade
                      W       pval  equal_var
          ------  -------  ---------  -----------
          levene  2.03155  0.0888169  True
          Conclusão: Distribuições possuem variâncias estatisticamente SEMELHANTES (homoscedasticidade).

          Teste de Kruskal
          statistic = 182.22539784431183, pvalue = 2.480716493859747e-38
          Conclusão: Estatisticamente as amostras correspondem a distribuições DIFERENTES (Kruskal-Wallis).

          Teste de Conover
                      1             2             3             4             5
          1  1.000000e+00  3.280180e-04  8.963739e-01  1.632161e-08  6.805120e-21
          2  3.280180e-04  1.000000e+00  5.316246e-04  3.410392e-02  2.724152e-35
          3  8.963739e-01  5.316246e-04  1.000000e+00  3.335991e-08  2.296912e-21
          4  1.632161e-08  3.410392e-02  3.335991e-08  1.000000e+00  1.024363e-44
          5  6.805120e-21  2.724152e-35  2.296912e-21  1.024363e-44  1.000000e+00

              dist1    dist2  same?
          --  -------  -------  -------
          0        0        1  False
          1        0        2  False
          2        0        3  False
          3        0        4  False
          4        1        2  False
          5        1        3  True
          6        1        4  False
          7        2        3  False
          8        2        4  False
          9        3        4  False



   .. py:method:: fit_distribution(df: pandas.DataFrame, verbose: bool = False) -> pandas.DataFrame
      :staticmethod:

      Find the distribution that best fits the input data.

      :param df: Input data (must contain only one distribution).
      :type df: Pandas DataFrame
      :param verbose: Flag that controls whether detailed messages are displayed. Defaults to False.
      :type verbose: bool, optional

      :raises ValueError: Input data must contain only one distribution.

      :returns: DataFrame containing information about the distribution that best fit the input data, as well as the most common distributions (``norm``, ``beta``, ``chi2``, ``uniform``, ``expon``). The columns of the DataFrame are: ``Distribution_Type``, ``P_Value``, ``Statistics``, and ``Parameters``.
      :rtype: (Pandas DataFrame)

      .. seealso::

          `scipy.stats.kstest <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html>`_: teste de Kolmogorov-Smirnov (uma ou duas amostras) para verificar a qualidade do ajuste.

      Example usage:

      .. code-block::

          >>> import pandas as pd
          >>> from sklearn.datasets import load_wine
          >>> X, y  = load_wine(as_frame=True, return_X_y=True)
          >>> result = asapy.Analysis.fit_distribution(X[['magnesium']], verbose = True)
          Distribution_Type      P_Value    Statistics  Parameters
          -------------------  ---------  ------------  -------------------------------------
          weibull_min           0.666605     0.0535577  (1.65, 77.23, 25.3)
          beta                  0.585262     0.0571824  (6.06, 5334914.75, 65.16, 30436461.8)
          norm                  0.110071     0.0892933  (99.74, 14.24)
          expon                 0            0.317447   (70.0, 29.74)
          uniform               0            0.386541   (70.0, 92.0)
          chi2                  0            0.915856   (0.64, 70.0, 3.93)

      .. image:: /../../../../../image/output_fit_distribution.png


   .. py:method:: feature_score(df: pandas.DataFrame, x: List[str], y: List[str], scoring_function: str, verbose: bool = False) -> pandas.DataFrame
      :staticmethod:

      Calculate the score of input data.

      :param df: DataFrame with input data.
      :type df: Pandas DataFrame
      :param x: Names of input variables (same name as the corresponding column of ``df``).
      :type x: List[str]
      :param y: Names of output variables (same name as the corresponding column of ``df``).
      :type y: List[str]
      :param scoring_function: Name of the scoring function.
      :type scoring_function: str
      :param verbose: Flag to display detailed messages. Defaults to False.
      :type verbose: bool, optional

      :raises ValueError: Invalid scoring_function name.

      :returns: DataFrame with scores of input variables.
      :rtype: (Pandas DataFrame)

      .. warning::

          Beware not to use a regression scoring function with a classification problem, you will get useless results

          For regression: ``r_regression``, ``f_regression``, ``mutual_info_regression``.

          For classification: ``chi2``, ``f_classif``, ``mutual_info_classif``.

      .. seealso::

          `sklearn.feature_selection.SelectKBest <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>`_: seleciona as features de acordo com os k scores mais altos.

          `sklearn.feature_selection.SelectPercentile <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile>`_: seleciona as features de acordo com um percentil dos scores mais altos.

      Example usage:

      .. code-block::

          >>> import asapy
          >>> from sklearn.datasets import load_diabetes
          >>> # load dataset
          >>> X, y  = load_diabetes(as_frame=True, return_X_y=True)
          >>> # getting the input variable names
          >>> feature_list = X.columns.tolist()
          >>> # adding the output (target variable) in the data frame
          >>> X['target'] = y
          >>> scores = asapy.Analysis.feature_score(X,feature_list, ['target'], 'f_regression', verbose = True)

              bmi      s5      bp      s4     s3    s6     s1    age     s2    sex
          --  ------  ------  ------  ------  -----  ----  -----  -----  -----  -----
          0  230.65  207.27  106.52  100.07  81.24  75.4  20.71   16.1  13.75   0.82


   .. py:method:: pareto(df: pandas.DataFrame, min_list: List[str], max_list: List[str]) -> pandas.DataFrame
      :staticmethod:

      Returns a subset of the input DataFrame consisting of Pareto optimal points based on the specified columns.

      :param df: The input DataFrame.
      :type df: pd.DataFrame
      :param min_list: A list of column names that should be minimized in the Pareto optimality calculation.
      :type min_list: List[str]
      :param max_list: A list of column names that should be maximized in the Pareto optimality calculation.
      :type max_list: List[str]

      :returns: A DataFrame that contains the Pareto optimal points of the input DataFrame based on the specified columns.
      :rtype: pd.DataFrame

      Example usage:

      .. code-block::

          >>> import pandas as pd
          >>> from sklearn.datasets import load_wine
          >>> X, y  = load_wine(as_frame=True, return_X_y=True)
          >>> p = pareto(X, ['alcohol'], ['malic_acid','ash'])
          >>> print(p.index.tolist())
          [77, 88, 110, 112, 113, 115, 120, 121, 122, 123, 124, 136, 137, 169, 173]

      .. note::

          This function drops any row that contains missing values before performing the Pareto optimality calculation.

          The columns specified in the `min_list` parameter will be multiplied by -1 to convert them into maximization criteria.


   .. py:method:: get_best_pareto_point(df: pandas.DataFrame, list_variable: List[str], weights: List[float], verbose: bool = False) -> pandas.DataFrame
      :staticmethod:

      Calculate the best Pareto optimal point in the input DataFrame based on the specified variables and weights.

      :param df: The input DataFrame.
      :type df: pd.DataFrame
      :param list_variable: A list of column names that should be considered in the Pareto optimality calculation.
      :type list_variable: List[str]
      :param weights: A list of weights that determine the relative importance of each variable.
      :type weights: List[float]
      :param verbose: A flag that determines whether to print the best Pareto optimal point or not.
      :type verbose: bool

      :returns: A DataFrame that contains the best Pareto optimal point based on the specified variables and weights.
      :rtype: pd.DataFrame

      Example usage:

      .. code-block::

          >>> import asapy
          >>> from sklearn.datasets import load_wine
          >>> X, y  = load_wine(as_frame=True, return_X_y=True)
          >>> p = asapy.Analysis.pareto(X, ['alcohol'], ['malic_acid','ash'])
          >>> best = asapy.Analysis.get_best_pareto_point(p,['alcohol', 'malic_acid', 'ash'],[0.0,0.9,0.1], True)
          Melhor opção de acordo com a decomposição: Ponto 115 - [11.03  1.51  2.2 ]

      .. note::

          This function assumes that the input DataFrame contains only Pareto optimal points.

          The weights parameter should contain a value for each variable specified in the list_variable parameter.


   .. py:method:: detect_outliers(df: pandas.DataFrame, method: str = 'IQR', thr: float = 3, verbose: bool = False) -> Tuple[pandas.DataFrame]
      :staticmethod:

      Detect outliers in a Pandas DataFrame using IQR or zscore method.

      :param df: Input DataFrame containing numerical data.
      :type df: Pandas DataFrame
      :param method: Method to use for outlier detection. Available options: 'IQR' or 'zscore'. Defaults to 'IQR'.
      :type method: str, optional
      :param thr: Threshold value for zscore method. Defaults to 3.
      :type thr: int, optional
      :param verbose: Determines whether to display detailed messages. Defaults to False.
      :type verbose: bool, optional

      :raises ValueError: If method is not equal to one of the following options: 'IQR' or 'zscore'.

      :returns: tuple containing

                - (Pandas DataFrame): DataFrame containing the index of the outliers.
                - (Pandas DataFrame): The columns of the DataFrame are: ``column``, ``min_thres``, ``max_thres``. Values smaller than ``min_thres`` and larger than ``max_thres`` are considered outliers for IQR method.

      Example usage:

      .. code-block::

          >>> import asapy
          >>> from sklearn.datasets import load_diabetes
          >>> # load dataset
          >>> X, y  = load_diabetes(as_frame=True, return_X_y=True)
          >>> df, df_thres = asapy.Analysis().detect_outliers(X, verbose = True)
                outliers_index
          --  ----------------
          0                23
          1                35
          2                58
          ...
          28               406
          29               428
          30               441



   .. py:method:: remove_outliers(df: pandas.DataFrame, verbose: bool = False) -> Tuple[pandas.DataFrame, List[int]]

      Remove outliers from a Pandas DataFrame using the Interquartile Range (IQR) method.

      :param df: DataFrame containing the data.
      :type df: Pandas DataFrame
      :param verbose: If True, print the number of lines removed. Defaults to False.
      :type verbose: bool, optional

      :returns: tuple containing

                - df_new (Pandas DataFrame): DataFrame with the outliers removed.
                - drop_lines (list): List of indexes of the rows that were removed.

      Example usage:

      .. code-block::

          >>> import asapy
          >>> from sklearn.datasets import load_diabetes
          >>> # load dataset
          >>> X, y  = load_diabetes(as_frame=True, return_X_y=True)
          >>> df_new, drop_lines = asapy.Analysis().remove_outliers(X, verbose = True)
          Foram removidas 31 linhas.


   .. py:method:: cramer_V(df: pandas.DataFrame, verbose: bool = False, save: bool = False, path: str = None, format: str = 'png') -> pandas.DataFrame
      :staticmethod:

      Calculate Cramer's V statistic for categorical feature association in a DataFrame.

      Cramer's V is a measure of association between two categorical variables. It is based on the ``chi-squared`` statistic
      and considers both the strength and direction of association. This function calculates Cramer's V for all pairs of
      categorical variables in a given DataFrame and returns the results in a new DataFrame.

      :param df: The input DataFrame containing the categorical variables.
      :type df: pandas DataFrame
      :param verbose: If True, a heatmap of the Cramer's V values will be displayed using Seaborn. Default is False.
      :type verbose: bool, optional

      :returns: A DataFrame containing Cramer's V values for all pairs of categorical variables.
      :rtype: (pandas DataFrame)

      Example usage:

      .. code-block::

          >>> import pandas as pd
          >>> import asapy
          >>> # Create a sample DataFrame
          >>> df = pd.DataFrame({'A': ['cat', 'dog', 'bird', 'cat', 'dog'],
          ...                    'B': ['small', 'large', 'medium', 'medium', 'small'],
          ...                    'C': ['red', 'blue', 'green', 'red', 'blue']})
          >>> # Calculate Cramer's V
          >>> cramer_df = asapy.Analysis.cramer_V(df, verbose=True)

      .. image:: /../../../../../image/output_cramer_v.png


   .. py:method:: EDA(df: pandas.DataFrame, save: bool = False, path: str = None, format: str = 'png') -> None

      Perform exploratory data analysis (EDA) on a given pandas DataFrame.

      The function displays a summary table of the DataFrame, a table of class balance for categorical variables,
      and histograms and boxplots with information on the number of outliers for numerical variables.

      :param df: Input DataFrame to be analyzed.
      :type df: pandas.DataFrame
      :param save: If True, save the plots. Defaults to False.
      :type save: bool, optional
      :param path: Path to save the plots. Defaults to None.
      :type path: str, optional
      :param format: Format for the plot files. Defaults to 'png'.
      :type format: str, optional

      :returns: None

      Example Usage:

      .. code::

          >>> import asapy
          >>> import pandas as pd
          >>> df = pd.read_csv('path-to-dataset.csv')
          >>> asapy.Analysis().EDA(df)

          Variáveis Categóricas:

                  occupation      education      educational-num
          ------  --------------  -----------  -----------------
          nan     0               0                            0
          count   48842           48842                    48842
          unique  15              16                          16
          top     Prof-specialty  HS-grad                      9
          freq    6172            15784                    15784


          Associação:

      .. image:: /../../../../../image/association.png

      .. code::

          Histogramas:

      .. image:: /../../../../../image/occupation.png

      .. image:: /../../../../../image/education.png

      .. image:: /../../../../../image/educational-num.png

      .. code::

          Variáveis Numéricas:
                      age           fnlwgt
          -----  ----------  ---------------
          nan        0            0
          count  48842        48842
          mean      38.6436  189664
          std       13.7105  105604
          min       17        12285
          25%       28       117550
          50%       37       178144
          75%       48       237642
          max       90            1.4904e+06

          Correlação:

      .. image:: /../../../../../image/correlation.png

      .. code::

          Histogramas e boxplots:

      .. image:: /../../../../../image/age.png

      .. code::

          Detecção de outlier da variável 'age':
          Quantidade: 216 de 48842.
          Método: Intervalo Interquartil (IQR - Interquatile Range).
          Critério: Os valores menores que -2.0 ou maiores que 78.0 foram considerados outliers.

      .. image:: /../../../../../image/fnlwgt.png

      .. code::

          Detecção de outlier da variável 'fnlwgt':
          Quantidade: 1453 de 48842.
          Método: Intervalo Interquartil (IQR - Interquatile Range).
          Critério: Os valores menores que -62586.75 ou maiores que 417779.25 foram considerados outliers.




