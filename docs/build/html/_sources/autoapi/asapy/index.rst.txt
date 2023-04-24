:py:mod:`asapy`
===============

.. py:module:: asapy


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   analysis/index.rst
   doe/index.rst
   execution_controller/index.rst
   models/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.Doe
   asapy.Analysis
   asapy.NN
   asapy.RandomForest
   asapy.Scaler
   asapy.AsaML




.. py:class:: Doe

   .. py:method:: _get_aliases(sim)
      :staticmethod:

      Returns a generator that yields dictionaries containing information about aliases found within the input `sim` parameter.

      :param sim: A dictionary representing the simulation object.
      :type sim: dict

      :Yields: *dict* -- A dictionary containing information about an alias, including its `id`, `alias_attribute`, and `alias`.

      Example Usage:

      .. code::

          >>> #To extract aliases from a simulation object `sim`, you can use the following code:
          >>> for alias_info in Doe._get_aliases(sim):
          ...     print(alias_info)

      .. note::

          - This method checks if the input `sim` has an `alias` key with a non-empty value. If found, it iterates through the key-value pairs of the value dictionary and yields a dictionary containing information about the alias.

          - If the value of the key-value pair is a dictionary, the method recursively calls itself with the dictionary as the new input `sim` parameter.

          - If the value of the key-value pair is a list, it iterates through the list and recursively calls itself with each dictionary in the list as the new input `sim` parameter.


   .. py:method:: _get_configs(configs, alias)
      :staticmethod:

      Recursively searches for the attribute of the given alias in a list of configurations.

      Args:
      - configs (list of dicts): List of configuration dictionaries
      - alias (dict): Dictionary containing the alias attribute to be searched and the id to match

      Returns:
      - Generator object that yields the value of the alias attribute whenever it is found


   .. py:method:: process_aliases_by_sim(sim, configs) -> pandas.DataFrame

      Process aliases based on similarity and configurations.

      Parameters:
      sim (list): A list of dictionaries, containing 'id' and 'text' keys.
      configs (list): A list of dictionaries, containing 'identifier', 'attributes', and 'alias_attribute' keys.

      Returns:
      DataFrame: A DataFrame containing aliases and their corresponding attributes.


   .. py:method:: create(df_T, samples)
      :staticmethod:

      Creates a design of experiments (DOE) based on the input DataFrame ``df_T``.
      The DOE is created using a Latin Hypercube Sampling (LHS) method and a sample size ``samples``.
      The function returns a new DataFrame with the input variables' names and values according to their type.

      :param df_T: A DataFrame of the variables' metadata. The index should contain the variables' names, and the following columns
      :param should exist:
                           - "type": Variable's type: integer, double, boolean, select, multi_select, or string.
                           - "default": Default value for the variable.
                           - "min": Minimum value for the variable (if applicable).
                           - "max": Maximum value for the variable (if applicable).
                           - "options": Available options for the variable (if applicable).
      :param samples: An integer value indicating the number of samples in the design.

      :returns: A new DataFrame containing the input variables' names and values, generated using LHS sampling.

      :raises TypeError: If ``df_T`` is not a pandas DataFrame or ``samples`` is not an integer.
      :raises ValueError: If ``df_T`` does not contain the required columns or the ``default`` value is not within the ``min`` and ``max`` range.


   .. py:method:: prepare_experiments(df) -> List[dict]
      :staticmethod:

      Prepare a list of experiments from a Pandas DataFrame.

      :param df: A Pandas DataFrame containing experiment data.

      :returns: A list of dictionaries, where each dictionary represents an experiment and its attributes.

      :raises None.:


   .. py:method:: _get_metrics(sim)
      :staticmethod:

      Recursively extract monitor metrics from a simulation dictionary.

      :param sim: A simulation dictionary.
      :type sim: dict

      :Yields: *dict* -- A dictionary containing the workspace name, simulation identifier, and monitor metrics.


   .. py:method:: process_metrics(sim)

      Process the metrics obtained from a simulation.

      :param sim: A dictionary containing the simulation data.

      :returns: A pandas DataFrame containing the metrics indexed by workspace name.

      :raises None:



.. py:class:: Analysis

   The Analysis object.

   .. py:method:: hypothesis(df, alpha=0.05, verbose=False)
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



   .. py:method:: fit_distribution(df, verbose=False)
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


   .. py:method:: feature_score(df, x, y, scoring_function, verbose=False)
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


   .. py:method:: detect_outliers(df, method='IQR', thr=3, verbose=False)
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



   .. py:method:: remove_outliers(df, verbose=False)

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


   .. py:method:: cramer_V(df, verbose=False, save=False, path=None, format='png')
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


   .. py:method:: EDA(df, save=False, path=None, format='png')

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




.. py:class:: NN(model=None)

   Bases: :py:obj:`Model`

   The class NN is a wrapper around Keras Sequential API, which provides an easy way to create and train neural network models. It can perform hyperparameters search and model building with specified hyperparameters.

   Attributes:
   -----------
       - model: the built Keras model.
       - loss: the loss function used to compile the Keras model.
       - metrics: the metrics used to compile the Keras model.
       - dir_name: a string that defines the name of the directory to save the hyperparameters search results.
       - input_shape: the shape of the input data for the Keras model.
       - output_shape: the shape of the output data for the Keras model.


   .. py:method:: _model_search(hp, **kwargs)

      Searches for the best hyperparameters to create a Keras model using the given hyperparameters space.

      :param hp: Object that holds
                 the hyperparameters space to search.
      :type hp: keras_tuner.engine.hyperparameters.HyperParameters

      :returns: A compiled Keras model with the best hyperparameters found.


   .. py:method:: search_hyperparams(X, y, project_name='', verbose=False)

      Perform hyperparameter search for the neural network using Keras Tuner.

      :param X: Input data.
      :type X: numpy.ndarray
      :param y: Target data.
      :type y: numpy.ndarray
      :param project_name: Name of the Keras Tuner project (default '').
      :type project_name: str
      :param verbose: Whether or not to print out information about the search progress (default False).
      :type verbose: bool

      :returns: A dictionary containing the optimal hyperparameters found by the search.
      :rtype: dict

      :raises ValueError: If `self.loss` is not a supported loss function.


   .. py:method:: build(input_shape=(1, ), output_shape=(1, ), n_neurons=[1], n_layers=1, learning_rate=0.001, activation='relu', **kwargs)

      Builds a Keras neural network model with the given hyperparameters.

      :param input_shape: The shape of the input data. Defaults to (1,).
      :type input_shape: tuple, optional
      :param output_shape: The shape of the output data. Defaults to (1,).
      :type output_shape: tuple, optional
      :param n_neurons: A list of integers representing the number of neurons in each hidden layer.
                        The length of the list determines the number of hidden layers. Defaults to [1].
      :type n_neurons: list, optional
      :param n_layers: The number of hidden layers in the model. Defaults to 1.
      :type n_layers: int, optional
      :param learning_rate: The learning rate of the optimizer. Defaults to 1e-3.
      :type learning_rate: float, optional
      :param activation: The activation function used for the hidden layers. Defaults to 'relu'.
      :type activation: str, optional

      :returns: None.


   .. py:method:: load(path)

      Load a Keras model from an H5 file.

      :param path: Path to the H5 file containing the Keras model.
      :type path: str

      :raises ValueError: If the file extension is not '.h5'.

      :returns: None


   .. py:method:: predict(x)

      Uses the trained neural network to make predictions on input data.

      :param x: Input data to be used for prediction. It must have the same number of features
                as the input_shape used to build the network.
      :type x: numpy.ndarray

      :returns: Predicted outputs for the input data.
      :rtype: numpy.ndarray

      :raises ValueError: If the input data x does not have the same number of features as the input_shape
          used to build the network.


   .. py:method:: fit(x, y, validation_data, batch_size=32, epochs=500, save=True, patience=5, path='')

      Trains the neural network model using the given input and output data.

      :param x: The input data used to train the model.
      :type x: numpy array
      :param y: The output data used to train the model.
      :type y: numpy array
      :param validation_data: A tuple containing the validation data as input and output data.
      :type validation_data: tuple
      :param batch_size: The batch size used for training the model (default=32).
      :type batch_size: int
      :param epochs: The number of epochs used for training the model (default=500).
      :type epochs: int
      :param save: Whether to save the model after training (default=True).
      :type save: bool
      :param patience: The number of epochs to wait before early stopping if the validation loss does not improve (default=5).
      :type patience: int
      :param path: The path to save the trained model (default='').
      :type path: str

      :returns: None


   .. py:method:: save(path)

      Saves the trained neural network model to a file.

      :param path: A string specifying the path and filename for the saved model. The ".h5" file extension
                   will be appended to the provided filename if not already present.

      :raises ValueError: If the provided file extension is not ".h5".

      :returns: None



.. py:class:: RandomForest(model=None)

   Bases: :py:obj:`Model`

   This class is used to build and search hyperparameters for a random forest model in scikit-learn.

   Attributes:
   -----------
       - model: the built RandomForest scikit-learn model.

   .. py:method:: search_hyperparams(X, y, verbose=False, **kwargs)

      Perform a hyperparameter search for a Random Forest model using RandomizedSearchCV.

      :param X: The feature matrix of the data.
      :type X: numpy array
      :param y: The target vector of the data.
      :type y: numpy array
      :param verbose: If True, print the optimal hyperparameters. Defaults to False.
      :type verbose: bool, optional
      :param \*\*kwargs: Additional keyword arguments. The following hyperparameters can be set:
                         - n_estimators (int): Number of trees in the forest. Defaults to sp_randint(10, 1000).
                         - max_features (list): The number of features to consider when looking for the best split.
                         Allowed values are 'sqrt', 'log2' or a float between 0 and 1. Defaults to ['sqrt', 'log2'].
                         - max_depth (list): The maximum depth of the tree. Defaults to [None, 5, 10, 15, 20, 30, 40].
                         - min_samples_split (int): The minimum number of samples required to split an internal node.
                         Defaults to sp_randint(2, 20).
                         - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
                         Defaults to sp_randint(1, 20).
                         - bootstrap (bool): Whether bootstrap samples are used when building trees.
                         Defaults to [True, False].
                         - y_type (str): Type of target variable. Either 'num' for numeric or 'cat' for categorical.
                         Defaults to 'cat'.

      :returns: A dictionary with the best hyperparameters found during the search.
      :rtype: dict


   .. py:method:: build(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', **kwargs)

      Builds a new Random Forest model with the specified hyperparameters.

      :param n_estimators: The number of trees in the forest. Default is 100.
      :type n_estimators: int, optional
      :param max_depth: The maximum depth of each tree. None means unlimited. Default is None.
      :type max_depth: int or None, optional
      :param min_samples_split: The minimum number of samples required to split an internal node. Default is 2.
      :type min_samples_split: int, optional
      :param min_samples_leaf: The minimum number of samples required to be at a leaf node. Default is 1.
      :type min_samples_leaf: int, optional
      :param max_features: The maximum number of features to consider when looking for the best split.
                           Can be 'sqrt', 'log2', an integer or None. Default is 'sqrt'.
      :type max_features: str or int, optional
      :param \*\*kwargs: Additional keyword arguments. Must include a 'y_type' parameter, which should be set to 'num' for
                         regression problems and 'cat' for classification problems.

      :returns: None


   .. py:method:: load(path)

      Load a saved random forest model.

      :param path: The path to the saved model file. The file must be a joblib file with the extension '.joblib'.
      :type path: str

      :raises ValueError: If the extension of the file is not '.joblib'.

      :returns: None.


   .. py:method:: predict(x)

      Makes predictions using the trained Random Forest model on the given input data.

      :param x: The input data to make predictions on.

      :returns: An array of predicted target values.


   .. py:method:: fit(x, y)

      Trains the Random Forest model on the given input and target data.

      :param x: The input data to train the model on.
      :param y: The target data to train the model on.

      :returns: None


   .. py:method:: save(path)

      Saves the trained model to a file with the specified path.

      :param path: The file path where the model should be saved. The file extension should be '.joblib'.
      :type path: str

      :raises ValueError: If the file extension is invalid or missing.

      :returns: None



.. py:class:: Scaler(scaler=None)

   The Scaler class is designed to scale and transform data using various scaling techniques. It contains methods for fitting and transforming data, as well as saving and loading scaler objects to and from files.

   .. py:method:: fit_transform(data)

      Fit to data, then transform it.

      :param data: The data to be transformed.
      :type data: array-like

      :returns: The transformed data.
      :rtype: array-like


   .. py:method:: transform(data)

      Perform standardization on an array.

      :param data: The data to be standardized.
      :type data: array-like

      :returns: The standardized data.
      :rtype: array-like


   .. py:method:: inverse_transform(data)

      Scale back the data to the original representation.

      :param data: The data to be scaled back.
      :type data: array-like

      :returns: The original representation of the data.
      :rtype: array-like


   .. py:method:: save(path)

      Save the scaler object to a file.

      :param path: The path where the scaler object will be saved.
      :type path: str


   .. py:method:: load(path)

      Load a saved scaler object from a file.

      :param path: The path where the scaler object is saved.
      :type path: str



.. py:class:: AsaML(dir_name=None)

   .. py:method:: identify_categorical_data(df)
      :staticmethod:

      Identifies categorical data.

      :param df: The DataFrame containing the data.
      :type df: pandas.DataFrame

      :returns: A tuple containing two DataFrames - one containing the categorical columns and the other containing the numerical columns.
      :rtype: tuple

      :raises ValueError: If the input DataFrame is empty or if it contains no categorical or numerical data.

      Example usage:

      .. code-block::

          >>> import pandas as pd
          >>> df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3], 'col3': ['x', 'y', 'z']})
          >>> df_cat, df_num = identify_categorical_data(df)
          >>> df_cat
          col1
          0    a
          1    b
          2    c
          >>> df_num
          col2
          0     1
          1     2
          2     3


   .. py:method:: pre_processing_train(X, y, remove_outlier=False)

      Perform pre-processing steps for the training dataset.

      :param X: pandas.DataFrame - input features.
      :param y: pandas.DataFrame - target variable.
      :param remove_outlier: bool - if True, removes the outliers from the dataset.

      :returns: pandas.DataFrame - pre-processed input features.
                y: pandas.DataFrame - pre-processed target variable.
      :rtype: X


   .. py:method:: __add_random_value_to_max(row)


   .. py:method:: train_model(X=None, y=None, name_model=None, save=True, scaling=True, scaler_Type='StandardScaler', remove_outlier=False, search=False, params=None, **kwargs)

      Train a model on a given dataset.

      :param X: A pandas dataframe containing the feature data. Default is None.
      :param y: A pandas dataframe containing the target data. Default is None.
      :param name_model: The name of the model to train. Default is None.
      :param save: A boolean indicating whether to save the model or not. Default is True.
      :param scaling: A boolean indicating whether to perform data scaling or not. Default is True.
      :param scaler_Type: The type of data scaling to perform. Must be one of 'StandardScaler', 'Normalizer', or 'MinMaxScaler'. Default is 'StandardScaler'.
      :param remove_outlier: A boolean indicating whether to remove outliers or not. Default is False.
      :param search: A boolean indicating whether to search for the best hyperparameters or not. Default is False.
      :param params: A dictionary containing the hyperparameters to use for training. Default is None.
      :param \*\*kwargs: Additional arguments to be passed to the training function.

      :returns: None


   .. py:method:: load_model(path='')

      Loads a saved model from the specified path and returns a dictionary of models
      with their corresponding parameters.

      :param path: The path where the model and its associated files are saved.
                   Defaults to an empty string.
      :type path: str

      :returns: A dictionary containing the loaded models with their corresponding parameters.
      :rtype: dict

      :raises ValueError: If the path argument is empty.

      .. note::
          The path variable must be the address of the 'dirname' folder and MUST contain the scaler.pkl file and each subdirectory MUST contain the 'paramenters', 'model' files.



   .. py:method:: pre_processing_predict(X, input_list, var_type)

      Pre-processes the input data before prediction by scaling numerical features and creating dummy variables
      for categorical features. Also handles missing and extra features in the input data.

      :param X: The input data to be pre-processed.
      :type X: pandas.DataFrame
      :param input_list: A list of expected input features.
      :type input_list: list
      :param var_type: A dictionary with the types of the input features. The keys 'cat' and 'num' contain lists
                       of categorical and numerical feature names respectively.
      :type var_type: dict

      :returns:

                The pre-processed input data with scaled numerical features and dummy variables
                    for categorical features. Any missing or extra features are handled accordingly.
      :rtype: pandas.DataFrame


   .. py:method:: pos_processing(y, output_list)

      Post-processes the output of a model prediction to transform it into a more usable format.

      :param y: The output of the model prediction, as a NumPy array.
      :type y: np.ndarray
      :param output_list: A list of column names representing the output variables.
      :type output_list: list

      :returns: A pandas DataFrame containing the post-processed output values.
      :rtype: pd.DataFrame

      This function takes the output of a model prediction, which is typically a NumPy array of raw output values, and transforms it into a more usable format. The output variables are expected to have been one-hot encoded with the use of triple underscores ('___') as separator, and possibly have a random value added to the max value of each row. The function first separates the categorical and numerical variables, then processes the categorical variables by selecting the maximum value for each row and one-hot encoding them. Finally, it concatenates the categorical and numerical variables back together to produce a pandas DataFrame containing the post-processed output values.



   .. py:method:: predict_all(X, model_dict)

      Apply all models in the model dictionary to the input data frame X and return the predictions.

      :param X: A pandas DataFrame representing the input data.
      :param model_dict: A dictionary containing the models and their associated metadata. The keys are the names of the
                         models and the values are themselves dictionaries containing the following keys:
                         - 'model': A trained machine learning model.
                         - 'input_list': A list of the names of the input features used by the model.
                         - 'output_list': A list of the names of the output features produced by the model.
                         - 'var_type': A dictionary containing the types of the input and output features, with the keys
                                         'X' and 'y', respectively, and the values being dictionaries themselves with
                                         the following keys:
                                         - 'cat': A list of the categorical input features.
                                         - 'num': A list of the numerical input features.

      :returns: A pandas DataFrame containing the predictions of all models in the model dictionary. The columns of the
                DataFrame are the names of the models, and the rows correspond to the input rows in X.

      :raises ValueError: If X is empty or None, or if the model dictionary is empty or None.


   .. py:method:: full_cycle(X_pred, load=False, **kwargs)

      Performs the full cycle of the machine learning pipeline: loads or trains the models, preprocesses the input data,
      generates predictions, and post-processes the output data.

      :param X_pred: Input data to generate predictions for.
      :type X_pred: pandas.DataFrame
      :param load: If True, loads the trained models from disk instead of training new ones. Default is False.
      :type load: bool, optional
      :param \*\*kwargs: Additional keyword arguments passed to either `load_model()` or `train_model()` method.

      :returns: Dataframe with the generated predictions.
      :rtype: pandas.DataFrame

      :raises ValueError: If `load` is True and `path` is not provided in `kwargs`.



