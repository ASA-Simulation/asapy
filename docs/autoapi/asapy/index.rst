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
   prediction/index.rst
   prediction copy/index.rst
   preprocessing/index.rst
   utils/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.Doe
   asapy.ExecutionController
   asapy.Analysis
   asapy.NN
   asapy.RandomForest
   asapy.Scaler
   asapy.AsaML
   asapy.Preprocessing



Functions
~~~~~~~~~

.. autoapisummary::

   asapy.gen_dict_extract
   asapy.transform_stringified_dict
   asapy.basic_simulate
   asapy.batch_simulate
   asapy.basic_stop
   asapy.stop_func
   asapy.non_stop_func
   asapy.prepare_simulation_batch
   asapy.prepare_simulation_tacview
   asapy.load_simulation
   asapy.json_to_df
   asapy.list_to_df
   asapy.unique_list
   asapy.get_parents_dict
   asapy.check_samples_similar
   asapy.test_t
   asapy.convert_nested_string_to_dict
   asapy.find_key
   asapy.gen_dict_extract
   asapy.transform_stringified_dict
   asapy.meters_to_micrometers
   asapy.micrometers_to_meters
   asapy.meters_to_centimeters
   asapy.centimeters_to_meters
   asapy.meters_to_kilometers
   asapy.kilometers_to_meters
   asapy.meters_to_inches
   asapy.inches_to_meters
   asapy.meters_to_feet
   asapy.feet_to_meters
   asapy.kilometers_to_nautical_miles
   asapy.nautical_miles_to_kilometers
   asapy.kilometers_to_statute_miles
   asapy.statute_miles_to_kilometers
   asapy.nautical_miles_to_statute_miles
   asapy.statute_miles_to_nautical_miles
   asapy.degrees_to_radians
   asapy.degrees_to_semicircles
   asapy.radians_to_degrees
   asapy.radians_to_semicircles
   asapy.semicircles_to_radians
   asapy.semicircles_to_degrees
   asapy.aepcd_deg
   asapy.aepcd_rad
   asapy.alimd
   asapy.gbd2ll
   asapy.fbd2ll
   asapy.gll2bd
   asapy.fll2bd
   asapy.convert_ecef_to_geod
   asapy.convert_geod_to_ecef
   asapy.prepare_simulation_batch
   asapy.prepare_simulation_tacview
   asapy.load_simulation
   asapy.json_to_df
   asapy.list_to_df
   asapy.unique_list
   asapy.get_parents_dict
   asapy.check_samples_similar
   asapy.test_t
   asapy.convert_nested_string_to_dict
   asapy.find_key
   asapy.gen_dict_extract
   asapy.transform_stringified_dict
   asapy.meters_to_micrometers
   asapy.micrometers_to_meters
   asapy.meters_to_centimeters
   asapy.centimeters_to_meters
   asapy.meters_to_kilometers
   asapy.kilometers_to_meters
   asapy.meters_to_inches
   asapy.inches_to_meters
   asapy.meters_to_feet
   asapy.feet_to_meters
   asapy.kilometers_to_nautical_miles
   asapy.nautical_miles_to_kilometers
   asapy.kilometers_to_statute_miles
   asapy.statute_miles_to_kilometers
   asapy.nautical_miles_to_statute_miles
   asapy.statute_miles_to_nautical_miles
   asapy.degrees_to_radians
   asapy.degrees_to_semicircles
   asapy.radians_to_degrees
   asapy.radians_to_semicircles
   asapy.semicircles_to_radians
   asapy.semicircles_to_degrees
   asapy.aepcd_deg
   asapy.aepcd_rad
   asapy.alimd
   asapy.gbd2ll
   asapy.fbd2ll
   asapy.gll2bd
   asapy.fll2bd
   asapy.convert_ecef_to_geod
   asapy.convert_geod_to_ecef



Attributes
~~~~~~~~~~

.. autoapisummary::

   asapy.FT2M
   asapy.M2FT
   asapy.IN2M
   asapy.M2IN
   asapy.NM2M
   asapy.M2NM
   asapy.NM2FT
   asapy.FT2NM
   asapy.SM2M
   asapy.M2SM
   asapy.SM2FT
   asapy.FT2SM
   asapy.KM2M
   asapy.M2KM
   asapy.CM2M
   asapy.M2CM
   asapy.UM2M
   asapy.M2UM
   asapy.D2SC
   asapy.SC2D
   asapy.R2SC
   asapy.SC2R
   asapy.R2DCC
   asapy.D2RCC
   asapy.earth_model_data
   asapy.FT2M
   asapy.M2FT
   asapy.IN2M
   asapy.M2IN
   asapy.NM2M
   asapy.M2NM
   asapy.NM2FT
   asapy.FT2NM
   asapy.SM2M
   asapy.M2SM
   asapy.SM2FT
   asapy.FT2SM
   asapy.KM2M
   asapy.M2KM
   asapy.CM2M
   asapy.M2CM
   asapy.UM2M
   asapy.M2UM
   asapy.D2SC
   asapy.SC2D
   asapy.R2SC
   asapy.SC2R
   asapy.R2DCC
   asapy.D2RCC
   asapy.earth_model_data


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


   .. py:method:: create(df_T, samples, seed=42)
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



.. py:function:: gen_dict_extract(key, var)

   A generator function to iterate and yield values from a dictionary or list nested inside the dictionary, given a key.

   :param key: The key to search for in the dictionary.
   :type key: str
   :param var: The dictionary or list to search.
   :type var: dict or list

   :Yields: *value* -- The value from the dictionary or list that corresponds to the given key.


.. py:function:: transform_stringified_dict(data)

   Recursively converts stringified JSON parts of a dictionary or list into actual dictionaries or lists.

   This function checks if an item is a string and attempts to convert it to a dictionary or list
   using `json.loads()`. If the conversion is successful, the function recursively processes the new
   dictionary or list. If a string is not a valid JSON representation, it remains unchanged.

   :param data: Input data that might contain stringified JSON parts.
   :type data: Union[dict, list, str]

   :returns: The transformed data with all stringified JSON parts converted
             to dictionaries or lists.
   :rtype: Union[dict, list, str]

   :raises json.JSONDecodeError: If there's an issue decoding a JSON string. This is caught internally
   :raises and the original string is returned.:


.. py:function:: basic_simulate(batch: asaclient.Batch, current: pandas.DataFrame, processed: pandas.DataFrame, all: pandas.DataFrame, asa_custom_types=[], pbar=None) -> pandas.DataFrame

   Performs basic simulation on a chunk of data.

   :param batch: The ASA batch object.
   :type batch: asaclient.Batch
   :param current: The current chunk of data to simulate.
   :type current: pd.DataFrame
   :param processed: The previously processed data.
   :type processed: pd.DataFrame
   :param all: The complete dataset.
   :type all: pd.DataFrame
   :param asa_custom_types: List of custom ASA types to retrieve in the simulation records.
   :type asa_custom_types: list, optional
   :param pbar: The tqdm progress bar object.
   :type pbar: tqdm, optional

   :returns: The simulation results for the current chunk of data.
   :rtype: pd.DataFrame


.. py:function:: batch_simulate(batch: asaclient.Batch, asa_custom_types=[])

   Returns a partial function for batch simulation.

   :param batch: The ASA batch object.
   :type batch: asaclient.Batch
   :param asa_custom_types: List of custom ASA types to retrieve in the simulation records.
   :type asa_custom_types: list, optional

   :returns: The partial function for batch simulation.
   :rtype: callable


.. py:function:: basic_stop(current: pandas.DataFrame, all_previous: pandas.DataFrame, metric: str, threshold: float, side: str) -> bool

   Determines whether to stop the simulation based on the current and previous results.

   :param current: The current chunk of simulation results.
   :type current: pd.DataFrame
   :param all_previous: All previously processed simulation results.
   :type all_previous: pd.DataFrame
   :param metric: The metric to compare.
   :type metric: str
   :param threshold: The threshold value for stopping the simulation.
   :type threshold: float
   :param side: The side information to select the specific metric.
   :type side: str

   :returns: True if the simulation should stop, False otherwise.
   :rtype: bool


.. py:function:: stop_func(metric: str, threshold: float, side: str)

   Returns a partial function for stopping the simulation.

   :param metric: The metric to compare.
   :type metric: str
   :param threshold: The threshold value for stopping the simulation.
   :type threshold: float

   :returns: The partial function for stopping the simulation.
   :rtype: callable


.. py:function:: non_stop_func(current: pandas.DataFrame, all_previous: pandas.DataFrame) -> bool

   Determines that the simulation should never stop.

   :param current: The current chunk of simulation results.
   :type current: pd.DataFrame
   :param all_previous: All previously processed simulation results.
   :type all_previous: pd.DataFrame

   :returns: Always False.
   :rtype: bool


.. py:class:: ExecutionController(sim_func: callable, stop_func: callable, chunk_size: int = 0)


   A class for controlling the execution of a simulation function on a Design of Experiments (DOE) dataset.

   .. py:attribute:: HIDDEN_FOLDER
      :value: '.execution_state'

      

   .. py:method:: save_state(file_name: str)

      Saves the current execution state to a file.


   .. py:method:: load_state(file_name: str, sim_func: callable, stop_func: callable, chunk_size: int)
      :classmethod:

      Loads the saved execution state from a file.


   .. py:method:: resume()

      Resumes the execution from the saved state if available.


   .. py:method:: pause()

      Pauses the current execution and saves the state.


   .. py:method:: _safe_pbar_update(n)

      Safely updates the progress bar by `n` steps.


   .. py:method:: _safe_pbar_close()

      Safely closes the progress bar.


   .. py:method:: run(doe: pandas.DataFrame, resume=False) -> pandas.DataFrame

      Runs the simulation on the DOE by dividing it into chunks and stops if `_stop_func` returns True.



.. py:class:: Analysis


   Class for performing Analysis on a DataFrame with simulation data.

   .. py:method:: detect_outliers(df: pandas.DataFrame, method: str = 'IQR', thr: float = 3) -> Tuple[pandas.DataFrame, pandas.DataFrame]
      :staticmethod:

      Detect outliers in a Pandas DataFrame using either Inter-Quartile Range (IQR) or Z-Score method.

      :param df: The input DataFrame containing numerical data.
      :type df: pd.DataFrame
      :param method: The method used for outlier detection, options are 'IQR' or 'zscore'. Default is 'IQR'.
      :type method: str
      :param thr: The threshold value for Z-Score method. Default is 3.
      :type thr: float

      :returns:     - The first DataFrame contains the index, column name, and values of the outliers.
                    - The second DataFrame contains the outlier thresholds for each column.
      :rtype: Tuple[pd.DataFrame, pd.DataFrame]

      :raises ValueError: If the method is not 'IQR' or 'zscore'.


   .. py:method:: remove_outliers(df: pandas.DataFrame, verbose: bool = False) -> Tuple[pandas.DataFrame, List[int]]
      :staticmethod:

      Remove outliers from a Pandas DataFrame using the Interquartile Range (IQR) method.

      :param df: DataFrame containing the data.
      :type df: pd.DataFrame
      :param verbose: If True, print the number of lines removed. Defaults to False.
      :type verbose: bool, optional

      :returns:

                DataFrame with the outliers removed,
                                                List of indexes of the rows that were removed (unique indices).
      :rtype: Tuple[pd.DataFrame, List[int]]


   .. py:method:: cramer_v(df: pandas.DataFrame, verbose: bool = False, save: bool = False, path: Optional[str] = None, format: str = 'png') -> pandas.DataFrame
      :staticmethod:

      Calculate Cramer's V statistic for categorical feature association in a DataFrame.

      This function takes a DataFrame and calculates Cramer's V, a measure of association between two
      categorical variables, for all pairs of categorical columns. The result is a symmetric DataFrame where
      each cell [i, j] contains the Cramer's V value between column i and column j.

      :param df: Input DataFrame containing categorical variables.
      :param verbose: If True, a heatmap of Cramer's V values is displayed. Default is False.
      :param save: If True, the resulting heatmap is saved to a file. Default is False.
      :param path: The directory path where the heatmap is to be saved, if `save` is True. Default is None.
      :param format: The file format to save the heatmap, if `save` is True. Default is 'png'.

      :returns: A DataFrame containing Cramer's V values for all pairs of categorical variables in df.

      :raises ValueError: If `save` is True but `path` is None.


   .. py:method:: eda(df: pandas.DataFrame, save: bool = False, path: Optional[str] = None, format: str = 'png') -> None

      Perform exploratory data analysis (EDA) on a pandas DataFrame.

      The function provides a summary of the DataFrame, showing class balance for categorical variables,
      and displaying histograms and boxplots with outlier information for numerical variables. If desired,
      the function can save these plots to a specified location.

      :param df: DataFrame to be analyzed.
      :type df: pd.DataFrame
      :param save: Whether to save the plots. Defaults to False.
      :type save: bool, optional
      :param path: Directory where the plots should be saved. If save is True, this parameter
                   must be provided. Defaults to None.
      :type path: str, optional
      :param format: The file format for saved plots. Acceptable formats include png, pdf, ps, eps and svg.
                     Defaults to 'png'.
      :type format: str, optional

      :raises ValueError: If 'save' is set to True but 'path' is not specified.

      :returns: None


   .. py:method:: _process_categorical(df_cat, save, path, format)

      Processes categorical features of a DataFrame. Moves columns from df_num to df_cat based on a condition.
      Then, it summarizes, visualizes and saves the processed DataFrame.

      :param df_cat: Categorical DataFrame to be processed.
      :type df_cat: pd.DataFrame
      :param df_num: Numerical DataFrame to be processed.
      :type df_num: pd.DataFrame
      :param save: Whether to save the generated plots.
      :type save: bool
      :param path: Path to save the plots.
      :type path: str
      :param format: Format for the plot files.
      :type format: str


   .. py:method:: _process_numerical(df_num, save, path, format)

      Processes numerical features of a DataFrame. It summarizes, visualizes and saves the processed DataFrame.

      :param df_num: Numerical DataFrame to be processed.
      :type df_num: pd.DataFrame
      :param save: Whether to save the generated plots.
      :type save: bool
      :param path: Path to save the plots.
      :type path: str
      :param format: Format for the plot files.
      :type format: str


   .. py:method:: _describe(df)

      Prints a summary of the DataFrame, including the count of NaN values.

      :param df: DataFrame to be summarized.
      :type df: pd.DataFrame


   .. py:method:: _plot_histograms(df, save, path, format)

      Plots histograms of all columns in a DataFrame.

      :param df: DataFrame to be plotted.
      :type df: pd.DataFrame
      :param save: Whether to save the plots.
      :type save: bool
      :param path: Path to save the plots.
      :type path: str
      :param format: Format for the plot files.
      :type format: str


   .. py:method:: _plot_single_histogram(df, column, save, path, format)

      Plots a single histogram for a specific column in a DataFrame.

      :param df: DataFrame to be plotted.
      :type df: pd.DataFrame
      :param column: Column to be plotted.
      :type column: str
      :param save: Whether to save the plot.
      :type save: bool
      :param path: Path to save the plot.
      :type path: str
      :param format: Format for the plot file.
      :type format: str


   .. py:method:: _plot_correlation(df, save, path, format)

      Plots a correlation heatmap of a numerical DataFrame.

      :param df: Numerical DataFrame to be plotted.
      :type df: pd.DataFrame
      :param save: Whether to save the plot.
      :type save: bool
      :param path: Path to save the plot.
      :type path: str
      :param format: Format for the plot file.
      :type format: str


   .. py:method:: _plot_histograms_boxplots(df, save, path, format)

      Plots histograms and boxplots of all columns in a numerical DataFrame.

      :param df: Numerical DataFrame to be plotted.
      :type df: pd.DataFrame
      :param save: Whether to save the plots.
      :type save: bool
      :param path: Path to save the plots.
      :type path: str
      :param format: Format for the plot files.
      :type format: str


   .. py:method:: _plot_single_histogram_boxplot(df, column, save, path, format)

      Plots a histogram and boxplot for a given column in a DataFrame.

      :param df: DataFrame to be plotted.
      :type df: pd.DataFrame
      :param column: Column to be plotted.
      :type column: str
      :param save: If True, saves the plot.
      :type save: bool
      :param path: File path to save the plot.
      :type path: str
      :param format: File format to save the plot.
      :type format: str


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

      .. seealso::

          `pingouin.homoscedasticity <https://pingouin-stats.org/build/html/generated/pingouin.homoscedasticity.html#pingouin.homoscedasticity>`_: teste de igualdade de variância.

          `pingouin.normality <https://pingouin-stats.org/build/html/generated/pingouin.normality.html#pingouin.normality>`_: teste de normalidade.

          `scipy.stats.f_oneway <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html>`_: one-way ANOVA.

          `scipy.stats.tukey_hsd <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html>`_: teste HSD de Tukey para igualdade de médias.

          `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_: teste H de Kruskal-Wallis para amostras independentes.

          `scikit_posthocs.posthoc_conover <https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html>`_: teste de Conover.



   .. py:method:: fit_distribution(df: pandas.DataFrame, verbose: bool = False) -> pandas.DataFrame
      :staticmethod:

      Find the best fitting distribution for the input data.

      This function compares 93 available distributions in the scipy library and finds the one
      that best fits the input data. The best fit is determined by the Kolmogorov-Smirnov test.

      :param df: Input data, which must contain only one distribution.
      :type df: pd.DataFrame
      :param verbose: Flag that controls whether detailed messages are displayed. Defaults to False.
      :type verbose: bool, optional

      :raises ValueError: Raised if the input data contains more than one distribution.

      :returns: DataFrame with information about the distribution that best fits the input data,
                as well as the most common distributions (``norm``, ``beta``, ``chi2``, ``uniform``, ``expon``).
                The DataFrame's columns are: ``Distribution_Type``, ``P_Value``, ``Statistics``, and ``Parameters``.
      :rtype: pd.DataFrame


   .. py:method:: feature_score(df: pandas.DataFrame, x: List[str], y: str, scoring_function: str, verbose: bool = False, save_path=None) -> pandas.DataFrame
      :staticmethod:

      Calculate the score of input features using the specified scoring function.

      This function applies a specified scoring function to evaluate the relevance of each input feature for the output
      variable in a given DataFrame. The supported scoring functions are those provided by sklearn's feature_selection module.

      :param df: The input data, where each column is a feature and each row is an observation.
      :type df: pd.DataFrame
      :param x: A list of names of the input features in 'df'.
      :type x: List[str]
      :param y: The name of the output feature in 'df'.
      :type y: str
      :param scoring_function: The name of the scoring function to be used. Should be one of the following:
                               - 'r_regression'
                               - 'f_regression'
                               - 'mutual_info_regression'
                               - 'chi2'
                               - 'f_classif'
                               - 'mutual_info_classif'
      :type scoring_function: str
      :param verbose: Whether to print detailed output. Default is False.
      :type verbose: bool, optional
      :param save_path: Path to save the plotted figure. If not specified, the figure is simply shown.
      :type save_path: str, optional

      :raises ValueError: If 'scoring_function' is not one of the supported scoring functions.

      :returns:

                A DataFrame where each row corresponds to an input feature, and the 'score' column contains
                    the corresponding score. The DataFrame is sorted by score in descending order.
      :rtype: pd.DataFrame


   .. py:method:: pareto_front(df, list_min=None, list_max=None, verbose=False, max_points=None)
      :staticmethod:

      Identifies the Pareto front of a DataFrame based on objectives to minimize and maximize.

      :param df: Input DataFrame containing the data.
      :type df: pd.DataFrame
      :param list_min: List of variable names to minimize. Defaults to None.
      :type list_min: list of str, optional
      :param list_max: List of variable names to maximize. Defaults to None.
      :type list_max: list of str, optional
      :param verbose: If True, displays detailed information. Defaults to False.
      :type verbose: bool, optional
      :param max_points: Maximum number of points to include in the Pareto front. Defaults to None.
      :type max_points: int, optional

      :returns: DataFrame containing the Pareto optimal points.
      :rtype: pd.DataFrame


   .. py:method:: get_best_pareto_point(df, list_min=None, list_max=None, weights_min=None, weights_max=None, minimization_weight=0.5, verbose=False)
      :staticmethod:

      Determine the optimal Pareto point from the input DataFrame, considering specified variables and their weights.

      :param df: The input DataFrame.
      :type df: pd.DataFrame
      :param list_min: A list of column names to minimize in the Pareto optimality calculation. Defaults to None.
      :type list_min: List[str], optional
      :param list_max: A list of column names to maximize in the Pareto optimality calculation. Defaults to None.
      :type list_max: List[str], optional
      :param weights_min: A list of weights defining the relative importance of each variable to minimize. Defaults to None.
      :type weights_min: List[float], optional
      :param weights_max: A list of weights defining the relative importance of each variable to maximize. Defaults to None.
      :type weights_max: List[float], optional
      :param minimization_weight: The global weight for the minimization part, between 0 and 1. Defaults to 0.5.
      :type minimization_weight: float, optional
      :param verbose: Flag to display detailed messages. Defaults to False.
      :type verbose: bool, optional

      :returns: A Pandas Series containing the best Pareto optimal point based on the specified variables and weights.
      :rtype: pd.Series

      .. note:: The function assumes that the input DataFrame contains only Pareto optimal points.


   .. py:method:: anova(df: pandas.DataFrame, columns: List[str] = None, alpha: float = 0.05, show_plots: bool = True, save_path: Optional[str] = None, boxplot_title: str = 'Distributions of Samples', boxplot_xlabel: str = 'Samples', boxplot_ylabel: str = 'Value', boxplot_names: Optional[List[str]] = None) -> Tuple[pandas.DataFrame, Optional[pandas.DataFrame]]
      :staticmethod:

      Perform ANOVA test on the given DataFrame columns and conduct Multiple pairwise comparisons (Post-hoc test)
      if more than two variables are being compared.

      :param df: DataFrame containing the samples.
      :type df: pd.DataFrame
      :param columns: Columns to be analyzed. If None, all columns are used. Defaults to None.
      :type columns: List[str], optional
      :param alpha: Significance level. Defaults to 0.05.
      :type alpha: float, optional
      :param show_plots: If True, plots will be displayed for visual analysis. Defaults to True.
      :type show_plots: bool, optional
      :param save_path: Path to save the generated plots and results. If None, the plots are
                        displayed and results are printed without saving. If provided, plots and results will be saved to the
                        specified path. Directory structure will be created if not exists. Defaults to None.
      :type save_path: Optional[str], optional
      :param boxplot_title: Title for the box plots. Defaults to 'Distributions of Samples'.
      :type boxplot_title: str, optional
      :param boxplot_xlabel: Label for the X-axis of the box plot. Defaults to 'Samples'.
      :type boxplot_xlabel: str, optional
      :param boxplot_ylabel: Label for the Y-axis of the box plot. Defaults to 'Value'.
      :type boxplot_ylabel: str, optional
      :param boxplot_names: Names for the box plots. Defaults to None.
      :type boxplot_names: Optional[List[str]], optional

      :returns: ANOVA summary and optionally Post-hoc test results.
      :rtype: Tuple[pd.DataFrame, Optional[pd.DataFrame]]


   .. py:method:: analyze_relationship(df, col1, col2, save_path=None)
      :staticmethod:

      Analyzes the relationship between two columns in a DataFrame.

      This function performs a series of analyses to understand the relationship
      between two numeric columns in the given DataFrame. It produces:
      - Descriptive statistics.
      - A scatter plot.
      - Pearson, Spearman, and Kendall correlations.
      - A correlation heatmap.
      - Linear regression and a regression plot.
      - A residual plot.

      :param df: The DataFrame containing the data.
      :type df: pandas.DataFrame
      :param col1: The name of the first column.
      :type col1: str
      :param col2: The name of the second column.
      :type col2: str
      :param save_path: The directory where the results, plots, and analysis
                        will be saved. If not specified, results are just displayed.
      :type save_path: str, optional

      :returns: None. Displays or saves plots and textual analysis depending on `save_path`.

      :raises ValueError: If the specified columns are not found in the DataFrame.
      :raises TypeError: If input data is not in the expected format.


   .. py:method:: plot_histograms(df, columns, figsize=(15, 5), alpha=0.7, save_path=None)
      :staticmethod:

      Plots histograms for the given columns in the DataFrame.

      :param df: The DataFrame containing the data.
      :type df: DataFrame
      :param columns: List of column names to plot.
      :type columns: list
      :param figsize: Size of the figure for each row of histograms. Default is (15, 5).
      :type figsize: tuple
      :param alpha: Alpha value for the histograms. Default is 0.7.
      :type alpha: float
      :param save_path: Path to save the plotted figure. If not specified, the figure is simply shown.
      :type save_path: str, optional


   .. py:method:: bootstrap(dataframe, columns, n_iterations=1000, alpha=0.05, show_plots=False, boxplot_xlabel: str = 'Samples', boxplot_ylabel: str = 'Value', boxplot_names: Optional[List[str]] = None)
      :staticmethod:

      Perform bootstrap hypothesis tests to determine if the mean of one sample
      is statistically greater or lesser than the other for each pair of columns
      in the provided list and optionally visualize the distributions with box plots.

      :param dataframe: The dataframe containing the samples.
      :type dataframe: pd.DataFrame
      :param columns: List of column names to be compared.
      :type columns: list
      :param n_iterations: Number of bootstrap iterations. Default is 1,000.
      :type n_iterations: int, optional
      :param alpha: Significance level. Default is 0.05.
      :type alpha: float, optional
      :param show_plots: Flag to display plots. Default is False.
      :type show_plots: bool, optional
      :param boxplot_xlabel: Label for the X-axis of the box plot. Default is 'Samples'.
      :type boxplot_xlabel: str, optional
      :param boxplot_ylabel: Label for the Y-axis of the box plot. Default is 'Value'.
      :type boxplot_ylabel: str, optional
      :param boxplot_names: Names for the box plots. Default is None.
      :type boxplot_names: Optional[List[str]], optional

      :returns: Prints the test outcome for each pair and optionally displays a box plot.
      :rtype: None


   .. py:method:: create_2d_scatter_plot(df, x_col, y_col, size_col, title='2D Scatter Plot', xlabel='X-axis', ylabel='Y-axis', size_label='Size', cmap='coolwarm', figsize=(12, 8), alpha=0.5, grid=True, ref_size_value=0.5)
      :staticmethod:

      Create a 2D scatter plot with variable circle sizes.

      :param df: DataFrame containing the data.
      :type df: DataFrame
      :param x_col: Name of the column in df for the x-axis.
      :type x_col: str
      :param y_col: Name of the column in df for the y-axis.
      :type y_col: str
      :param size_col: Name of the column in df for determining the size of the scatter points.
      :type size_col: str
      :param title: Title of the plot. Defaults to '2D Scatter Plot'.
      :type title: str, optional
      :param xlabel: Label for the x-axis. Defaults to 'X-axis'.
      :type xlabel: str, optional
      :param ylabel: Label for the y-axis. Defaults to 'Y-axis'.
      :type ylabel: str, optional
      :param size_label: Label for the size legend. Defaults to 'Size'.
      :type size_label: str, optional
      :param cmap: Colormap for the scatter points. Defaults to 'coolwarm'.
      :type cmap: str, optional
      :param figsize: Size of the figure. Defaults to (12, 8).
      :type figsize: tuple, optional
      :param alpha: Alpha blending value for the scatter points, between 0 and 1. Defaults to 0.5.
      :type alpha: float, optional
      :param grid: Flag to add grid to the plot. Defaults to True.
      :type grid: bool, optional
      :param ref_size_value: Value for calculating the reference size circle. Defaults to 0.5.
      :type ref_size_value: float, optional

      :returns: The function creates a matplotlib scatter plot and does not return any value.
      :rtype: None

      .. rubric:: Example

      create_2d_scatter_plot(df=my_dataframe,
                          x_col='speed',
                          y_col='altitude',
                          size_col='fuel_consumed',
                          title='Flight Characteristics',
                          xlabel='Speed (knots)',
                          ylabel='Altitude (feet)',
                          size_label='Fuel Consumed (normalized)')


   .. py:method:: create_3d_surface_plot(df, x_col, y_col, z_col, title='3D Surface Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis', cmap=cm.coolwarm, figsize=(16, 12), elev=30, azim=45)
      :staticmethod:

      Create a 3D surface plot from three columns in a DataFrame.

      :param df: DataFrame containing the data.
      :type df: DataFrame
      :param x_col: Name of the column in df for the x-axis.
      :type x_col: str
      :param y_col: Name of the column in df for the y-axis.
      :type y_col: str
      :param z_col: Name of the column in df for the z-axis (surface height).
      :type z_col: str
      :param title: Title of the plot. Defaults to '3D Surface Plot'.
      :type title: str, optional
      :param xlabel: Label for the x-axis. Defaults to 'X-axis'.
      :type xlabel: str, optional
      :param ylabel: Label for the y-axis. Defaults to 'Y-axis'.
      :type ylabel: str, optional
      :param zlabel: Label for the z-axis. Defaults to 'Z-axis'.
      :type zlabel: str, optional
      :param cmap: Colormap for the surface plot. Defaults to cm.coolwarm.
      :type cmap: Colormap, optional
      :param figsize: Size of the figure. Defaults to (16, 12).
      :type figsize: tuple, optional
      :param elev: Elevation angle in the z plane for the 3D plot. Defaults to 30.
      :type elev: int, optional
      :param azim: Azimuth angle in the x,y plane for the 3D plot. Defaults to 45.
      :type azim: int, optional

      :returns: The function creates a matplotlib 3D surface plot and does not return any value.
      :rtype: None



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


   .. py:method:: search_hyperparams(X, y, project_name='', y_type='num', verbose=False)

      Perform hyperparameter search for the neural network using Keras Tuner.

      :param X: Input data.
      :type X: numpy.ndarray
      :param y: Target data.
      :type y: numpy.ndarray
      :param project_name: Name of the Keras Tuner project (default '').
      :type project_name: str
      :param y_type: Type of target variable. Either 'num' for numeric or 'cat' for categorical (default 'num').
      :type y_type: str
      :param verbose: Whether or not to print out information about the search progress (default False).
      :type verbose: bool

      :returns: A dictionary containing the optimal hyperparameters found by the search.
      :rtype: dict


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


   .. py:method:: fit(x, y, validation_data=None, batch_size=32, epochs=500, save=True, patience=5, path='')

      Trains the Random Forest model on the given input and target data.

      :param x: The input data to train the model on.
      :type x: numpy array
      :param y: The target data to train the model on.
      :type y: numpy array
      :param validation_data: Not used in this context. For compatibility only.
      :type validation_data: tuple
      :param batch_size: Not used in this context. For compatibility only.
      :type batch_size: int
      :param epochs: Not used in this context. For compatibility only.
      :type epochs: int
      :param save: If True, saves the model to the specified path.
      :type save: bool
      :param patience: Not used in this context. For compatibility only.
      :type patience: int
      :param path: The path to save the trained model.
      :type path: str

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



.. py:class:: Preprocessing


   .. py:method:: aliases(x)
      :staticmethod:

      Preprocessing the simulation aliases. Adjusts the index of the dataframe by creating a new column 'experiment' that
      receives the current indices of the DataFrame.

      :param x: Input DataFrame
      :type x: pandas.DataFrame

      :returns: DataFrame with modified index.
      :rtype: pandas.DataFrame


   .. py:method:: team_metrics(df)
      :staticmethod:

      Preprocesses the monitor report data with an additional filter for monitor_type.

      :param df: Input DataFrame
      :type df: pandas.DataFrame
      :param monitor_type: The specific monitor type to filter by. For example, 'AsaTeamMetrics@AsaModels' or 'AsaAirThreatMetric@AsaModels'.
      :type monitor_type: str

      :returns: The preprocessed dataframe.
      :rtype: pandas.DataFrame


   .. py:method:: air_threat_metric(df)
      :staticmethod:

      Preprocesses the monitor report data with an additional filter for monitor_type.

      :param df: Input DataFrame
      :type df: pandas.DataFrame

      :returns: The preprocessed dataframe.
      :rtype: pandas.DataFrame


   .. py:method:: weapon_detonation(df)
      :staticmethod:

      Function for pre-processing weapon detonation data.

      :param df: The dataframe to preprocess.
      :type df: pandas.DataFrame

      :returns: The preprocessed dataframe.
      :rtype: pandas.DataFrame


   .. py:method:: convert_categorical_to_dummies(df, column_name, prefix=None)
      :staticmethod:

      Convert a categorical column into dummy/indicator columns, and add these new columns
      into the DataFrame at the same position of the original one.

      :param df: The DataFrame to process.
      :type df: pd.DataFrame
      :param column_name: The name of the categorical column to convert.
      :type column_name: str
      :param prefix: The prefix to apply to the dummy column names.
      :type prefix: str

      :returns: The DataFrame with the original column replaced by dummy columns.
      :rtype: pd.DataFrame



.. py:function:: prepare_simulation_batch(sim: asaclient.Simulation) -> asaclient.Simulation

   Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run in batch mode.

   :param sim: The simulation instance for which the simulation setup needs to be prepared.
   :type sim: Simulation

   :returns: The updated Simulation instance with the added recorder configurations in its subcomponents.
   :rtype: Simulation


.. py:function:: prepare_simulation_tacview(sim: asaclient.Simulation) -> asaclient.Simulation

   Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run on Tacview.

   :param sim: The simulation instance for which the simulation setup needs to be prepared.
   :type sim: Simulation

   :returns: The updated Simulation instance with the added recorder configurations in its subcomponents.
   :rtype: Simulation


.. py:function:: load_simulation(path: str) -> asaclient.Simulation

   Loads a Simulation object from a JSON file.

   This method accepts a path to a JSON file, reads the content of the file and
   creates a Simulation object using the data parsed from the file.

   :param path: The absolute or relative path to the JSON file to be loaded.
   :type path: str

   :returns: The Simulation object created from the loaded JSON data.
   :rtype: Simulation


.. py:function:: json_to_df(self, json, id='id') -> pandas.DataFrame

   Convert a JSON object to a pandas DataFrame and set the index to the given id column.

   :param json: A JSON object.
   :type json: dict
   :param id: The name of the column to set as the index. Default is 'id'.
   :type id: str

   :returns: A DataFrame representation of the JSON object.
   :rtype: pandas.DataFrame


.. py:function:: list_to_df(arr, id='id')

   Convert a list of dictionaries to a pandas DataFrame and set the index to the given id column.

   :param arr: A list of dictionaries.
   :type arr: list
   :param id: The name of the column to set as the index. Default is 'id'.
   :type id: str

   :returns: A DataFrame representation of the list of dictionaries.
   :rtype: pandas.DataFrame


.. py:function:: unique_list(list1)

   Return a list of unique values in the given list.

   :param list1: A list of values.
   :type list1: list

   :returns: A list of unique values in the input list.
   :rtype: list


.. py:function:: get_parents_dict(dic, value)

   Return a list of keys that lead to the given value in the given dictionary.

   :param dic: A dictionary to search.
   :type dic: dict
   :param value: The value to search for in the dictionary.

   :returns: A list of keys that lead to the given value in the dictionary.
   :rtype: list


.. py:function:: check_samples_similar(new_sample, last_sample, threshold)

   Checks if two samples are similar based on a given threshold.

   :param new_sample: The new sample to compare.
   :type new_sample: np.ndarray
   :param last_sample: The last sample to compare.
   :type last_sample: np.ndarray
   :param threshold: The threshold to use for comparison.
   :type threshold: float

   :returns: True if the samples are similar, False otherwise.
   :rtype: bool


.. py:function:: test_t(sample1, sample2, alpha=0.05)

   Performs a t-test and compares the p-value with a given alpha value.

   :param sample1: The first sample.
   :type sample1: np.ndarray
   :param sample2: The second sample.
   :type sample2: np.ndarray
   :param alpha: The alpha value to use for comparison. Defaults to 0.05.
   :type alpha: float, optional

   :returns: True if the samples are similar, False otherwise.
   :rtype: bool


.. py:function:: convert_nested_string_to_dict(s)

   Converts a string that contains a dictionary and JSON-formatted strings into a nested dictionary.

   :param s: The input string containing a dictionary and JSON-formatted strings.
   :type s: str

   :returns: The output dictionary after conversion of JSON-formatted strings.
   :rtype: dict


.. py:function:: find_key(nested_dict, target_key)

   Find a key in a nested dictionary.

   :param nested_dict: The dictionary to search.
   :type nested_dict: dict
   :param target_key: The key to find.
   :type target_key: str

   :returns: The value of the found key, or None if the key was not found.
   :rtype: value


.. py:function:: gen_dict_extract(key, var)

   A generator function to iterate and yield values from a dictionary or list nested inside the dictionary, given a key.

   :param key: The key to search for in the dictionary.
   :type key: str
   :param var: The dictionary or list to search.
   :type var: dict or list

   :Yields: *value* -- The value from the dictionary or list that corresponds to the given key.


.. py:function:: transform_stringified_dict(data)

   Recursively converts stringified JSON parts of a dictionary or list into actual dictionaries or lists.

   This function checks if an item is a string and attempts to convert it to a dictionary or list
   using `json.loads()`. If the conversion is successful, the function recursively processes the new
   dictionary or list. If a string is not a valid JSON representation, it remains unchanged.

   :param data: Input data that might contain stringified JSON parts.
   :type data: Union[dict, list, str]

   :returns: The transformed data with all stringified JSON parts converted
             to dictionaries or lists.
   :rtype: Union[dict, list, str]

   :raises json.JSONDecodeError: If there's an issue decoding a JSON string. This is caught internally
   :raises and the original string is returned.:


.. py:data:: FT2M
   :value: 0.3048

   Conversion factor from feet to meters.

.. py:data:: M2FT

   Conversion factor from meters to feet.

.. py:data:: IN2M
   :value: 0.0254

   Conversion factor from inches to meters.

.. py:data:: M2IN

   Conversion factor from meters to inches.

.. py:data:: NM2M
   :value: 1852.0

   Conversion factor from nautical miles to meters.

.. py:data:: M2NM

   Conversion factor from meters to nautical miles.

.. py:data:: NM2FT

   Conversion factor from nautical miles to feet.

.. py:data:: FT2NM

   Conversion factor from feet to nautical miles.

.. py:data:: SM2M
   :value: 1609.344

   Conversion factor from statute miles to meters.

.. py:data:: M2SM

   Conversion factor from meters to statute miles.

.. py:data:: SM2FT
   :value: 5280.0

   Conversion factor from statute miles to feet.

.. py:data:: FT2SM

   Conversion factor from feet to statute miles.

.. py:data:: KM2M
   :value: 1000.0

   Conversion factor from kilometers to meters.

.. py:data:: M2KM

   Conversion factor from meters to kilometers.

.. py:data:: CM2M
   :value: 0.01

   Conversion factor from centimeters to meters.

.. py:data:: M2CM

   Conversion factor from meters to centimeters.

.. py:data:: UM2M
   :value: 1e-06

   Conversion factor from micrometers to meters.

.. py:data:: M2UM

   Conversion factor from meters to micrometers.

.. py:function:: meters_to_micrometers(v)

   Convert meters to micrometers.

   :param v: Value in meters.
   :type v: float

   :returns: Value in micrometers.
   :rtype: float


.. py:function:: micrometers_to_meters(v)

   Convert micrometers to meters.

   :param v: Value in micrometers.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_centimeters(v)

   Convert meters to centimeters.

   :param v: Value in meters.
   :type v: float

   :returns: Value in centimeters.
   :rtype: float


.. py:function:: centimeters_to_meters(v)

   Convert centimeters to meters.

   :param v: Value in centimeters.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_kilometers(v)

   Convert meters to kilometers.

   :param v: Value in meters.
   :type v: float

   :returns: Value in kilometers.
   :rtype: float


.. py:function:: kilometers_to_meters(v)

   Convert kilometers to meters.

   :param v: Value in kilometers.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_inches(v)

   Convert meters to inches.

   :param v: Value in meters.
   :type v: float

   :returns: Value in inches.
   :rtype: float


.. py:function:: inches_to_meters(v)

   Convert inches to meters.

   :param v: Value in inches.
   :type v: float

   :returns: Value
   :rtype: float


.. py:function:: meters_to_feet(v)

   Converts a distance in meters to feet.

   :param v: distance in meters
   :type v: float

   :returns: distance in feet
   :rtype: float


.. py:function:: feet_to_meters(v)

   Converts a distance in feet to meters.

   :param v: distance in feet
   :type v: float

   :returns: distance in meters
   :rtype: float


.. py:function:: kilometers_to_nautical_miles(v)

   Converts a distance in kilometers to nautical miles.

   :param v: distance in kilometers
   :type v: float

   :returns: distance in nautical miles
   :rtype: float


.. py:function:: nautical_miles_to_kilometers(v)

   Converts a distance in nautical miles to kilometers.

   :param v: distance in nautical miles
   :type v: float

   :returns: distance in kilometers
   :rtype: float


.. py:function:: kilometers_to_statute_miles(v)

   Converts a distance in kilometers to statute miles.

   :param v: distance in kilometers
   :type v: float

   :returns: distance in statute miles
   :rtype: float


.. py:function:: statute_miles_to_kilometers(v)

   Converts a distance in statute miles to kilometers.

   :param v: distance in statute miles
   :type v: float

   :returns: distance in kilometers
   :rtype: float


.. py:function:: nautical_miles_to_statute_miles(v)

   Converts a distance in nautical miles to statute miles.

   :param v: distance in nautical miles
   :type v: float

   :returns: distance in statute miles
   :rtype: float


.. py:function:: statute_miles_to_nautical_miles(v)

   Converts a distance in statute miles to nautical miles.

   :param v: distance in statute miles
   :type v: float

   :returns: distance in nautical miles
   :rtype: float


.. py:data:: D2SC
   :value: 0.0055555555555556

   Conversion factor for converting degrees to semicircles.

.. py:data:: SC2D
   :value: 180.0

   Conversion factor for converting semicircles to degrees.

.. py:data:: R2SC
   :value: 0.3183098861837906

   Conversion factor for converting radians to semicircles.

.. py:data:: SC2R

   Conversion factor for converting semicircles to radians.

.. py:data:: R2DCC

   Conversion factor for converting radians to degrees.

.. py:data:: D2RCC

   Conversion factor for converting degrees to radians.

.. py:function:: degrees_to_radians(v)

   Converts a value from degrees to radians.

   :param v: The value in degrees to convert to radians.
   :type v: float

   :returns: The value in radians.
   :rtype: float


.. py:function:: degrees_to_semicircles(v)

   Converts a value from degrees to semicircles.

   :param v: The value in degrees to convert to semicircles.
   :type v: float

   :returns: The value in semicircles.
   :rtype: float


.. py:function:: radians_to_degrees(v)

   Converts a value from radians to degrees.

   :param v: The value in radians to convert to degrees.
   :type v: float

   :returns: The value in degrees.
   :rtype: float


.. py:function:: radians_to_semicircles(v)

   Converts a value from radians to semicircles.

   :param v: The value in radians to convert to semicircles.
   :type v: float

   :returns: The value in semicircles.
   :rtype: float


.. py:function:: semicircles_to_radians(v)

   Converts a value from semicircles to radians.

   :param v: The value in semicircles to convert to radians.
   :type v: float

   :returns: The value in radians.
   :rtype: float


.. py:function:: semicircles_to_degrees(v)

   Converts a value from semicircles to degrees.

   :param v: The value in semicircles to convert to degrees.
   :type v: float

   :returns: The value in degrees.
   :rtype: float


.. py:function:: aepcd_deg(x)

   The method aepcd_deg keeps an angle within the range -180.0 to 180.0 as presented in the figure bellow.
   In the example of this figure, the angle of 225 degrees is converted to -135 degrees through aepcd_deg.

   :param x: Angle in degrees.
   :type x: float

   :returns: The angle in degrees adjusted to lie within the range -180.0 to 180.0.
   :rtype: float


.. py:function:: aepcd_rad(x)

   Keeps an angle within the range -pi to pi, equivalent to the function aepcd_deg but in radians.

   :param x: float, the angle to be checked in radians.

   :returns: float, the angle within the range -pi to pi, with the same orientation as the original angle.


.. py:function:: alimd(x, limit)

   Limits the value of `x` to +/- `limit`.

   :param x: The value to be limited.
   :type x: float
   :param limit: The maximum absolute value allowed for `x`.
   :type limit: float

   :returns:

             The limited value of `x`. If `x` is greater than `limit`, returns `limit`.
                    If `x` is less than negative `limit`, returns `-limit`. Otherwise, returns `x`.
   :rtype: float


.. py:data:: earth_model_data
   :value: [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,...

   Data from Different Models of the Earth
   Data from 22 Earth surface models are stored in the array earthModelData.
   Each cell of this array corresponds to a type of model and has the following information: semi major axis (a), in meters; and flattening (f).

.. py:function:: gbd2ll(slat, slon, brg, dist, index_earth_model)

   This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

   This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - bearing:
   :type - bearing: brg
   :param - distance:
   :type - distance: dist) or ground range, in nautical miles (nm
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - latitude (dlat) and longitude (dlon) of the destination point.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21


.. py:function:: fbd2ll(slat, slon, brg, dist)

   This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

   This method considers the flat-earth projection and a spherical earth radius of 'ERAD60'. This method is similar to the method of the file nav_utils.inl of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - bearing:
   :type - bearing: brg
   :param - distance:
   :type - distance: dist) or ground range, in nautical miles (nm

   :returns:

             - latitude (dlat) and longitude (dlon) of the destination point.


.. py:function:: gll2bd(slat, slon, dlat, dlon, index_earth_model)

   This function computes the initial bearing and the distance from the starting point to the destination point.

   This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - latitude:
   :type - latitude: dlat) and longitude (dlon
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - bearing (brg), in degrees, between the starting and destination points; and

             - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21


.. py:function:: fll2bd(slat, slon, dlat, dlon)

   This function computes the initial bearing and the distance from the starting point to the destination point.

   This method considers a flat earth projection and a spherical earth radius of 'ERAD60'.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - latitude:
   :type - latitude: dlat) and longitude (dlon

   :returns:

             - bearing (brg), in degrees, between the starting and destination points; and

             - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.


.. py:function:: convert_ecef_to_geod(x, y, z, index_earth_model)

   This function converts Earth Centered, Earth Fixed (ECEF) coordinates (x,y,z) to geodetic coordinates (latitude,longitude,altitude).

   :param - ECEF coordinates:
   :type - ECEF coordinates: x,y,z
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - geodetic coordinates (lat, lon, alt), considering lat and lon in degrees, and alt in meters.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21



.. py:function:: convert_geod_to_ecef(lat, lon, alt, index_earth_model)

   This function converts Geodetic ((Latitude,Longitude,Altitude) coordinates) to ECEF ((X,Y,Z) coordinates).

   :param - geodetic coordinates:
   :type - geodetic coordinates: lat, lon, alt
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - ECEF coordinates (x,y,z), in meters.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21


.. py:function:: prepare_simulation_batch(sim: asaclient.Simulation) -> asaclient.Simulation

   Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run in batch mode.

   :param sim: The simulation instance for which the simulation setup needs to be prepared.
   :type sim: Simulation

   :returns: The updated Simulation instance with the added recorder configurations in its subcomponents.
   :rtype: Simulation


.. py:function:: prepare_simulation_tacview(sim: asaclient.Simulation) -> asaclient.Simulation

   Prepares a simulation by adding specific recorder configurations to the simulation's station subcomponents to run on Tacview.

   :param sim: The simulation instance for which the simulation setup needs to be prepared.
   :type sim: Simulation

   :returns: The updated Simulation instance with the added recorder configurations in its subcomponents.
   :rtype: Simulation


.. py:function:: load_simulation(path: str) -> asaclient.Simulation

   Loads a Simulation object from a JSON file.

   This method accepts a path to a JSON file, reads the content of the file and
   creates a Simulation object using the data parsed from the file.

   :param path: The absolute or relative path to the JSON file to be loaded.
   :type path: str

   :returns: The Simulation object created from the loaded JSON data.
   :rtype: Simulation


.. py:function:: json_to_df(self, json, id='id') -> pandas.DataFrame

   Convert a JSON object to a pandas DataFrame and set the index to the given id column.

   :param json: A JSON object.
   :type json: dict
   :param id: The name of the column to set as the index. Default is 'id'.
   :type id: str

   :returns: A DataFrame representation of the JSON object.
   :rtype: pandas.DataFrame


.. py:function:: list_to_df(arr, id='id')

   Convert a list of dictionaries to a pandas DataFrame and set the index to the given id column.

   :param arr: A list of dictionaries.
   :type arr: list
   :param id: The name of the column to set as the index. Default is 'id'.
   :type id: str

   :returns: A DataFrame representation of the list of dictionaries.
   :rtype: pandas.DataFrame


.. py:function:: unique_list(list1)

   Return a list of unique values in the given list.

   :param list1: A list of values.
   :type list1: list

   :returns: A list of unique values in the input list.
   :rtype: list


.. py:function:: get_parents_dict(dic, value)

   Return a list of keys that lead to the given value in the given dictionary.

   :param dic: A dictionary to search.
   :type dic: dict
   :param value: The value to search for in the dictionary.

   :returns: A list of keys that lead to the given value in the dictionary.
   :rtype: list


.. py:function:: check_samples_similar(new_sample, last_sample, threshold)

   Checks if two samples are similar based on a given threshold.

   :param new_sample: The new sample to compare.
   :type new_sample: np.ndarray
   :param last_sample: The last sample to compare.
   :type last_sample: np.ndarray
   :param threshold: The threshold to use for comparison.
   :type threshold: float

   :returns: True if the samples are similar, False otherwise.
   :rtype: bool


.. py:function:: test_t(sample1, sample2, alpha=0.05)

   Performs a t-test and compares the p-value with a given alpha value.

   :param sample1: The first sample.
   :type sample1: np.ndarray
   :param sample2: The second sample.
   :type sample2: np.ndarray
   :param alpha: The alpha value to use for comparison. Defaults to 0.05.
   :type alpha: float, optional

   :returns: True if the samples are similar, False otherwise.
   :rtype: bool


.. py:function:: convert_nested_string_to_dict(s)

   Converts a string that contains a dictionary and JSON-formatted strings into a nested dictionary.

   :param s: The input string containing a dictionary and JSON-formatted strings.
   :type s: str

   :returns: The output dictionary after conversion of JSON-formatted strings.
   :rtype: dict


.. py:function:: find_key(nested_dict, target_key)

   Find a key in a nested dictionary.

   :param nested_dict: The dictionary to search.
   :type nested_dict: dict
   :param target_key: The key to find.
   :type target_key: str

   :returns: The value of the found key, or None if the key was not found.
   :rtype: value


.. py:function:: gen_dict_extract(key, var)

   A generator function to iterate and yield values from a dictionary or list nested inside the dictionary, given a key.

   :param key: The key to search for in the dictionary.
   :type key: str
   :param var: The dictionary or list to search.
   :type var: dict or list

   :Yields: *value* -- The value from the dictionary or list that corresponds to the given key.


.. py:function:: transform_stringified_dict(data)

   Recursively converts stringified JSON parts of a dictionary or list into actual dictionaries or lists.

   This function checks if an item is a string and attempts to convert it to a dictionary or list
   using `json.loads()`. If the conversion is successful, the function recursively processes the new
   dictionary or list. If a string is not a valid JSON representation, it remains unchanged.

   :param data: Input data that might contain stringified JSON parts.
   :type data: Union[dict, list, str]

   :returns: The transformed data with all stringified JSON parts converted
             to dictionaries or lists.
   :rtype: Union[dict, list, str]

   :raises json.JSONDecodeError: If there's an issue decoding a JSON string. This is caught internally
   :raises and the original string is returned.:


.. py:data:: FT2M
   :value: 0.3048

   Conversion factor from feet to meters.

.. py:data:: M2FT

   Conversion factor from meters to feet.

.. py:data:: IN2M
   :value: 0.0254

   Conversion factor from inches to meters.

.. py:data:: M2IN

   Conversion factor from meters to inches.

.. py:data:: NM2M
   :value: 1852.0

   Conversion factor from nautical miles to meters.

.. py:data:: M2NM

   Conversion factor from meters to nautical miles.

.. py:data:: NM2FT

   Conversion factor from nautical miles to feet.

.. py:data:: FT2NM

   Conversion factor from feet to nautical miles.

.. py:data:: SM2M
   :value: 1609.344

   Conversion factor from statute miles to meters.

.. py:data:: M2SM

   Conversion factor from meters to statute miles.

.. py:data:: SM2FT
   :value: 5280.0

   Conversion factor from statute miles to feet.

.. py:data:: FT2SM

   Conversion factor from feet to statute miles.

.. py:data:: KM2M
   :value: 1000.0

   Conversion factor from kilometers to meters.

.. py:data:: M2KM

   Conversion factor from meters to kilometers.

.. py:data:: CM2M
   :value: 0.01

   Conversion factor from centimeters to meters.

.. py:data:: M2CM

   Conversion factor from meters to centimeters.

.. py:data:: UM2M
   :value: 1e-06

   Conversion factor from micrometers to meters.

.. py:data:: M2UM

   Conversion factor from meters to micrometers.

.. py:function:: meters_to_micrometers(v)

   Convert meters to micrometers.

   :param v: Value in meters.
   :type v: float

   :returns: Value in micrometers.
   :rtype: float


.. py:function:: micrometers_to_meters(v)

   Convert micrometers to meters.

   :param v: Value in micrometers.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_centimeters(v)

   Convert meters to centimeters.

   :param v: Value in meters.
   :type v: float

   :returns: Value in centimeters.
   :rtype: float


.. py:function:: centimeters_to_meters(v)

   Convert centimeters to meters.

   :param v: Value in centimeters.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_kilometers(v)

   Convert meters to kilometers.

   :param v: Value in meters.
   :type v: float

   :returns: Value in kilometers.
   :rtype: float


.. py:function:: kilometers_to_meters(v)

   Convert kilometers to meters.

   :param v: Value in kilometers.
   :type v: float

   :returns: Value in meters.
   :rtype: float


.. py:function:: meters_to_inches(v)

   Convert meters to inches.

   :param v: Value in meters.
   :type v: float

   :returns: Value in inches.
   :rtype: float


.. py:function:: inches_to_meters(v)

   Convert inches to meters.

   :param v: Value in inches.
   :type v: float

   :returns: Value
   :rtype: float


.. py:function:: meters_to_feet(v)

   Converts a distance in meters to feet.

   :param v: distance in meters
   :type v: float

   :returns: distance in feet
   :rtype: float


.. py:function:: feet_to_meters(v)

   Converts a distance in feet to meters.

   :param v: distance in feet
   :type v: float

   :returns: distance in meters
   :rtype: float


.. py:function:: kilometers_to_nautical_miles(v)

   Converts a distance in kilometers to nautical miles.

   :param v: distance in kilometers
   :type v: float

   :returns: distance in nautical miles
   :rtype: float


.. py:function:: nautical_miles_to_kilometers(v)

   Converts a distance in nautical miles to kilometers.

   :param v: distance in nautical miles
   :type v: float

   :returns: distance in kilometers
   :rtype: float


.. py:function:: kilometers_to_statute_miles(v)

   Converts a distance in kilometers to statute miles.

   :param v: distance in kilometers
   :type v: float

   :returns: distance in statute miles
   :rtype: float


.. py:function:: statute_miles_to_kilometers(v)

   Converts a distance in statute miles to kilometers.

   :param v: distance in statute miles
   :type v: float

   :returns: distance in kilometers
   :rtype: float


.. py:function:: nautical_miles_to_statute_miles(v)

   Converts a distance in nautical miles to statute miles.

   :param v: distance in nautical miles
   :type v: float

   :returns: distance in statute miles
   :rtype: float


.. py:function:: statute_miles_to_nautical_miles(v)

   Converts a distance in statute miles to nautical miles.

   :param v: distance in statute miles
   :type v: float

   :returns: distance in nautical miles
   :rtype: float


.. py:data:: D2SC
   :value: 0.0055555555555556

   Conversion factor for converting degrees to semicircles.

.. py:data:: SC2D
   :value: 180.0

   Conversion factor for converting semicircles to degrees.

.. py:data:: R2SC
   :value: 0.3183098861837906

   Conversion factor for converting radians to semicircles.

.. py:data:: SC2R

   Conversion factor for converting semicircles to radians.

.. py:data:: R2DCC

   Conversion factor for converting radians to degrees.

.. py:data:: D2RCC

   Conversion factor for converting degrees to radians.

.. py:function:: degrees_to_radians(v)

   Converts a value from degrees to radians.

   :param v: The value in degrees to convert to radians.
   :type v: float

   :returns: The value in radians.
   :rtype: float


.. py:function:: degrees_to_semicircles(v)

   Converts a value from degrees to semicircles.

   :param v: The value in degrees to convert to semicircles.
   :type v: float

   :returns: The value in semicircles.
   :rtype: float


.. py:function:: radians_to_degrees(v)

   Converts a value from radians to degrees.

   :param v: The value in radians to convert to degrees.
   :type v: float

   :returns: The value in degrees.
   :rtype: float


.. py:function:: radians_to_semicircles(v)

   Converts a value from radians to semicircles.

   :param v: The value in radians to convert to semicircles.
   :type v: float

   :returns: The value in semicircles.
   :rtype: float


.. py:function:: semicircles_to_radians(v)

   Converts a value from semicircles to radians.

   :param v: The value in semicircles to convert to radians.
   :type v: float

   :returns: The value in radians.
   :rtype: float


.. py:function:: semicircles_to_degrees(v)

   Converts a value from semicircles to degrees.

   :param v: The value in semicircles to convert to degrees.
   :type v: float

   :returns: The value in degrees.
   :rtype: float


.. py:function:: aepcd_deg(x)

   The method aepcd_deg keeps an angle within the range -180.0 to 180.0 as presented in the figure bellow.
   In the example of this figure, the angle of 225 degrees is converted to -135 degrees through aepcd_deg.

   :param x: Angle in degrees.
   :type x: float

   :returns: The angle in degrees adjusted to lie within the range -180.0 to 180.0.
   :rtype: float


.. py:function:: aepcd_rad(x)

   Keeps an angle within the range -pi to pi, equivalent to the function aepcd_deg but in radians.

   :param x: float, the angle to be checked in radians.

   :returns: float, the angle within the range -pi to pi, with the same orientation as the original angle.


.. py:function:: alimd(x, limit)

   Limits the value of `x` to +/- `limit`.

   :param x: The value to be limited.
   :type x: float
   :param limit: The maximum absolute value allowed for `x`.
   :type limit: float

   :returns:

             The limited value of `x`. If `x` is greater than `limit`, returns `limit`.
                    If `x` is less than negative `limit`, returns `-limit`. Otherwise, returns `x`.
   :rtype: float


.. py:data:: earth_model_data
   :value: [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,...

   Data from Different Models of the Earth
   Data from 22 Earth surface models are stored in the array earthModelData.
   Each cell of this array corresponds to a type of model and has the following information: semi major axis (a), in meters; and flattening (f).

.. py:function:: gbd2ll(slat, slon, brg, dist, index_earth_model)

   This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

   This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - bearing:
   :type - bearing: brg
   :param - distance:
   :type - distance: dist) or ground range, in nautical miles (nm
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - latitude (dlat) and longitude (dlon) of the destination point.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21


.. py:function:: fbd2ll(slat, slon, brg, dist)

   This function computes the destination (target) point from starting (ref) point given distance and initial bearing.

   This method considers the flat-earth projection and a spherical earth radius of 'ERAD60'. This method is similar to the method of the file nav_utils.inl of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - bearing:
   :type - bearing: brg
   :param - distance:
   :type - distance: dist) or ground range, in nautical miles (nm

   :returns:

             - latitude (dlat) and longitude (dlon) of the destination point.


.. py:function:: gll2bd(slat, slon, dlat, dlon, index_earth_model)

   This function computes the initial bearing and the distance from the starting point to the destination point.

   This method considers an elliptical earth model, and it is similar to the method of the file nav_utils.cpp of MIXR.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - latitude:
   :type - latitude: dlat) and longitude (dlon
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - bearing (brg), in degrees, between the starting and destination points; and

             - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21


.. py:function:: fll2bd(slat, slon, dlat, dlon)

   This function computes the initial bearing and the distance from the starting point to the destination point.

   This method considers a flat earth projection and a spherical earth radius of 'ERAD60'.

   :param - latitude:
   :type - latitude: slat) and longitude (slon
   :param - latitude:
   :type - latitude: dlat) and longitude (dlon

   :returns:

             - bearing (brg), in degrees, between the starting and destination points; and

             - distance (dist) or ground range, in nautical miles (nm), between the starting and destination points.


.. py:function:: convert_ecef_to_geod(x, y, z, index_earth_model)

   This function converts Earth Centered, Earth Fixed (ECEF) coordinates (x,y,z) to geodetic coordinates (latitude,longitude,altitude).

   :param - ECEF coordinates:
   :type - ECEF coordinates: x,y,z
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - geodetic coordinates (lat, lon, alt), considering lat and lon in degrees, and alt in meters.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21



.. py:function:: convert_geod_to_ecef(lat, lon, alt, index_earth_model)

   This function converts Geodetic ((Latitude,Longitude,Altitude) coordinates) to ECEF ((X,Y,Z) coordinates).

   :param - geodetic coordinates:
   :type - geodetic coordinates: lat, lon, alt
   :param - an index of an optional earth model (default: WGS-84 (indexEarthModel = 0)).

   :returns:

             - ECEF coordinates (x,y,z), in meters.

   .. note::

       possible values for indexEarthModel.

       wgs84 -> indexEarthModel = 0

       airy -> indexEarthModel = 1

       australianNational -> indexEarthModel = 2

       bessel1841 -> indexEarthModel = 3

       clark1866 -> indexEarthModel = 4

       clark1880 -> indexEarthModel = 5

       everest -> indexEarthModel = 6

       fischer1960 -> indexEarthModel = 7

       fischer1968 -> indexEarthModel = 8

       grs1967 -> indexEarthModel = 9

       grs1980 -> indexEarthModel = 10

       helmert1906 -> indexEarthModel = 11

       hough -> indexEarthModel = 12

       international -> indexEarthModel = 13

       kravosky -> indexEarthModel = 14

       modAiry -> indexEarthModel = 15

       modEverest -> indexEarthModel = 16

       modFischer -> indexEarthModel = 17

       southAmerican1969 -> indexEarthModel = 18

       wgs60 -> indexEarthModel = 19

       wgs66 -> indexEarthModel = 20

       wgs72 -> indexEarthModel = 21


