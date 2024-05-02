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



