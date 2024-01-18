import os
from itertools import combinations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
from pymoo.decomposition.asf import ASF
from scikit_posthocs import posthoc_conover
from scipy import stats
from scipy.stats import chi2_contingency, spearmanr, kendalltau
from tabulate import tabulate
import statsmodels.api as sm
import warnings
import math
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, r_regression, f_regression, mutual_info_regression, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy.stats import shapiro
from scipy.stats import levene
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
from itertools import combinations



class Analysis:
    """
    Class for performing Analysis on a DataFrame with simulation data.
    """

    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'IQR', thr: float = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers in a Pandas DataFrame using either Inter-Quartile Range (IQR) or Z-Score method.

        Args:
            df (pd.DataFrame): The input DataFrame containing numerical data.
            method (str): The method used for outlier detection, options are 'IQR' or 'zscore'. Default is 'IQR'.
            thr (float): The threshold value for Z-Score method. Default is 3.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                - The first DataFrame contains the index, column name, and values of the outliers.
                - The second DataFrame contains the outlier thresholds for each column.

        Raises:
            ValueError: If the method is not 'IQR' or 'zscore'.
        """

        # Validate method argument
        if method not in ['IQR', 'zscore']:
            raise ValueError("Method should be one of 'IQR' or 'zscore'.")

        # Select numerical columns from the DataFrame
        df_numeric = df.select_dtypes(include=[np.number])

        # Initialize lists to store outliers and thresholds
        outliers = []
        min_thresh_list = []
        max_thresh_list = []

        # Detect outliers using the IQR method
        if method == 'IQR':
            for col in df_numeric:
                # Compute Q1, Q3, and Inter-Quartile Range (IQR)
                Q1 = df_numeric[col].quantile(0.25)
                Q3 = df_numeric[col].quantile(0.75)
                IQR = Q3 - Q1

                # Compute minimum and maximum thresholds
                min_thresh = Q1 - 1.5 * IQR
                max_thresh = Q3 + 1.5 * IQR

                # Identify outliers in the column
                outliers_col = df_numeric[(df_numeric[col] < min_thresh) | (df_numeric[col] > max_thresh)]

                # Append the column's outliers to the overall outliers list
                outliers.extend([(idx, col, outliers_col.loc[idx, col]) for idx in outliers_col.index])

                # Store the thresholds for this column
                min_thresh_list.append(min_thresh)
                max_thresh_list.append(max_thresh)

        # Detect outliers using the Z-Score method
        elif method == 'zscore':
            for col in df_numeric:
                # Remove missing values before computing Z-Scores
                df_col_clean = df_numeric[col].dropna()
                z_scores = np.abs(stats.zscore(df_col_clean))

                # Identify indices of the column's outliers
                outliers_indices = df_col_clean.iloc[np.where(z_scores > thr)[0]].index

                # Append the column's outliers to the overall outliers list
                outliers.extend([(idx, col, df_numeric.loc[idx, col]) for idx in outliers_indices])

                # In the case of Z-Score, no explicit threshold, hence None
                min_thresh_list.append(None)
                max_thresh_list.append(None)

        # Convert the outliers list to a DataFrame
        outliers_df = pd.DataFrame(outliers, columns=['index', 'column', 'outlier_value'])

        # Convert the thresholds list to a DataFrame
        thresholds_df = pd.DataFrame({
            'column': df_numeric.columns.tolist(),
            'min_thresh': min_thresh_list,
            'max_thresh': max_thresh_list
        }).set_index('column')

        return outliers_df, thresholds_df

    @staticmethod
    def remove_outliers(df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, List[int]]:
        """
        Remove outliers from a Pandas DataFrame using the Interquartile Range (IQR) method.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            verbose (bool, optional): If True, print the number of lines removed. Defaults to False.
        
        Returns:
            Tuple[pd.DataFrame, List[int]]: DataFrame with the outliers removed, 
                                            List of indexes of the rows that were removed (unique indices).
        """
        # Use the detect_outliers method to find outliers.
        # This method returns a DataFrame of outliers and their thresholds.
        outliers, _ = Analysis.detect_outliers(df)
        
        # The 'index' column of the 'outliers' DataFrame contains the indices of the original DataFrame
        # where outliers were detected. Extract these indices and convert them to a list.
        outlier_indices = outliers['index'].values.tolist()

        # Convert to set to get unique indices. This is because a single row could have more than one outlier.
        unique_outlier_indices = list(set(outlier_indices))

        # Drop the rows in the original DataFrame that correspond to the outliers.
        # The drop method removes rows based on their index.
        # The reset_index method is used to reset the index of the DataFrame after dropping rows.
        df_cleaned = df.drop(unique_outlier_indices).reset_index(drop=True)

        # If the verbose flag is set to True, print the number of rows (outliers) removed.
        if verbose:
            # Count unique indices for an accurate number of removed rows.
            print(f"Removed {len(unique_outlier_indices)} rows.\n")

        # Return the cleaned DataFrame (with outliers removed) and the indices of the rows that were removed.
        return df_cleaned, unique_outlier_indices


    @staticmethod
    def cramer_v(df: pd.DataFrame, verbose: bool = False, save: bool = False, path: Optional[str] = None, format: str = 'png') -> pd.DataFrame:
        """
        Calculate Cramer's V statistic for categorical feature association in a DataFrame.

        This function takes a DataFrame and calculates Cramer's V, a measure of association between two 
        categorical variables, for all pairs of categorical columns. The result is a symmetric DataFrame where
        each cell [i, j] contains the Cramer's V value between column i and column j.

        Args:
            df: Input DataFrame containing categorical variables.
            verbose: If True, a heatmap of Cramer's V values is displayed. Default is False.
            save: If True, the resulting heatmap is saved to a file. Default is False.
            path: The directory path where the heatmap is to be saved, if `save` is True. Default is None.
            format: The file format to save the heatmap, if `save` is True. Default is 'png'.

        Returns:
            A DataFrame containing Cramer's V values for all pairs of categorical variables in df.

        Raises:
            ValueError: If `save` is True but `path` is None.
        """
        # Label encoding
        label_encoder = preprocessing.LabelEncoder()
        df_encoded = df.apply(label_encoder.fit_transform)

        # Calculating Cramer's V for each pair of columns
        n_cols = len(df_encoded.columns)
        cramers_v_matrix = np.eye(n_cols)
        for i in range(n_cols):
            for j in range(i+1, n_cols):
                contingency_table = pd.crosstab(df_encoded.iloc[:, i], df_encoded.iloc[:, j])
                chi2_stat = chi2_contingency(contingency_table)[0]
                n_observations = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n_observations * min(contingency_table.shape) - 1))
                cramers_v_matrix[i, j] = cramers_v_matrix[j, i] = cramers_v

        df_results = pd.DataFrame(cramers_v_matrix, columns=df_encoded.columns, index=df_encoded.columns)

        # Plotting the results if verbose is True
        if verbose or save:
            plt.figure(figsize=(12, 8))
            plt.title('Association (Cramér\'s V)', fontsize=14, fontweight="bold")
            sns.heatmap(df_results, vmin=0., vmax=1, linewidths=.1, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), annot=True)
            plt.yticks(rotation=0)

            # Save the figure if save is True
            if save:
                if path is None:
                    raise ValueError("path must be provided if save is set to True.")
                plt.savefig(f"{path}/association.{format}")

            # Show the plot if verbose is True
            if verbose:
                plt.show()

        return df_results
    
    
    def eda(self, df: pd.DataFrame, save: bool = False, path: Optional[str] = None, format: str = 'png') -> None:
        """
        Perform exploratory data analysis (EDA) on a pandas DataFrame.

        The function provides a summary of the DataFrame, showing class balance for categorical variables,
        and displaying histograms and boxplots with outlier information for numerical variables. If desired,
        the function can save these plots to a specified location.

        Args:
            df (pd.DataFrame): DataFrame to be analyzed.
            save (bool, optional): Whether to save the plots. Defaults to False.
            path (str, optional): Directory where the plots should be saved. If save is True, this parameter
                                  must be provided. Defaults to None.
            format (str, optional): The file format for saved plots. Acceptable formats include png, pdf, ps, eps and svg. 
                                    Defaults to 'png'.

        Raises:
            ValueError: If 'save' is set to True but 'path' is not specified.

        Returns:
            None
        """
        if save and path is None:
            raise ValueError("Must specify 'path' if 'save' is set to True.")
        
        # Separate categorical and numerical columns for different analysis
        df_cat = df.select_dtypes(exclude=["number"])
        df_num = df.select_dtypes(include=["number"])
        
        # Initialize an empty list to collect columns that are to be moved from df_num to df_cat
        cols_to_move = []

        # For each column in the numerical DataFrame df_num
        for col in df_num.columns:
            # Compute the number of unique values
            unique_values = df_num[col].nunique()

            # If the number of unique values is less than 20 and the ratio of unique values to total entries is less than 0.05
            # Append the column to the list cols_to_move
            if unique_values < 20 and unique_values / len(df_num) < 0.05:
                cols_to_move.append(col)

        # For each column in the list cols_to_move
        for col in cols_to_move:
            # Convert the numerical column in df_num to categorical and append it to df_cat
            df_cat[col] = pd.Categorical(df_num[col])

            # Drop the column from the numerical DataFrame df_num
            df_num.drop(columns=[col], inplace=True)

        # If the categorical DataFrame df_cat is not empty
        if not df_cat.empty:
            # Process categorical features: describe them, drop missing values, show associations, and plot histograms
            self._process_categorical(df_cat, save, path, format)
        
        # If the numerical DataFrame df_num is not empty
        if not df_num.empty:
            # Process numerical features: describe them, drop missing values, show correlations, plot histograms and boxplots
            self._process_numerical(df_num, save, path, format)
            

    def _process_categorical(self, df_cat, save, path, format):
        """
        Processes categorical features of a DataFrame. Moves columns from df_num to df_cat based on a condition. 
        Then, it summarizes, visualizes and saves the processed DataFrame.

        Args:
            df_cat (pd.DataFrame): Categorical DataFrame to be processed.
            df_num (pd.DataFrame): Numerical DataFrame to be processed.
            save (bool): Whether to save the generated plots.
            path (str): Path to save the plots.
            format (str): Format for the plot files.
        """
        # Indicates the feature type
        print("\nCategorial Features:\n")
        
        # Describe the DataFrame
        self._describe(df_cat)
        
        # Drop missing values
        df_cat.dropna(inplace=True)

        # Calculate and display Cramer's V statistic for the DataFrame
        self.cramer_v(df_cat, verbose=True, save=save, path=path, format=format)

        # Plot histograms for the DataFrame
        self._plot_histograms(df_cat, save, path, format)


    def _process_numerical(self, df_num, save, path, format):
        """
        Processes numerical features of a DataFrame. It summarizes, visualizes and saves the processed DataFrame.

        Args:
            df_num (pd.DataFrame): Numerical DataFrame to be processed.
            save (bool): Whether to save the generated plots.
            path (str): Path to save the plots.
            format (str): Format for the plot files.
        """
        # Indicates the feature type
        print("\nNumerical Features:\n")
        
        # Describe the DataFrame
        self._describe(df_num)
        
        # Drop missing values
        df_num.dropna(inplace=True)

        # Plot correlation heatmap for the DataFrame
        self._plot_correlation(df_num, save, path, format)
        
        # Plot histograms and boxplots for the DataFrame
        self._plot_histograms_boxplots(df_num, save, path, format)

        # Detect and print outlier information
        # Loop through each column in the DataFrame
        for column in df_num:
            outliers, df_thresh = self.detect_outliers(df_num[[column]])
            print(f"\nOutlier detection for variable '{column}':")
            print(f"\tQuantity: {len(outliers)} of {len(df_num[column])}.")
            print(f"\tMethod: Interquartile Range (IQR).")
            print(f"\tCriteria: Values smaller than {round(df_thresh['min_thresh'][column], 2)} or larger than {round(df_thresh['max_thresh'][column], 2)} were considered outliers.\n\n")
    
    def _describe(self, df):
        """
        Prints a summary of the DataFrame, including the count of NaN values.

        Args:
            df (pd.DataFrame): DataFrame to be summarized.
        """
        # Generate statistical summary for df
        desc = df.describe()
        
        # Count NaNs in each column and create a DataFrame
        nan_count = pd.DataFrame([df.isna().sum()], columns=desc.columns, index=['nan'])
        
        # Combine the statistical summary and NaN counts
        desc = pd.concat([nan_count, desc])
        
        # Print the final DataFrame as a table
        print(tabulate(desc, headers='keys', tablefmt='tab'))


    def _plot_histograms(self, df, save, path, format):
        """
        Plots histograms of all columns in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be plotted.
            save (bool): Whether to save the plots.
            path (str): Path to save the plots.
            format (str): Format for the plot files.
        """
        # Print the section title
        print("\nHistograms:\n")
        
        # Loop through each column in the DataFrame
        for column in df:
            # Plot a single histogram for each column
            self._plot_single_histogram(df, column, save, path, format)


    def _plot_single_histogram(self, df, column, save, path, format):
        """
        Plots a single histogram for a specific column in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be plotted.
            column (str): Column to be plotted.
            save (bool): Whether to save the plot.
            path (str): Path to save the plot.
            format (str): Format for the plot file.
        """
        # Prepare data for plotting
        x = df[column].value_counts().index.tolist()
        y = np.round(df[column].value_counts() / df[column].count() * 100, 2)

        # Initialize the figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot a bar graph
        graph = sns.barplot(x=x, y=y)

        # Add a horizontal line to the graph
        graph.axhline(100 / len(df[column].unique()))
        
        # Add title and labels to the plot
        plt.title(column, fontsize=18, fontweight="bold")
        plt.ylabel("Frequency(%)", fontsize=12)

        # Adjust x-axis labels if they are overlapping
        fig.canvas.draw()
        labels = [label.get_window_extent().width for label in ax.get_xticklabels()]
        if max(labels) > fig.get_size_inches()[0] / len(labels) * fig.dpi:
            ax.tick_params(axis='x', labelrotation=85)
        
        # Save the plot if save flag is true
        if save:
            name = os.path.join(path, f"{column}.{format}")
            plt.savefig(name)

        # Show the plot
        plt.show()


    def _plot_correlation(self, df, save, path, format):
        """
        Plots a correlation heatmap of a numerical DataFrame.

        Args:
            df (pd.DataFrame): Numerical DataFrame to be plotted.
            save (bool): Whether to save the plot.
            path (str): Path to save the plot.
            format (str): Format for the plot file.
        """
        # Print the heading "Correlation"
        print("\nCorrelation:\n")
        
        # Initialize the figure with specified size
        plt.figure(figsize=(12, 8))
        
        # Set the title of the plot
        plt.title('Correlation  (Pearson)', fontsize=18, fontweight="bold")
        
        # Plot the heatmap with seaborn's heatmap function
        sns.heatmap(df.corr(), linewidths=.1, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), annot=True)
        
        # Rotate the y-axis labels to be horizontal
        plt.yticks(rotation=0)
        
        # If 'save' flag is true, save the figure in the specified path with specified format
        if save:
            name = os.path.join(path, f"correlation.{format}")
            plt.savefig(name)
        
        # Display the plot
        plt.show()


    def _plot_histograms_boxplots(self, df, save, path, format):
        """
        Plots histograms and boxplots of all columns in a numerical DataFrame.

        Args:
            df (pd.DataFrame): Numerical DataFrame to be plotted.
            save (bool): Whether to save the plots.
            path (str): Path to save the plots.
            format (str): Format for the plot files.
        """
        # Print the section title
        print("\nHistograms and Boxplots:\n")
        
        # Loop through each column in the DataFrame
        for column in df:
            # Plot a single histogram and boxplot for each column
            self._plot_single_histogram_boxplot(df, column, save, path, format)


    def _plot_single_histogram_boxplot(self, df, column, save, path, format):
        """
        Plots a histogram and boxplot for a given column in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to be plotted.
            column (str): Column to be plotted.
            save (bool): If True, saves the plot.
            path (str): File path to save the plot.
            format (str): File format to save the plot.
        """
        # Create subplot for histogram and boxplot
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(12, 8))
        
        # Set plot title
        plt.title(column, fontsize=18, fontweight="bold")
        
        # Generate boxplot
        sns.boxplot(x=df[column], orient='h', ax=ax_box)
        ax_box.set(xlabel='')
        
        # Generate histogram
        sns.histplot(x=df[column], stat='percent', ax=ax_hist, kde=True)
        plt.ylabel("Frequency(%)", fontsize=10)
        
        # Save plot if 'save' is True
        if save:
            name = os.path.join(path, f"{column}.{format}")
            plt.savefig(name)
        
        # Display plot
        plt.show()


    @staticmethod
    def hypothesis(df: pd.DataFrame, alpha: float = 0.05, verbose: bool = False) -> pd.DataFrame:    
        """ Method that performs hypothesis testing

        Args:
            df : (Pandas DataFrame)
                Input data (must contain at least two distributions).
            alpha : (float)
                Significance level. Represents a cutoff value, a criterion that we set to reject or not H0. Default 0.05.
            verbose :  (bool, optional)
                Variable that defines whether or not to display detailed messages. Defaults to False.

        Raises:
            ValueError: Input variable is empty.
            ValueError: Input data must match at least two distributions.

        Returns:
            (Pandas DataFrame): Indicates which distributions are statistically similar.
        
        .. seealso::
        
            `pingouin.homoscedasticity <https://pingouin-stats.org/build/html/generated/pingouin.homoscedasticity.html#pingouin.homoscedasticity>`_: teste de igualdade de variância.

            `pingouin.normality <https://pingouin-stats.org/build/html/generated/pingouin.normality.html#pingouin.normality>`_: teste de normalidade.

            `scipy.stats.f_oneway <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html>`_: one-way ANOVA.

            `scipy.stats.tukey_hsd <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html>`_: teste HSD de Tukey para igualdade de médias.

            `scipy.stats.kruskal <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html>`_: teste H de Kruskal-Wallis para amostras independentes.
            
            `scikit_posthocs.posthoc_conover <https://scikit-posthocs.readthedocs.io/en/latest/tutorial.html>`_: teste de Conover.

        """

        warnings.filterwarnings('ignore')

        # Check if the input DataFrame is empty
        if df.empty:
            raise ValueError('Variável de entrada está vazia.')
	
	    # Check if the input data has at least two distributions
        if len(df.columns) < 2:
            raise ValueError('Input data must correspond to at least two distributions.')

        # Normality test
        normality = False
        if len(df) > 20:
            method = 'normaltest'
        else:
            method = 'shapiro'
        norm_df = pg.normality(df, method=method)
        if all(norm_df.normal):
            normality = True

        if verbose:
            print("Normality test")
            print(tabulate(norm_df, headers='keys', tablefmt='tab'))
            if normality:
                print('Conclusion: Distributions can be considered Gaussian (normal).\n')
            else:
                print('Conclusion: At least one distribution does not resemble a Gaussian (normal) distribution.\n')

        # Homoscedasticity test (if distributions have the same variance)
        var_df = pg.homoscedasticity(df)
        variance = var_df.equal_var.item()
        if verbose:
            print("Homoscedasticity test")
            print(tabulate(var_df, headers='keys', tablefmt='tab'))
            if variance:
                print('Conclusion: Distributions have statistically similar variances (homoscedasticity).\n')
            else:
                print('Conclusion: Distributions have statistically different variances (heteroscedasticity).\n')

        # Prepare the parameters for the statistical tests
        params_dict = {}
        params = ""
        for column in df:
            params_dict[column] = df[column].values.tolist()
            params = params + 'params_dict[\"' + column + '\"],'

        # Prepare the output DataFrame
        indexes = list(combinations(range(len(df.columns)), 2))
        for i in range(len(indexes)):
            indexes[i] = indexes[i] + (False,)
        output = pd.DataFrame(indexes, columns=['dist1', 'dist2', 'same?'])

        # If the variances are equal
        if variance:
            # If distributions are normal
            if normality:
                # ANOVA
                # H0: Two or more groups have the same population mean
                # p <= alpha: Reject H0
                # p > alpha: Accept H0
                statistic, p = eval(f"stats.f_oneway({params})")
                if verbose:
                        print("Teste de ANOVA")
                        print(f"statistic = {statistic}, pvalue = {p}")
                if p > alpha:
                    same = True
                    output.iloc[:, 2] = True
                    if verbose:
                        if same:
                            print('Conclusion: Statistically, the samples correspond to the same distribution (ANOVA).\n')

                else:
                    # Tukey's HSD test
                    res = eval(f"stats.tukey_hsd({params})")
                    if verbose:
                        print('Conclusion: Statistically, the samples are different distributions (ANOVA).\n')
                        print("Tukey test\n")
                        print(res,"\n")
                    for i in range(0, len(params_dict)):
                        for j in range(i, len(params_dict)):
                            if res.pvalue[i, j] < alpha:
                                same = False
                            else:
                                same = True
                            output.loc[(output.dist1 == i) & (output.dist2 == j), 'same?'] = same

                if verbose:
                    print(tabulate(output, headers='keys', tablefmt='tab'))
            # If distributions are not normal
            else:
                # Kruskal-Wallis test
                # H0: Two or more groups have the same population mean
                # p <= alpha: Reject H0
                # p > alpha: Accept H0
                statistic, p = eval(f"stats.kruskal({params})")
                if verbose:
                    print("Kruskal test")
                    print(f"statistic = {statistic}, pvalue = {p}")
                if p > alpha:
                    same = True
                    output.iloc[:, 2] = same
                    if verbose:
                        print('Conclusion: Statistically, the samples correspond to the same distribution (Kruskal-Wallis).\n')
                        print(tabulate(output, headers='keys', tablefmt='tab'))
                
                else:
                    # Conover's post hoc test
                    con = posthoc_conover(df.transpose().values.tolist())
                    if verbose:
                        print('Conclusion: Statistically, the samples correspond to different distributions (Kruskal-Wallis).\n')
                        
                        print("Conover test")
                        print(con,"\n")
                    for i in range(1, len(con) + 1):
                        for j in range(i + 1, len(con) + 1):
                            if con[i][ j] < alpha:
                                same = False
                            else:
                                same = True
                            output.loc[(output['dist1'] == i) & (output['dist2'] == j), 'same?'] = same
                    if verbose:
                        print(tabulate(output, headers='keys', tablefmt='tab'))

        # If the variances are different
        else:
            # Welch's ANOVA
            # H0: Two or more groups have the same population mean
            # p <= alpha: Reject H0
            # p > alpha: Accept H0
            normality = False
            col = df.columns.to_list()

            for i in range(0, len(col)):
                for j in range(i + 1, len(col)):
                    norm_df = pg.normality(df[[col[i], col[j]]], method=method)
                    if all(norm_df.normal):
                        normality = True

                    if normality:
                        aov = pg.welch_anova(dv=col[i], between=col[j], data=df)

                        if aov['p-unc'][0] > alpha:
                            same = True
                        else:
                            same = False

                        output.loc[(output['dist1'] == i) & (output['dist2'] == j), 'same?'] = same

            if verbose:
                print("One-Way Welch ANOVA test")
                print(tabulate(output, headers='keys', tablefmt='tab'))

        return output


    @staticmethod
    def fit_distribution(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Find the best fitting distribution for the input data.

        This function compares 93 available distributions in the scipy library and finds the one 
        that best fits the input data. The best fit is determined by the Kolmogorov-Smirnov test.

        Args:
            df (pd.DataFrame): Input data, which must contain only one distribution.
            verbose (bool, optional): Flag that controls whether detailed messages are displayed. Defaults to False.

        Raises:
            ValueError: Raised if the input data contains more than one distribution.

        Returns:
            pd.DataFrame: DataFrame with information about the distribution that best fits the input data, 
            as well as the most common distributions (``norm``, ``beta``, ``chi2``, ``uniform``, ``expon``). 
            The DataFrame's columns are: ``Distribution_Type``, ``P_Value``, ``Statistics``, and ``Parameters``.            
        """

        import warnings
        warnings.filterwarnings('ignore')
        
        # Ensure that the dataframe contains a single distribution.
        if len(df.columns) != 1:
            raise ValueError("Input data 'df' must correspond to only one distribution.")

        # List with 93 distribution objects available in the scipy package
        list_of_dists = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'burr12', 'cauchy',
                        'chi', 'chi2', 'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib',
                        'exponpow', 'f', 'fatiguelife', 'fisk', 'foldcauchy', 'foldnorm', 'genlogistic', 'genpareto',
                        'gennorm', 'genexpon', 'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic',
                        'gilbrat', 'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm',
                        'halfgennorm', 'hypsecant', 'invgamma', 'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu',
                        'kstwobign', 'laplace', 'levy', 'levy_l', 'logistic', 'loggamma', 'loglaplace', 'lognorm',
                        'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm', 'pareto', 'pearson3',
                        'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh', 'rice',
                        'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda',
                        'uniform', 'vonmises', 'vonmises_line', 'wald', 'weibull_min', 'weibull_max']

        # Initialize a list to store the results of fitting and testing each distribution
        results = []
        column = df.columns.tolist()[0]
        
        # Fit each distribution and perform a Kolmogorov-Smirnov test to compare it with the data
        for i in list_of_dists:
            dist = getattr(stats, i)
            try:
                param = dist.fit(df[column])
                a = stats.kstest(df[column], i, args=param)
                param = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, param))
                results.append((i, a[0], a[1], param))
            except Exception as e:
                print(f"Skipping distribution {i} due to error: {e}")

        # Convert the lists to DataFrame  
        result_df = pd.DataFrame.from_records(results)
        # Renaming the columns according to the respective outputs: 'Distribution_Type', 'Statistics' and 'P_Value'
        result_df = result_df.rename(columns={0: 'Distribution_Type', 1: 'Statistics', 2: 'P_Value', 3: 'Parameters'})
        # Filling the data of the Parameters column in the DF with P-Value data
        result_df = result_df.set_index('Distribution_Type').reset_index()
        # Sorting the consolidated Data Frame by P-Value
        result_df = result_df.sort_values(by=['P_Value'], ascending=False)
        # Generating a Data Frame with the Best fit by P-Value (this DF, of only one line will be added at the end of the previous DF)
        best_fit = result_df.drop(result_df.index.to_list()[1:], axis=0)
        # Filtering the initial Data Frame by pre-selected distributions
        distr_selec = result_df.loc[result_df['Distribution_Type'].isin(['norm', 'beta', 'chi2', 'uniform', 'expon'])]
        # Concatenating the two Dataframes (of pre-selected distributions and of the best result)
        dfs = [best_fit, distr_selec]
        result = pd.concat(dfs)
        # Setting the index to the distribution column and ordering the columns
        result.set_index('Distribution_Type', inplace=True)
        result = result[['P_Value', 'Statistics', 'Parameters']]
        # Limiting the decimal places of the P-Value
        result['P_Value'] = result['P_Value'].round(decimals=6)

        # If verbose, print the result DataFrame and plot the fitting distributions
        if verbose:
            # Display the result DataFrame in a tabular format.
            print(tabulate(result, headers='keys', tablefmt='tab'),"\n")

            # Create a figure to plot the distributions.
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get unique values and their counts from the DataFrame.
            unique, counts = np.unique(df[column].tolist(), return_counts=True)

            # Generate 1000 points between the minimum and maximum values in the DataFrame.
            x = np.linspace(unique[0], unique[-1], num=1000)

            # Loop through each distribution in the result DataFrame.
            for i in result.index.values:
                # Plot the distribution if its P-value is greater than 0.001.
                if result['P_Value'][i] > 0.001:
                    
                    # Format the parameters of the distribution into a string that can be evaluated.
                    param_string = str(result['Parameters'][i]).replace('(', '').replace(')', '')
                    dist_func = 'stats.' + i + '.pdf(x,' + param_string + ')'
                    y = eval(dist_func)

                    # Plot the probability density function of the distribution.
                    ax.plot(x, y, label=i)

            # Plot the original data as a normalized histogram.
            ax.bar(unique, counts / counts.sum(), label=column + ' data', color='lightblue')

            # Set the title and labels of the plot.
            plt.title('Fitting Data Distribution', fontweight='bold', fontsize=20)
            plt.ylabel('Frequency', fontsize=15)
            plt.xlabel('Value', fontsize=15)
            
            # Add a legend to the plot.
            plt.legend()

            # Display the plot.
            plt.show()
                
        return result

    @staticmethod
    def feature_score(df: pd.DataFrame, x: List[str], y: str, scoring_function: str, verbose: bool = False, save_path=None) -> pd.DataFrame:
        """
        Calculate the score of input features using the specified scoring function.

        This function applies a specified scoring function to evaluate the relevance of each input feature for the output
        variable in a given DataFrame. The supported scoring functions are those provided by sklearn's feature_selection module.

        Args:
            df (pd.DataFrame): The input data, where each column is a feature and each row is an observation.
            x (List[str]): A list of names of the input features in 'df'.
            y (str): The name of the output feature in 'df'.
            scoring_function (str): The name of the scoring function to be used. Should be one of the following:
                - 'r_regression'
                - 'f_regression'
                - 'mutual_info_regression'
                - 'chi2'
                - 'f_classif'
                - 'mutual_info_classif'
            verbose (bool, optional): Whether to print detailed output. Default is False.
            save_path (str, optional): Path to save the plotted figure. If not specified, the figure is simply shown.

        Raises:
            ValueError: If 'scoring_function' is not one of the supported scoring functions.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to an input feature, and the 'score' column contains
                the corresponding score. The DataFrame is sorted by score in descending order.
        """
        
        # Mapping of provided scoring function string to actual function
        s_function = {
            'r_regression': r_regression,
            'f_regression': f_regression,
            'mutual_info_regression': mutual_info_regression,
            'chi2': chi2,
            'f_classif': f_classif,
            'mutual_info_classif': mutual_info_classif
        }
        
        # Check if the provided scoring function is supported
        if scoring_function not in s_function:
            raise ValueError(f"Unsupported scoring function {scoring_function}. Supported functions: {', '.join(s_function.keys())}")
        
        # Initialize the feature selector
        fs = SelectKBest(score_func=s_function[scoring_function], k='all')
        
        # Fit the feature selector to the data
        fs.fit(df[x], df[y])
        
        # Create a DataFrame with the scores for each feature
        scores_df = pd.DataFrame({'feature': x, 'score': fs.scores_})
        
        # Sort the DataFrame by score
        scores_df = scores_df.sort_values(by='score', ascending=False)
        
        # Round scores to 2 decimal places
        scores_df['score'] = scores_df['score'].round(2)
        
        # If verbose is set to True, print the scores
        if verbose:
            print(tabulate(scores_df, headers='keys', tablefmt='plain'))
        
        # Plot the scores
        plt.figure(figsize=(12,8))
        sns.barplot(x='score', y='feature', data=scores_df, orient='h')
        plt.title('Feature Importance')
        plt.xlabel('Score')
        plt.ylabel('Feature')
        if save_path:
            # Check if the directory exists, if not, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            file_name = "feature_score.png"
            full_path = os.path.join(save_path, file_name)
            plt.savefig(full_path)
        else:
            plt.show()

        return scores_df

    
    @staticmethod
    def pareto_front(df, list_min=None, list_max=None, verbose=False, max_points=None):
        """
        Identifies the Pareto front of a DataFrame based on objectives to minimize and maximize.

        Args:
            df (pd.DataFrame): Input DataFrame containing the data.
            list_min (list of str, optional): List of variable names to minimize. Defaults to None.
            list_max (list of str, optional): List of variable names to maximize. Defaults to None.
            verbose (bool, optional): If True, displays detailed information. Defaults to False.
            max_points (int, optional): Maximum number of points to include in the Pareto front. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the Pareto optimal points.
        """
        if list_min is None:
            list_min = []

        if list_max is None:
            list_max = []

        pareto = []

        # Iterate over each row in the DataFrame
        for i, row in df.iterrows():
            dominated = False

            # Check if the current row is dominated by any other row
            for j, other_row in df.iterrows():
                if i != j:
                    is_dominated = True

                    # Check the minimization criteria
                    for var_min in list_min:
                        if row[var_min] < other_row[var_min]:
                            is_dominated = False
                            break

                    # Check the maximization criteria
                    for var_max in list_max:
                        if row[var_max] > other_row[var_max]:
                            is_dominated = False
                            break

                    if is_dominated:
                        dominated = True
                        break

            # If the current row is not dominated, add it to the Pareto front
            if not dominated:
                pareto.append(row)

                # Display detailed information if verbose is True
                if verbose:
                    print(f"Point {i} is Pareto optimal.")

        # Create a DataFrame from the Pareto front points
        pareto_df = pd.DataFrame(pareto)

        # If max_points is specified, you need to define a metric for sorting.
        # This could be based on a combination of list_min and list_max, or a specific domain logic.
        if max_points is not None:
            # This is a placeholder. Replace with the appropriate sorting logic.
            pareto_df = pareto_df.head(max_points)

        return pareto_df       


    @staticmethod
    def get_best_pareto_point(df, list_min=None, list_max=None, weights_min=None, weights_max=None, minimization_weight=0.5, verbose=False):
        """
        Determine the optimal Pareto point from the input DataFrame, considering specified variables and their weights.

        Args:
            df (pd.DataFrame): The input DataFrame.
            list_min (List[str], optional): A list of column names to minimize in the Pareto optimality calculation. Defaults to None.
            list_max (List[str], optional): A list of column names to maximize in the Pareto optimality calculation. Defaults to None.
            weights_min (List[float], optional): A list of weights defining the relative importance of each variable to minimize. Defaults to None.
            weights_max (List[float], optional): A list of weights defining the relative importance of each variable to maximize. Defaults to None.
            minimization_weight (float, optional): The global weight for the minimization part, between 0 and 1. Defaults to 0.5.
            verbose (bool, optional): Flag to display detailed messages. Defaults to False.

        Returns:
            pd.Series: A Pandas Series containing the best Pareto optimal point based on the specified variables and weights.

        Note:
            The function assumes that the input DataFrame contains only Pareto optimal points.
        """

        # Check if the length of weights and list_variable is the same for both minimization and maximization
        if list_min and weights_min and len(weights_min) != len(list_min):
            raise ValueError("weights_min and list_min must have the same length.")
        if list_max and weights_max and len(weights_max) != len(list_max):
            raise ValueError("weights_max and list_max must have the same length.")

        # Validate that the sum of weights is 1.0
        if list_min and weights_min and not math.isclose(sum(weights_min), 1.0):
            raise ValueError("The sum of weights_min must be 1.0.")
        if list_max and weights_max and not math.isclose(sum(weights_max), 1.0):
            raise ValueError("The sum of weights_max must be 1.0.")

        if list_min is None:
            list_min = []
        if list_max is None:
            list_max = []

        if weights_min is None and list_min:
            weights_min = [1] * len(list_min)
        if weights_max is None and list_max:
            weights_max = [1] * len(list_max)

        maximization_weight = 1 - minimization_weight

        # Calculate the weighted sum for minimization and maximization
        weighted_sum_min = df[list_min].mul(weights_min).sum(axis=1)
        weighted_sum_max = df[list_max].mul(weights_max).sum(axis=1)

        # Calculate the total weighted sum, applying the global weights
        total_weighted_sum = (-weighted_sum_min * minimization_weight) + (weighted_sum_max * maximization_weight)

        # Find the index of the row with the maximum total weighted sum
        max_index = total_weighted_sum.idxmax()

        # Get the row corresponding to the maximum total weighted sum
        best_pareto_point = df.loc[max_index]

        # Display detailed information if verbose is True
        if verbose:
            print("Best Pareto Optimal Point:")
            print(best_pareto_point)

        return best_pareto_point

    @staticmethod
    def anova(df: pd.DataFrame, 
              columns: List[str] = None, 
              alpha: float = 0.05, 
              show_plots: bool = True, 
              save_path: Optional[str] = None, 
              boxplot_title: str = 'Distributions of Samples', 
              boxplot_xlabel: str = 'Samples', 
              boxplot_ylabel: str = 'Value', 
              boxplot_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Perform ANOVA test on the given DataFrame columns and conduct Multiple pairwise comparisons (Post-hoc test)
        if more than two variables are being compared.

        Args:
            df (pd.DataFrame): DataFrame containing the samples.
            columns (List[str], optional): Columns to be analyzed. If None, all columns are used. Defaults to None.
            alpha (float, optional): Significance level. Defaults to 0.05.
            show_plots (bool, optional): If True, plots will be displayed for visual analysis. Defaults to True.
            save_path (Optional[str], optional): Path to save the generated plots and results. If None, the plots are 
                displayed and results are printed without saving. If provided, plots and results will be saved to the 
                specified path. Directory structure will be created if not exists. Defaults to None.
            boxplot_title (str, optional): Title for the box plots. Defaults to 'Distributions of Samples'.
            boxplot_xlabel (str, optional): Label for the X-axis of the box plot. Defaults to 'Samples'.
            boxplot_ylabel (str, optional): Label for the Y-axis of the box plot. Defaults to 'Value'.
            boxplot_names (Optional[List[str]], optional): Names for the box plots. Defaults to None.

        Returns:
            Tuple[pd.DataFrame, Optional[pd.DataFrame]]: ANOVA summary and optionally Post-hoc test results.
        """

        def custom_print(message: str):
            """Utility function to print and optionally save text."""
            print(message)
            if save_path:
                with open(f"{save_path}/anova_results.txt", "a") as file:
                    file.write(message + '\n')

        if df.empty or not columns:
            custom_print("DataFrame or columns are empty. Please provide valid data.")
            return

        # Create directory if save_path is provided and it doesn't exist
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        
        sub_df = df[columns].reset_index()
        sub_df.rename(columns={sub_df.columns[0]: 'index_col'}, inplace=True) 
        df_melt = pd.melt(sub_df, id_vars=['index_col'], value_vars=columns)
        df_melt.columns = ['index', 'samples', 'value']

        palette = sns.color_palette("coolwarm", n_colors=len(columns))

        if show_plots:
            plt.figure(figsize=(8, 6))
            
            # Use the custom names for boxplots if provided
            if boxplot_names:
                box_names = boxplot_names
                # Create a mapping dictionary
                name_mapping = dict(zip(columns, box_names))

                # Update the 'samples' column with the new names
                df_melt['samples'] = df_melt['samples'].map(name_mapping)
            else:
                box_names = columns

            boxprops = dict(linewidth=1.5)
            medianprops = dict(linestyle='-', linewidth=1.5, color="red")
            meanprops = dict(marker='^', markeredgecolor='black', markerfacecolor='green', markersize=8)

            ax = sns.boxplot(x='samples', y='value', data=df_melt, width=0.3,  
                            showmeans=True, meanprops=meanprops, boxprops=boxprops, medianprops=medianprops, palette="viridis")
            # Get the upper limit of the y-axis to position the annotations
            y_upper = ax.get_ylim()[1]
            offset_from_top = y_upper * 0.07  # Adjust this factor as necessary to set a suitable distance

            # Annotate median and mean values
            for i, column in enumerate(columns):
                # Median
                median_y = df[column].median()
                ax.annotate(f'Median: {median_y:.2f}', (i, y_upper + offset_from_top), ha='center', fontsize=8, color='red')
                # Mean
                mean_y = df[column].mean()
                ax.annotate(f'Mean: {mean_y:.2f}', (i, y_upper + offset_from_top * 1.5), ha='center', fontsize=8, color='green')

            # Adjust the y-axis limits to ensure annotations are visible
            ax.set_ylim(bottom=ax.get_ylim()[0], top=y_upper + offset_from_top * 3)

            plt.title(boxplot_title, fontsize=14)
            plt.xlabel(boxplot_xlabel, fontsize=12)
            plt.ylabel(boxplot_ylabel, fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            sns.despine()


            if save_path:
                plt.savefig(f"{save_path}/distribution_of_samples.png", bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
            plt.close()
                                                                                                                   

        if df.shape[1] < 2:
            custom_print("At least two columns are required for ANOVA.")
            return
        
        custom_print("Running ANOVA test\n")

        model = sm.formula.ols('value ~ C(samples)', data=df_melt).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        _, p_normality = shapiro(model.resid)
        if p_normality < alpha:
            custom_print("Residuals DO NOT follow a normal distribution. The ANOVA results may not be valid.\n")
        else:
            custom_print("Residuals DO follow a normal distribution.\n")

        levene_test = levene(*[df[col] for col in columns])
        if levene_test.pvalue < alpha:
            custom_print("Variances ARE NOT homogeneous across the groups. The result may not be valid.\n")
        else:
            custom_print("Variances ARE homogeneous across the groups.\n")

        sns.set_style("whitegrid")
        sns.set_palette("pastel")

        if show_plots:
            # QQ-plot of Residuals
            qq_fig = sm.qqplot(model.resid, line='45', fit=True)
            qq_fig.set_size_inches(8, 6)  # Ajusta o tamanho da figura
            plt.title('QQ-Plot of Residuals', fontsize=15)
            plt.xlabel("Theoretical Quantiles", fontsize=12)
            plt.ylabel("Standardized Residuals", fontsize=12)
            plt.gca().get_lines()[0].set_color('skyblue')
            plt.gca().get_lines()[1].set_color('red')

            if save_path:
                plt.savefig(f"{save_path}/qq_plot.png", bbox_inches='tight')

            plt.tight_layout()
            plt.show()
            plt.close()

            plt.show()
            plt.close()

            # Histogram of Residuals
            plt.figure(figsize=(8, 6))
            sns.histplot(model.resid, bins='auto', kde=True, color='lightblue', edgecolor='k')
            plt.title('Histogram of Residuals', fontsize=15)
            plt.xlabel("Residuals", fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            sns.despine(left=True)

            if save_path:
                plt.savefig(f"{save_path}/histogram.png", bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
            plt.close()
            plt.close()

            # Residual vs Fitted plot
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=model.fittedvalues, y=model.resid, color='lightblue', edgecolor='k')
            plt.axhline(0, color='red', linestyle='--')
            plt.title('Residuals vs Fitted Values', fontsize=15)
            plt.xlabel("Fitted Values", fontsize=12)
            plt.ylabel("Residuals", fontsize=12)
            sns.despine(left=True)

            if save_path:
                plt.savefig(f"{save_path}/residual_vs_fitted.png", bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
            plt.close()
            plt.close()

        custom_print("\nANOVA Summary:\n")
        custom_print(str(anova_table))

        p_value = anova_table['PR(>F)'].iloc[0]
        post_hoc_res = None
        if p_value < alpha:
            custom_print("The ANOVA test result is significant. There is a statistical difference among the samples.\n")
            post_hoc_res = pairwise_tukeyhsd(df_melt['value'], df_melt['samples']).summary()
            custom_print("\nPost-hoc (Tukey HSD) Test Results:\n")
            custom_print(str(post_hoc_res))
        else:
            custom_print("\nThe ANOVA test result is not significant. There is no statistical difference among the samples.\n")

        return anova_table, post_hoc_res

    @staticmethod
    def analyze_relationship(df, col1, col2, save_path=None):
        """
        Analyzes the relationship between two columns in a DataFrame.

        This function performs a series of analyses to understand the relationship
        between two numeric columns in the given DataFrame. It produces:
        - Descriptive statistics.
        - A scatter plot.
        - Pearson, Spearman, and Kendall correlations.
        - A correlation heatmap.
        - Linear regression and a regression plot.
        - A residual plot.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.
            col1 (str): The name of the first column.
            col2 (str): The name of the second column.
            save_path (str, optional): The directory where the results, plots, and analysis 
                will be saved. If not specified, results are just displayed.

        Returns:
            None. Displays or saves plots and textual analysis depending on `save_path`.

        Raises:
            ValueError: If the specified columns are not found in the DataFrame.
            TypeError: If input data is not in the expected format.
        """        
        def custom_print(message: str):
            """Utility function to print and optionally save text."""
            print(message)
            if save_path:
                with open(f"{save_path}/analysis_results.txt", "a") as file:
                    file.write(message + '\n')

        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define a custom color palette
        PALETTE = {
            'scatter': '#1f77b4',
            'edge': '#c7c7c7',
            'heatmap': 'coolwarm',
            'regression_line': '#d62728',
            'residual_line': '#2ca02c'
        }

        # Common styling for the plots
        FONT_SIZE = 10
        FIG_SIZE = (6, 4)

        # Descriptive Analysis
        description = df[[col1, col2]].describe().to_string()
        custom_print("Descriptive Statistics\n" + description + "\n")

        # Scatter Plot
        plt.figure(figsize=FIG_SIZE)
        sns.scatterplot(x=col1, y=col2, data=df, s=50, color=PALETTE['scatter'], edgecolor=PALETTE['edge'])
        plt.title(f'Scatter Plot: {col1} vs. {col2}', fontsize=FONT_SIZE + 2)
        plt.xlabel(col1, fontsize=FONT_SIZE)
        plt.ylabel(col2, fontsize=FONT_SIZE)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/scatter_plot.png", bbox_inches='tight')
        plt.show()

        # Correlation Analysis
        correlation = df[col1].corr(df[col2])
        spearman_corr, _ = spearmanr(df[col1], df[col2])
        kendall_corr, _ = kendalltau(df[col1], df[col2])
        custom_print(f"Pearson correlation between {col1} and {col2}: {correlation:.2f}")
        custom_print(f"Spearman correlation between {col1} and {col2}: {spearman_corr:.2f}")
        custom_print(f"Kendall correlation between {col1} and {col2}: {kendall_corr:.2f}\n")

        # Correlation Heatmap
        corr_matrix = df[[col1, col2]].corr()
        plt.figure(figsize=FIG_SIZE)
        sns.heatmap(corr_matrix, annot=True, cmap=PALETTE['heatmap'], fmt=".2f", annot_kws={'fontsize': FONT_SIZE})
        plt.title('Correlation Heatmap', fontsize=FONT_SIZE + 2)
        plt.xticks(fontsize=FONT_SIZE)
        plt.yticks(fontsize=FONT_SIZE)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/correlation_heatmap.png", bbox_inches='tight')
        plt.show()

        # Linear Regression
        X = df[[col1]].values
        y = df[col2].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Regression metrics
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        custom_print(f'Mean Squared Error: {mse:.2f}')
        custom_print(f'R² Score: {r2:.2f}\n')

        # Linear Regression Plot
        plt.figure(figsize=FIG_SIZE)
        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'s':50, 'color':PALETTE['scatter'], 'edgecolor':PALETTE['edge']}, line_kws={'color':PALETTE['regression_line']})
        plt.title(f'Regression: {col1} vs. {col2}', fontsize=FONT_SIZE + 2)
        plt.xlabel(col1, fontsize=FONT_SIZE)
        plt.ylabel(col2, fontsize=FONT_SIZE)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/linear_regression_plot.png", bbox_inches='tight')
        plt.show()

        # Residual Plot
        residuals = y - y_pred
        plt.figure(figsize=FIG_SIZE)
        sns.residplot(x=y_pred, y=residuals, lowess=True, color=PALETTE['scatter'], scatter_kws={'s':50, 'edgecolor':PALETTE['edge']}, line_kws={'color':PALETTE['residual_line'], 'lw':1.5})
        plt.title('Residual Plot', fontsize=FONT_SIZE + 2)
        plt.xlabel('Fitted values', fontsize=FONT_SIZE)
        plt.ylabel('Residuals', fontsize=FONT_SIZE)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/residual_plot.png", bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_histograms(df, columns, figsize=(15, 5), alpha=0.7, save_path=None):
        """
        Plots histograms for the given columns in the DataFrame.
        
        Parameters:
            df (DataFrame): The DataFrame containing the data.
            columns (list): List of column names to plot.
            figsize (tuple): Size of the figure for each row of histograms. Default is (15, 5).
            alpha (float): Alpha value for the histograms. Default is 0.7.
            save_path (str, optional): Path to save the plotted figure. If not specified, the figure is simply shown.
        """
        colors = sns.color_palette("viridis", len(columns))
        
        for i in range(0, len(columns), 3):
            cols_to_plot = columns[i:i+3]
            fig, ax = plt.subplots(1, len(cols_to_plot), figsize=figsize)
            axes = ax if isinstance(ax, np.ndarray) else [ax]  
                      
            for j, col in enumerate(cols_to_plot):
                data = df[col]
                
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(data):
                    # Check if all values are integers
                    if np.issubdtype(data.dtype, np.integer):
                        bins = np.linspace(data.min() - 0.5, data.max() + 0.5, data.nunique() + 1)
                        shrink = 0.3
                    else:
                        shrink = 0.8
                        bins = 'auto'
                    
                    sns.histplot(data, bins=bins, color=colors[i+j], kde=True, ax=axes[j], shrink=shrink)
                    mean_value = data.mean()
                    median_value = data.median()
                    axes[j].axvline(mean_value, color='g', linestyle='--', label=f'Mean: {mean_value:.2f}')
                    axes[j].axvline(median_value, color='r', linestyle='-', label=f'Median: {median_value:.2f}')
                    axes[j].set_title(col)
                    axes[j].set_xlabel('Value')
                    axes[j].set_ylabel('Frequency')
                    axes[j].grid(True, linestyle='--')
                    axes[j].legend()
                else:
                    axes[j].text(0.5, 0.5, 'Non-numeric data', ha='center', va='center')
                    axes[j].set_title(f'{col} (Non-numeric)')
            
            plt.tight_layout()
            if save_path:
                # Check if the directory exists, if not, create it
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                file_name = f"{i//3}.png"  # Name the plot files based on their order
                full_path = os.path.join(save_path, file_name)
                plt.savefig(full_path)
            else:
                plt.show()

    @staticmethod
    def bootstrap(dataframe, 
                  columns, 
                  n_iterations=1000, 
                  alpha=0.05, 
                  show_plots=False,
                  boxplot_xlabel: str = 'Samples', 
                  boxplot_ylabel: str = 'Value',
                  boxplot_names: Optional[List[str]] = None):
        """
        Perform bootstrap hypothesis tests to determine if the mean of one sample 
        is statistically greater or lesser than the other for each pair of columns 
        in the provided list and optionally visualize the distributions with box plots.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the samples.
            columns (list): List of column names to be compared.
            n_iterations (int, optional): Number of bootstrap iterations. Default is 1,000.
            alpha (float, optional): Significance level. Default is 0.05.
            show_plots (bool, optional): Flag to display plots. Default is False.
            boxplot_xlabel (str, optional): Label for the X-axis of the box plot. Default is 'Samples'.
            boxplot_ylabel (str, optional): Label for the Y-axis of the box plot. Default is 'Value'.
            boxplot_names (Optional[List[str]], optional): Names for the box plots. Default is None.

        Returns:
            None: Prints the test outcome for each pair and optionally displays a box plot.
        """
        if boxplot_names:
            box_names = boxplot_names
            # Create a mapping dictionary
            name_mapping = dict(zip(columns, box_names))
        else:
            name_mapping = dict(zip(columns, columns))

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1, col2 = columns[i], columns[j]

                # Original observed difference
                observed_difference = dataframe[col1].mean() - dataframe[col2].mean()

                combined_data = np.concatenate([dataframe[col1], dataframe[col2]])
                bootstrap_differences = []

                for _ in range(n_iterations):
                    np.random.shuffle(combined_data)
                    new_sample_1 = combined_data[:len(dataframe[col1])]
                    new_sample_2 = combined_data[len(dataframe[col1]):]
                    bootstrap_difference = new_sample_1.mean() - new_sample_2.mean()
                    bootstrap_differences.append(bootstrap_difference)

                # Null hypothesis adjustment
                adjusted_bootstrap_differences = np.array(bootstrap_differences) - np.mean(bootstrap_differences)

                p_value = np.mean(np.abs(adjusted_bootstrap_differences) >= np.abs(observed_difference))

                result_header = f"Bootstrap Hypothesis Test Results ({name_mapping[col1]} vs {name_mapping[col2]})"
                separator = "-" * len(result_header)
                print(separator)
                print(result_header)
                print(separator)
                print(f"Observed difference in means: {observed_difference:.4f}")
                print(f"Adjusted Bootstrap p-value: {p_value:.4f}")

                if p_value < alpha:
                    if observed_difference > 0:
                        conclusion = f"Sample '{name_mapping[col1]}' is statistically greater than sample '{name_mapping[col2]}' (p = {p_value:.4f})."
                    else:
                        conclusion = f"Sample '{name_mapping[col1]}' is statistically lesser than sample '{name_mapping[col2]}' (p = {p_value:.4f})."
                else:
                    conclusion = "There's no sufficient evidence to claim one sample is statistically greater or lesser than the other."

                print(conclusion)
                print(separator)
                print()

                # Plotting
                if show_plots:
                    plt.figure(figsize=(8, 6))
                    
                    # Preparing data for plotting
                    df_melt = pd.melt(dataframe[[col1, col2]], var_name='samples', value_name='value')

                    # Use the custom names for boxplots if provided
                    if boxplot_names:
                        # Update the 'samples' column with the new names
                        df_melt['samples'] = df_melt['samples'].map(name_mapping)
                    else:
                        box_names = columns

                    boxprops = dict(linewidth=1.5)
                    medianprops = dict(linestyle='-', linewidth=1.5, color="red")
                    meanprops = dict(marker='^', markeredgecolor='black', markerfacecolor='green', markersize=8)

                    ax = sns.boxplot(x='samples', y='value', data=df_melt, width=0.3,  
                                    showmeans=True, meanprops=meanprops, boxprops=boxprops, medianprops=medianprops, palette="viridis")

                    y_upper = ax.get_ylim()[1]
                    offset_from_top = y_upper * 0.07

                    columns_pair = [col1, col2]
                    for k, column in enumerate(columns_pair):
                        median_y = dataframe[column].median()
                        ax.annotate(f'Median: {median_y:.2f}', (k, y_upper + offset_from_top), ha='center', fontsize=8, color='red')
                        mean_y = dataframe[column].mean()
                        ax.annotate(f'Mean: {mean_y:.2f}', (k, y_upper + offset_from_top * 1.5), ha='center', fontsize=8, color='green')

                    ax.set_ylim(bottom=ax.get_ylim()[0], top=y_upper + offset_from_top * 3)
                    plt.title(f"Distributions of Samples '{name_mapping[col1]}' and '{name_mapping[col2]}'", fontsize=14)
                    plt.xlabel(boxplot_xlabel, fontsize=12)
                    plt.ylabel(boxplot_ylabel, fontsize=12)
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    sns.despine()
                    plt.show()


    
    @staticmethod
    def create_2d_scatter_plot(df, x_col, y_col, size_col, title='2D Scatter Plot', 
                            xlabel='X-axis', ylabel='Y-axis', size_label='Size', 
                            cmap='coolwarm', figsize=(12, 8), alpha=0.5, grid=True, ref_size_value=0.5):
        """
        Create a 2D scatter plot with variable circle sizes.

        Args:
            df (DataFrame): DataFrame containing the data.
            x_col (str): Name of the column in df for the x-axis.
            y_col (str): Name of the column in df for the y-axis.
            size_col (str): Name of the column in df for determining the size of the scatter points.
            title (str, optional): Title of the plot. Defaults to '2D Scatter Plot'.
            xlabel (str, optional): Label for the x-axis. Defaults to 'X-axis'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Y-axis'.
            size_label (str, optional): Label for the size legend. Defaults to 'Size'.
            cmap (str, optional): Colormap for the scatter points. Defaults to 'coolwarm'.
            figsize (tuple, optional): Size of the figure. Defaults to (12, 8).
            alpha (float, optional): Alpha blending value for the scatter points, between 0 and 1. Defaults to 0.5.
            grid (bool, optional): Flag to add grid to the plot. Defaults to True.
            ref_size_value (float, optional): Value for calculating the reference size circle. Defaults to 0.5.

        Returns:
            None: The function creates a matplotlib scatter plot and does not return any value.

        Example:
            create_2d_scatter_plot(df=my_dataframe, 
                                x_col='speed', 
                                y_col='altitude', 
                                size_col='fuel_consumed',
                                title='Flight Characteristics', 
                                xlabel='Speed (knots)', 
                                ylabel='Altitude (feet)', 
                                size_label='Fuel Consumed (normalized)')
        """
        # Normalizing the size column data for circle size
        normalized_size = df[size_col] / df[size_col].max()
        point_size = np.log1p(normalized_size) * 100  # Logarithmic scale for size, adjusted by a factor

        plt.figure(figsize=figsize)

        # Creating the scatter plot
        sc = plt.scatter(df[x_col], df[y_col], c=normalized_size, s=point_size, cmap=cmap, alpha=alpha)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.colorbar(sc, label=size_label)

        # Adding grid, minor ticks and a reference size circle for scale
        if grid:
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.minorticks_on()

        # Reference size circle
        ref_size = np.log1p(ref_size_value) * 100  # Logarithmic scale for reference size
        plt.scatter([], [], c='k', alpha=0.5, s=ref_size, label=f'Reference Size for {int(ref_size_value * 100)}% of Max {size_label}')
        plt.legend(scatterpoints=1, frameon=False, labelspacing=1, loc='lower right', bbox_to_anchor=(1, -0.01))

        plt.show()

    @staticmethod
    def create_3d_surface_plot(df, x_col, y_col, z_col, title='3D Surface Plot', 
                            xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis', 
                            cmap=cm.coolwarm, figsize=(16, 12), elev=30, azim=45):
        """
        Create a 3D surface plot from three columns in a DataFrame.

        Args:
            df (DataFrame): DataFrame containing the data.
            x_col (str): Name of the column in df for the x-axis.
            y_col (str): Name of the column in df for the y-axis.
            z_col (str): Name of the column in df for the z-axis (surface height).
            title (str, optional): Title of the plot. Defaults to '3D Surface Plot'.
            xlabel (str, optional): Label for the x-axis. Defaults to 'X-axis'.
            ylabel (str, optional): Label for the y-axis. Defaults to 'Y-axis'.
            zlabel (str, optional): Label for the z-axis. Defaults to 'Z-axis'.
            cmap (Colormap, optional): Colormap for the surface plot. Defaults to cm.coolwarm.
            figsize (tuple, optional): Size of the figure. Defaults to (16, 12).
            elev (int, optional): Elevation angle in the z plane for the 3D plot. Defaults to 30.
            azim (int, optional): Azimuth angle in the x,y plane for the 3D plot. Defaults to 45.

        Returns:
            None: The function creates a matplotlib 3D surface plot and does not return any value.
        """
        # Data for the plot
        x = df[x_col]
        y = df[y_col]
        z = df[z_col]

        # Creating grid data for the surface plot
        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # Plotting the surface
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(xi, yi, zi, cmap=cmap, linewidth=0, antialiased=True)

        # Setting labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel, labelpad=1)
        plt.title(title)

        # Enhancing the view
        ax.view_init(elev=elev, azim=azim)

        # Adding a color bar
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
        cbar.set_label(zlabel)

        plt.show()

            









