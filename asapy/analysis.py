import numpy as np
import pandas as pd
from scipy import stats
import pingouin as pg
import os 
import sys
import seaborn as sns
#import pypref as p
import re

from scipy.stats import chi2_contingency
from matplotlib import pyplot as plt
from sklearn import preprocessing
from itertools import combinations
from scikit_posthocs import posthoc_conover
from tabulate import tabulate
from pymoo.decomposition.asf import ASF
from sklearn.feature_selection import SelectKBest, r_regression, f_regression, mutual_info_regression, chi2, f_classif, \
    mutual_info_classif

"""ASA Analysis module."""

class Analysis:
    """The Analysis object."""

    @staticmethod
    def hypothesis(df, alpha = 0.05, verbose=False):    
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

        """

        import warnings
        warnings.filterwarnings('ignore')
        # Checar se DataFrame de entrada está vazio
        if df.empty:
            raise ValueError('Variável de entrada está vazia.')

        if len(df.columns) < 2:
            raise ValueError('Dados de entrada têm de corresponder a, no mínimo, duas distribições')

        # teste de normalidade
        normality = False
        if len(df) > 20:
            method = 'normaltest'
        else:
            method = 'shapiro'
        norm_df = pg.normality(df, method=method)
        if all(norm_df.normal):
            normality = True

        if verbose:
            print("Teste de normalidade")
            print(tabulate(norm_df, headers='keys', tablefmt='tab'))
            if normality:
                print('Conclusão: Distribuições podem ser consideradas gaussianas (normais).\n')
            else:
                print('Conclusão: Ao menos uma distribuição não se assemelha à gaussiana (normal).\n')

        # Teste de homoscedasticidade (se as distribuições têm a mesma variância)
        var_df = pg.homoscedasticity(df)
        variance = var_df.equal_var.item()
        if verbose:
            if verbose:
                print("Teste de homocedasticidade")
                print(tabulate(var_df, headers='keys', tablefmt='tab'))
                if variance:
                    print('Conclusão: Distribuições possuem variâncias estatisticamente SEMELHANTES (homoscedasticidade).\n')
                else:
                    print('Conclusão: Distribuições possuem variâncias estatisticamente DIFERENTES (heteroscedasticidade).\n')

        params_dict = {}
        params = ""
        for column in df:
            params_dict[column] = df[column].values.tolist()
            params = params + 'params_dict[\"' + column + '\"],'

        # Criar arquivo de saída
        indexes = list(combinations(range(len(df.columns)), 2))
        for i in range(len(indexes)):
            indexes[i] = indexes[i] + (False,)
        output = pd.DataFrame(indexes, columns=['dist1', 'dist2', 'same?'])

        # Mesma variância
        if variance:
            # Se for normal
            if normality:
                # ANOVA
                # H0: dois ou mais grupos têm a mesma média populacional
                # p <= alpha: rejeita H0
                # p > alpha: aceita H0
                statistic, p = eval(f"stats.f_oneway({params})")
                if verbose:
                        print("Teste de ANOVA")
                        print(f"statistic = {statistic}, pvalue = {p}")
                if p > alpha:
                    same = True
                    output.iloc[:, 2] = True
                    if verbose:
                        if same:
                            print('Conclusão: Estatisticamente as amostras correspondem à MESMA distribuição (ANOVA).\n')                           

                else:
                    # Tukey
                    res = eval(f"stats.tukey_hsd({params})")
                    if verbose:
                        print('Conclusão: Estatisticamente as amostras são distribuições DIFERENTES (ANOVA).\n')
                        print("Teste de Tukey\n")
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
            # Se não for normal   
            else:
                # Kruskal
                # H0: dois ou mais grupos têm a mesma média populacional
                # p <= alpha: rejeita H0
                # p > alpha: aceita H0
                statistic, p = eval(f"stats.kruskal({params})")
                if verbose:
                        print("Teste de Kruskal")
                        print(f"statistic = {statistic}, pvalue = {p}")
                if p > alpha:
                    same = True
                    output.iloc[:, 2] = same
                    if verbose:
                        print('Conclusão: Estatisticamente as amostras correspondem à MESMA distribuição (Kruskal-Wallis).\n')
                        print(tabulate(output, headers='keys', tablefmt='tab'))
                
                else:
                    # Teste de Conover
                    con = posthoc_conover(df.transpose().values.tolist())
                    if verbose:
                        print('Conclusão: Estatisticamente as amostras correspondem a distribuições DIFERENTES (Kruskal-Wallis).\n')
                        
                        print("Teste de Conover")
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

        # Variâncias diferentes
        else:
            # One-Way Welch ANOVA
            # Variância diferente e distribuição normal
            # p <= alpha: rejeita H0
            # p > alpha: aceita H0
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
                print("Teste One-Way Welch ANOVA")
                print(tabulate(output, headers='keys', tablefmt='tab'))

        return output

    @staticmethod
    def fit_distribution(df, verbose=False):
        """Find the distribution that best fits the input data.

        Args:
            df (Pandas DataFrame): Input data (must contain only one distribution).
            verbose (bool, optional): Flag that controls whether detailed messages are displayed. Defaults to False.

        Raises:
            ValueError: Input data must contain only one distribution.
            
        Returns:
            (Pandas DataFrame): DataFrame containing information about the distribution that best fit the input data, as well as the most common distributions (``norm``, ``beta``, ``chi2``, ``uniform``, ``expon``). The columns of the DataFrame are: ``Distribution_Type``, ``P_Value``, ``Statistics``, and ``Parameters``.

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
        """


        import warnings
        warnings.filterwarnings('ignore')
        if len(df.columns) != 1:
            raise ValueError("Os dados de entrada 'df' têm de corresponder a apenas uma distribição.")

        # Lista com 93 objetos de distribuições disponíveis no pacote scipy
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

        # Obtendos os dados estatísticos e P-Value das distribuições
        results = []
        column = df.columns.tolist()[0]
        for i in list_of_dists:
            dist = getattr(stats, i)
            param = dist.fit(df[column])
            a = stats.kstest(df[column], i, args=param)
            param = tuple(map(lambda x: isinstance(x, float) and round(x, 2) or x, param))
            results.append((i, a[0], a[1], param))

        # Convertendo as listas em DataFrame  
        result_df = pd.DataFrame.from_records(results)
        # Renomeando as colunas conforme as respectivas saídas: 'Distribution_Type', 'Statistics' e 'P_Value'
        result_df = result_df.rename(columns={0: 'Distribution_Type', 1: 'Statistics', 2: 'P_Value', 3: 'Parameters'})
        # Preenchendo os dados da coluna Parameters no DF com dados do P-Value
        result_df = result_df.set_index('Distribution_Type').reset_index()
        # Ordenando o Data Frame consolidado pelo P-Value
        result_df = result_df.sort_values(by=['P_Value'], ascending=False)
        # Gerando um Data Frame com o Melhor fit pelo P-Value (este Df, de aenas uma linha será adicionado no final do DF anterior)
        best_fit = result_df.drop(result_df.index.to_list()[1:], axis=0)
        # Filtando o Data Frame inicial pelas distribuições pré-selecionadas
        distr_selec = result_df.loc[result_df['Distribution_Type'].isin(['norm', 'beta', 'chi2', 'uniform', 'expon'])]
        # Concatenando os dois Dataframes (das distribuições pré-selecionadas e do melhor resultado)
        dfs = [best_fit, distr_selec]
        result = pd.concat(dfs)
        # Setando o index para a coluna da distribuição e ordenando as colunas
        result.set_index('Distribution_Type', inplace=True)
        result = result[['P_Value', 'Statistics', 'Parameters']]
        # Limitando as casas decimais do P-Value
        result['P_Value'] = result['P_Value'].round(decimals=6)

        if verbose:
            print(tabulate(result, headers='keys', tablefmt='tab'),"\n")
            fig = plt.subplots(figsize=(12, 8))
            unique, counts = np.unique(df[column].tolist(), return_counts=True)
            x = np.linspace(unique[0], unique[-1], num=1000)
            for i in result.index.values:
                if result['P_Value'][i] > 0.001:
                    s = str(result['Parameters'][i])
                    s = re.sub('[()]', '', s)
                    s = 'stats.' + i + '.pdf(x,' + s + ')'
                    y = eval(f"{s}")
                    plt.plot(x, y, label=i)
            plt.bar(unique, counts / counts.sum(), label=column + ' data', color='lightblue')
            plt.title('Fitting Data Distribution', fontweight='bold', fontsize=20)
            plt.ylabel('Frequency', fontsize=15)
            plt.xlabel('Value', fontsize=15)
            plt.legend()
            plt.show()

        return result

    @staticmethod
    def feature_score(df, x, y, scoring_function, verbose = False):
        """Calculate the score of input data.

        Args:
            df (Pandas DataFrame): DataFrame with input data.
            x (List[str]): Names of input variables (same name as the corresponding column of ``df``).
            y (List[str]): Names of output variables (same name as the corresponding column of ``df``).
            scoring_function (str): Name of the scoring function.
            verbose (bool, optional): Flag to display detailed messages. Defaults to False.

        Raises:
            ValueError: Invalid scoring_function name.

        Returns:
            (Pandas DataFrame): DataFrame with scores of input variables.
        
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
        """
        #  

        s_function = ['r_regression', 'f_regression', 'mutual_info_regression', 'chi2', 'f_classif',
                      'mutual_info_classif']
        if scoring_function not in s_function:
            raise ValueError( f"Função {scoring_function} não suportada. Funções válidas: r_regression, f_regression, mutual_info_regression, chi2, f_classif, mutual_info_classif")
        
        fs = SelectKBest(score_func=eval(f"{scoring_function}"), k=len(x))
        X_new = fs.fit_transform(df[x], df[y].transpose().values.tolist()[0])
        scores_df = pd.DataFrame([fs.scores_], columns=x)
        scores_df = scores_df.T
        scores_df = scores_df.sort_values(by=[0], ascending=False)
        scores_df = (scores_df.T).round(2)
        if verbose:
            print(tabulate(scores_df, headers='keys', tablefmt='tab'))

        return scores_df

    '''
    @staticmethod
    def pareto_front(df, list_min, list_max, verbose=False):
        """
        Calculate the Pareto front of a DataFrame given a set of minimum and maximum criteria.

        Args:
            df (Pandas DataFrame): The DataFrame to calculate the Pareto front for.
            list_min (list): A list of column names that represent minimum criteria for the Pareto front.
            list_max (list): A list of column names that represent maximum criteria for the Pareto front.
            verbose (bool): If True, print the resulting Pareto front DataFrame. Defaults to False.

        Raises:
            ValueError: If fewer than two criteria are provided.

        Returns:
            (Pandas DataFrame): The Pareto front of the DataFrame.
        """
        if (len(list_min) + len(list_max)) < 2:
            raise ValueError('Para calcular a fronteira de Pareto é necessário selecionar pelo menos duas variáveis.')

        pref = None
        if list_min:
            pref = p.low(list_min[0])
            for i in range(1, len(list_min)):
                pref = pref * p.low(list_min[i])

        if list_max:
            if pref:
                pref = pref * p.high(list_max[0])
            else:
                pref = p.max(list_max[0])
            for i in range(1, len(list_max)):
                pref = pref * p.high(list_max[i])

        pareto = pref.psel(df)

        if verbose:
            print(tabulate(pareto, headers='keys', tablefmt='tab'))

        return pareto

    @staticmethod
    def get_best_pareto_point(df, list_variable, weigths, verbose=False):
        """
        Calculate the best Pareto point of a DataFrame given a set of variables and weights.

        Args:
            df (Pandas DataFrame): The DataFrame to calculate the best Pareto point for.
            list_variable (list): A list of column names to consider in the calculation.
            weights (list): A list of weights to use in the calculation.
            verbose (bool): If True, print the DataFrame and the best Pareto point found. Defaults to False.

        Returns:
            tuple: A tuple containing the index of the best Pareto point and the coordinates of the point.

        """
        df = df[list_variable].reset_index(drop=True)
        f = np.array(df.values.tolist())
        decomp = ASF()
        I = decomp(f, weigths).argmin()
        if verbose:
            print(tabulate(df, headers='keys', tablefmt='tab'))
            print("Melhor opção de acordo com a decomposição: Ponto %s - %s" % (I, f[I]))
        
        return I, f[I]
    '''
    
    @staticmethod
    def detect_outliers(df, method = 'IQR', thr = 3, verbose = False):
        """Detect outliers in a Pandas DataFrame using IQR or zscore method.

        Args:
            df (Pandas DataFrame): Input DataFrame containing numerical data.
            method (str, optional): Method to use for outlier detection. Available options: 'IQR' or 'zscore'. Defaults to 'IQR'.
            thr (int, optional): Threshold value for zscore method. Defaults to 3.
            verbose (bool, optional): Determines whether to display detailed messages. Defaults to False.

        Raises:
            ValueError: If method is not equal to one of the following options: 'IQR' or 'zscore'.

        Returns: 
            tuple containing

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
        
        """
        df = df.select_dtypes("number")
        m = ['IQR', 'zscore']
        if method not in m:
            raise ValueError(f"method deve igual à uma das seguintes opções: 'IQR', 'zcore.'")
        min_thres_list = []
        max_thres_list = []
        df_thres = pd.DataFrame()
        indx = np.array([])
        if method == 'IQR':
            for column in df:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)    
                IQR = Q3 - Q1
                max_thres = Q3 + 1.5*IQR
                min_thres = Q1 - 1.5*IQR
                min_thres_list.append(Q1 - 1.5*IQR)
                max_thres_list.append(max_thres)
                indx = np.concatenate((indx, np.array(df[(df[column] > max_thres) | (df[column] < min_thres) ][column].index.values.tolist())), axis=None)
            df_thres['column'] = df.columns.tolist()
            df_thres['min_thres'] = min_thres_list
            df_thres['max_thres'] = max_thres_list
            df_thres = df_thres.set_index('column')
        elif method == 'zscore':
            for column in df:
                z = np.abs(stats.zscore(df[column]))
                indx = np.concatenate((indx, np.where(z > thr)[0]), axis=None)
        df_output = pd.DataFrame({'outliers_index': np.unique(indx)})        
        if verbose:        
            print(tabulate(df_output, headers='keys', tablefmt='tab'),"\n")
        return df_output, df_thres

    def remove_outliers(self, df, verbose = False):
        """
        Remove outliers from a Pandas DataFrame using the Interquartile Range (IQR) method.
        
        Args:
            df (Pandas DataFrame): DataFrame containing the data.
            verbose (bool, optional): If True, print the number of lines removed. Defaults to False.
        
        Returns:
            tuple containing
            
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
        """
        df_new = df.copy()
        drop_lines = []
        columns = df_new.columns.tolist()
        outliers, _ = self.detect_outliers(df)
        idx = outliers.outliers_index.values.tolist()
        for i in idx:
            df_new = df_new.drop(index = i)
            if i not in drop_lines:
                drop_lines.append(i)
        df_new = df_new.reset_index(drop=True)
        if verbose:
            print(f"Foram removidas {len(idx)} linhas.\n")
        return df_new, idx

    @staticmethod
    def cramer_V(df, verbose = False, save = False, path = None, format = 'png'):
        """
        Calculate Cramer's V statistic for categorical feature association in a DataFrame.

        Cramer's V is a measure of association between two categorical variables. It is based on the ``chi-squared`` statistic
        and considers both the strength and direction of association. This function calculates Cramer's V for all pairs of
        categorical variables in a given DataFrame and returns the results in a new DataFrame.

        Args:
            df (pandas DataFrame): The input DataFrame containing the categorical variables.
            verbose (bool, optional): If True, a heatmap of the Cramer's V values will be displayed using Seaborn. Default is False.

        Returns:
            (pandas DataFrame): A DataFrame containing Cramer's V values for all pairs of categorical variables.

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
        """
        label = preprocessing.LabelEncoder()
        data_encoded = pd.DataFrame() 
        for i in df.columns:
            data_encoded[i]=label.fit_transform(df[i]) 

        rows= []
        col_data = data_encoded.columns.tolist()
        len_data = len(data_encoded.columns)
        for i in range(len_data):
            row = []
            for k in range(i):
                row.append(rows[k][i])

            for j in range(i, len_data):
                crosstab =np.array(pd.crosstab(data_encoded[col_data[i]],data_encoded[col_data[j]], rownames=None, colnames=None)) # Cross table building
                stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
                obs = np.sum(crosstab) # Number of observations
                mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
                cramers = (stat/(obs*mini))
                row.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
            rows.append(row)

        cramers_results = np.array(rows)
        df = pd.DataFrame(cramers_results, columns = data_encoded.columns, index =data_encoded.columns)
        plt.figure(figsize=(12,9))
        plt.title('Association (Cramér\'s V)', fontsize=18,fontweight="bold")
        sns.heatmap(df, vmin=0., vmax=1,linewidths=.1,cmap=sns.color_palette("ch:s=.25,rot=-.25",as_cmap=True), annot=True)
        plt.yticks(rotation=0)
        if save:
            name = path + "/association." + format
            plt.savefig(name)
        if verbose:
            plt.show()

        return df
    
    
    def EDA(self, df, save = False, path = None, format = 'png'):
        """Perform exploratory data analysis (EDA) on a given pandas DataFrame.
        
        The function displays a summary table of the DataFrame, a table of class balance for categorical variables,
        and histograms and boxplots with information on the number of outliers for numerical variables.
        
        Args:
            df (pandas.DataFrame): Input DataFrame to be analyzed.
            save (bool, optional): If True, save the plots. Defaults to False.
            path (str, optional): Path to save the plots. Defaults to None.
            format (str, optional): Format for the plot files. Defaults to 'png'.

        Returns:
            None

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

        """
        df_cat = df.select_dtypes(exclude=["number"])
        df_number = df.select_dtypes("number")
        for col in df_number:
            unique_values = df_number[col].nunique()
            if unique_values < 20 and unique_values/len(df_number) <0.05:
                df_cat[col] = pd.Categorical(df_number[col])
                df_number = df_number.drop(columns=[col])
        
        if not df_cat.empty:
            print("Variáveis Categóricas:\n")
            des = df_cat.describe()
            des = pd.concat([pd.DataFrame([df_cat.isna().sum().values.tolist()], columns = des.columns, index = ['nan']),des])
            print(tabulate(des, headers='keys', tablefmt='tab'),"\n")
           
            # Dropna
            df_cat = df_cat.dropna()

            #Associação
            print("\nAssociação:\n")
            self.cramer_V(df_cat, verbose=True, save=save, path=path, format=format)
            
            sns.set_theme(style="darkgrid")
            palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors = 8)[2:]
            sns.set_palette(palette)
            
            print("\nHistogramas:\n")
            for column in df_cat:
                x = df_cat[column].value_counts().index.to_list()
                palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors = len(x)+2)[2:]
                sns.set_palette(palette)
                y = np.round(np.array(df_cat[column].value_counts().to_list())/df_cat[column].count()*100,decimals = 2 )
                fig,ax = plt.subplots(figsize=(12,9))
                graph = sns.barplot(x = x, y = y)
                graph.axhline(100/len(df_cat[column].unique()))
                plt.title(column, fontsize=18,fontweight="bold")
                plt.ylabel("Frequency(%)", fontsize=12)
                fig.canvas.draw()
                labels = ax.get_xticklabels()
                label_sizes = [label.get_window_extent().width for label in labels]
                max_label_size = max(label_sizes)
                fig_size = fig.get_size_inches() * fig.dpi  # figure size in pixels
                w, h = fig_size
                if max_label_size > w / len(labels):
                    # rotate the labels if they overlap
                    ax.tick_params(axis='x', labelrotation=85)
                #plt.xticks(rotation=80, fontsize = 8)
                
                
                if save:
                    name = path + "/" +column + "." + format
                    plt.savefig(name)
                plt.show()
        
        if not df_number.empty:
            print("\nVariáveis Numéricas:")
            des = df_number.describe()
            des = pd.concat([pd.DataFrame([df_number.isna().sum().values.tolist()], columns = des.columns, index = ['nan']),des])
            print(tabulate(des, headers='keys', tablefmt='tab'),"\n")
            # Dropna
            df_number = df_number.dropna()
            #Correlação
            print("\nCorrelação:\n")
            plt.figure(figsize=(12,9))
            plt.title('Correlation  (Pearson)', fontsize=18,fontweight="bold")
            sns.heatmap(df_number.corr(),linewidths=.1,cmap=sns.color_palette("ch:s=.25,rot=-.25",as_cmap=True), annot=True)
            plt.yticks(rotation=0)
            if save:
                name = path + "/correlation." + format
                plt.savefig(name)
            plt.show()

            sns.set_theme(style="darkgrid")
            palette = sns.color_palette("ch:s=.25,rot=-.25", n_colors = 8)[3:]
            sns.set_palette(palette)
            print("\nHistogramas e boxplots:\n")
            for column in df_number:
                f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(12,9))
                plt.title(column, fontsize=18,fontweight="bold")
                sns.boxplot(x=df_number[column], orient='h', ax=ax_box)
                ax_box.set(xlabel='')
                sns.histplot(x = df_number[column], stat='percent', ax=ax_hist, kde = True)
                plt.ylabel("Frequency(%)", fontsize=10)
                
                if save:
                    name = path + "/" +column + "." + format
                    plt.savefig(name)
                plt.show()
                outliers, df_thresh = self.detect_outliers(df_number[[column]])
                print(f"\nDetecção de outlier da variável '{column}':")
                print(f"\tQuantidade: {len(outliers)} de {len(df_number[column])}.")
                print(f"\tMétodo: Intervalo Interquartil (IQR - Interquatile Range).")
                print(f"\tCritério: Os valores menores que {round(df_thresh['min_thres'][column],2)} ou maiores que {round(df_thresh['max_thres'][column],2)} foram considerados outliers.\n\n")