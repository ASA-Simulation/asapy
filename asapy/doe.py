from pandas import DataFrame
from pyDOE import lhs
from math import isnan
from typing import List
import pandas as pd
import numpy as np

class Doe:
    def __init__(self):
        self.aliases_df = None
        self.aliases = None

    @staticmethod
    def _get_aliases(sim):
        """
        Returns a generator that yields dictionaries containing information about aliases found within the input `sim` parameter.

        Args:
            sim (dict): A dictionary representing the simulation object.

        Yields:
            dict: A dictionary containing information about an alias, including its `id`, `alias_attribute`, and `alias`.

        Example Usage:

        .. code::

            >>> #To extract aliases from a simulation object `sim`, you can use the following code:
            >>> for alias_info in Doe._get_aliases(sim):
            ...     print(alias_info)

        .. note::
        
            - This method checks if the input `sim` has an `alias` key with a non-empty value. If found, it iterates through the key-value pairs of the value dictionary and yields a dictionary containing information about the alias.

            - If the value of the key-value pair is a dictionary, the method recursively calls itself with the dictionary as the new input `sim` parameter.

            - If the value of the key-value pair is a list, it iterates through the list and recursively calls itself with each dictionary in the list as the new input `sim` parameter.
        """
        if hasattr(sim, 'items'):
            for k, v in sim.items():
                if k == 'alias' and len(v) != 0:
                    for attribute, alias in v.items():
                        yield dict(id=sim['identifier'], alias_attribute=attribute, alias=alias)
                if isinstance(v, dict):
                    for result in Doe._get_aliases(v):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in Doe._get_aliases(d):
                            yield result

    @staticmethod
    def _get_configs(configs, alias):
        """
        Recursively searches for the attribute of the given alias in a list of configurations.

        Args:
        - configs (list of dicts): List of configuration dictionaries
        - alias (dict): Dictionary containing the alias attribute to be searched and the id to match

        Returns:
        - Generator object that yields the value of the alias attribute whenever it is found
        """
        for config in configs:
            if hasattr(config, 'items'):
                for k, v in config.items():
                    if (k == 'identifier') and (v == alias['id']):
                        yield config['attributes'][alias['alias_attribute']]
                    if isinstance(v, dict):
                        for result in Doe._get_configs(v, alias):
                            yield result
                    elif isinstance(v, list):
                        for d in v:
                            for result in Doe._get_configs(d, alias):
                                yield result

    def process_aliases_by_sim(self, sim, configs) -> DataFrame:
        """
        Process aliases based on similarity and configurations.

        Parameters:
        sim (list): A list of dictionaries, containing 'id' and 'text' keys.
        configs (list): A list of dictionaries, containing 'identifier', 'attributes', and 'alias_attribute' keys.

        Returns:
        DataFrame: A DataFrame containing aliases and their corresponding attributes.
        """
        required_columns = ['label', 'type', 'default', 'min', 'max', 'alias_attribute']
        
        aliases_dic = dict()
        
        # Extract aliases and iterate
        self.aliases = list(self._get_aliases(sim))
        for alias in self.aliases:
            for i in list(self._get_configs(configs, alias)):
                i['alias_attribute'] = alias['alias_attribute']
                aliases_dic[alias['alias']] = i
                
        # Convert the dictionary to a dataframe and transpose it
        self.aliases_df = pd.DataFrame(aliases_dic).T
        
        # Check and add any missing required columns to the dataframe with NaN values
        for col in required_columns:
            if col not in self.aliases_df.columns:
                self.aliases_df[col] = np.nan
                
        return self.aliases_df[required_columns]

    @staticmethod
    def create(df_T, samples, seed=42):
        """
        Creates a design of experiments (DOE) based on the input DataFrame ``df_T``.
        The DOE is created using a Latin Hypercube Sampling (LHS) method and a sample size ``samples``.
        The function returns a new DataFrame with the input variables' names and values according to their type.

        Args:
            df_T: A DataFrame of the variables' metadata. The index should contain the variables' names, and the following columns
            should exist:
                - "type": Variable's type: integer, double, boolean, select, multi_select, or string.
                - "default": Default value for the variable.
                - "min": Minimum value for the variable (if applicable).
                - "max": Maximum value for the variable (if applicable).
                - "options": Available options for the variable (if applicable).
            samples: An integer value indicating the number of samples in the design.

        Returns:
            A new DataFrame containing the input variables' names and values, generated using LHS sampling.

        Raises:
            TypeError: If ``df_T`` is not a pandas DataFrame or ``samples`` is not an integer.
            ValueError: If ``df_T`` does not contain the required columns or the ``default`` value is not within the ``min`` and ``max`` range.
        """
        np.random.seed(seed)
        df = df_T.T
        n = df.shape[1]
        doe = DataFrame(lhs(n=n, samples=samples))
        doe.columns = df.columns
        for col in doe.columns:
            if (df[col].loc['type'] == 'multi_select') or (df[col].loc['type'] == 'string'):
                df = df.drop(col, axis=1)
                doe = doe.drop(col, axis=1)
                continue
            elif df[col].loc['type'] == 'select':
                values = []
                for dic in df[col].loc['options']:
                    values.append(dic['value'])
                doe[col] = doe[col].apply(lambda x: values[int(x * len(values))])
            elif df[col].loc['type'] == 'boolean':
                doe[col] = doe[col].apply(lambda x: round(x))
            else:
                if isnan((df[col].loc['max'])):
                    max_value = 5 * df[col].loc['default']
                else:
                    max_value = df[col].loc['max']
                if isnan((df[col].loc['min'])):
                    min_value = 0.5 * df[col].loc['default']
                else:
                    min_value = df[col].loc['min']
                doe[col] = doe[col].apply(lambda x: x * (max_value - min_value) + min_value)
                if df[col].loc['type'] == 'integer':
                    doe[col] = doe[col].apply(lambda x: int(x))
                if df[col].loc['type'] == 'double':
                    doe[col] = doe[col].apply(lambda x: round(x,2))
        return doe

    @staticmethod
    def prepare_experiments(df) -> List[dict]:
        """
        Prepare a list of experiments from a Pandas DataFrame.

        Args:
            df: A Pandas DataFrame containing experiment data.

        Returns:
            A list of dictionaries, where each dictionary represents an experiment and its attributes.

        Raises:
            None.

        """
        return list(df.T.to_dict().values())

    @staticmethod
    def _get_metrics(sim):
        """
        Recursively extract monitor metrics from a simulation dictionary.

        Args:
            sim (dict): A simulation dictionary.

        Yields:
            dict: A dictionary containing the workspace name, simulation identifier, and monitor metrics.
        """
        if hasattr(sim, 'items'):
            for k, v in sim.items():
                if k == 'monitors':
                    for monitor in sim['monitors']:
                        # yield {monitor['attributes']['side']: monitor['attributes']['metrics']}
                        yield dict(ws_name=monitor['attributes']['ws_name'], identifier=monitor['identifier'], metrics=monitor['attributes']['metrics'])
                if isinstance(v, dict):
                    for result in Doe._get_metrics(v):
                        yield result
                elif isinstance(v, list):
                    for d in v:
                        for result in Doe._get_metrics(d):
                            yield result

    def process_metrics(self, sim):
        """
        Process the metrics obtained from a simulation.

        Args:
            sim: A dictionary containing the simulation data.

        Returns:
            A pandas DataFrame containing the metrics indexed by workspace name.

        Raises:
            None
        """
        return pd.DataFrame(list(self._get_metrics(sim))).set_index('ws_name')