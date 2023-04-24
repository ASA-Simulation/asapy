:py:mod:`asapy.doe`
===================

.. py:module:: asapy.doe


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.doe.Doe




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



