:py:mod:`asapy.preprocessing`
=============================

.. py:module:: asapy.preprocessing


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.preprocessing.Preprocessing




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



