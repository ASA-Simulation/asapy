:py:mod:`asapy.execution_controller`
====================================

.. py:module:: asapy.execution_controller


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.execution_controller.ExecutionController
   asapy.execution_controller.ExecutionController



Functions
~~~~~~~~~

.. autoapisummary::

   asapy.execution_controller.basic_simulate
   asapy.execution_controller.batch_simulate
   asapy.execution_controller.basic_stop
   asapy.execution_controller.stop_func
   asapy.execution_controller.non_stop_func



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


.. py:class:: ExecutionController



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



