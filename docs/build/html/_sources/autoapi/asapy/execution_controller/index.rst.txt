:py:mod:`asapy.execution_controller`
====================================

.. py:module:: asapy.execution_controller


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.execution_controller.ExecutionController




.. py:class:: ExecutionController(sim_func: callable, stop_func: callable, chunk_size: int = 0)

   A class for controlling the execution of a simulation function on a Design of Experiments (DOE) dataset.

   .. attribute:: _sim_func

      A function that performs the simulation on a chunk of the DOE dataset.

      :type: callable

   .. attribute:: _stop_func

      A function that determines whether to stop the simulation based on the current and partial results.

      :type: callable

   .. attribute:: _chunk_size

      The size of each chunk of the DOE dataset.

      :type: int

   .. method:: run(doe

      pd.DataFrame) -> None:
      Runs the simulation on the entire DOE dataset by dividing it into chunks of size `_chunk_size` (if provided).
      Stops the simulation if the `stop_func` returns True, and returns the result.
      

   :raises Exception: If the `_sim_func` returns an invalid result (not a pandas DataFrame).

   .. py:method:: run(doe: pandas.DataFrame) -> None

      Runs the simulation on the entire DOE dataset by dividing it into chunks of size `_chunk_size` (if provided).
      Stops the simulation if the `stop_func` returns True, and returns the result.

      :param doe: The Design of Experiments dataset to be used in the simulation.
      :type doe: pd.DataFrame

      :returns: None

      :raises Exception: If the `_sim_func` returns an invalid result (not a pandas DataFrame).



