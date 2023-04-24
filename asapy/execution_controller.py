import pandas as pd
import numpy as np
"""
def simulate_example(doe: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()

def stop_example(result: pd.DataFrame, last_result: pd.DataFrame) -> bool:
    return False
"""

class ExecutionController:
    """ A class for controlling the execution of a simulation function on a Design of Experiments (DOE) dataset.
    
    Attributes:
        _sim_func (callable): A function that performs the simulation on a chunk of the DOE dataset.
        _stop_func (callable): A function that determines whether to stop the simulation based on the current and partial results.
        _chunk_size (int): The size of each chunk of the DOE dataset.

    Methods:
        run(doe: pd.DataFrame) -> None:
            Runs the simulation on the entire DOE dataset by dividing it into chunks of size `_chunk_size` (if provided).
            Stops the simulation if the `stop_func` returns True, and returns the result.

    Raises:
        Exception: If the `_sim_func` returns an invalid result (not a pandas DataFrame).
    """
    
    def __init__(self, sim_func: callable, stop_func: callable, chunk_size: int = 0) -> None:
        """
        Initializes a new instance of the ExecutionController class.

        Args:
            sim_func (callable): A function that performs the simulation on a chunk of the DOE dataset.
            stop_func (callable): A function that determines whether to stop the simulation based on the current and partial results.
            chunk_size (int, optional): The size of each chunk of the DOE dataset. Defaults to 0.
        """
        self._sim_func = sim_func
        self._stop_func = stop_func
        self._chunk_size = chunk_size

    def run(self, doe: pd.DataFrame) -> None:
        """
        Runs the simulation on the entire DOE dataset by dividing it into chunks of size `_chunk_size` (if provided).
        Stops the simulation if the `stop_func` returns True, and returns the result.

        Args:
            doe (pd.DataFrame): The Design of Experiments dataset to be used in the simulation.

        Returns:
            None

        Raises:
            Exception: If the `_sim_func` returns an invalid result (not a pandas DataFrame).
        """
        n = len(doe)/self._chunk_size if self._chunk_size > 0 else 1
        result = pd.DataFrame()
        for data in np.array_split(doe, n):
            partial_result = self._sim_func(data)
            if not isinstance(partial_result, pd.DataFrame):
                raise Exception(
                    "Invalid result returned by the simulation function")
            
            stop = self._stop_func(result, partial_result)
            result.append(partial_result)
            if stop:
                return result
            
            