import pandas as pd
import numpy as np
import asaclient
from functools import partial
import time
import math
import json
from tqdm import tqdm
from .utils import gen_dict_extract, transform_stringified_dict
import pickle
import os


def basic_simulate(batch: asaclient.Batch, current: pd.DataFrame, processed: pd.DataFrame, all: pd.DataFrame, asa_custom_types=[], pbar=None) -> pd.DataFrame:
    """Performs basic simulation on a chunk of data.

    Args:
        batch (asaclient.Batch): The ASA batch object.
        current (pd.DataFrame): The current chunk of data to simulate.
        processed (pd.DataFrame): The previously processed data.
        all (pd.DataFrame): The complete dataset.
        asa_custom_types (list, optional): List of custom ASA types to retrieve in the simulation records.
        pbar (tqdm, optional): The tqdm progress bar object. 

    Returns:
        pd.DataFrame: The simulation results for the current chunk of data.
    """
    experiments = list(current.T.to_dict().values())
    execs = batch.add_chunks(experiments)
    while True:
        status = batch.status()
        # print(status)
        total_finished = status.stopped + status.finished + status.failed + status.error + status.timeout + status.canceled
        pbar.update(total_finished - pbar.n)
        if total_finished == status.sent:
            break
        else:
            time.sleep(5)
    uuids = [exec.uuid for exec in execs]
    records = batch.records(uuids=uuids, types=asa_custom_types)
    # print(f"{len(current)+len(processed)}/{len(all)}")
    if len(records):
        df = pd.DataFrame(records)
        df['experiment'] = 0
        for i in range(len(execs)):
            exec = execs[i]
            df.loc[df['execution_uuid'] == exec.uuid,
                   'experiment'] = current.index[i]
        return df
    return pd.DataFrame()


def batch_simulate(batch: asaclient.Batch, asa_custom_types=[]):
    """Returns a partial function for batch simulation.

    Args:
        batch (asaclient.Batch): The ASA batch object.
        asa_custom_types (list, optional): List of custom ASA types to retrieve in the simulation records.

    Returns:
        callable: The partial function for batch simulation.
    """
    return partial(basic_simulate, batch=batch, asa_custom_types=asa_custom_types)


def basic_stop(current: pd.DataFrame, all_previous: pd.DataFrame, metric: str, threshold: float, side: str) -> bool:
    """Determines whether to stop the simulation based on the current and previous results.

    Args:
        current (pd.DataFrame): The current chunk of simulation results.
        all_previous (pd.DataFrame): All previously processed simulation results.
        metric (str): The metric to compare.
        threshold (float): The threshold value for stopping the simulation.
        side (str): The side information to select the specific metric.

    Returns:
        bool: True if the simulation should stop, False otherwise.
    """
    if len(all_previous) == 0:
        return False

    payload_key = 'asa_custom_type'
    payload_value = 'asa::recorder::AsaMonitorReport'

    current_total = pd.concat([all_previous, current])

    payload_current = current_total[current_total[payload_key] == payload_value]['payload']
    dic_current = payload_current.apply(transform_stringified_dict)
    metrics_current = pd.DataFrame(gen_dict_extract(metric, dic_current), columns=[metric])
    side_current = pd.DataFrame(gen_dict_extract('side', dic_current), columns=['side'])
    result_current = pd.concat([metrics_current, side_current], axis=1)
    result_current = result_current[result_current['side'] == side]

    payload_previous = all_previous[all_previous[payload_key] == payload_value]['payload']
    dic_previous = payload_previous.apply(transform_stringified_dict)
    metrics_previous = pd.DataFrame(gen_dict_extract(metric, dic_previous), columns=[metric])
    side_previous = pd.DataFrame(gen_dict_extract('side', dic_previous), columns=['side'])
    result_previous = pd.concat([metrics_previous, side_previous], axis=1)
    result_previous = result_previous[result_previous['side'] == side]

    relative_change_mean = (result_current.mean(numeric_only=True) - result_previous.mean(numeric_only=True)) / result_previous.mean(numeric_only=True)
    relative_change_std = (result_current.std(numeric_only=True) - result_previous.std(numeric_only=True)) / result_previous.std(numeric_only=True)

    if (relative_change_mean.abs().values < threshold) and (relative_change_std.abs().values < threshold):
        print("\n\nThis is an early stop! The batch execution has been completed.\n")
        return True
    else: 
        return False
        

def stop_func(metric: str, threshold: float, side: str):
    """Returns a partial function for stopping the simulation.

    Args:
        metric (str): The metric to compare.
        threshold (float): The threshold value for stopping the simulation.

    Returns:
        callable: The partial function for stopping the simulation.
    """
    return partial(basic_stop, metric=metric, threshold=threshold, side=side)


def non_stop_func(current: pd.DataFrame, all_previous: pd.DataFrame) -> bool:
    """Determines that the simulation should never stop.

    Args:
        current (pd.DataFrame): The current chunk of simulation results.
        all_previous (pd.DataFrame): All previously processed simulation results.

    Returns:
        bool: Always False.
    """
    return False


class ExecutionController:
    pass


class ExecutionController:
    """A class for controlling the execution of a simulation function on a Design of Experiments (DOE) dataset."""
    
    # Hidden folder for saving the execution state
    HIDDEN_FOLDER = ".execution_state"

    def __init__(self, sim_func: callable, stop_func: callable, chunk_size: int = 0) -> None:
        """
        Initializes a new instance of the ExecutionController class.

        Args:
            sim_func (callable): A function that performs the simulation on a chunk of the DOE dataset.
            stop_func (callable): A function that determines whether to stop the simulation based on the current and partial results.
            chunk_size (int, optional): The size of each chunk of the DOE dataset. Defaults to 0.

        """
        # Assigning the simulation function, stop function, and chunk size
        self._sim_func = sim_func
        self._stop_func = stop_func
        self._chunk_size = chunk_size
        
        # Initializing result, processed, and remaining dataframes
        self._result = pd.DataFrame()
        self._processed = pd.DataFrame()
        self._remaining = pd.DataFrame()
        
        # Other initial attributes
        self._state_saved = False
        self._pause = False
        self._progress = 0

    def save_state(self, file_name: str):
        """Saves the current execution state to a file."""
        
        # Define the state to be saved
        state = {
            "result": self._result,
            "processed": self._processed,
            "remaining": self._remaining,
            "state_saved": self._state_saved,
            "progress": self._progress
        }
        
        # Save the state using pickle
        path_to_save = os.path.join(self.HIDDEN_FOLDER, file_name)
        with open(path_to_save, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, file_name: str, sim_func: callable, stop_func: callable, chunk_size: int):
        """Loads the saved execution state from a file."""

        # Path to the saved state
        path_to_read = os.path.join(cls.HIDDEN_FOLDER, file_name)

        # Load the saved state
        with open(path_to_read, 'rb') as f:
            state = pickle.load(f)

        # Create a new instance and populate with the saved state
        instance = cls(sim_func, stop_func, chunk_size)
        instance._result = state["result"]
        instance._processed = state["processed"]
        instance._remaining = state["remaining"]
        instance._state_saved = state["state_saved"]
        instance._progress = state["progress"]

        return instance

    def resume(self):
        """Resumes the execution from the saved state if available."""

        # Check if state is already saved (execution finished)
        if self._state_saved:
            print("This execution has already completed successfully.")
            return self._result
        else:
            print("Resuming the execution...")
            return self.run(self._remaining, resume=True)

    def pause(self):
        """Pauses the current execution and saves the state."""
        if self.pbar is None:
            return

        self._pause = True
        self.pbar.close()
        self.pbar = None
        self.save_state("execution_state.pkl")

    def _safe_pbar_update(self, n):
        """Safely updates the progress bar by `n` steps."""
        # Update the progress bar if it exists
        if self.pbar:
            self._progress += n
            self.pbar.update(n)

    def _safe_pbar_close(self):
        """Safely closes the progress bar."""
        # Close the progress bar if it exists
        if self.pbar:
            self.pbar.close()
            self.pbar = None
 
    def run(self, doe: pd.DataFrame, resume=False) -> pd.DataFrame:
        """Runs the simulation on the DOE by dividing it into chunks and stops if `_stop_func` returns True."""
        # Reset pause flag
        self._pause = False 

        # Total number of simulations for this run
        total_simulations = len(doe)

        # Initialize the progress bar only if it doesn't exist or if it's a new run
        if not hasattr(self, 'pbar') or self.pbar is None:
            self.pbar = tqdm(total=total_simulations, initial=self._progress, position=0, leave=True)

        # Determine number of chunks based on chunk size
        n_chunks = math.ceil(total_simulations / self._chunk_size) if self._chunk_size > 0 else 1

        self._remaining = doe
        
        # Loop through each chunk and perform simulation
        for current in np.array_split(doe, n_chunks):
            if self._pause:
                self._safe_pbar_close()
                print("Execution paused. You can resume the simulation later.")
                self._state_saved = False
                self.save_state("execution_state.pkl")
                return self._result
            
            # Try running the simulation function
            try:
                partial_result = self._sim_func(current=current,
                                                processed=self._processed,
                                                all=doe,
                                                pbar=self.pbar)

                # Update progress after a successful simulation
                self._safe_pbar_update(len(current))
            except Exception as e:
                # Handle any exceptions and save the state
                self._safe_pbar_close()
                print(f"An error occurred during simulation: {str(e)}")
                self._state_saved = False
                self.save_state("execution_state.pkl")
                print("Execution state saved. You can resume the simulation later.")
                return self._result

            # Validate that the result of simulation function is a dataframe
            if not isinstance(partial_result, pd.DataFrame):
                raise Exception("Invalid result returned by the simulation function")

            # Update the processed data and the result
            self._processed = pd.concat([self._processed, current])
            self._safe_pbar_update(len(current))

            # Check if the simulation should stop
            stop = self._stop_func(current=partial_result, all_previous=self._result)

            self._result = pd.concat([self._result, partial_result])
            if stop:
                self._safe_pbar_close()
                self._state_saved = True
                return self._result

        # Close the progress bar and mark the state as saved
        self._safe_pbar_close()
        self._state_saved = True
        return self._result