# `asapy.execution_controller`

## Module Contents

### Classes

| `ExecutionController`
 | A class for controlling the execution of a simulation function on a Design of Experiments (DOE) dataset.

 |

### _class_ asapy.execution_controller.ExecutionController(sim_func: callable, stop_func: callable, chunk_size: int = 0)
A class for controlling the execution of a simulation function on a Design of Experiments (DOE) dataset.


#### _sim_func()
A function that performs the simulation on a chunk of the DOE dataset.


* **Type**

    callable



#### _stop_func()
A function that determines whether to stop the simulation based on the current and partial results.


* **Type**

    callable



#### _chunk_size()
The size of each chunk of the DOE dataset.


* **Type**

    int



### run(doe()
pd.DataFrame) -> None:
Runs the simulation on the entire DOE dataset by dividing it into chunks of size _chunk_size (if provided).
Stops the simulation if the stop_func returns True, and returns the result.


* **Raises**

    **Exception** – If the _sim_func returns an invalid result (not a pandas DataFrame).



#### run(doe: pandas.DataFrame)
Runs the simulation on the entire DOE dataset by dividing it into chunks of size _chunk_size (if provided).
Stops the simulation if the stop_func returns True, and returns the result.


* **Parameters**

    **doe** (*pd.DataFrame*) – The Design of Experiments dataset to be used in the simulation.



* **Returns**

    None



* **Raises**

    **Exception** – If the _sim_func returns an invalid result (not a pandas DataFrame).
