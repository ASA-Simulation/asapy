:py:mod:`asapy.prediction.NeuralNetwork`
========================================

.. py:module:: asapy.prediction.NeuralNetwork


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.prediction.NeuralNetwork.NeuralNetwork




.. py:class:: NeuralNetwork(target, name=None, seed=None)


   Bases: :py:obj:`asapy.prediction.model.Model`

   A class for constructing and training neural networks, with built-in methods for preprocessing,
   hyperparameter optimization, training, and inference. Inherits from the Model base class.

   .. attribute:: n_input

      Number of input features.

      :type: int

   .. attribute:: n_neurons

      Number of neurons in each hidden layer.

      :type: int

   .. attribute:: n_output

      Number of output neurons.

      :type: int

   .. attribute:: metrics

      List of Keras metrics to be used for model evaluation.

      :type: list

   .. attribute:: callbacks

      List of Keras Callbacks to be used during model training.

      :type: list

   .. method:: build(data, **kwargs)

      Prepares the neural network model based on the provided dataset and hyperparameters.

   .. method:: _make_nn(dropout, layers, optimizer)

      Constructs the neural network architecture.

   .. method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning.

   .. method:: hyperparameter_optimization(n_trials=1, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna.

   .. method:: load(foldername)

      Loads the model and preprocessor from the specified folder.

   .. method:: fit(return_history=False, graph=True, graph_save_extension=None, verbose=0, **kwargs)

      Trains the neural network on preprocessed data.

   .. method:: predict(x, verbose=0)

      Makes predictions using the trained neural network model.

   .. method:: save()

      Saves the model and preprocessor to disk.
      

   .. py:method:: build(data, **kwargs)

      Prepares the neural network model based on the provided dataset and hyperparameters. This includes preprocessing
      the data and initializing the model architecture based on the data's characteristics and specified hyperparameters.

      :param data: The dataset to be used for building the model.
      :type data: Pandas DataFrame
      :param \*\*kwargs: Additional keyword arguments for preprocessing and model configuration.


   .. py:method:: _make_nn(dropout, layers, optimizer)

      Constructs the neural network architecture with the specified number of layers, dropout rate, and optimizer.

      :param dropout: The dropout rate to be applied to each hidden layer.
      :type dropout: float
      :param layers: The number of hidden layers in the neural network.
      :type layers: int
      :param optimizer: The name of the optimizer to be used for training the neural network.
      :type optimizer: str

      :returns: The constructed Keras Sequential model.
      :rtype: keras.models.Sequential


   .. py:method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
      a callback within an Optuna optimization study.

      :param trial: An Optuna trial object.
      :type trial: optuna.trial.Trial
      :param \*\*kwargs: Additional keyword arguments for configuring the optimization process.

      :returns: The average validation loss across all folds for the current trial.
      :rtype: float


   .. py:method:: hyperparameter_optimization(n_trials=1, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
      and updates the model's hyperparameters with the best found values.

      :param n_trials: The number of optimization trials to perform. Defaults to 1.
      :type n_trials: int, optional
      :param info: Whether to print detailed information about each trial. Defaults to False.
      :type info: bool, optional
      :param \*\*kwargs: Additional keyword arguments for configuring the optimization process.

      :returns: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
      :rtype: pd.DataFrame


   .. py:method:: load(foldername)

      Loads the model and preprocessor from the specified folder.

      :param foldername: The name of the folder where the model and preprocessor are saved.
      :type foldername: str


   .. py:method:: fit(return_history=False, graph=False, graph_save_extension=None, verbose=0, **kwargs)

      Trains the neural network on preprocessed data. This method supports early stopping and learning rate reduction
      based on the performance on the validation set.

      :param return_history: Whether to return the training history object. Defaults to False.
      :type return_history: bool, optional
      :param graph: Whether to plot training and validation loss and metrics. Defaults to True.
      :type graph: bool, optional
      :param graph_save_extension: Extension to save the graphs (e.g., 'png', 'svg'). If None, graphs are not saved. Defaults to None.
      :type graph_save_extension: str, optional
      :param verbose: Verbosity mode for training progress. Defaults to 0.
      :type verbose: int, optional
      :param \*\*kwargs: Additional keyword arguments for configuring the training process.

      :returns: The training history object, if `return_history` is True. Otherwise, None.
      :rtype: keras.callbacks.History


   .. py:method:: predict(x, verbose=0)

      Makes predictions using the trained neural network model.

      :param x: The input data for making predictions.
      :type x: Pandas DataFrame
      :param verbose: Verbosity mode for prediction. Defaults to 0.
      :type verbose: int, optional

      :returns: The input data with an additional column for predictions.
      :rtype: Pandas DataFrame


   .. py:method:: save()

      Saves the model and preprocessor to disk.



