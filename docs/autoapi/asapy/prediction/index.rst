:py:mod:`asapy.prediction`
==========================

.. py:module:: asapy.prediction


Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   DBSCAN/index.rst
   KMeans/index.rst
   NeuralNetwork/index.rst
   XgBoost/index.rst
   model/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.prediction.NeuralNetwork
   asapy.prediction.XgBoost
   asapy.prediction.DBSCAN
   asapy.prediction.KMeans




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



.. py:class:: XgBoost(target, name=None, seed=None)


   Bases: :py:obj:`asapy.prediction.model.Model`

   A class for constructing and training XGBoost models, with built-in methods for preprocessing,
   hyperparameter optimization, training, and inference. Inherits from the Model base class.

   .. attribute:: metrics

      List of evaluation metrics to be used for model evaluation.

      :type: list

   .. attribute:: patience_early_stopping

      Number of rounds without improvement to wait before stopping training.

      :type: int

   .. method:: build(data, **kwargs)

      Prepares the XGBoost model based on provided dataset and hyperparameters.

   .. method:: _make_xgBooster(**kwargs)

      Constructs the XGBoost model with specified hyperparameters.

   .. method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning.

   .. method:: hyperparameter_optimization(n_trials=1, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna.

   .. method:: load(foldername)

      Loads the model and preprocessor from the specified folder.

   .. method:: fit(return_history=False, graph=True, graph_save_extension=None, verbose=0, **kwargs)

      Trains the XGBoost model on preprocessed data.

   .. method:: predict(x)

      Makes predictions using the trained XGBoost model.

   .. method:: save()

      Saves the model and preprocessor to disk.
      

   .. py:method:: build(data, **kwargs)

      Prepares the XGBoost model based on the provided dataset and hyperparameters. This includes preprocessing
      the data and initializing the model parameters based on the data's characteristics and specified hyperparameters.

      :param data: The dataset to be used for building the model.
      :type data: Pandas DataFrame
      :param \*\*kwargs: Additional keyword arguments for preprocessing and model configuration.


   .. py:method:: _make_xgBooster(tree_method, booster, learning_rate, min_split_loss, max_depth, min_child_weight, max_delta_step, subsample, sampling_method, colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda, reg_alpha, scale_pos_weight, grow_policy, max_leaves, max_bin, num_parallel_tree, verbose=0)

      Constructs the XGBoost model with the specified hyperparameters.

      :param \*\*kwargs: Hyperparameters for the XGBoost model.

      :returns: The constructed XGBoost model.
      :rtype: xgboost.XGBModel


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

      Trains the XGBoost model on preprocessed data. This method supports early stopping based on the performance
      on the validation set.

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
      :rtype: dict


   .. py:method:: predict(x)

      Makes predictions using the trained XGBoost model.

      :param x: The input data for making predictions.
      :type x: Pandas DataFrame

      :returns: The input data with an additional column for predictions.
      :rtype: Pandas DataFrame


   .. py:method:: save()

      Saves the model and preprocessor to disk.



.. py:class:: DBSCAN(name=None, seed=None)


   Bases: :py:obj:`asapy.prediction.model.Model`

   A class for constructing and running the DBSCAN clustering algorithm, with built-in methods for preprocessing,
   hyperparameter optimization, and silhouette analysis. Inherits from the Model base class.

   .. attribute:: have_categ

      Indicates whether the dataset contains categorical features.

      :type: bool

   .. attribute:: distance_matrix

      The computed distance matrix for the dataset.

      :type: numpy.ndarray

   .. attribute:: clusters

      The cluster labels for each point in the dataset.

      :type: numpy.ndarray

   .. method:: build(data, **kwargs)

      Prepares the DBSCAN model based on provided dataset and hyperparameters.

   .. method:: _make_dbscan(eps, min_samples, algorithm, leaf_size, p)

      Constructs the DBSCAN model with specified hyperparameters.

   .. method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning.

   .. method:: hyperparameter_optimization(n_trials=100, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna.

   .. method:: load(foldername)

      Loads the preprocessor, preprocessed data, clusters, and model from the specified folder.

   .. method:: fit()

      Applies the DBSCAN algorithm to the preprocessed data.

   .. method:: predict(projection='2d', graph_save_extension=None)

      Generates and displays a 2D or 3D t-SNE plot of the clusters.

   .. method:: save()

      Saves the preprocessor, preprocessed data, clusters, and model to disk.
      

   .. py:method:: build(data, **kwargs)

      Prepares the DBSCAN model based on the provided dataset and hyperparameters. This includes preprocessing
      the data and calculating the distance matrix if necessary.

      :param data: The dataset to be used for clustering.
      :type data: Pandas DataFrame
      :param \*\*kwargs: Additional keyword arguments for preprocessing.


   .. py:method:: _make_dbscan(eps, min_samples, algorithm, leaf_size, p)

      Constructs the DBSCAN model with the specified hyperparameters.

      :param eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
      :type eps: float
      :param min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
      :type min_samples: int
      :param algorithm: The algorithm to be used by the DBSCAN model.
      :type algorithm: str
      :param leaf_size: Leaf size passed to the underlying BallTree or KDTree.
      :type leaf_size: int
      :param p: The power of the Minkowski metric to be used to calculate distance between points.
      :type p: float

      :returns: The constructed DBSCAN model.
      :rtype: sklearn.cluster._dbscan.DBSCAN


   .. py:method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
      a callback within an Optuna optimization study.

      :param trial: An Optuna trial object.
      :type trial: optuna.trial.Trial
      :param \*\*kwargs: Additional keyword arguments for configuring the optimization process.

      :returns: The silhouette score for the clustering configuration defined by the trial.
      :rtype: float


   .. py:method:: hyperparameter_optimization(n_trials=100, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
      and identifies the best hyperparameters for DBSCAN clustering.

      :param n_trials: The number of optimization trials to perform. Defaults to 100.
      :type n_trials: int, optional
      :param info: Whether to print detailed information about each trial. Defaults to False.
      :type info: bool, optional
      :param \*\*kwargs: Additional keyword arguments for configuring the optimization process.

      :returns: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
      :rtype: pd.DataFrame


   .. py:method:: load(foldername)

      Loads the preprocessor, preprocessed data, and cluster labels from the specified folder.

      :param foldername: The name of the folder where the data and model are saved.
      :type foldername: str


   .. py:method:: fit()

      Applies DBSCAN clustering to the preprocessed dataset and updates the 'clusters' attribute with the cluster labels.


   .. py:method:: predict(projection='2d', graph_save_extension=None)

      Projects the clustered data into 2D or 3D space using t-SNE and visualizes the clusters.

      :param projection: The type of projection for visualization ('2d' or '3d'). Defaults to '2d'.
      :type projection: str, optional
      :param graph_save_extension: Extension to save the graphs (e.g., 'png', 'svg'). If None, graphs are not saved. Defaults to None.
      :type graph_save_extension: str, optional


   .. py:method:: save()

      Saves the preprocessor, preprocessed data, cluster labels, and model to disk.



.. py:class:: KMeans(name=None, seed=None)


   Bases: :py:obj:`asapy.prediction.model.Model`

   A class for constructing and applying KMeans clustering algorithm, with built-in methods for preprocessing,
   hyperparameter optimization, and visualization. Inherits from the Model base class.

   .. attribute:: clusters

      Stores the cluster labels for each sample.

      :type: array

   .. method:: build(data, **kwargs)

      Prepares the dataset for KMeans clustering.

   .. method:: _make_kmeans(n_clusters, init, n_init, tol, algorithm, verbose=0)

      Constructs the KMeans model with specified hyperparameters.

   .. method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning.

   .. method:: hyperparameter_optimization(n_trials=100, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna.

   .. method:: load(foldername)

      Loads the preprocessor, preprocessed data, and cluster labels from the specified folder.

   .. method:: fit(verbose=0)

      Applies KMeans clustering to the preprocessed dataset.

   .. method:: predict(projection='2d', graph_save_extension=None)

      Projects the clustered data into 2D or 3D space and visualizes the clusters.

   .. method:: save()

      Saves the preprocessor, preprocessed data, cluster labels, and model to disk.
      

   .. py:method:: build(data, **kwargs)

      Prepares the dataset for KMeans clustering. This includes preprocessing the data based on its characteristics and specified parameters.

      :param data: The dataset to be used for clustering.
      :type data: Pandas DataFrame
      :param \*\*kwargs: Additional keyword arguments for preprocessing.


   .. py:method:: _make_kmeans(n_clusters, init, n_init, tol, algorithm, verbose=0)

      Constructs the KMeans model with the specified hyperparameters.

      :param n_clusters: The number of clusters to form as well as the number of centroids to generate.
      :type n_clusters: int
      :param init: Method for initialization ('k-means++', 'random' or an ndarray).
      :type init: str
      :param n_init: Number of time the k-means algorithm will be run with different centroid seeds.
      :type n_init: int
      :param tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
      :type tol: float
      :param algorithm: K-means algorithm to use ('auto', 'full' or 'elkan').
      :type algorithm: str
      :param verbose: Verbosity mode.
      :type verbose: int

      :returns: The constructed KMeans clustering model.
      :rtype: sklearn.cluster._kmeans.KMeans


   .. py:method:: _optimizer(trial, **kwargs)

      Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
      a callback within an Optuna optimization study.

      :param trial: An Optuna trial object.
      :type trial: optuna.trial.Trial
      :param \*\*kwargs: Additional keyword arguments for configuring the optimization process.

      :returns: The silhouette score for the clustering configuration defined by the trial.
      :rtype: float


   .. py:method:: hyperparameter_optimization(n_trials=100, info=False, **kwargs)

      Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
      and identifies the best hyperparameters for KMeans clustering.

      :param n_trials: The number of optimization trials to perform. Defaults to 100.
      :type n_trials: int, optional
      :param info: Whether to print detailed information about each trial. Defaults to False.
      :type info: bool, optional
      :param \*\*kwargs: Additional keyword arguments for configuring the optimization process.

      :returns: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
      :rtype: pd.DataFrame


   .. py:method:: load(foldername)

      Loads the preprocessor, preprocessed data, and cluster labels from the specified folder.

      :param foldername: The name of the folder where the data and model are saved.
      :type foldername: str


   .. py:method:: fit(verbose=0)

      Applies KMeans clustering to the preprocessed dataset and updates the 'clusters' attribute with the cluster labels.

      :param verbose: Verbosity mode.
      :type verbose: int


   .. py:method:: predict(projection='2d', graph_save_extension=None)

      Projects the clustered data into 2D or 3D space using t-SNE and visualizes the clusters.

      :param projection: The type of projection for visualization ('2d' or '3d'). Defaults to '2d'.
      :type projection: str, optional
      :param graph_save_extension: Extension to save the graphs (e.g., 'png', 'svg'). If None, graphs are not saved. Defaults to None.
      :type graph_save_extension: str, optional


   .. py:method:: save()

      Saves the preprocessor, preprocessed data, cluster labels, and model to disk.



