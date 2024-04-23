:py:mod:`asapy.prediction.DBSCAN`
=================================

.. py:module:: asapy.prediction.DBSCAN


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.prediction.DBSCAN.DBSCAN




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



