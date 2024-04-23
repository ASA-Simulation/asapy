:py:mod:`asapy.prediction.KMeans`
=================================

.. py:module:: asapy.prediction.KMeans


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.prediction.KMeans.KMeans




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



