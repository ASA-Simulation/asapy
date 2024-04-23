:py:mod:`asapy.prediction.model`
================================

.. py:module:: asapy.prediction.model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.prediction.model.Model




.. py:class:: Model(target, name, seed=None)


   Bases: :py:obj:`abc.ABC`

   Base class for all models in the library. This abstract class provides a template for the methods and attributes
   that all models should implement and maintain.

   .. attribute:: target

      The name of the target variable in the dataset.

      :type: str

   .. attribute:: name

      The name of the model, used for identification and referencing.

      :type: str

   .. attribute:: seed

      Random seed used to ensure reproducibility. Defaults to None.

      :type: int, optional

   .. attribute:: preprocessor

      Stores the preprocessing pipelines for features and target. Defaults to None.

      :type: dict

   .. attribute:: preprocessed_data

      Stores the preprocessed training, validation, and test data. Defaults to None.

      :type: dict

   .. attribute:: task

      Task type (e.g., 'classification', 'regression'). Defaults to None.

      :type: str, optional

   .. attribute:: model

      The underlying model object. Defaults to None.

      :type: model object, optional

   .. attribute:: hyperparameter

      Stores hyperparameters used by the model. Defaults to None.

      :type: dict, optional

   .. attribute:: history_kfold

      Stores the training history for each fold during k-fold cross-validation. Defaults to None.

      :type: list, optional

   .. attribute:: have_cat

      Indicates whether the dataset contains categorical features. Defaults to False.

      :type: bool

   .. method:: build()

      Placeholder method for building the model structure. Should be overridden by subclasses.

   .. method:: _optimizer()

      Placeholder method for setting up the optimization algorithm. Should be overridden by subclasses.

   .. method:: hyperparameter_optimization()

      Placeholder for hyperparameter optimization. Should be overridden by subclasses.

   .. method:: load()

      Placeholder for loading a saved model from disk. Should be overridden by subclasses.

   .. method:: fit()

      Placeholder for fitting the model on training data. Should be overridden by subclasses.

   .. method:: predict()

      Placeholder for making predictions with the trained model. Should be overridden by subclasses.

   .. method:: save()

      Placeholder for saving the current state of the model to disk. Should be overridden by subclasses.

   .. method:: _preprocess(data, target_one_hot_encoder=False, **kwargs)

      Preprocesses the data according to the specified parameters.

   .. method:: _cluster_preprocess(data, **kwargs)

      Preprocesses the data for clustering tasks according to the specified parameters.
      

   .. py:method:: build()

      Placeholder method for setting up the optimization algorithm. This method should be overridden by
      subclasses to specify how the model should be optimized during training (e.g., SGD, Adam).


   .. py:method:: _optimizer()

      Placeholder method for setting up the optimization algorithm. This method should be overridden by
      subclasses to specify how the model should be optimized during training (e.g., SGD, Adam).


   .. py:method:: hyperparameter_optimization()

      Placeholder method for performing hyperparameter optimization. This method should be overridden by
      subclasses to implement hyperparameter tuning techniques (e.g., grid search, random search).


   .. py:method:: load()

      Placeholder method for loading a saved model from disk. This method should be overridden by subclasses
      to enable loading model state, allowing for model persistence across sessions.


   .. py:method:: fit()

      Placeholder method for fitting the model on the training data. This method should be overridden by
      subclasses to implement the training process, including any preprocessing, training iterations, and
      validation.


   .. py:method:: predict()

      Placeholder method for making predictions with the trained model. This method should be overridden
      by subclasses to use the model for making predictions on new or unseen data.


   .. py:method:: save()

      Placeholder method for saving the current state of the model to disk. This method should be overridden
      by subclasses to provide a way to serialize and save the model structure and trained weights.


   .. py:method:: _preprocess(data, target_one_hot_encoder=False, **kwargs)

      Preprocesses the data based on the provided parameters and updates the instance attributes accordingly.

      :param data: The dataset to preprocess.
      :type data: Pandas DataFrame
      :param target_one_hot_encoder: Indicates whether to apply one-hot encoding to the target variable. Defaults to False.
      :type target_one_hot_encoder: bool, optional
      :param \*\*kwargs: Additional keyword arguments for preprocessing options.

      Note: This method updates the 'preprocessed_data' and 'preprocessor' attributes of the instance.


   .. py:method:: _cluster_preprocess(data, **kwargs)

      Preprocesses the data for clustering tasks based on the provided parameters and updates the instance attributes accordingly.

      :param data: The dataset to preprocess.
      :type data: Pandas DataFrame
      :param \*\*kwargs: Additional keyword arguments for preprocessing options.

      Note: This method updates the 'preprocessed_data' and 'preprocessor' attributes of the instance.



