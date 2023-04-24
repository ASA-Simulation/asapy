:py:mod:`asapy.models`
======================

.. py:module:: asapy.models


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   asapy.models.Model
   asapy.models.NN
   asapy.models.RandomForest
   asapy.models.Scaler
   asapy.models.AsaML




.. py:class:: Model

   Bases: :py:obj:`abc.ABC`

   Abstract base class for machine learning models.

   Attributes:
   -----------
   None

   Methods:
   --------
   build():
       Builds the machine learning model.

   load(path: str):
       Loads the machine learning model from a file.

   fit(X_train: np.ndarray, y_train: np.ndarray):
       Trains the machine learning model on the input data.

   predict(X: np.ndarray):
       Makes predictions using the trained machine learning model.

   save(path: str):
       Saves the trained machine learning model to a file.

   Raises:
   -------
   None

   .. py:method:: build()


   .. py:method:: load()


   .. py:method:: fit()


   .. py:method:: predict()


   .. py:method:: save()



.. py:class:: NN(model=None)

   Bases: :py:obj:`Model`

   The class NN is a wrapper around Keras Sequential API, which provides an easy way to create and train neural network models. It can perform hyperparameters search and model building with specified hyperparameters.

   Attributes:
   -----------
       - model: the built Keras model.
       - loss: the loss function used to compile the Keras model.
       - metrics: the metrics used to compile the Keras model.
       - dir_name: a string that defines the name of the directory to save the hyperparameters search results.
       - input_shape: the shape of the input data for the Keras model.
       - output_shape: the shape of the output data for the Keras model.


   .. py:method:: _model_search(hp, **kwargs)

      Searches for the best hyperparameters to create a Keras model using the given hyperparameters space.

      :param hp: Object that holds
                 the hyperparameters space to search.
      :type hp: keras_tuner.engine.hyperparameters.HyperParameters

      :returns: A compiled Keras model with the best hyperparameters found.


   .. py:method:: search_hyperparams(X, y, project_name='', verbose=False)

      Perform hyperparameter search for the neural network using Keras Tuner.

      :param X: Input data.
      :type X: numpy.ndarray
      :param y: Target data.
      :type y: numpy.ndarray
      :param project_name: Name of the Keras Tuner project (default '').
      :type project_name: str
      :param verbose: Whether or not to print out information about the search progress (default False).
      :type verbose: bool

      :returns: A dictionary containing the optimal hyperparameters found by the search.
      :rtype: dict

      :raises ValueError: If `self.loss` is not a supported loss function.


   .. py:method:: build(input_shape=(1, ), output_shape=(1, ), n_neurons=[1], n_layers=1, learning_rate=0.001, activation='relu', **kwargs)

      Builds a Keras neural network model with the given hyperparameters.

      :param input_shape: The shape of the input data. Defaults to (1,).
      :type input_shape: tuple, optional
      :param output_shape: The shape of the output data. Defaults to (1,).
      :type output_shape: tuple, optional
      :param n_neurons: A list of integers representing the number of neurons in each hidden layer.
                        The length of the list determines the number of hidden layers. Defaults to [1].
      :type n_neurons: list, optional
      :param n_layers: The number of hidden layers in the model. Defaults to 1.
      :type n_layers: int, optional
      :param learning_rate: The learning rate of the optimizer. Defaults to 1e-3.
      :type learning_rate: float, optional
      :param activation: The activation function used for the hidden layers. Defaults to 'relu'.
      :type activation: str, optional

      :returns: None.


   .. py:method:: load(path)

      Load a Keras model from an H5 file.

      :param path: Path to the H5 file containing the Keras model.
      :type path: str

      :raises ValueError: If the file extension is not '.h5'.

      :returns: None


   .. py:method:: predict(x)

      Uses the trained neural network to make predictions on input data.

      :param x: Input data to be used for prediction. It must have the same number of features
                as the input_shape used to build the network.
      :type x: numpy.ndarray

      :returns: Predicted outputs for the input data.
      :rtype: numpy.ndarray

      :raises ValueError: If the input data x does not have the same number of features as the input_shape
          used to build the network.


   .. py:method:: fit(x, y, validation_data, batch_size=32, epochs=500, save=True, patience=5, path='')

      Trains the neural network model using the given input and output data.

      :param x: The input data used to train the model.
      :type x: numpy array
      :param y: The output data used to train the model.
      :type y: numpy array
      :param validation_data: A tuple containing the validation data as input and output data.
      :type validation_data: tuple
      :param batch_size: The batch size used for training the model (default=32).
      :type batch_size: int
      :param epochs: The number of epochs used for training the model (default=500).
      :type epochs: int
      :param save: Whether to save the model after training (default=True).
      :type save: bool
      :param patience: The number of epochs to wait before early stopping if the validation loss does not improve (default=5).
      :type patience: int
      :param path: The path to save the trained model (default='').
      :type path: str

      :returns: None


   .. py:method:: save(path)

      Saves the trained neural network model to a file.

      :param path: A string specifying the path and filename for the saved model. The ".h5" file extension
                   will be appended to the provided filename if not already present.

      :raises ValueError: If the provided file extension is not ".h5".

      :returns: None



.. py:class:: RandomForest(model=None)

   Bases: :py:obj:`Model`

   This class is used to build and search hyperparameters for a random forest model in scikit-learn.

   Attributes:
   -----------
       - model: the built RandomForest scikit-learn model.

   .. py:method:: search_hyperparams(X, y, verbose=False, **kwargs)

      Perform a hyperparameter search for a Random Forest model using RandomizedSearchCV.

      :param X: The feature matrix of the data.
      :type X: numpy array
      :param y: The target vector of the data.
      :type y: numpy array
      :param verbose: If True, print the optimal hyperparameters. Defaults to False.
      :type verbose: bool, optional
      :param \*\*kwargs: Additional keyword arguments. The following hyperparameters can be set:
                         - n_estimators (int): Number of trees in the forest. Defaults to sp_randint(10, 1000).
                         - max_features (list): The number of features to consider when looking for the best split.
                         Allowed values are 'sqrt', 'log2' or a float between 0 and 1. Defaults to ['sqrt', 'log2'].
                         - max_depth (list): The maximum depth of the tree. Defaults to [None, 5, 10, 15, 20, 30, 40].
                         - min_samples_split (int): The minimum number of samples required to split an internal node.
                         Defaults to sp_randint(2, 20).
                         - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
                         Defaults to sp_randint(1, 20).
                         - bootstrap (bool): Whether bootstrap samples are used when building trees.
                         Defaults to [True, False].
                         - y_type (str): Type of target variable. Either 'num' for numeric or 'cat' for categorical.
                         Defaults to 'cat'.

      :returns: A dictionary with the best hyperparameters found during the search.
      :rtype: dict


   .. py:method:: build(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', **kwargs)

      Builds a new Random Forest model with the specified hyperparameters.

      :param n_estimators: The number of trees in the forest. Default is 100.
      :type n_estimators: int, optional
      :param max_depth: The maximum depth of each tree. None means unlimited. Default is None.
      :type max_depth: int or None, optional
      :param min_samples_split: The minimum number of samples required to split an internal node. Default is 2.
      :type min_samples_split: int, optional
      :param min_samples_leaf: The minimum number of samples required to be at a leaf node. Default is 1.
      :type min_samples_leaf: int, optional
      :param max_features: The maximum number of features to consider when looking for the best split.
                           Can be 'sqrt', 'log2', an integer or None. Default is 'sqrt'.
      :type max_features: str or int, optional
      :param \*\*kwargs: Additional keyword arguments. Must include a 'y_type' parameter, which should be set to 'num' for
                         regression problems and 'cat' for classification problems.

      :returns: None


   .. py:method:: load(path)

      Load a saved random forest model.

      :param path: The path to the saved model file. The file must be a joblib file with the extension '.joblib'.
      :type path: str

      :raises ValueError: If the extension of the file is not '.joblib'.

      :returns: None.


   .. py:method:: predict(x)

      Makes predictions using the trained Random Forest model on the given input data.

      :param x: The input data to make predictions on.

      :returns: An array of predicted target values.


   .. py:method:: fit(x, y)

      Trains the Random Forest model on the given input and target data.

      :param x: The input data to train the model on.
      :param y: The target data to train the model on.

      :returns: None


   .. py:method:: save(path)

      Saves the trained model to a file with the specified path.

      :param path: The file path where the model should be saved. The file extension should be '.joblib'.
      :type path: str

      :raises ValueError: If the file extension is invalid or missing.

      :returns: None



.. py:class:: Scaler(scaler=None)

   The Scaler class is designed to scale and transform data using various scaling techniques. It contains methods for fitting and transforming data, as well as saving and loading scaler objects to and from files.

   .. py:method:: fit_transform(data)

      Fit to data, then transform it.

      :param data: The data to be transformed.
      :type data: array-like

      :returns: The transformed data.
      :rtype: array-like


   .. py:method:: transform(data)

      Perform standardization on an array.

      :param data: The data to be standardized.
      :type data: array-like

      :returns: The standardized data.
      :rtype: array-like


   .. py:method:: inverse_transform(data)

      Scale back the data to the original representation.

      :param data: The data to be scaled back.
      :type data: array-like

      :returns: The original representation of the data.
      :rtype: array-like


   .. py:method:: save(path)

      Save the scaler object to a file.

      :param path: The path where the scaler object will be saved.
      :type path: str


   .. py:method:: load(path)

      Load a saved scaler object from a file.

      :param path: The path where the scaler object is saved.
      :type path: str



.. py:class:: AsaML(dir_name=None)

   .. py:method:: identify_categorical_data(df)
      :staticmethod:

      Identifies categorical data.

      :param df: The DataFrame containing the data.
      :type df: pandas.DataFrame

      :returns: A tuple containing two DataFrames - one containing the categorical columns and the other containing the numerical columns.
      :rtype: tuple

      :raises ValueError: If the input DataFrame is empty or if it contains no categorical or numerical data.

      Example usage:

      .. code-block::

          >>> import pandas as pd
          >>> df = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1, 2, 3], 'col3': ['x', 'y', 'z']})
          >>> df_cat, df_num = identify_categorical_data(df)
          >>> df_cat
          col1
          0    a
          1    b
          2    c
          >>> df_num
          col2
          0     1
          1     2
          2     3


   .. py:method:: pre_processing_train(X, y, remove_outlier=False)

      Perform pre-processing steps for the training dataset.

      :param X: pandas.DataFrame - input features.
      :param y: pandas.DataFrame - target variable.
      :param remove_outlier: bool - if True, removes the outliers from the dataset.

      :returns: pandas.DataFrame - pre-processed input features.
                y: pandas.DataFrame - pre-processed target variable.
      :rtype: X


   .. py:method:: __add_random_value_to_max(row)


   .. py:method:: train_model(X=None, y=None, name_model=None, save=True, scaling=True, scaler_Type='StandardScaler', remove_outlier=False, search=False, params=None, **kwargs)

      Train a model on a given dataset.

      :param X: A pandas dataframe containing the feature data. Default is None.
      :param y: A pandas dataframe containing the target data. Default is None.
      :param name_model: The name of the model to train. Default is None.
      :param save: A boolean indicating whether to save the model or not. Default is True.
      :param scaling: A boolean indicating whether to perform data scaling or not. Default is True.
      :param scaler_Type: The type of data scaling to perform. Must be one of 'StandardScaler', 'Normalizer', or 'MinMaxScaler'. Default is 'StandardScaler'.
      :param remove_outlier: A boolean indicating whether to remove outliers or not. Default is False.
      :param search: A boolean indicating whether to search for the best hyperparameters or not. Default is False.
      :param params: A dictionary containing the hyperparameters to use for training. Default is None.
      :param \*\*kwargs: Additional arguments to be passed to the training function.

      :returns: None


   .. py:method:: load_model(path='')

      Loads a saved model from the specified path and returns a dictionary of models
      with their corresponding parameters.

      :param path: The path where the model and its associated files are saved.
                   Defaults to an empty string.
      :type path: str

      :returns: A dictionary containing the loaded models with their corresponding parameters.
      :rtype: dict

      :raises ValueError: If the path argument is empty.

      .. note::
          The path variable must be the address of the 'dirname' folder and MUST contain the scaler.pkl file and each subdirectory MUST contain the 'paramenters', 'model' files.



   .. py:method:: pre_processing_predict(X, input_list, var_type)

      Pre-processes the input data before prediction by scaling numerical features and creating dummy variables
      for categorical features. Also handles missing and extra features in the input data.

      :param X: The input data to be pre-processed.
      :type X: pandas.DataFrame
      :param input_list: A list of expected input features.
      :type input_list: list
      :param var_type: A dictionary with the types of the input features. The keys 'cat' and 'num' contain lists
                       of categorical and numerical feature names respectively.
      :type var_type: dict

      :returns:

                The pre-processed input data with scaled numerical features and dummy variables
                    for categorical features. Any missing or extra features are handled accordingly.
      :rtype: pandas.DataFrame


   .. py:method:: pos_processing(y, output_list)

      Post-processes the output of a model prediction to transform it into a more usable format.

      :param y: The output of the model prediction, as a NumPy array.
      :type y: np.ndarray
      :param output_list: A list of column names representing the output variables.
      :type output_list: list

      :returns: A pandas DataFrame containing the post-processed output values.
      :rtype: pd.DataFrame

      This function takes the output of a model prediction, which is typically a NumPy array of raw output values, and transforms it into a more usable format. The output variables are expected to have been one-hot encoded with the use of triple underscores ('___') as separator, and possibly have a random value added to the max value of each row. The function first separates the categorical and numerical variables, then processes the categorical variables by selecting the maximum value for each row and one-hot encoding them. Finally, it concatenates the categorical and numerical variables back together to produce a pandas DataFrame containing the post-processed output values.



   .. py:method:: predict_all(X, model_dict)

      Apply all models in the model dictionary to the input data frame X and return the predictions.

      :param X: A pandas DataFrame representing the input data.
      :param model_dict: A dictionary containing the models and their associated metadata. The keys are the names of the
                         models and the values are themselves dictionaries containing the following keys:
                         - 'model': A trained machine learning model.
                         - 'input_list': A list of the names of the input features used by the model.
                         - 'output_list': A list of the names of the output features produced by the model.
                         - 'var_type': A dictionary containing the types of the input and output features, with the keys
                                         'X' and 'y', respectively, and the values being dictionaries themselves with
                                         the following keys:
                                         - 'cat': A list of the categorical input features.
                                         - 'num': A list of the numerical input features.

      :returns: A pandas DataFrame containing the predictions of all models in the model dictionary. The columns of the
                DataFrame are the names of the models, and the rows correspond to the input rows in X.

      :raises ValueError: If X is empty or None, or if the model dictionary is empty or None.


   .. py:method:: full_cycle(X_pred, load=False, **kwargs)

      Performs the full cycle of the machine learning pipeline: loads or trains the models, preprocesses the input data,
      generates predictions, and post-processes the output data.

      :param X_pred: Input data to generate predictions for.
      :type X_pred: pandas.DataFrame
      :param load: If True, loads the trained models from disk instead of training new ones. Default is False.
      :type load: bool, optional
      :param \*\*kwargs: Additional keyword arguments passed to either `load_model()` or `train_model()` method.

      :returns: Dataframe with the generated predictions.
      :rtype: pandas.DataFrame

      :raises ValueError: If `load` is True and `path` is not provided in `kwargs`.



