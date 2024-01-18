import tensorflow as tf
import pandas as pd
import keras_tuner as kt
import datetime
import os
import joblib 
import numpy as np

from numpy import sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras import regularizers
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from .analysis import Analysis
from keras.models import load_model
from abc import ABC, abstractmethod 
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from keras_tuner import HyperParameters
from scipy.stats import randint as sp_randint


class Model(ABC):
    """
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
    """
    def build():
        pass

    def load():
        pass

    def fit():
        pass

    def predict():
        pass

    def save():
        pass

class NN(Model):
    """
    The class NN is a wrapper around Keras Sequential API, which provides an easy way to create and train neural network models. It can perform hyperparameters search and model building with specified hyperparameters.

    Attributes:
    -----------
        - model: the built Keras model.
        - loss: the loss function used to compile the Keras model.
        - metrics: the metrics used to compile the Keras model.
        - dir_name: a string that defines the name of the directory to save the hyperparameters search results.
        - input_shape: the shape of the input data for the Keras model.
        - output_shape: the shape of the output data for the Keras model.

    """
    def __init__(self, model = None):
        self.model = None
        self.loss = None
        self.metrics = None
        self.dir_name =None
        self.input_shape = None
        self.output_shape = None

    def _model_search(self, hp, **kwargs):
        """
        Searches for the best hyperparameters to create a Keras model using the given hyperparameters space.

        Args:
            hp (keras_tuner.engine.hyperparameters.HyperParameters): Object that holds
                the hyperparameters space to search.

        Returns:
            A compiled Keras model with the best hyperparameters found.
        """
        defaultKwargs = { 'n': [2, 256],'n_l': [1,20], 'list_lr': [1e-1,1e-2, 1e-3, 1e-4,1e-5]}
        kwargs = { **defaultKwargs, **kwargs }
        n_layers = hp.Int('n_layers', min_value=kwargs['n_l'][0], max_value=kwargs['n_l'][1])
        learning_rate = hp.Choice('learning_rate', values= kwargs['list_lr'])
        loss_function = self.loss
        metrics = self.metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = Sequential()
        neurons = hp.Int('neurons_0', min_value=kwargs['n'][0], max_value=kwargs['n'][1])
        model.add(Dense(neurons, activation='relu', input_shape=self.input_shape))
        for i in range(1,n_layers):
            name = 'neurons_' + str(i)
            neurons = hp.Int(name, min_value=kwargs['n'][0], max_value=kwargs['n'][1])
            model.add(Dense(neurons, activation='relu'))
        model.add(Dense(self.output_shape[0]))
        model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)
        return model

    def search_hyperparams(self, X, y, project_name='', y_type='num', verbose=False):
        """
        Perform hyperparameter search for the neural network using Keras Tuner.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): Target data.
            project_name (str): Name of the Keras Tuner project (default '').
            y_type (str): Type of target variable. Either 'num' for numeric or 'cat' for categorical (default 'num').
            verbose (bool): Whether or not to print out information about the search progress (default False).

        Returns:
            dict: A dictionary containing the optimal hyperparameters found by the search.
        """
        
        objective = 'val_loss' if y_type == 'num' else 'val_accuracy'
        
        callback = tf.keras.callbacks.EarlyStopping(monitor=objective, patience=5)
        tuner = kt.RandomSearch(self._model_search,
                                objective=objective,
                                max_trials=30,
                                directory='hpSearch/' + self.dir_name + '_hpSearch',
                                project_name=project_name
                                )
        
        tuner.search(X, y, epochs=200, validation_split=0.2, verbose=True, shuffle=True, callbacks=[callback])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        n = [best_hps.get('neurons_' + str(i)) for i in range(best_hps.get('n_layers'))]
        params = {'n_layers': best_hps.get('n_layers'), 'n_neurons': n, 'learning_rate': best_hps.get('learning_rate')}

        if verbose:
            print(f"Optimal hyperparameters for the neural network are:\n"
                  f"Number of layers: {params['n_layers']}\n"
                  f"Number of neurons: {params['n_neurons']}\n"
                  f"Learning rate: {params['learning_rate']}.")

        return params

    def build(self,input_shape = (1,), output_shape = (1,), n_neurons = [1], n_layers =  1, learning_rate = 1e-3, activation = 'relu', **kwargs):
        """
        Builds a Keras neural network model with the given hyperparameters.

        Args:
            input_shape (tuple, optional): The shape of the input data. Defaults to (1,).
            output_shape (tuple, optional): The shape of the output data. Defaults to (1,).
            n_neurons (list, optional): A list of integers representing the number of neurons in each hidden layer.
                                        The length of the list determines the number of hidden layers. Defaults to [1].
            n_layers (int, optional): The number of hidden layers in the model. Defaults to 1.
            learning_rate (float, optional): The learning rate of the optimizer. Defaults to 1e-3.
            activation (str, optional): The activation function used for the hidden layers. Defaults to 'relu'.

        Returns:
            None.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model = Sequential()
        model.add(keras.layers.Dense(n_neurons[0], activation='relu', input_shape=input_shape))
        for i in range(1, n_layers):
            neurons = n_neurons[i]
            model.add(Dense(neurons, activation=activation))

        model.add(Dense(output_shape[0], activation='linear'))
        model.compile(optimizer=optimizer, loss=self.loss, metrics=self.metrics)
        model.summary()

        self.model = model

    def load(self, path):
        """Load a Keras model from an H5 file.

        Args:
            path (str): Path to the H5 file containing the Keras model.

        Raises:
            ValueError: If the file extension is not '.h5'.

        Returns:
            None
        """
        extension = os.path.splitext(path)[1]
        if not extension:
            path = path + '.h5'
        elif extension != '.h5':
            raise ValueError("Extensão inválida para o modelo de rede neural. A extensão do arquivo deve ser '.h5'.")
    
        self.model = load_model(path)

    def predict(self, x):
        """
        Uses the trained neural network to make predictions on input data.

        Args:
            x (numpy.ndarray): Input data to be used for prediction. It must have the same number of features
                            as the input_shape used to build the network.

        Returns:
            numpy.ndarray: Predicted outputs for the input data.

        Raises:
            ValueError: If the input data x does not have the same number of features as the input_shape
                        used to build the network.
        """
        return self.model.predict(x)

    def fit(self, x, y, validation_data, batch_size = 32, epochs = 500, save = True, patience = 5, path = ''):
        """
        Trains the neural network model using the given input and output data.

        Args:
            x (numpy array): The input data used to train the model.
            y (numpy array): The output data used to train the model.
            validation_data (tuple): A tuple containing the validation data as input and output data.
            batch_size (int): The batch size used for training the model (default=32).
            epochs (int): The number of epochs used for training the model (default=500).
            save (bool): Whether to save the model after training (default=True).
            patience (int): The number of epochs to wait before early stopping if the validation loss does not improve (default=5).
            path (str): The path to save the trained model (default='').

        Returns:
            None
        """
        early_stop = EarlyStopping(monitor='val_loss', verbose=1, patience=patience)
        callbacks = [early_stop]
        if save:
            csv_logger = keras.callbacks.CSVLogger(path + "/training.log")
            callbacks.append(csv_logger)
        self.model.fit(x, y,
                       batch_size = batch_size,
                       epochs = epochs, 
                       validation_data = validation_data, 
                       callbacks = callbacks)

    def save(self, path):
        """
        Saves the trained neural network model to a file.

        Args:
            path: A string specifying the path and filename for the saved model. The ".h5" file extension 
                will be appended to the provided filename if not already present.

        Raises:
            ValueError: If the provided file extension is not ".h5".

        Returns:
            None
        """
        path = os.path.splitext(path)[0]
        self.model.save(path + '.h5')

class RandomForest(Model):
    """
    This class is used to build and search hyperparameters for a random forest model in scikit-learn.

    Attributes:
    -----------
        - model: the built RandomForest scikit-learn model.
    """
    def __init__(self, model = None):
        self.model = model

    def search_hyperparams(self, X, y, verbose = False, **kwargs):
        """
        Perform a hyperparameter search for a Random Forest model using RandomizedSearchCV.

        Args:
            X (numpy array): The feature matrix of the data.
            y (numpy array): The target vector of the data.
            verbose (bool, optional): If True, print the optimal hyperparameters. Defaults to False.
            **kwargs: Additional keyword arguments. The following hyperparameters can be set:
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

        Returns:
            dict: A dictionary with the best hyperparameters found during the search.
        """
        if y.shape[1]==1:
            y = np.reshape(y, (y.shape[0],))
        defaultKwargs = {
                        'n_estimators': sp_randint(10, 1000),
                        'max_features': ['sqrt', 'log2'],
                        'max_depth': [None, 5, 10, 15, 20, 30, 40],
                        'min_samples_split': sp_randint(2, 20),
                        'min_samples_leaf': sp_randint(1, 20),
                        'bootstrap': [True, False]
                        }
        kwargs = { **defaultKwargs, **kwargs }
        if kwargs['y_type'] == 'num':
            rf = RandomForestRegressor()
        else:
            rf = RandomForestClassifier()
        param_dist = {
                        'n_estimators': kwargs['n_estimators'],
                        'max_features': kwargs['max_features'],
                        'max_depth': kwargs['max_depth'],
                        'min_samples_split': kwargs['min_samples_split'],
                        'min_samples_leaf': kwargs['min_samples_leaf'],
                        'bootstrap': kwargs['bootstrap'],
                        }
        random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=50, cv=5)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        random_search.fit(X, y)
        
        print("Best hyperparameters:", random_search.best_params_)
        if verbose:
            print("Os valores ótimos dos hyperparamentros para a Random Forest são:")
            for key, value in random_search.best_params_.items():
                print(key,': ', value)
        return random_search.best_params_

    def build(self, n_estimators = 100, max_depth =  None, min_samples_split = 2, min_samples_leaf = 1, max_features = 'sqrt', **kwargs):  
        """
        Builds a new Random Forest model with the specified hyperparameters.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Default is 100.
            max_depth (int or None, optional): The maximum depth of each tree. None means unlimited. Default is None.
            min_samples_split (int, optional): The minimum number of samples required to split an internal node. Default is 2.
            min_samples_leaf (int, optional): The minimum number of samples required to be at a leaf node. Default is 1.
            max_features (str or int, optional): The maximum number of features to consider when looking for the best split.
                Can be 'sqrt', 'log2', an integer or None. Default is 'sqrt'.
            **kwargs: Additional keyword arguments. Must include a 'y_type' parameter, which should be set to 'num' for
                regression problems and 'cat' for classification problems.

        Returns:
            None
        """
        if kwargs['y_type'] == 'num':
            rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features)
        else:
            rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, max_features = max_features)
        self.model = rf

    def load(self, path):
        """Load a saved random forest model.

        Args:
            path (str): The path to the saved model file. The file must be a joblib file with the extension '.joblib'.

        Raises:
            ValueError: If the extension of the file is not '.joblib'.

        Returns:
            None.
        """
        extension = os.path.splitext(path)[1]
        if extension != '.joblib':
            raise ValueError("Extensão inválida para o modelo de Random Forest. A extensão do arquivo deve ser '.joblib'.")
        
        elif not extension:
            path = path + '.joblib'
        self.model = joblib.load(path)

    def predict(self, x):
        """
        Makes predictions using the trained Random Forest model on the given input data.
        
        Args:
            x: The input data to make predictions on.
            
        Returns:
            An array of predicted target values.
        """
        return self.model.predict(x)

    def fit(self, x, y, validation_data=None, batch_size=32, epochs=500, save=True, patience=5, path=''):
        """
        Trains the Random Forest model on the given input and target data.
        
        Args:
            x (numpy array): The input data to train the model on.
            y (numpy array): The target data to train the model on.
            validation_data (tuple): Not used in this context. For compatibility only.
            batch_size (int): Not used in this context. For compatibility only.
            epochs (int): Not used in this context. For compatibility only.
            save (bool): If True, saves the model to the specified path.
            patience (int): Not used in this context. For compatibility only.
            path (str): The path to save the trained model.

        Returns:
            None
        """
        if y.ndim == 2 and y.shape[1] == 1:
            y = np.ravel(y)
        self.model.fit(x, y)

        if save and path:
            self.save(os.path.join(path, "random_forest_model.joblib"))

    def save(self, path):
        """
        Saves the trained model to a file with the specified path.

        Args:
            path (str): The file path where the model should be saved. The file extension should be '.joblib'.

        Raises:
            ValueError: If the file extension is invalid or missing.

        Returns:
            None
        """
        path = os.path.splitext(path)[0]
        joblib.dump(self.model, path + '.joblib')

class Scaler():
    """The Scaler class is designed to scale and transform data using various scaling techniques. It contains methods for fitting and transforming data, as well as saving and loading scaler objects to and from files."""
    def __init__(self, scaler = None):
        if scaler:
            self.scaler = eval(f"{scaler}()")
        else:
            self.scaler = scaler

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Args:
            data (array-like): The data to be transformed.

        Returns:
            array-like: The transformed data.
        """
        return self.scaler.fit_transform(data)

    def transform(self, data):
        """
        Perform standardization on an array.

        Args:
            data (array-like): The data to be standardized.

        Returns:
            array-like: The standardized data.
        """
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        """
        Scale back the data to the original representation.

        Args:
            data (array-like): The data to be scaled back.

        Returns:
            array-like: The original representation of the data.
        """
        return self.scaler.inverse_transform(data)

    def save(self, path):
        """
        Save the scaler object to a file.

        Args:
            path (str): The path where the scaler object will be saved.
        """
        path = os.path.splitext(path)[0]
        joblib.dump(self.scaler, path + '.pkl')

    def load(self, path):
        """
        Load a saved scaler object from a file.

        Args:
            path (str): The path where the scaler object is saved.
        """
        self.scaler = joblib.load(path)

class AsaML():
    def __init__(self, dir_name = None):
        if not dir_name:
            self.dir_name = datetime.datetime.now().strftime("%Y_%m_%d_(%H-%M-%S)")
        else:
            self.dir_name = dir_name
        self.y_type = {}
        self.var_type = {'X':{}, 'y':{}}
        self.losses_dict = {'bi_cat': 'categorical_crossentropy', 'multi_cat': 'binary_crossentropy', 'num': 'mean_squared_error'}
        self.metrics_dict = {'bi_cat': ['accuracy'], 'multi_cat': ['accuracy'], 'num': ['mean_squared_error']}
        self.scaler = None
    
    @staticmethod
    def identify_categorical_data(df):
        """Identifies categorical data.

        Args:
            df (pandas.DataFrame): The DataFrame containing the data.

        Returns:
            tuple: A tuple containing two DataFrames - one containing the categorical columns and the other containing the numerical columns.

        Raises:
            ValueError: If the input DataFrame is empty or if it contains no categorical or numerical data.
        
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
        """
        df_cat = df.select_dtypes(exclude=["number"])
        df_num = df.select_dtypes("number")
        for col in df_num:
            unique_values = df_num[col].nunique()
            if unique_values < 20 and unique_values/len(df_num) <0.05:
                df_cat[col] = pd.Categorical(df_num[col])
                df_num = df_num.drop(columns=[col])
        return df_cat, df_num

    def pre_processing_train(self, X, y, remove_outlier = False):
        """Perform pre-processing steps for the training dataset.
    
        Args:
            X: pandas.DataFrame - input features.
            y: pandas.DataFrame - target variable.
            remove_outlier: bool - if True, removes the outliers from the dataset.

        Returns:
            X: pandas.DataFrame - pre-processed input features.
            y: pandas.DataFrame - pre-processed target variable.
        """
        # Removendo Outliers
        if remove_outlier:
            y, drop_lines = Analysis().remove_outliers(y, y.columns.tolist())
            for i in drop_lines:
                X = X.drop(index = i)
            X = X.reset_index(drop=True)
            y = y.reset_index(drop=True)

        y_cat, y_num = self.identify_categorical_data(y)
        self.var_type['y']['cat'] = y_cat.columns.tolist()
        self.var_type['y']['num'] = y_num.columns.tolist()
    
        for col in y_cat:
            n_unique = y_cat[col].nunique()
            if n_unique>2:
                self.y_type[col] = 'multi_cat'
            else:
                self.y_type[col] = 'bi_cat'
            dummies = pd.get_dummies(y_cat[col], prefix=col, prefix_sep="___")
            y_cat = pd.concat([y_cat, dummies], axis=1)
            y_cat = y_cat.drop(col, axis=1)
        for col in y_num:
            self.y_type[col] = 'num'
        y = pd.concat([y_cat, y_num], axis=1)
        X_cat, X_num = self.identify_categorical_data(X)
        for col in X_cat:
            dummies = pd.get_dummies(X_cat[col], prefix=col, prefix_sep="___")
            X_cat = pd.concat([X_cat, dummies], axis=1)
            X_cat = X_cat.drop(col, axis=1)

        X = pd.concat([X_cat, X_num], axis=1)

        return X, y

    def __add_random_value_to_max(self, row):
        max_idx = row.idxmax()
        row[max_idx] += np.random.rand()
        return row
 
    def train_model(self, X = None, y =None, name_model = None, save =True, scaling = True, scaler_Type = 'StandardScaler', remove_outlier = False, search = False, params = None, **kwargs):
        """
        Train a model on a given dataset.

        Args:
            X: A pandas dataframe containing the feature data. Default is None.
            y: A pandas dataframe containing the target data. Default is None.
            name_model: The name of the model to train. Default is None.
            save: A boolean indicating whether to save the model or not. Default is True.
            scaling: A boolean indicating whether to perform data scaling or not. Default is True.
            scaler_Type: The type of data scaling to perform. Must be one of 'StandardScaler', 'Normalizer', or 'MinMaxScaler'. Default is 'StandardScaler'.
            remove_outlier: A boolean indicating whether to remove outliers or not. Default is False.
            search: A boolean indicating whether to search for the best hyperparameters or not. Default is False.
            params: A dictionary containing the hyperparameters to use for training. Default is None.
            **kwargs: Additional arguments to be passed to the training function.

        Returns:
            None
        """

        if  (X is None and y is  None) or (X.empty or y.empty):
            raise ValueError("As variáveis de entrada 'X' e 'y' não podem ser vazias.")
        path = './models/'+ self.dir_name
        if save:
            if not os.path.exists(path):
                os.makedirs(path)
                with open(path + "/READ.md", 'w') as f:
                    d = {'X_var_list': X.columns.tolist(), 'y_car_list': y.columns.tolist()}
                    f.writelines(f"{d}\n")
                    f.writelines(f"pre processing optinal paramenters: save = {save},  scaling = {scaling}, scaler = {scaler_Type}, remove_outlier = {remove_outlier} \n")
                for column in y:
                    p = path + '/' + column
                    if not os.path.exists(p):
                        os.makedirs(p)
        # Removendo Nan values
        df = pd.concat([X, y], axis=1)
        df = df.dropna()
        df = df.reset_index(drop=True)
        X = df[X.columns.tolist()]
        y = df[y.columns.tolist()]
        
        X_cat, X_num = self.identify_categorical_data(X)
        self.var_type['X']['cat'] = X_cat.columns.tolist()
        self.var_type['X']['num'] = X_num.columns.tolist()
        method_list = ['StandardScaler', 'Normalizer', 'MinMaxScaler']
        if scaling:
            if scaler_Type not in method_list:
                raise NameError(f"O método {scaler_Type} não consta na lista de métodos suportados para data scaling. Por favor, tente:'StandardScaler',  Normalizer' ou 'MinMaxScaler'.")
            # Scalling 
            scaler = Scaler(scaler_Type)
            sc_X_num = scaler.fit_transform(X_num)
            self.scaler = scaler
            if save:
                scaler.save(path + '/scaler.pkl')
            X_num = pd.DataFrame(sc_X_num, columns= X_num.columns.tolist())
            X = pd.concat([X_cat, X_num], axis=1)
        
        model_dict = {}
        for col in y.columns.tolist():
            X_, y_ = self.pre_processing_train(X, y[[col]], remove_outlier)
            X_pp = np.array(X_.values.tolist())
            y_pp = np.array(y_.values.tolist())
            self.input_shape = (X_pp.shape[1],)
            self.output_shape = (y_pp.shape[1],)
            model = eval(f"{name_model}()") 
            if name_model == 'NN':
                model.loss = self.losses_dict[self.y_type[col]]
                model.metrics = self.metrics_dict[self.y_type[col]]
                model.dir_name = self.dir_name
                model.input_shape = (X_pp.shape[1],)
                model.output_shape = (y_pp.shape[1],)
            if search:
                params = model.search_hyperparams(X_pp, y_pp, project_name = col, y_type = self.y_type[col])
            elif params == None:
                raise ValueError("A variável 'params' não pode ser vazia. Essa variável corresponde aos parâmetros para a construção do model.")
            
            params['input_shape'] = (X_pp.shape[1],)
            params['output_shape'] = (y_pp.shape[1],)
            model.build(y_type = self.y_type[col], **params)
            kfold = KFold(n_splits=5, shuffle=True)
            metrics = []
            X_train, X_test, y_train, y_test = train_test_split(X_pp, y_pp, test_size=0.2)
            path = './models/'+ self.dir_name+'/' + col
            for fold, (train_index, val_index) in enumerate(kfold.split(X_train, y_train)):
                xtrain, ytrain = X_train[train_index], y_train[train_index]
                xval, yval = X_train[val_index], y_train[val_index]
                model.fit(xtrain, ytrain, validation_data=(xval, yval), path = path, **kwargs)              
                if self.y_type[col] == 'num':
                    y_pred = model.predict(X_test)
                    metrics.append([fold, mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), 
                      sqrt(mean_squared_error(y_test, y_pred)), r2_score(y_test, y_pred)])
                else: 
                    y_pred = model.predict(X_test)
                    df_pred = pd.DataFrame(y_pred, columns = y_.columns.tolist())
                    df_pred = df_pred.apply(self.__add_random_value_to_max, axis=1)
                    df_pred = df_pred.eq(df_pred.max(axis=1),axis=0).astype(int)
                    y_pred = np.array(df_pred.values.tolist())
                    metrics.append([fold,accuracy_score(y_test, y_pred), precision_score(y_test, y_pred, average='micro'), 
                    recall_score(y_test, y_pred,average='micro'), f1_score(y_test, y_pred,average='micro')])
                
            if save:
                with open(path + "/parameters.txt", 'w') as f:
                    d = {'model_name': name_model, 'input_list': X_.columns.tolist(), 'output_list' : y_.columns.tolist(), 'var_type': self.var_type}
                    f.writelines(f"{d}\n")
                    f.writelines(f"pre processing optinal parameters: save = {save},  scaling = {scaling}, scaler = {scaler_Type}, remove_outlier = {remove_outlier}\n")
                    f.writelines(f"model parameters: {params}\n")  

                model.save(path + "/model_" + col)

                if self.y_type[col] == 'num':
                    columns = ['fold', 'mae', 'mse', 'rmse', 'r2']
                else: 
                    columns = ['fold', 'accuracy', 'precision', 'recall', 'f1_score']
                
                df_metrics = pd.DataFrame(metrics, columns=columns)
                df_metrics.to_csv(path + "/metrics.csv")

            model_dict[col] = {'model': model,  'model_name': name_model , 'var_type': self.var_type,
                                'input_list': X_.columns.tolist(), 'output_list' : y_.columns.tolist()}
                    
        return model_dict

    def load_model(self, path = ''):
        """
        Loads a saved model from the specified path and returns a dictionary of models
        with their corresponding parameters.

        Args:
            path (str): The path where the model and its associated files are saved.
                        Defaults to an empty string.

        Returns:
            dict: A dictionary containing the loaded models with their corresponding parameters.

        Raises:
            ValueError: If the path argument is empty.

        .. note::
            The path variable must be the address of the 'dirname' folder and MUST contain the scaler.pkl file and each subdirectory MUST contain the 'paramenters', 'model' files.
        
        """
        if not path:
            raise ValueError("A variável 'path' não pode ser vazia.")

        parent_dir = path
        subdirs = [os.path.join(parent_dir, name) for name in os.listdir(parent_dir)
           if os.path.isdir(os.path.join(parent_dir, name))]
        
        scaler = Scaler()
        scaler.load(path +'/scaler.pkl')
        self.scaler = scaler
        model_dict = {}

        for p in subdirs:
            with open( p + '/parameters.txt', 'r') as file:
                first_line = file.readline()
            col = os.path.basename(p)
            model_dict[col] = eval(f"{first_line}")
            path =  p +'/model_' + col
            model =  eval(f"{model_dict[col]['model_name']}()")
            model.load(path)
            model_dict[col]['model'] =  model
        
        return model_dict
        

    def pre_processing_predict(self, X, input_list, var_type):
        """
        Pre-processes the input data before prediction by scaling numerical features and creating dummy variables
        for categorical features. Also handles missing and extra features in the input data.

        Args:
            X (pandas.DataFrame): The input data to be pre-processed.
            input_list (list): A list of expected input features.
            var_type (dict): A dictionary with the types of the input features. The keys 'cat' and 'num' contain lists 
                of categorical and numerical feature names respectively.

        Returns:
            pandas.DataFrame: The pre-processed input data with scaled numerical features and dummy variables 
                for categorical features. Any missing or extra features are handled accordingly.
        """
        X = X.dropna()
        X = X.reset_index(drop=True)
        cat_list = var_type['cat']
        num_list = var_type['num']
        X_num = X[num_list]
        sc_X_num = self.scaler.transform(X_num)
        X_num = pd.DataFrame(sc_X_num, columns= X_num.columns.tolist())
        X_cat = X[cat_list]  
        for col in X_cat:
            dummies = pd.get_dummies(X_cat[col], prefix=col, prefix_sep="___")
            X_cat = pd.concat([X_cat, dummies], axis=1)
            X_cat = X_cat.drop(col, axis=1)
        if len(cat_list) > 0:
            if len(num_list) > 0: 
                X = pd.concat([X_cat, X_num], axis=1)
            else:
                X = X_cat
        else:
            if len(num_list) > 0: 
                X = X_num
        
        X_cols = X.columns.tolist()
        missing_list = [x for x in input_list if x not in X_cols]
        extra_list = [x for x in X_cols if x not in input_list]
        X = X.drop(columns= extra_list)
        for col in missing_list:
            X[col] = 0
        return X

    def pos_processing(self, y, output_list):
        """
        Post-processes the output of a model prediction to transform it into a more usable format.

        Args:
            y (np.ndarray): The output of the model prediction, as a NumPy array.
            output_list (list): A list of column names representing the output variables.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the post-processed output values.

        This function takes the output of a model prediction, which is typically a NumPy array of raw output values, and transforms it into a more usable format. The output variables are expected to have been one-hot encoded with the use of triple underscores ('___') as separator, and possibly have a random value added to the max value of each row. The function first separates the categorical and numerical variables, then processes the categorical variables by selecting the maximum value for each row and one-hot encoding them. Finally, it concatenates the categorical and numerical variables back together to produce a pandas DataFrame containing the post-processed output values.

        """
        df_y = pd.DataFrame(y, columns = output_list)
        df_num = df_y.loc[:, ~df_y.columns.str.contains('___')]
        df_cat = df_y.filter(like='___')
        if not df_cat.empty:
            df_cat = df_cat.apply(self.__add_random_value_to_max, axis=1)
            df_cat=df_cat.eq(df_cat.max(axis=1),axis=0).astype(int)
            df_cat=pd.from_dummies(df_cat, sep="___")
        if len(df_cat.columns) > 0:
            if len(df_num.columns) > 0: 
                y = pd.concat([df_num, df_cat])
            else:
                y = df_cat
        else:
            if len(df_num.columns) > 0: 
                y = df_num
        return y

    def predict_all(self, X, model_dict):
        """
        Apply all models in the model dictionary to the input data frame X and return the predictions.

        Args:
            X: A pandas DataFrame representing the input data.
            model_dict: A dictionary containing the models and their associated metadata. The keys are the names of the 
                        models and the values are themselves dictionaries containing the following keys:
                        - 'model': A trained machine learning model.
                        - 'input_list': A list of the names of the input features used by the model.
                        - 'output_list': A list of the names of the output features produced by the model.
                        - 'var_type': A dictionary containing the types of the input and output features, with the keys 
                                        'X' and 'y', respectively, and the values being dictionaries themselves with 
                                        the following keys:
                                        - 'cat': A list of the categorical input features.
                                        - 'num': A list of the numerical input features.

        Returns:
            A pandas DataFrame containing the predictions of all models in the model dictionary. The columns of the 
            DataFrame are the names of the models, and the rows correspond to the input rows in X.

        Raises:
            ValueError: If X is empty or None, or if the model dictionary is empty or None.
        """
        out = pd.DataFrame()
        for col, model_dict_col in model_dict.items():
            model = model_dict_col['model']
            X_pp = self.pre_processing_predict(X, model_dict_col['input_list'], model_dict_col['var_type']['X'])
            X_pp = np.array(X_pp.values.tolist())
            Y_pred = model.predict(X_pp)
            Y_pred = self.pos_processing(Y_pred, model_dict_col['output_list'])
            out[col] = Y_pred[col]
        return out

    def full_cycle(self, X_pred, load = False, **kwargs):
        """
        Performs the full cycle of the machine learning pipeline: loads or trains the models, preprocesses the input data,
        generates predictions, and post-processes the output data.

        Args:
            X_pred (pandas.DataFrame): Input data to generate predictions for.
            load (bool, optional): If True, loads the trained models from disk instead of training new ones. Default is False.
            **kwargs: Additional keyword arguments passed to either `load_model()` or `train_model()` method.

        Returns:
            pandas.DataFrame: Dataframe with the generated predictions.

        Raises:
            ValueError: If `load` is True and `path` is not provided in `kwargs`.
        """
        if load:
            model_dict = self.load_model(**kwargs)
        else:
            model_dict = self.train_model(**kwargs)
        out = self.predict_all(X_pred, model_dict)
        return out