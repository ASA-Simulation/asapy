import warnings
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OUTDATED_IGNORE'] = '1'

import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 400

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import gower

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans as _KMeans, DBSCAN as _DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances, confusion_matrix, r2_score

from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.models import Sequential, load_model 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.metrics import AUC, Precision, Recall, CategoricalAccuracy, MeanSquaredError, RootMeanSquaredError, \
                          MeanSquaredLogarithmicError, MeanAbsoluteError, MeanAbsolutePercentageError                        

from abc import ABC
from xgboost import XGBClassifier, XGBRegressor
from matplotlib.ticker import MaxNLocator
from datetime import datetime
from joblib import dump, load
from time import time
from math import log, ceil
from functools import wraps


def validate_split_size(func):

    @wraps(func)
    def wrapper(self, data, **kwargs):
        split_size = kwargs.get('split_size', (0.7, 0.15, 0.15))

        if not isinstance(split_size, tuple):
            print("split_size deve ser uma tupla!")

        elif len(split_size) != 3:
            print("split_size deve ter 3 elementos")

        elif any(ss <= 0 for ss in split_size):
            print("Todos os valores em 'split_size' devem ser maiores que zero.")
        
        elif sum(split_size) != 1:
            print("A soma dos valores em 'split_size' deve ser igual a 1.")
        
        else:
            return func(self, data, **kwargs)
        
    return wrapper


def validate_task(func):

    @wraps(func)
    def wrapper(self, data, **kwargs):
        task = kwargs.get('task', None)

        if task is not None and task not in ['regression', 'classification']:
            print("Já que deseja informar a task explicitamente, ela deve ser 'regression' ou 'classification'")
        
        else:
            return func(self, data, **kwargs)
        
    return wrapper


def validate_target(func):

    @wraps(func)
    def wrapper(self, data, **kwargs):

        if self.target not in data.columns:
            print(f"O dataset não contém a variável '{self.target}'")
        
        else:
            return func(self, data, **kwargs)
        
    return wrapper


class Model(ABC):
    """
    Base class for all models in the library. This abstract class provides a template for the methods and attributes
    that all models should implement and maintain.

    Attributes:
        target (str): The name of the target variable in the dataset.
        name (str): The name of the model, used for identification and referencing.
        seed (int, optional): Random seed used to ensure reproducibility. Defaults to None.
        preprocessor (dict): Stores the preprocessing pipelines for features and target. Defaults to None.
        preprocessed_data (dict): Stores the preprocessed training, validation, and test data. Defaults to None.
        task (str, optional): Task type (e.g., 'classification', 'regression'). Defaults to None.
        model (model object, optional): The underlying model object. Defaults to None.
        hyperparameter (dict, optional): Stores hyperparameters used by the model. Defaults to None.
        history_kfold (list, optional): Stores the training history for each fold during k-fold cross-validation. Defaults to None.
        have_cat (bool): Indicates whether the dataset contains categorical features. Defaults to False.
    
    Methods:
        build(): Placeholder method for building the model structure. Should be overridden by subclasses.
        _optimizer(): Placeholder method for setting up the optimization algorithm. Should be overridden by subclasses.
        hyperparameter_optimization(): Placeholder for hyperparameter optimization. Should be overridden by subclasses.
        load(): Placeholder for loading a saved model from disk. Should be overridden by subclasses.
        fit(): Placeholder for fitting the model on training data. Should be overridden by subclasses.
        predict(): Placeholder for making predictions with the trained model. Should be overridden by subclasses.
        save(): Placeholder for saving the current state of the model to disk. Should be overridden by subclasses.
        _preprocess(data, target_one_hot_encoder=False, **kwargs): Preprocesses the data according to the specified parameters.
        _cluster_preprocess(data, **kwargs): Preprocesses the data for clustering tasks according to the specified parameters.
    """

    def __init__(self, target, name, seed=None):
        """
        Initializes a new instance of the Model class.

        Args:
            target (str): The name of the target variable.
            name (str): The name of the model.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
       
        self.target = target
        self.name = name
        self.seed = seed
        self.preprocessor = None
        self.preprocessed_data = None
        self.task  = None
        self.model = None
        self.hyperparameter = None
        self.history_kfold = None
        self.have_cat = False


    def build(self):
        """
        Placeholder method for setting up the optimization algorithm. This method should be overridden by
        subclasses to specify how the model should be optimized during training (e.g., SGD, Adam).
        """

        pass


    def _optimizer(self):
        """
        Placeholder method for setting up the optimization algorithm. This method should be overridden by
        subclasses to specify how the model should be optimized during training (e.g., SGD, Adam).
        """

        pass


    def hyperparameter_optimization(self):
        """
        Placeholder method for performing hyperparameter optimization. This method should be overridden by
        subclasses to implement hyperparameter tuning techniques (e.g., grid search, random search).
        """

        pass


    def load(self):
        """
        Placeholder method for loading a saved model from disk. This method should be overridden by subclasses
        to enable loading model state, allowing for model persistence across sessions.
        """

        pass


    def fit(self):
        """
        Placeholder method for fitting the model on the training data. This method should be overridden by
        subclasses to implement the training process, including any preprocessing, training iterations, and
        validation.
        """

        pass


    def predict(self):
        """
        Placeholder method for making predictions with the trained model. This method should be overridden
        by subclasses to use the model for making predictions on new or unseen data.
        """

        pass


    def save(self):
        """
        Placeholder method for saving the current state of the model to disk. This method should be overridden
        by subclasses to provide a way to serialize and save the model structure and trained weights.
        """

        pass
    

    def _preprocess(self, data, target_one_hot_encoder=False, **kwargs):
        """
        Preprocesses the data based on the provided parameters and updates the instance attributes accordingly.

        Args:
            data (Pandas DataFrame): The dataset to preprocess.
            target_one_hot_encoder (bool, optional): Indicates whether to apply one-hot encoding to the target variable. Defaults to False.
            **kwargs: Additional keyword arguments for preprocessing options.

        Note: This method updates the 'preprocessed_data' and 'preprocessor' attributes of the instance.
        """

        max_cat_nunique = kwargs.get('max_cat_nunique', 10)
        split_size = kwargs.get('split_size', (0.7, 0.15, 0.15))
        info = kwargs.get('info', False)
        task = kwargs.get('task', None)  

        train_size = split_size[0]
        valid_size = split_size[1]
        test_size = split_size[2]

        _data = data.copy()

        num_rows_with_nan = _data.isna().any(axis=1).sum()
        _data.dropna(axis=0, inplace=True)
        
        for feature in _data.columns:
            if _data[feature].dtype == 'object' and pd.to_numeric(_data[feature], errors='coerce').notna().all():
                _data[feature] = pd.to_numeric(_data[feature])

        types_num = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                     'float16', 'float32', 'float64']

    
        ######### X #########
        _X = _data.drop(columns=[self.target])
        df_preprocessor = None

        categorical_cols = list()
        numerical_cols= list()
        high_cardinality_cols = list()

        for feature in _X.columns:

            if _X[feature].dtype in types_num:
                numerical_cols.append(feature)

            elif np.unique(_X[feature]).size <= max_cat_nunique:
                categorical_cols.append(feature)
                self.have_cat = True

            else:
                high_cardinality_cols.append(feature)

        
        _X[categorical_cols] = _X[categorical_cols].astype('category')

        df_preprocessor_num = Pipeline(steps=[
            ('standardlization_num', StandardScaler(with_mean=True))
        ])
    
        df_preprocessor_cat = Pipeline(steps=[
            ('onehot_cat', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        df_preprocessor = ColumnTransformer(
            transformers=[
                ('df_preprocessor_num', df_preprocessor_num, numerical_cols),
                ('df_preprocessor_cat', df_preprocessor_cat, categorical_cols)
            ],
            remainder='drop',
            sparse_threshold=0
        )
    
        ######### Y #########
        _y = _data[[self.target]]
        target_preprocessor = None

        if task in ['classification', 'regression']:
            self.task = task

        elif _y[self.target].dtypes not in types_num:
            self.task = 'classification'

        elif np.unique(_y[self.target]).size > max_cat_nunique:
            self.task = 'regression'

        else:
            self.task = 'classification'


        ######### transformation #########
        if self.task == 'classification':

            if target_one_hot_encoder:
                encoder = OneHotEncoder(handle_unknown='ignore')

            else:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=999)
            
            target_preprocessor_cat = Pipeline(steps=[
                ('target_encoder_cat', encoder)
            ])
            
            target_preprocessor = ColumnTransformer(
                transformers=[
                    ('target_preprocessor_cat', target_preprocessor_cat, [self.target])
                ],
                remainder='drop',
                sparse_threshold=0
            )
    
            _y = target_preprocessor.fit_transform(_y)
        
        else:
            # _y = _y.to_numpy()

            target_preprocessor_num = Pipeline(steps=[
                ('standardlization_num', StandardScaler(with_mean=True))
            ])
           
            target_preprocessor = ColumnTransformer(
                transformers=[
                    ('target_preprocessor_cat', target_preprocessor_num, [self.target])
                ],
                remainder='drop',
                sparse_threshold=0
            )
    
            _y = target_preprocessor.fit_transform(_y)

    
        ######### TRAIN / VALID / TEST SPLIT #########
        _X_train, _X_temp, y_train, _y_temp = train_test_split(_X, _y, test_size=1-train_size, random_state=self.seed)

        X_train = df_preprocessor.fit_transform(_X_train)
        _X_temp = df_preprocessor.transform(_X_temp)
    
        X_test, X_val, y_test, y_val = train_test_split(_X_temp, _y_temp, test_size=valid_size/(valid_size + test_size), random_state=self.seed)
    
        X_train_val = np.concatenate((X_train, X_val), axis=0)
        y_train_val = np.concatenate((y_train, y_val), axis=0)
        
        if not target_one_hot_encoder:
            y_train = y_train.reshape(-1)
            y_val = y_val.reshape(-1)
            y_train_val = y_train_val.reshape(-1)
            y_test = y_test.reshape(-1)
        
        self.preprocessed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_train_val': X_train_val,
            'y_train_val': y_train_val,
            'X_test': X_test,
            'y_test': y_test
        }
    
        self.preprocessor = {
            "features": df_preprocessor,
            "target": target_preprocessor 
        }

        if info:
            msg = f"""
                Task: {self.task}

                Total of registers: {len(data)}
                Total of valid registers: {len(_X)}
                Total of invalid registers: {num_rows_with_nan}

                Total of training registers: {X_train.shape[0]}
                Total of validation registers: {X_val.shape[0]}
                Total of test registers: {X_test.shape[0]}

                Features before preprocessing: {_X_train.shape[1]}
                Features after preprocessing: {X_train.shape[1]}

                Numerical Features: {numerical_cols}
                Categorical Features: {categorical_cols}
                Categorical Features removed due to high cardinality: {high_cardinality_cols}

                Target: ['{self.target}']
            """

            if self.task == 'classification':
                if target_one_hot_encoder == False:
                    msg += f"\tCardinality (Target): {np.unique(self.preprocessed_data['y_train']).size}"

                else:
                    msg += f"\tCardinality (Target): {self.preprocessed_data['y_train'].shape[1]}"

            print(msg)


    def _cluster_preprocess(self, data, **kwargs):
        """
        Preprocesses the data for clustering tasks based on the provided parameters and updates the instance attributes accordingly.

        Args:
            data (Pandas DataFrame): The dataset to preprocess.
            **kwargs: Additional keyword arguments for preprocessing options.

        Note: This method updates the 'preprocessed_data' and 'preprocessor' attributes of the instance.
        """

        info = kwargs.get('info', False)
        max_cat_nunique = kwargs.get('max_cat_nunique', 10)

        types_num = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                     'float16', 'float32', 'float64']
            
        ######### X #########
        _X = data.copy()

        num_rows_with_nan = _X.isna().any(axis=1).sum()
        _X.dropna(axis=0, inplace=True)

        for feature in _X.columns:
            if _X[feature].dtype == 'object' and pd.to_numeric(_X[feature], errors='coerce').notna().all():
                _X[feature] = pd.to_numeric(_X[feature])
        

        df_preprocessor = None

        categorical_cols = list()
        numerical_cols= list()
        high_cardinality_cols = list()

        for feature in _X.columns:

            if _X[feature].dtype in types_num:
                numerical_cols.append(feature)

            elif np.unique(_X[feature]).size <= max_cat_nunique:
                categorical_cols.append(feature)
                self.have_cat = True

            else:
                high_cardinality_cols.append(feature)

        
        _X[categorical_cols] = _X[categorical_cols].astype('category')
        
        df_preprocessor_num = Pipeline(steps=[
            ('standardlization_num', StandardScaler(with_mean=True))
        ])

    
        df_preprocessor_cat = Pipeline(steps=[
            ('identidade', FunctionTransformer(None))
        ])

        df_preprocessor = ColumnTransformer(
            transformers=[
                ('df_preprocessor_num', df_preprocessor_num, numerical_cols),
                ('df_preprocessor_cat', df_preprocessor_cat, categorical_cols)
            ],
            remainder='drop',
            sparse_threshold=0
        )

        ### One-hot Encoding
        df_preprocessor_cat_ohe = Pipeline(steps=[
            ('onehot_cat', OneHotEncoder(handle_unknown='ignore'))
        ])

        df_preprocessor_ohe = ColumnTransformer(
            transformers=[
                ('df_preprocessor_num', df_preprocessor_num, numerical_cols),
                ('df_preprocessor_cat', df_preprocessor_cat_ohe, categorical_cols)
            ],
            remainder='drop',
            sparse_threshold=0
        )
      
        X = df_preprocessor.fit_transform(_X)
        X_ohe = df_preprocessor_ohe.fit_transform(_X)
  
        self.preprocessed_data = {'X': X, 'X_ohe': X_ohe}
        self.preprocessor = {"features": df_preprocessor, "features_ohe": df_preprocessor_ohe}

        if info:
            msg = f"""
                Total of registers: {len(data)}
                Total of valid registers: {len(_X)}
                Total of invalid registers: {num_rows_with_nan}

                Total of Features: {X.shape[1]}
                Numerical Features: {numerical_cols}
                Categorical Features: {categorical_cols}
                Categorical Features removed due to high cardinality: {high_cardinality_cols}
            """

            print(msg)


class NeuralNetwork(Model):
    """
    A class for constructing and training neural networks, with built-in methods for preprocessing,
    hyperparameter optimization, training, and inference. Inherits from the Model base class.

    Attributes:
        n_input (int): Number of input features.
        n_neurons (int): Number of neurons in each hidden layer.
        n_output (int): Number of output neurons.
        metrics (list): List of Keras metrics to be used for model evaluation.
        callbacks (list): List of Keras Callbacks to be used during model training.
    
    Methods:
        build(data, **kwargs): Prepares the neural network model based on the provided dataset and hyperparameters.
        _make_nn(dropout, layers, optimizer): Constructs the neural network architecture.
        _optimizer(trial, **kwargs): Defines and runs the optimization trial for hyperparameter tuning.
        hyperparameter_optimization(n_trials=1, info=False, **kwargs): Performs hyperparameter optimization using Optuna.
        load(foldername): Loads the model and preprocessor from the specified folder.
        fit(return_history=False, graphic=True, graphic_save_extension=None, verbose=0, **kwargs): Trains the neural network on preprocessed data.
        predict(x, verbose=0): Makes predictions using the trained neural network model.
        save(): Saves the model and preprocessor to disk.
    """

    def __init__(self, target, name=None, seed=None):
        """
        Initializes a new instance of the NeuralNetwork class.

        Args:
            target (str): The name of the target variable.
            name (str, optional): The name of the model. Defaults to a generated name based on the current datetime.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        
        if name is None:
            _name = f'neuralNetwork {datetime.now().strftime("%d-%m-%y %Hh%Mmin")}'

        else:
            _name = name

        super().__init__(target, _name, seed)

        self.n_input = None
        self.n_neurons = None
        self.n_output = None
        self._metrics = None
        self.callbacks = None


    @validate_target
    @validate_task  
    @validate_split_size
    def build(self, data, **kwargs):
        """
        Prepares the neural network model based on the provided dataset and hyperparameters. This includes preprocessing
        the data and initializing the model architecture based on the data's characteristics and specified hyperparameters.

        Args:
            data (Pandas DataFrame): The dataset to be used for building the model.
            **kwargs: Additional keyword arguments for preprocessing and model configuration.
        """

        super()._preprocess(data, target_one_hot_encoder=True, **kwargs)
        
        self.n_input = self.preprocessed_data['X_train'].shape[1]
        self.n_neurons = 2 ** ceil(log(self.n_input, 2))
        self.n_output = self.preprocessed_data['y_train'].shape[1]

        dict_metrics = {
            # Regression
            'mse': MeanSquaredError(),
            'rmse': RootMeanSquaredError(),
            'msle': MeanSquaredLogarithmicError(),
            'mae': MeanAbsoluteError(),
            'mape': MeanAbsolutePercentageError(),

            # Classification
            'auc': AUC(),
            'precision': Precision(),
            'recall': Recall(),
            'accuracy': CategoricalAccuracy()
        }

        if self.task == 'regression':
            _metrics = kwargs.get('metrics', [])
            self._metrics = [dict_metrics[metric] for metric in _metrics]

        else:
            _metrics = kwargs.get('metrics', [])
            self._metrics = [dict_metrics[metric] for metric in _metrics]

        patience_early_stopping = kwargs.get('patience_early_stopping', 20)
        patience_early_reduceLR = kwargs.get('patience_early_reduceLR', 4)

        earlystop = EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=patience_early_stopping,
            verbose=0,
            restore_best_weights=True
        )

        reduceLr = ReduceLROnPlateau(
            monitor='loss',
            factor=0.2,
            patience=patience_early_reduceLR,
            mode="min",
            verbose=0,
            min_delta=0.0001,
            min_lr=0
        )
        
        self.callbacks = [earlystop, reduceLr, TerminateOnNaN()]


    def _make_nn(self, dropout, layers, optimizer):
        """
        Constructs the neural network architecture with the specified number of layers, dropout rate, and optimizer.

        Args:
            dropout (float): The dropout rate to be applied to each hidden layer.
            layers (int): The number of hidden layers in the neural network.
            optimizer (str): The name of the optimizer to be used for training the neural network.
        
        Returns:
            keras.models.Sequential: The constructed Keras Sequential model.
        """
       
        model = Sequential()
        model.add(Input(shape=(self.n_input,)))
        
        #################################################################
        for _ in range(layers):
            
            model.add(Dense(self.n_neurons))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(dropout))
        ##################################################################

        if self.task == 'classification':
            model.add(Dense(self.n_output, activation='softmax'))
            
        else:
            model.add(Dense(1, activation='linear'))

        model.compile(
            loss = 'mean_squared_error' if self.task == 'regression' else 'categorical_crossentropy',
            metrics = self._metrics,
            optimizer = optimizer
        )

        return model


    def _optimizer(self, trial, **kwargs):
        """
        Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
        a callback within an Optuna optimization study.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            float: The average validation loss across all folds for the current trial.
        """

        num_folds = kwargs.get('num_folds', 5)
       
        search_space_dropout = kwargs.get('search_space_dropout', [0, 0.05])
        search_space_layers = kwargs.get('search_space_layers', [2, 3, 4])
        search_space_batch_size = kwargs.get('search_space_batch_size', [32, 64, 128])
        search_space_optimizer = kwargs.get('search_space_optimizer', ['Adam'])
        
        dropout = trial.suggest_categorical('dropout', search_space_dropout)
        layers = trial.suggest_categorical('layers', search_space_layers)
        batch_size = trial.suggest_categorical('batch_size', search_space_batch_size)
        optimizer = trial.suggest_categorical('optimizer', search_space_optimizer)

        kfold = KFold(n_splits=num_folds, shuffle=True)

        scores = []
        
        X_train_val = self.preprocessed_data['X_train_val']
        y_train_val = self.preprocessed_data['y_train_val']

        for index_train, index_val in kfold.split(X_train_val, y_train_val):

            modelStudy = self._make_nn(
                dropout=dropout,
                layers=layers,
                optimizer=optimizer
            )
            
            history = modelStudy.fit(
                X_train_val[index_train],
                y_train_val[index_train],
                epochs=10_000,
                batch_size=batch_size,
                validation_data=(
                    X_train_val[index_val],
                    y_train_val[index_val],
                ),
                callbacks = self.callbacks,
                verbose=0
            )

            scores.append(min(history.history['val_loss']))

        new_trial = pd.DataFrame([scores], columns=self.history_kfold.columns)
        self.history_kfold = self.history_kfold.append(new_trial, ignore_index=True)
        self.history_kfold.rename_axis('Trial (nº)', inplace=True)

        return sum(scores) / num_folds
    

    def hyperparameter_optimization(self, n_trials=1, info=False, **kwargs):
        """
        Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
        and updates the model's hyperparameters with the best found values.

        Args:
            n_trials (int, optional): The number of optimization trials to perform. Defaults to 1.
            info (bool, optional): Whether to print detailed information about each trial. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            pd.DataFrame: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
        """

        num_folds = kwargs.get('num_folds', 5)
        columns_name = [f'Fold nº {i}' for i in range(1, num_folds + 1)]

        self.history_kfold = pd.DataFrame(columns=columns_name).rename_axis('Trial (nº)')

        self.hyperparameter = optuna.create_study(study_name='optimization', direction='minimize')
        self.hyperparameter.optimize(lambda trial: self._optimizer(trial, **kwargs), n_trials = n_trials)
       
        if info:
            trial = self.hyperparameter.trials_dataframe()
            trial = trial.set_index("number")
            trial.rename_axis('Trial (nº)', inplace=True)
            trial.rename(columns={'value': 'Folds mean'}, inplace=True)

            self.history_kfold['Folds std'] = self.history_kfold.std(axis=1)      

            df_info = self.history_kfold.join(trial.drop(['datetime_start', 'datetime_complete', 'duration', 'state'], axis=1))
            df_info = df_info.sort_values(by='Folds mean', ascending=True)
            df_info.reset_index(inplace=True)
            df_info.index = [f'{i}º' for i in df_info.index + 1]
            df_info.rename_axis('Ranking', inplace=True)

            fist_level_multiindex = 'Categorical Crossentropy' if self.task == "classification" else "Mean Squared Error"

            trial_columns = [(fist_level_multiindex, col) for col in df_info.columns[: num_folds + 3]]
            hyperparameter_columns =[('Hyperparameters', col) for col in df_info.columns[num_folds + 3 :]]
            
            multi_columns = pd.MultiIndex.from_tuples(trial_columns + hyperparameter_columns)
            df_info.columns = multi_columns

            return df_info.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])


    def load(self, foldername):
        """
        Loads the model and preprocessor from the specified folder.

        Args:
            foldername (str): The name of the folder where the model and preprocessor are saved.
        """

        if not os.path.exists(f'./saved/{foldername}'):
            print("There is no folder with that name!")
            return

        self.name = foldername
        self.preprocessor = load(f'./saved/{foldername}/preprocessor.joblib')
        self.model = load_model(f'./saved/{foldername}/model.h5')

    
    def fit(self, return_history=False, graphic=False, graphic_save_extension=None, verbose=0, **kwargs):
        """
        Trains the neural network on preprocessed data. This method supports early stopping and learning rate reduction
        based on the performance on the validation set.

        Args:
            return_history (bool, optional): Whether to return the training history object. Defaults to False.
            graphic (bool, optional): Whether to plot training and validation loss and metrics. Defaults to True.
            graphic_save_extension (str, optional): Extension to save the graphics (e.g., 'png', 'svg'). If None, graphics are not saved. Defaults to None.
            verbose (int, optional): Verbosity mode for training progress. Defaults to 0.
            **kwargs: Additional keyword arguments for configuring the training process.
        
        Returns:
            keras.callbacks.History: The training history object, if `return_history` is True. Otherwise, None.
        """

        if self.hyperparameter is None:
           self.hyperparameter_optimization()
            
        self.model = self._make_nn(
            dropout = self.hyperparameter.best_params['dropout'],
            layers = self.hyperparameter.best_params['layers'],
            optimizer = self.hyperparameter.best_params['optimizer']
        )
        
        history = self.model.fit(
            self.preprocessed_data['X_train'],
            self.preprocessed_data['y_train'],
            epochs=10_000,
            batch_size= self.hyperparameter.best_params['batch_size'],
            validation_data = (
                self.preprocessed_data['X_val'],
                self.preprocessed_data['y_val']
            ),
            callbacks = self.callbacks,
            verbose=verbose
        )

        loss = self.model.evaluate(self.preprocessed_data['X_test'], self.preprocessed_data['y_test'], verbose=0)

        if graphic:

            height = kwargs.get('subplot_height', 4)
            width = kwargs.get('subplot_width', 8)

            color = kwargs.get('subplot_color', {
                "train": 'red',
                "validation": "blue",
                "test": "green" 
            })

            if self.task == "regression":
                fig, axs = plt.subplots(len(self._metrics) + 1, 1, figsize=(width, height * (len(self._metrics) + 1)))

            else:
                fig, axs = plt.subplots(len(self._metrics) + 2, 1, figsize=(width, height * (len(self._metrics) + 2)))
            
            title = "Bias-Variance Graphic (Neural Network)"
            fig.suptitle(title, fontweight='bold', fontsize=12)

            if not hasattr(axs, '__getitem__'):
                axs = [axs]

            if len(self._metrics) == 0:
                loss = [loss]

            if self.task == "regression":
                y_true = self.preprocessed_data['y_test']
                y_pred = self.model.predict(self.preprocessed_data['X_test'], verbose=0)

                r2 = r2_score(y_true, y_pred)

                axs[0].set_title(
                    f"R²: {r2:.3f} | cost function [mean square error] (train: {history.history['loss'][-1]:.5f}  val: {history.history['val_loss'][-1]:.5f}  test: {loss[0]:.5f})",
                    fontsize=12
                    )

            else:
                axs[0].set_title(
                    f"cost function [categorical crossentropy] (train: {history.history['loss'][-1]:.5f}  val: {history.history['val_loss'][-1]:.5f}  test: {loss[0]:.5f})",
                    fontsize=12
                    )
            
            axs[0].plot(history.history['loss'], linestyle='-', linewidth=2, label = 'Train', color=color['train'])
            axs[0].plot(history.history['val_loss'], linestyle='-', linewidth=1, label = 'Validation', color=color['validation'])
            axs[0].axhline(y=loss[0], linestyle='--', linewidth=1, label = 'Test', color=color['test'])
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Metric')
            axs[0].legend(loc = 'best')
            axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))


            if self.task == "regression":
                for i, metric in enumerate(self._metrics):
                    axs[i+1].set_title(f"R²: {r2:.3f} | {metric.name} (train: {history.history[metric.name][-1]:.5f}  val: {history.history[f'val_{metric.name}'][-1]:.5f}  test: {loss[i+1]:.5f})", fontsize=12)
                    axs[i+1].plot(history.history[f'{metric.name}'], linestyle='-', linewidth=2, label = 'Train', color=color['train'])
                    axs[i+1].plot(history.history[f'val_{metric.name}'], linestyle='-', linewidth=1, label = 'Validation', color=color['validation'])
                    axs[i+1].axhline(y=loss[i+1], linestyle='--', linewidth=1, label = 'Test', color=color['test'])
                    axs[i+1].set_xlabel('Epoch')
                    axs[i+1].set_ylabel('Metric')
                    axs[i+1].legend(loc = 'best')
                    axs[i+1].xaxis.set_major_locator(MaxNLocator(integer=True))

            else:
                for i, metric in enumerate(self._metrics):
                    axs[i+1].set_title(f"{metric.name} (train: {history.history[metric.name][-1]:.5f}  val: {history.history[f'val_{metric.name}'][-1]:.5f}  test: {loss[i+1]:.5f})", fontsize=12)
                    axs[i+1].plot(history.history[f'{metric.name}'], linestyle='-', linewidth=2, label = 'Train', color=color['train'])
                    axs[i+1].plot(history.history[f'val_{metric.name}'], linestyle='-', linewidth=1, label = 'Validation', color=color['validation'])
                    axs[i+1].axhline(y=loss[i+1], linestyle='--', linewidth=1, label = 'Test', color=color['test'])
                    axs[i+1].set_xlabel('Epoch')
                    axs[i+1].set_ylabel('Metric')
                    axs[i+1].legend(loc = 'best')
                    axs[i+1].xaxis.set_major_locator(MaxNLocator(integer=True))


            if self.task == "classification":
                y_true = np.argmax(self.preprocessed_data['y_test'], axis=1)
                y_pred = np.argmax(self.model.predict(self.preprocessed_data['X_test'], verbose=0), axis=1)

                conf_mat = confusion_matrix(y_true, y_pred)

                encoder = self.preprocessor['target'].named_transformers_['target_preprocessor_cat'].named_steps['target_encoder_cat']
                class_labels = encoder.categories_[0].tolist()

                ax_conf = axs[-1]

                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_conf,
                            xticklabels=class_labels, yticklabels=class_labels)
                
                ax_conf.set_xlabel(f'Predicted Values ({self.target})')
                ax_conf.set_ylabel(f'True Values ({self.target})')
                ax_conf.set_title('Confusion Matrix (Test Dataset)')

            plt.tight_layout(rect=[0, 0.05, 1, 0.98])

            if graphic_save_extension in ['png', 'svg', 'pdf', 'eps']:

                if not os.path.exists(f'./saved/{self.name}/figures'):
                    os.makedirs(f'./saved/{self.name}/figures')

                plt.savefig(
                    f'./saved/{self.name}/figures/{title}.{graphic_save_extension}',
                    format=f'{graphic_save_extension}'
                )

            plt.show()
            plt.close()

        if return_history:
            return history

    
    def predict(self, x, verbose=0):
        """
        Makes predictions using the trained neural network model.

        Args:
            x (Pandas DataFrame): The input data for making predictions.
            verbose (int, optional): Verbosity mode for prediction. Defaults to 0.
        
        Returns:
            Pandas DataFrame: The input data with an additional column for predictions.
        """

        _x = x.copy()

        if self.target in _x.columns:
            _y_real = _x[self.target]
            _x.drop(self.target, axis=1, inplace=True)
     
        ################### INFERENCE #######################
        start_time = time()

        _x_temp = self.preprocessor['features'].transform(_x)
        y = self.model.predict(_x_temp, verbose=verbose)

        if self.preprocessor['target'] is not None:
            target_preprocessor = self.preprocessor['target'].named_transformers_['target_preprocessor_cat']
            y = target_preprocessor.inverse_transform(y)

        end_time = time() 
        #####################################################

        inference_time = end_time - start_time
        print(f"Inference time: {inference_time * 1000:.2f} milliseconds ({len(x)} register(s))") 

        if '_y_real' in locals(): 
            _x[self.target] = _y_real

        _x[f'{self.target} (NN prediction)'] = y
        
        return _x

    
    def save(self):
        """
        Saves the model and preprocessor to disk.
        """

        if not os.path.exists(f'./saved/{self.name}'):
            os.makedirs(f'./saved/{self.name}')
        
        dump(self.preprocessor, f'./saved/{self.name}/preprocessor.joblib')
        self.model.save(f'./saved/{self.name}/model.h5')


class XgBoost(Model):
    """
    A class for constructing and training XGBoost models, with built-in methods for preprocessing,
    hyperparameter optimization, training, and inference. Inherits from the Model base class.

    Attributes:
        metrics (list): List of evaluation metrics to be used for model evaluation.
        patience_early_stopping (int): Number of rounds without improvement to wait before stopping training.

    Methods:
        build(data, **kwargs): Prepares the XGBoost model based on provided dataset and hyperparameters.
        _make_xgBooster(**kwargs): Constructs the XGBoost model with specified hyperparameters.
        _optimizer(trial, **kwargs): Defines and runs the optimization trial for hyperparameter tuning.
        hyperparameter_optimization(n_trials=1, info=False, **kwargs): Performs hyperparameter optimization using Optuna.
        load(foldername): Loads the model and preprocessor from the specified folder.
        fit(return_history=False, graphic=True, graphic_save_extension=None, verbose=0, **kwargs): Trains the XGBoost model on preprocessed data.
        predict(x): Makes predictions using the trained XGBoost model.
        save(): Saves the model and preprocessor to disk.
    """

    def __init__(self, target, name=None, seed=None):
        """
        Initializes a new instance of the XgBoost class.

        Args:
            target (str): The name of the target variable.
            name (str, optional): The name of the model. Defaults to a generated name based on the current datetime.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        if name is None:
            _name = f'xgBoost {datetime.now().strftime("%d-%m-%y %Hh%Mmin")}'

        else:
            _name = name

        super().__init__(target, _name, seed)
        
        self._metrics = None
        self.patience_early_stopping = None
          

    @validate_target
    @validate_task
    @validate_split_size
    def build(self, data, **kwargs):
        """
        Prepares the XGBoost model based on the provided dataset and hyperparameters. This includes preprocessing
        the data and initializing the model parameters based on the data's characteristics and specified hyperparameters.

        Args:
            data (Pandas DataFrame): The dataset to be used for building the model.
            **kwargs: Additional keyword arguments for preprocessing and model configuration.
        """

        super()._preprocess(data, **kwargs)
        
        if self.task == 'regression':
            self._metrics = ['rmse']

        else:
            self._metrics = ['mlogloss']


        self.patience_early_stopping = kwargs.get('patience_early_stopping', 20)


    def _make_xgBooster(self, tree_method, booster, learning_rate, min_split_loss, max_depth,
                        min_child_weight, max_delta_step, subsample, sampling_method,
                        colsample_bytree, colsample_bylevel, colsample_bynode, reg_lambda,
                        reg_alpha, scale_pos_weight, grow_policy, max_leaves, max_bin,
                        num_parallel_tree, verbose=0):
        """
        Constructs the XGBoost model with the specified hyperparameters.

        Args:
            **kwargs: Hyperparameters for the XGBoost model.
        
        Returns:
            xgboost.XGBModel: The constructed XGBoost model.
        """
        
        common_arguments = {
            'tree_method': tree_method,
            'n_estimators': 100_000,
            'early_stopping_rounds': self.patience_early_stopping,
            'booster': booster,
            'eval_metric': self._metrics,
            'validate_parameters': False,
            'learning_rate': learning_rate,
            'min_split_loss': min_split_loss,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step,
            'subsample': subsample,
            'sampling_method': sampling_method,
            'colsample_bytree': colsample_bytree,
            'colsample_bylevel': colsample_bylevel,
            'colsample_bynode': colsample_bynode,
            'reg_lambda': reg_lambda,
            'reg_alpha': reg_alpha,
            'scale_pos_weight': scale_pos_weight,
            'grow_policy': grow_policy,
            'max_leaves': max_leaves,
            'max_bin': max_bin,
            'num_parallel_tree': num_parallel_tree,
            'random_state': self.seed,
            'verbosity': verbose
        } 

        if self.task == "regression":
            model = XGBRegressor(
                objective='reg:squarederror',
                **common_arguments
            )

        else:
            model = XGBClassifier(
                objective='multi:softprob',
                num_class= np.unique(self.preprocessed_data['y_train']).size,
                use_label_encoder=False,
                **common_arguments
            )
            
        return model

    def _optimizer(self, trial, **kwargs):
        """
        Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
        a callback within an Optuna optimization study.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            float: The average validation loss across all folds for the current trial.
        """

        num_folds = kwargs.get('num_folds', 5)

        search_space_tree_method = kwargs.get('search_space_tree_method', ['auto'])
        search_space_booster = kwargs.get('search_space_booster', ['gbtree', 'gblinear', 'dart'])
        search_space_learning_rate = kwargs.get('search_space_learning_rate', [0.1, 0.3])
        search_space_min_split_loss = kwargs.get('search_space_min_split_loss', [0])
        search_space_max_depth = kwargs.get('search_space_max_depth', [5, 6, 7])
        search_space_min_child_weight = kwargs.get('search_space_min_child_weight', [1])
        search_space_max_delta_step = kwargs.get('search_space_max_delta_step', [0])
        search_space_subsample = kwargs.get('search_space_subsample', [1])
        search_space_sampling_method = kwargs.get('search_space_sampling_method', ['uniform'])
        search_space_colsample_bytree = kwargs.get('search_space_colsample_bytree', [1])
        search_space_colsample_bylevel = kwargs.get('search_space_colsample_bylevel', [1])
        search_space_colsample_bynode = kwargs.get('search_space_colsample_bynode', [1])
        search_space_reg_lambda = kwargs.get('search_space_reg_lambda', [1])
        search_space_reg_alpha = kwargs.get('search_space_reg_alpha', [0])
        search_space_scale_pos_weight = kwargs.get('search_space_scale_pos_weight', [1])
        search_space_grow_policy = kwargs.get('search_space_grow_policy', ['depthwise', 'lossguide'])
        search_space_max_leaves = kwargs.get('search_space_max_leaves', [0])
        search_space_max_bin = kwargs.get('search_space_max_bin', [256])
        search_space_num_parallel_tree = kwargs.get('search_space_num_parallel_tree', [1])

        tree_method = trial.suggest_categorical('tree_method', search_space_tree_method)
        booster = trial.suggest_categorical('booster', search_space_booster)
        learning_rate = trial.suggest_categorical('learning_rate', search_space_learning_rate)
        min_split_loss = trial.suggest_categorical('min_split_loss', search_space_min_split_loss)
        max_depth = trial.suggest_categorical('max_depth', search_space_max_depth)
        min_child_weight = trial.suggest_categorical('min_child_weight', search_space_min_child_weight)
        max_delta_step = trial.suggest_categorical('max_delta_step', search_space_max_delta_step)
        subsample = trial.suggest_categorical('subsample', search_space_subsample)
        sampling_method = trial.suggest_categorical('sampling_method', search_space_sampling_method)
        colsample_bytree = trial.suggest_categorical('colsample_bytree', search_space_colsample_bytree)
        colsample_bylevel = trial.suggest_categorical('colsample_bylevel', search_space_colsample_bylevel)
        colsample_bynode = trial.suggest_categorical('colsample_bynode', search_space_colsample_bynode)
        reg_lambda = trial.suggest_categorical('reg_lambda', search_space_reg_lambda)
        reg_alpha = trial.suggest_categorical('reg_alpha', search_space_reg_alpha)
        scale_pos_weight = trial.suggest_categorical('scale_pos_weight', search_space_scale_pos_weight)
        grow_policy = trial.suggest_categorical('grow_policy', search_space_grow_policy)
        max_leaves = trial.suggest_categorical('max_leaves', search_space_max_leaves)
        max_bin = trial.suggest_categorical('max_bin', search_space_max_bin)
        num_parallel_tree = trial.suggest_categorical('num_parallel_tree', search_space_num_parallel_tree)

        kfold = KFold(n_splits=num_folds, shuffle=True)

        scores = []

        X_train_val = self.preprocessed_data['X_train_val']
        y_train_val = self.preprocessed_data['y_train_val']

        for index_train, index_val in kfold.split(X_train_val, y_train_val):

            modelStudy = self._make_xgBooster(
                tree_method = tree_method,
                booster = booster,
                learning_rate = learning_rate,
                min_split_loss = min_split_loss,
                max_depth = max_depth,
                min_child_weight = min_child_weight,
                max_delta_step = max_delta_step,
                subsample = subsample,
                sampling_method = sampling_method,
                colsample_bytree = colsample_bytree,
                colsample_bylevel = colsample_bylevel,
                colsample_bynode = colsample_bynode,
                reg_lambda = reg_lambda,
                reg_alpha = reg_alpha,
                scale_pos_weight = scale_pos_weight,
                grow_policy = grow_policy,
                max_leaves = max_leaves,
                max_bin = max_bin,
                num_parallel_tree = num_parallel_tree
            )
        
            modelStudy.fit(
                X_train_val[index_train],
                y_train_val[index_train],
                eval_set = [(
                    X_train_val[index_val],
                    y_train_val[index_val]
                )],
                verbose=0
            )

            scores.append(modelStudy.best_score)

        new_trial = pd.DataFrame([scores], columns=self.history_kfold.columns)
        self.history_kfold = self.history_kfold.append(new_trial, ignore_index=True)
        self.history_kfold.rename_axis('Trial (nº)', inplace=True)

        return sum(scores) / num_folds

    
    def hyperparameter_optimization(self, n_trials=1, info=False, **kwargs):
        """
        Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
        and updates the model's hyperparameters with the best found values.

        Args:
            n_trials (int, optional): The number of optimization trials to perform. Defaults to 1.
            info (bool, optional): Whether to print detailed information about each trial. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            pd.DataFrame: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
        """

        num_folds = kwargs.get('num_folds', 5)
        columns_name = [f'Fold nº {i}' for i in range(1, num_folds + 1)]

        self.history_kfold = pd.DataFrame(columns=columns_name).rename_axis('Trial (nº)')

        self.hyperparameter = optuna.create_study(study_name='optimization', direction='minimize')
        self.hyperparameter.optimize(lambda trial: self._optimizer(trial, **kwargs), n_trials = n_trials)

        if info:
            trial = self.hyperparameter.trials_dataframe()
            trial = trial.set_index("number")
            trial.rename_axis('Trial (nº)', inplace=True)
            trial.rename(columns={'value': 'Folds mean'}, inplace=True)

            self.history_kfold['Folds std'] = self.history_kfold.std(axis=1)      

            df_info = self.history_kfold.join(trial.drop(['datetime_start', 'datetime_complete', 'duration', 'state'], axis=1))
            df_info = df_info.sort_values(by='Folds mean', ascending=True)
            df_info.reset_index(inplace=True)
            df_info.index = [f'{i}º' for i in df_info.index + 1]
            df_info.rename_axis('Ranking', inplace=True)

            fist_level_multiindex = 'Categorical Crossentropy' if self.task == "classification" else "Mean Squared Error"

            trial_columns = [(fist_level_multiindex, col) for col in df_info.columns[: num_folds + 3]]
            hyperparameter_columns =[('Hyperparameters', col) for col in df_info.columns[num_folds + 3 :]]
            
            multi_columns = pd.MultiIndex.from_tuples(trial_columns + hyperparameter_columns)
            df_info.columns = multi_columns

            return df_info.style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])


    def load(self, foldername):
        """
        Loads the model and preprocessor from the specified folder.

        Args:
            foldername (str): The name of the folder where the model and preprocessor are saved.
        """

        if not os.path.exists(f'./saved/{foldername}'):
            print("There is no folder with that name!")
            return
        
        self.name = foldername
        self.preprocessor = load(f'./saved/{foldername}/preprocessor.joblib')

        try:
            self.model = XGBRegressor()
            self.model.load_model(f'./saved/{foldername}/model.bin')

        except:
            self.model = XGBClassifier()
            self.model.load_model(f'./saved/{foldername}/model.bin')

    
    def fit(self, return_history=False, graphic=False, graphic_save_extension=None, verbose=0, **kwargs):
        """
        Trains the XGBoost model on preprocessed data. This method supports early stopping based on the performance
        on the validation set.

        Args:
            return_history (bool, optional): Whether to return the training history object. Defaults to False.
            graphic (bool, optional): Whether to plot training and validation loss and metrics. Defaults to True.
            graphic_save_extension (str, optional): Extension to save the graphics (e.g., 'png', 'svg'). If None, graphics are not saved. Defaults to None.
            verbose (int, optional): Verbosity mode for training progress. Defaults to 0.
            **kwargs: Additional keyword arguments for configuring the training process.
        
        Returns:
            dict: The training history object, if `return_history` is True. Otherwise, None.
        """

        if self.hyperparameter is None:
           self.hyperparameter_optimization()

        self.model = self._make_xgBooster(
            tree_method = self.hyperparameter.best_params['tree_method'],
            booster = self.hyperparameter.best_params['booster'],
            learning_rate = self.hyperparameter.best_params['learning_rate'],
            min_split_loss = self.hyperparameter.best_params['min_split_loss'],
            max_depth = self.hyperparameter.best_params['max_depth'],
            min_child_weight = self.hyperparameter.best_params['min_child_weight'],
            max_delta_step = self.hyperparameter.best_params['max_delta_step'],
            subsample = self.hyperparameter.best_params['subsample'],
            sampling_method = self.hyperparameter.best_params['sampling_method'],
            colsample_bytree = self.hyperparameter.best_params['colsample_bytree'],
            colsample_bylevel = self.hyperparameter.best_params['colsample_bylevel'],
            colsample_bynode = self.hyperparameter.best_params['colsample_bynode'],
            reg_lambda = self.hyperparameter.best_params['reg_lambda'],
            reg_alpha = self.hyperparameter.best_params['reg_alpha'],
            scale_pos_weight = self.hyperparameter.best_params['scale_pos_weight'],
            grow_policy = self.hyperparameter.best_params['grow_policy'],
            max_leaves = self.hyperparameter.best_params['max_leaves'],
            max_bin = self.hyperparameter.best_params['max_bin'],
            num_parallel_tree = self.hyperparameter.best_params['num_parallel_tree']
        )
    
        self.model.fit(
            self.preprocessed_data['X_train'],
            self.preprocessed_data['y_train'],
            
            ## Não ocorre data leaking. EarlyStopping utiliza somente eval_set[-1]
            eval_set = [(
                self.preprocessed_data['X_train'],
                self.preprocessed_data['y_train']
                ), (
                self.preprocessed_data['X_test'],
                self.preprocessed_data['y_test']
                ), (
                self.preprocessed_data['X_val'],
                self.preprocessed_data['y_val']
                )],

            verbose=verbose
        )

        history = self.model.evals_result()

        if graphic:
            height = kwargs.get('subplot_height', 4)
            width = kwargs.get('subplot_width', 8)

            color = kwargs.get('subplot_color', {
                "train": 'red',
                "validation": "blue",
                "test": "green" 
            })

            if self.task == "regression":
                fig, axs = plt.subplots(len(self._metrics), 1, figsize=(width, height * (len(self._metrics))))

            else:
                fig, axs = plt.subplots(len(self._metrics) + 1, 1, figsize=(width, height * (len(self._metrics) + 1)))

            if not hasattr(axs, '__getitem__'):
                axs = [axs]
            
            title = "Bias-Variance Graphic (XG Boost)"

            fig.suptitle(title, fontweight='bold', fontsize=12)

            for i, metric in enumerate(self._metrics):

                if i == 0 and self.task == "regression":
                    y_true = self.preprocessed_data['y_test']
                    y_pred = self.model.predict(self.preprocessed_data['X_test'])

                    r2 = r2_score(y_true, y_pred)

                    axs[i].set_title(
                        f"R²: {r2:.3f} | {metric} (train: {history['validation_0'][metric][-1]:.5f}  val: {history['validation_2'][metric][-1]:.5f}  test: {history['validation_1'][metric][-1]:.5f})",
                        fontsize=12
                        )

                elif i == 0:
                    ## mlogloss == categorical crossentropy (muticlassification problem)
                    axs[i].set_title(
                        f"cost function [categorical crossentropy] (train: {history['validation_0'][metric][-1]:.5f}  val: {history['validation_2'][metric][-1]:.5f}  test: {history['validation_1'][metric][-1]:.5f})",
                        fontsize=12
                        )

                else:
                    axs[i].set_title(f"{metric} (train: {history['validation_0'][metric][-1]:.5f}  val: {history['validation_2'][metric][-1]:.5f}  test: {history['validation_1'][metric][-1]:.5f})")
                
                axs[i].plot(history['validation_0'][metric], linestyle='-', linewidth=2, label = 'Train', color=color['train'])
                axs[i].plot(history['validation_2'][metric], linestyle='-', linewidth=1, label = 'Validation', color=color['validation'])
                axs[i].axhline(y=history['validation_1'][metric][-1], linestyle='--', linewidth=1, label = 'Test', color=color['test'])
                axs[i].set_xlabel('Estimators')
                axs[i].set_ylabel('Metric')
                axs[i].legend(loc = 'best')
                axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))

            if self.task == "classification":
                y_true = self.preprocessed_data['y_test']                
                y_pred = self.model.predict(self.preprocessed_data['X_test'])

                if y_pred.ndim == 2:
                    y_pred = np.argmax(y_pred, axis=1)

                conf_mat = confusion_matrix(y_true, y_pred)

                encoder = self.preprocessor['target'].named_transformers_['target_preprocessor_cat'].named_steps['target_encoder_cat']
                class_labels = encoder.categories_[0].tolist()

                ax_conf = axs[-1]

                sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_conf,
                            xticklabels=class_labels, yticklabels=class_labels)
                
                ax_conf.set_xlabel(f'Predicted Values ({self.target})')
                ax_conf.set_ylabel(f'True Values ({self.target})')
                ax_conf.set_title('Confusion Matrix (Test Dataset)')

            plt.tight_layout(rect=[0, 0.05, 1, 0.98])

            if graphic_save_extension in ['png', 'svg', 'pdf', 'eps']:

                if not os.path.exists(f'./saved/{self.name}/figures'):
                    os.makedirs(f'./saved/{self.name}/figures')

                plt.savefig(f'./saved/{self.name}/figures/{title}.{graphic_save_extension}', format=f'{graphic_save_extension}')

            plt.show()
            plt.close()

        if return_history:
            return history

    
    def predict(self, x):
        """
        Makes predictions using the trained XGBoost model.

        Args:
            x (Pandas DataFrame): The input data for making predictions.
        
        Returns:
            Pandas DataFrame: The input data with an additional column for predictions.
        """

        _x = x.copy()

        if self.target in _x.columns:
            _y_real = _x[self.target]
            _x.drop(self.target, axis=1, inplace=True)

        ################### INFERENCE #######################
        start_time = time()

        _x_temp = self.preprocessor['features'].transform(_x)
        y = self.model.predict(_x_temp)

        if y.ndim == 2:
            y = np.argmax(y, axis=1)

        if self.preprocessor['target'] is not None:
            target_preprocessor = self.preprocessor['target'].named_transformers_['target_preprocessor_cat']
            y = target_preprocessor.inverse_transform(y.reshape(-1, 1))

        end_time = time() 
        #####################################################

        inference_time = end_time - start_time
        print(f"Inference time: {inference_time * 1000:.2f} milliseconds ({len(x)} register(s))") 


        if '_y_real' in locals(): 
            _x[self.target] = _y_real

        _x[f'{self.target} (XGB prediction)'] = y
        
        return _x

    
    def save(self):
        """
        Saves the model and preprocessor to disk.
        """
    
        if not os.path.exists(f'./saved/{self.name}'):
            os.makedirs(f'./saved/{self.name}')
        
        dump(self.preprocessor, f'./saved/{self.name}/preprocessor.joblib')
        self.model.save_model(f'./saved/{self.name}/model.bin')


class DBSCAN(Model):
    """
    A class for constructing and running the DBSCAN clustering algorithm, with built-in methods for preprocessing,
    hyperparameter optimization, and silhouette analysis. Inherits from the Model base class.

    Attributes:
        have_categ (bool): Indicates whether the dataset contains categorical features.
        distance_matrix (numpy.ndarray): The computed distance matrix for the dataset.
        clusters (numpy.ndarray): The cluster labels for each point in the dataset.

    Methods:
        build(data, **kwargs): Prepares the DBSCAN model based on provided dataset and hyperparameters.
        _make_dbscan(eps, min_samples, algorithm, leaf_size, p): Constructs the DBSCAN model with specified hyperparameters.
        _optimizer(trial, **kwargs): Defines and runs the optimization trial for hyperparameter tuning.
        hyperparameter_optimization(n_trials=100, info=False, **kwargs): Performs hyperparameter optimization using Optuna.
        load(foldername): Loads the preprocessor, preprocessed data, clusters, and model from the specified folder.
        fit(): Applies the DBSCAN algorithm to the preprocessed data.
        predict(projection='2d', graphic_save_extension=None): Generates and displays a 2D or 3D t-SNE plot of the clusters.
        save(): Saves the preprocessor, preprocessed data, clusters, and model to disk.
    """

    def __init__(self, name=None, seed=None):
        """
        Initializes a new instance of the DBSCAN class.

        Args:
            name (str, optional): The name of the model. Defaults to a generated name based on the current datetime.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        if name is None:
            _name = f'dbscan {datetime.now().strftime("%d-%m-%y %Hh%Mmin")}'

        else:
            _name = name

        super().__init__(None, _name, seed)

        self.have_categ = None
        self.distance_matrix = None
        self.clusters = None

    def build(self, data, **kwargs):
        """
        Prepares the DBSCAN model based on the provided dataset and hyperparameters. This includes preprocessing
        the data and calculating the distance matrix if necessary.

        Args:
            data (Pandas DataFrame): The dataset to be used for clustering.
            **kwargs: Additional keyword arguments for preprocessing.
        """

        super()._cluster_preprocess(data, **kwargs)

        if self.have_cat:
            self.distance_matrix = gower.gower_matrix(self.preprocessed_data['X'])

        else:
            self.distance_matrix = pairwise_distances(self.preprocessed_data['X'], metric='euclidean')

    def _make_dbscan(self, eps, min_samples, algorithm, leaf_size, p):
        """
        Constructs the DBSCAN model with the specified hyperparameters.

        Args:
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            algorithm (str): The algorithm to be used by the DBSCAN model.
            leaf_size (int): Leaf size passed to the underlying BallTree or KDTree.
            p (float): The power of the Minkowski metric to be used to calculate distance between points.

        Returns:
            sklearn.cluster._dbscan.DBSCAN: The constructed DBSCAN model.
        """

        model = _DBSCAN(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric='precomputed'
        )
            
        return model

    def _optimizer(self, trial, **kwargs):
        """
        Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
        a callback within an Optuna optimization study.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            float: The silhouette score for the clustering configuration defined by the trial.
        """

        search_space_eps = kwargs.get('search_space_eps', [0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 1])
        search_space_min_samples = kwargs.get('search_space_min_samples', [3, 4, 5, 6, 7])
        search_space_algorithm = kwargs.get('search_space_algorithm', ['auto'])
        search_space_leaf_size = kwargs.get('search_space_leaf_size', [10, 20, 30, 40, 50])
        search_space_p = kwargs.get('search_space_p', [2])

        eps = trial.suggest_categorical('eps', search_space_eps)
        min_samples = trial.suggest_categorical('min_samples', search_space_min_samples)
        algorithm = trial.suggest_categorical('algorithm', search_space_algorithm)
        leaf_size = trial.suggest_categorical('leaf_size', search_space_leaf_size)
        p = trial.suggest_categorical('p', search_space_p)
           
        modelStudy = self._make_dbscan(
            eps=eps,
            min_samples=min_samples,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p
        )
        
        clusters = modelStudy.fit_predict(self.distance_matrix)
        clusters_filtered = clusters[clusters != -1]

        if len(np.unique(clusters_filtered)) > 1:
            score = silhouette_score(self.preprocessed_data['X_ohe'], clusters)
        
        else:
            score = -1
        
        return score
    

    def hyperparameter_optimization(self, n_trials=100, info=False, **kwargs):
        """
        Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
        and identifies the best hyperparameters for DBSCAN clustering.

        Args:
            n_trials (int, optional): The number of optimization trials to perform. Defaults to 100.
            info (bool, optional): Whether to print detailed information about each trial. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            pd.DataFrame: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
        """

        self.hyperparameter = optuna.create_study(study_name='optimization', direction='maximize')
        self.hyperparameter.optimize(lambda trial: self._optimizer(trial, **kwargs), n_trials = n_trials)

        if info:
            trial = self.hyperparameter.trials_dataframe()

            df_info = trial.drop(['datetime_start', 'datetime_complete', 'duration', 'state'], axis=1)
            df_info.set_index("number", inplace =True)
            df_info.rename_axis('Trial (nº)', inplace=True)
            df_info = df_info.sort_values('value', ascending=False)
            df_info.rename(columns={'value': 'Silhouette Score'}, inplace=True)      
            df_info.reset_index(inplace=True)
            df_info.index = [f'{i}º' for i in df_info.index + 1]
            df_info.rename_axis('Ranking', inplace=True)
            
            return df_info
        

    def load(self, foldername):
        """
        Loads the preprocessor, preprocessed data, and cluster labels from the specified folder.

        Args:
            foldername (str): The name of the folder where the data and model are saved.
        """

        if not os.path.exists(f'./saved/{foldername}'):
            print("There is no folder with that name!")
            return
        
        self.preprocessor = load(f'./saved/{foldername}/preprocessor.joblib')
        self.preprocessed_data = load(f'./saved/{foldername}/preprocessed_data.joblib')
        self.clusters = load(f'./saved/{foldername}/clusters.joblib')
        self.model = load(f'./saved/{foldername}/model.bin')


    def fit(self, return_cluster=False):
        """
        Applies DBSCAN clustering to the preprocessed dataset and updates the 'clusters' attribute with the cluster labels.
        """

        if self.hyperparameter is None:
           self.hyperparameter_optimization()
           
        self.model = self._make_dbscan(
            eps = self.hyperparameter.best_params['eps'],
            min_samples = self.hyperparameter.best_params['min_samples'],
            algorithm = self.hyperparameter.best_params['algorithm'],
            leaf_size = self.hyperparameter.best_params['leaf_size'],
            p = self.hyperparameter.best_params['p']
        )
        
        self.clusters = self.model.fit_predict(self.distance_matrix)

        if return_cluster:
            return self.clusters

    
    def predict(self, projection='2d', graphic_save_extension=None):
        """
        Projects the clustered data into 2D or 3D space using t-SNE and visualizes the clusters.

        Args:
            projection (str, optional): The type of projection for visualization ('2d' or '3d'). Defaults to '2d'.
            graphic_save_extension (str, optional): Extension to save the graphics (e.g., 'png', 'svg'). If None, graphics are not saved. Defaults to None.
        """

        if projection=='3d' and self.preprocessed_data['X_ohe'].shape[1] < 3:
            print('Os dados possuem dimensionalidade inferior a 3!')
            return

        tsne = TSNE(n_components=3 if projection == '3d' else 2, random_state=self.seed)
        X_reduced = tsne.fit_transform(self.preprocessed_data['X_ohe'])

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(np.unique(self.clusters)))]
        
        if projection == '3d':

            fig = plt.figure(figsize=(8, 6))
            title = '3-Dimensional t-SNE Clusters (DBSCAN)'

            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.set_zlabel('3rd component', fontsize=10)

            for color, i in zip(colors, np.unique(self.clusters)):

                ax.scatter(
                    X_reduced[self.clusters == i, 0],
                    X_reduced[self.clusters == i, 1],
                    X_reduced[self.clusters == i, 2],
                    edgecolor='k',
                    color='gray' if i == -1 else color,
                    s=10 if i == -1 else 40,
                    alpha=0.6,
                    linewidths=1,
                    depthshade=False,
                    marker='x' if i == -1 else 'o',
                    label='unmatch' if i == -1 else f'cluster {i + 1}',
                )
            
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor('#EAF2F8')
            ax.yaxis.pane.set_facecolor('#D4E6F1')
            ax.zaxis.pane.set_facecolor('#A9CCE3')

        else:

            fig = plt.figure(figsize=(6, 4))
            title = '2-Dimensional t-SNE Clusters (DBSCAN)'
            ax = fig.add_subplot(111)
            ax.set_facecolor('#EAF2F8')

            ax.set_title(title, fontweight='bold', fontsize=10)

            for color, i in zip(colors, np.unique(self.clusters)):

                ax.scatter(
                    X_reduced[self.clusters == i, 0],
                    X_reduced[self.clusters == i, 1],
                    edgecolor='k',
                    color='gray' if i == -1 else color,
                    s=10 if i == -1 else 40,
                    alpha=0.6,
                    linewidths=1,
                    marker='x' if i == -1 else 'o',
                    label=f'unmatch' if i == -1 else f'cluster {i + 1}'
                )

        ax.set_xlabel('1th Component', fontsize=10)
        ax.set_ylabel('2nd Component', fontsize=10)

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, framealpha=0.9, facecolor='white')

        plt.tight_layout()

        if graphic_save_extension in ['png', 'svg', 'pdf', 'eps']:

            if not os.path.exists(f'./saved/{self.name}/figures'):
                os.makedirs(f'./saved/{self.name}/figures')

            plt.savefig(f'./saved/{self.name}/figures/{title}.{graphic_save_extension}', format=f'{graphic_save_extension}')

        plt.show()
        plt.close()


    def save(self):
        """
        Saves the preprocessor, preprocessed data, cluster labels, and model to disk.
        """

        if not os.path.exists(f'./saved/{self.name}'):
            os.makedirs(f'./saved/{self.name}')
        
        dump(self.preprocessor, f'./saved/{self.name}/preprocessor.joblib')
        dump(self.preprocessed_data, f'./saved/{self.name}/preprocessed_data.joblib')
        dump(self.clusters, f'./saved/{self.name}/clusters.joblib')
        dump(self.model, f'./saved/{self.name}/model.bin')


class KMeans(Model):
    """
    A class for constructing and applying KMeans clustering algorithm, with built-in methods for preprocessing,
    hyperparameter optimization, and visualization. Inherits from the Model base class.

    Attributes:
        clusters (array): Stores the cluster labels for each sample.

    Methods:
        build(data, **kwargs): Prepares the dataset for KMeans clustering.
        _make_kmeans(n_clusters, init, n_init, tol, algorithm, verbose=0): Constructs the KMeans model with specified hyperparameters.
        _optimizer(trial, **kwargs): Defines and runs the optimization trial for hyperparameter tuning.
        hyperparameter_optimization(n_trials=100, info=False, **kwargs): Performs hyperparameter optimization using Optuna.
        load(foldername): Loads the preprocessor, preprocessed data, and cluster labels from the specified folder.
        fit(verbose=0): Applies KMeans clustering to the preprocessed dataset.
        predict(projection='2d', graphic_save_extension=None): Projects the clustered data into 2D or 3D space and visualizes the clusters.
        save(): Saves the preprocessor, preprocessed data, cluster labels, and model to disk.
    """

    def __init__(self, name=None, seed=None):
        """
        Initializes a new instance of the KMeans class.

        Args:
            name (str, optional): The name of the model. Defaults to a generated name based on the current datetime.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """

        if name is None:
            _name = f'KMeans {datetime.now().strftime("%d-%m-%y %Hh%Mmin")}'

        else:
            _name = name
    
        super().__init__(None, _name, seed)
        self.clusters = None


    def build(self, data, **kwargs):
        """
        Prepares the dataset for KMeans clustering. This includes preprocessing the data based on its characteristics and specified parameters.

        Args:
            data (Pandas DataFrame): The dataset to be used for clustering.
            **kwargs: Additional keyword arguments for preprocessing.
        """

        super()._cluster_preprocess(data, **kwargs)


    def _make_kmeans(self, n_clusters, init, n_init, tol, algorithm, verbose=0):
        """
        Constructs the KMeans model with the specified hyperparameters.

        Args:
            n_clusters (int): The number of clusters to form as well as the number of centroids to generate.
            init (str): Method for initialization ('k-means++', 'random' or an ndarray).
            n_init (int): Number of time the k-means algorithm will be run with different centroid seeds.
            tol (float): Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
            algorithm (str): K-means algorithm to use ('auto', 'full' or 'elkan').
            verbose (int): Verbosity mode.
        
        Returns:
            sklearn.cluster._kmeans.KMeans: The constructed KMeans clustering model.
        """

        model = _KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=300,
            tol=tol,
            algorithm=algorithm,
            verbose=verbose,
            random_state=self.seed
        )

        return model
    

    def _optimizer(self, trial, **kwargs):
        """
        Defines and runs the optimization trial for hyperparameter tuning. This method is intended to be used as
        a callback within an Optuna optimization study.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            float: The silhouette score for the clustering configuration defined by the trial.
        """

        search_space_n_clusters = kwargs.get('search_space_n_clusters', [8])
        search_space_init  = kwargs.get('search_space_init ', ['k-means++'])
        search_space_n_init = kwargs.get('search_space_n_init', ['auto'])
        search_space_tol = kwargs.get('search_space_tol', [1e-5, 1e-4, 1e-3]) 
        search_space_algorithm = kwargs.get('search_space_algorithm', ['lloyd', 'elkan']) 

        n_clusters = trial.suggest_categorical('n_clusters', search_space_n_clusters)
        init = trial.suggest_categorical('init', search_space_init)
        n_init = trial.suggest_categorical('n_init', search_space_n_init)
        tol = trial.suggest_categorical('tol', search_space_tol)
        algorithm = trial.suggest_categorical('algorithm', search_space_algorithm)

        modelStudy = self._make_kmeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            tol=tol,
            algorithm=algorithm
        )

        clusters = modelStudy.fit_predict(self.preprocessed_data['X_ohe'])
        clusters_filtered = clusters[clusters != -1]

        if len(np.unique(clusters_filtered)) > 1:
            score = silhouette_score(self.preprocessed_data['X_ohe'], clusters)

        else:
            score = -1

        return score
    

    def hyperparameter_optimization(self, n_trials=100, info=False, **kwargs):
        """
        Performs hyperparameter optimization using Optuna over a specified number of trials. Reports the results
        and identifies the best hyperparameters for KMeans clustering.

        Args:
            n_trials (int, optional): The number of optimization trials to perform. Defaults to 100.
            info (bool, optional): Whether to print detailed information about each trial. Defaults to False.
            **kwargs: Additional keyword arguments for configuring the optimization process.
        
        Returns:
            pd.DataFrame: A DataFrame containing detailed information about each trial if `info` is True. Otherwise, None.
        """

        self.hyperparameter = optuna.create_study(study_name='optimization', direction='maximize')
        self.hyperparameter.optimize(lambda trial: self._optimizer(trial, **kwargs), n_trials=n_trials)

        if info:
            trial = self.hyperparameter.trials_dataframe()

            df_info = trial.drop(['datetime_start', 'datetime_complete', 'duration', 'state'], axis=1)
            df_info.set_index("number", inplace =True)
            df_info.rename_axis('Trial (nº)', inplace=True)
            df_info = df_info.sort_values('value', ascending=False)
            df_info.rename(columns={'value': 'Silhouette Score'}, inplace=True)
            df_info.reset_index(inplace=True)
            df_info.index = [f'{i}º' for i in df_info.index + 1]
            df_info.rename_axis('Ranking', inplace=True)
            
            return df_info
        

    def load(self, foldername):
        """
        Loads the preprocessor, preprocessed data, and cluster labels from the specified folder.

        Args:
            foldername (str): The name of the folder where the data and model are saved.
        """

        if not os.path.exists(f'./saved/{foldername}'):
            print("There is no folder with that name!")
            return
        
        self.preprocessor = load(f'./saved/{foldername}/preprocessor.joblib')
        self.preprocessed_data = load(f'./saved/{foldername}/preprocessed_data.joblib')
        self.clusters = load(f'./saved/{foldername}/clusters.joblib')
        self.model = load(f'./saved/{foldername}/model.bin')


    def fit(self, return_cluster=False, verbose=0):
        """
        Applies KMeans clustering to the preprocessed dataset and updates the 'clusters' attribute with the cluster labels.
        
        Args:
            verbose (int): Verbosity mode.
        """

        if self.hyperparameter is None:
            self.hyperparameter_optimization(n_trials=100)

        n_clusters = self.hyperparameter.best_params['n_clusters']
        init = self.hyperparameter.best_params['init']
        n_init = self.hyperparameter.best_params['n_init']
        tol = self.hyperparameter.best_params['tol']
        algorithm = self.hyperparameter.best_params['algorithm']

        self.model = self._make_kmeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            tol=tol,
            algorithm=algorithm,
            verbose=verbose
        )

        self.clusters = self.model.fit_predict(self.preprocessed_data['X_ohe'])

        if return_cluster:
            return self.clusters


    def predict(self, projection='2d', graphic_save_extension=None):
        """
        Projects the clustered data into 2D or 3D space using t-SNE and visualizes the clusters.

        Args:
            projection (str, optional): The type of projection for visualization ('2d' or '3d'). Defaults to '2d'.
            graphic_save_extension (str, optional): Extension to save the graphics (e.g., 'png', 'svg'). If None, graphics are not saved. Defaults to None.
        """

        tsne = TSNE(n_components=3 if projection == '3d' else 2, random_state=self.seed)
        X_reduced = tsne.fit_transform(self.preprocessed_data['X_ohe'])

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(np.unique(self.clusters)))]
        
        if projection == '3d':

            fig = plt.figure(figsize=(8, 6))
            title = '3-Dimensional t-SNE Clusters (K-Means)'
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.set_zlabel('3rd component', fontsize=12)

            for color, i in zip(colors, np.unique(self.clusters)):

                ax.scatter(
                    X_reduced[self.clusters == i, 0],
                    X_reduced[self.clusters == i, 1],
                    X_reduced[self.clusters == i, 2],
                    edgecolor='k',
                    color='gray' if i == -1 else color,
                    s=10 if i == -1 else 40,
                    alpha=0.6,
                    linewidths=1,
                    depthshade=False,
                    marker='x' if i == -1 else 'o',
                    label='resíduo' if i == -1 else f'cluster {i + 1}',
                )
            
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            ax.xaxis.pane.set_facecolor('#EAF2F8')
            ax.yaxis.pane.set_facecolor('#D4E6F1')
            ax.zaxis.pane.set_facecolor('#A9CCE3')

        else:

            fig = plt.figure(figsize=(6, 4))
            title = '2-Dimensional t-SNE Clusters (K-Means)'
            ax = fig.add_subplot(111)
            ax.set_facecolor('#EAF2F8')

            ax.set_title(title, fontweight='bold', fontsize=10)

            for color, i in zip(colors, np.unique(self.clusters)):

                ax.scatter(
                    X_reduced[self.clusters == i, 0],
                    X_reduced[self.clusters == i, 1],
                    edgecolor='k',
                    color='gray' if i == -1 else color,
                    s=10 if i == -1 else 40,
                    alpha=0.6,
                    linewidths=1,
                    marker='x' if i == -1 else 'o',
                    label=f'resíduo' if i == -1 else f'cluster {i + 1}'
                )

        ax.set_xlabel('1th Component', fontsize=10)
        ax.set_ylabel('2nd Component', fontsize=10)

        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, framealpha=0.9, facecolor='white')

        plt.tight_layout()

        if graphic_save_extension in ['png', 'svg', 'pdf', 'eps']:

            if not os.path.exists(f'./saved/{self.name}/figures'):
                os.makedirs(f'./saved/{self.name}/figures')

            plt.savefig(f'./saved/{self.name}/figures/{title}.{graphic_save_extension}', format=f'{graphic_save_extension}')

        plt.show()
        plt.close()


    def save(self):
        """
        Saves the preprocessor, preprocessed data, cluster labels, and model to disk.
        """

        if not os.path.exists(f'./saved/{self.name}'):
            os.makedirs(f'./saved/{self.name}')
        
        dump(self.preprocessor, f'./saved/{self.name}/preprocessor.joblib')
        dump(self.preprocessed_data, f'./saved/{self.name}/preprocessed_data.joblib')
        dump(self.clusters, f'./saved/{self.name}/clusters.joblib')
        dump(self.model, f'./saved/{self.name}/model.bin')