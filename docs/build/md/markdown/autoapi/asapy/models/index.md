# `asapy.models`

## Module Contents

### Classes

| `Model`
 | Abstract base class for machine learning models.

 |
| `NN`
                  | The class NN is a wrapper around Keras Sequential API, which provides an easy way to create and train neural network models. It can perform hyperparameters search and model building with specified hyperparameters.

 |
| `RandomForest`
        | This class is used to build and search hyperparameters for a random forest model in scikit-learn.

                                                                                                                     |
| `Scaler`
              | The Scaler class is designed to scale and transform data using various scaling techniques. It contains methods for fitting and transforming data, as well as saving and loading scaler objects to and from files.

     |
| `AsaML`
               | 

                                                                                                                                                                                                                      |

### _class_ asapy.models.Model()
Bases: `abc.ABC`

Abstract base class for machine learning models.

#### Attributes:

None

#### Methods:

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

#### Raises:

None


#### build()

#### load()

#### fit()

#### predict()

#### save()

### _class_ asapy.models.NN(model=None)
Bases: `Model`

The class NN is a wrapper around Keras Sequential API, which provides an easy way to create and train neural network models. It can perform hyperparameters search and model building with specified hyperparameters.

#### Attributes:

> 
> * model: the built Keras model.


> * loss: the loss function used to compile the Keras model.


> * metrics: the metrics used to compile the Keras model.


> * dir_name: a string that defines the name of the directory to save the hyperparameters search results.


> * input_shape: the shape of the input data for the Keras model.


> * output_shape: the shape of the output data for the Keras model.


#### _model_search(hp, \*\*kwargs)
Searches for the best hyperparameters to create a Keras model using the given hyperparameters space.


* **Parameters**

    **hp** (*keras_tuner.engine.hyperparameters.HyperParameters*) – Object that holds
    the hyperparameters space to search.



* **Returns**

    A compiled Keras model with the best hyperparameters found.



#### search_hyperparams(X, y, project_name='', verbose=False)
Perform hyperparameter search for the neural network using Keras Tuner.


* **Parameters**

    
    * **X** (*numpy.ndarray*) – Input data.


    * **y** (*numpy.ndarray*) – Target data.


    * **project_name** (*str*) – Name of the Keras Tuner project (default ‘’).


    * **verbose** (*bool*) – Whether or not to print out information about the search progress (default False).



* **Returns**

    A dictionary containing the optimal hyperparameters found by the search.



* **Return type**

    dict



* **Raises**

    **ValueError** – If self.loss is not a supported loss function.



#### build(input_shape=(1,), output_shape=(1,), n_neurons=[1], n_layers=1, learning_rate=0.001, activation='relu', \*\*kwargs)
Builds a Keras neural network model with the given hyperparameters.


* **Parameters**

    
    * **input_shape** (*tuple**, **optional*) – The shape of the input data. Defaults to (1,).


    * **output_shape** (*tuple**, **optional*) – The shape of the output data. Defaults to (1,).


    * **n_neurons** (*list**, **optional*) – A list of integers representing the number of neurons in each hidden layer.
    The length of the list determines the number of hidden layers. Defaults to [1].


    * **n_layers** (*int**, **optional*) – The number of hidden layers in the model. Defaults to 1.


    * **learning_rate** (*float**, **optional*) – The learning rate of the optimizer. Defaults to 1e-3.


    * **activation** (*str**, **optional*) – The activation function used for the hidden layers. Defaults to ‘relu’.



* **Returns**

    None.



#### load(path)
Load a Keras model from an H5 file.


* **Parameters**

    **path** (*str*) – Path to the H5 file containing the Keras model.



* **Raises**

    **ValueError** – If the file extension is not ‘.h5’.



* **Returns**

    None



#### predict(x)
Uses the trained neural network to make predictions on input data.


* **Parameters**

    **x** (*numpy.ndarray*) – Input data to be used for prediction. It must have the same number of features
    as the input_shape used to build the network.



* **Returns**

    Predicted outputs for the input data.



* **Return type**

    numpy.ndarray



* **Raises**

    **ValueError** – If the input data x does not have the same number of features as the input_shape
    used to build the network.



#### fit(x, y, validation_data, batch_size=32, epochs=500, save=True, patience=5, path='')
Trains the neural network model using the given input and output data.


* **Parameters**

    
    * **x** (*numpy array*) – The input data used to train the model.


    * **y** (*numpy array*) – The output data used to train the model.


    * **validation_data** (*tuple*) – A tuple containing the validation data as input and output data.


    * **batch_size** (*int*) – The batch size used for training the model (default=32).


    * **epochs** (*int*) – The number of epochs used for training the model (default=500).


    * **save** (*bool*) – Whether to save the model after training (default=True).


    * **patience** (*int*) – The number of epochs to wait before early stopping if the validation loss does not improve (default=5).


    * **path** (*str*) – The path to save the trained model (default=’’).



* **Returns**

    None



#### save(path)
Saves the trained neural network model to a file.


* **Parameters**

    **path** – A string specifying the path and filename for the saved model. The “.h5” file extension
    will be appended to the provided filename if not already present.



* **Raises**

    **ValueError** – If the provided file extension is not “.h5”.



* **Returns**

    None



### _class_ asapy.models.RandomForest(model=None)
Bases: `Model`

This class is used to build and search hyperparameters for a random forest model in scikit-learn.

#### Attributes:

> 
> * model: the built RandomForest scikit-learn model.


#### search_hyperparams(X, y, verbose=False, \*\*kwargs)
Perform a hyperparameter search for a Random Forest model using RandomizedSearchCV.


* **Parameters**

    
    * **X** (*numpy array*) – The feature matrix of the data.


    * **y** (*numpy array*) – The target vector of the data.


    * **verbose** (*bool**, **optional*) – If True, print the optimal hyperparameters. Defaults to False.


    * **\*\*kwargs** – Additional keyword arguments. The following hyperparameters can be set:
    - n_estimators (int): Number of trees in the forest. Defaults to sp_randint(10, 1000).
    - max_features (list): The number of features to consider when looking for the best split.
    Allowed values are ‘sqrt’, ‘log2’ or a float between 0 and 1. Defaults to [‘sqrt’, ‘log2’].
    - max_depth (list): The maximum depth of the tree. Defaults to [None, 5, 10, 15, 20, 30, 40].
    - min_samples_split (int): The minimum number of samples required to split an internal node.
    Defaults to sp_randint(2, 20).
    - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
    Defaults to sp_randint(1, 20).
    - bootstrap (bool): Whether bootstrap samples are used when building trees.
    Defaults to [True, False].
    - y_type (str): Type of target variable. Either ‘num’ for numeric or ‘cat’ for categorical.
    Defaults to ‘cat’.



* **Returns**

    A dictionary with the best hyperparameters found during the search.



* **Return type**

    dict



#### build(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', \*\*kwargs)
Builds a new Random Forest model with the specified hyperparameters.


* **Parameters**

    
    * **n_estimators** (*int**, **optional*) – The number of trees in the forest. Default is 100.


    * **max_depth** (*int** or **None**, **optional*) – The maximum depth of each tree. None means unlimited. Default is None.


    * **min_samples_split** (*int**, **optional*) – The minimum number of samples required to split an internal node. Default is 2.


    * **min_samples_leaf** (*int**, **optional*) – The minimum number of samples required to be at a leaf node. Default is 1.


    * **max_features** (*str** or **int**, **optional*) – The maximum number of features to consider when looking for the best split.
    Can be ‘sqrt’, ‘log2’, an integer or None. Default is ‘sqrt’.


    * **\*\*kwargs** – Additional keyword arguments. Must include a ‘y_type’ parameter, which should be set to ‘num’ for
    regression problems and ‘cat’ for classification problems.



* **Returns**

    None



#### load(path)
Load a saved random forest model.


* **Parameters**

    **path** (*str*) – The path to the saved model file. The file must be a joblib file with the extension ‘.joblib’.



* **Raises**

    **ValueError** – If the extension of the file is not ‘.joblib’.



* **Returns**

    None.



#### predict(x)
Makes predictions using the trained Random Forest model on the given input data.


* **Parameters**

    **x** – The input data to make predictions on.



* **Returns**

    An array of predicted target values.



#### fit(x, y)
Trains the Random Forest model on the given input and target data.


* **Parameters**

    
    * **x** – The input data to train the model on.


    * **y** – The target data to train the model on.



* **Returns**

    None



#### save(path)
Saves the trained model to a file with the specified path.


* **Parameters**

    **path** (*str*) – The file path where the model should be saved. The file extension should be ‘.joblib’.



* **Raises**

    **ValueError** – If the file extension is invalid or missing.



* **Returns**

    None



### _class_ asapy.models.Scaler(scaler=None)
The Scaler class is designed to scale and transform data using various scaling techniques. It contains methods for fitting and transforming data, as well as saving and loading scaler objects to and from files.


#### fit_transform(data)
Fit to data, then transform it.


* **Parameters**

    **data** (*array-like*) – The data to be transformed.



* **Returns**

    The transformed data.



* **Return type**

    array-like



#### transform(data)
Perform standardization on an array.


* **Parameters**

    **data** (*array-like*) – The data to be standardized.



* **Returns**

    The standardized data.



* **Return type**

    array-like



#### inverse_transform(data)
Scale back the data to the original representation.


* **Parameters**

    **data** (*array-like*) – The data to be scaled back.



* **Returns**

    The original representation of the data.



* **Return type**

    array-like



#### save(path)
Save the scaler object to a file.


* **Parameters**

    **path** (*str*) – The path where the scaler object will be saved.



#### load(path)
Load a saved scaler object from a file.


* **Parameters**

    **path** (*str*) – The path where the scaler object is saved.



### _class_ asapy.models.AsaML(dir_name=None)

#### _static_ identify_categorical_data(df)
Identifies categorical data.


* **Parameters**

    **df** (*pandas.DataFrame*) – The DataFrame containing the data.



* **Returns**

    A tuple containing two DataFrames - one containing the categorical columns and the other containing the numerical columns.



* **Return type**

    tuple



* **Raises**

    **ValueError** – If the input DataFrame is empty or if it contains no categorical or numerical data.


Example usage:

```default
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
```


#### pre_processing_train(X, y, remove_outlier=False)
Perform pre-processing steps for the training dataset.


* **Parameters**

    
    * **X** – pandas.DataFrame - input features.


    * **y** – pandas.DataFrame - target variable.


    * **remove_outlier** – bool - if True, removes the outliers from the dataset.



* **Returns**

    pandas.DataFrame - pre-processed input features.
    y: pandas.DataFrame - pre-processed target variable.



* **Return type**

    X



#### \__add_random_value_to_max(row)

#### train_model(X=None, y=None, name_model=None, save=True, scaling=True, scaler_Type='StandardScaler', remove_outlier=False, search=False, params=None, \*\*kwargs)
Train a model on a given dataset.


* **Parameters**

    
    * **X** – A pandas dataframe containing the feature data. Default is None.


    * **y** – A pandas dataframe containing the target data. Default is None.


    * **name_model** – The name of the model to train. Default is None.


    * **save** – A boolean indicating whether to save the model or not. Default is True.


    * **scaling** – A boolean indicating whether to perform data scaling or not. Default is True.


    * **scaler_Type** – The type of data scaling to perform. Must be one of ‘StandardScaler’, ‘Normalizer’, or ‘MinMaxScaler’. Default is ‘StandardScaler’.


    * **remove_outlier** – A boolean indicating whether to remove outliers or not. Default is False.


    * **search** – A boolean indicating whether to search for the best hyperparameters or not. Default is False.


    * **params** – A dictionary containing the hyperparameters to use for training. Default is None.


    * **\*\*kwargs** – Additional arguments to be passed to the training function.



* **Returns**

    None



#### load_model(path='')
Loads a saved model from the specified path and returns a dictionary of models
with their corresponding parameters.


* **Parameters**

    **path** (*str*) – The path where the model and its associated files are saved.
    Defaults to an empty string.



* **Returns**

    A dictionary containing the loaded models with their corresponding parameters.



* **Return type**

    dict



* **Raises**

    **ValueError** – If the path argument is empty.


**NOTE**: The path variable must be the address of the ‘dirname’ folder and MUST contain the scaler.pkl file and each subdirectory MUST contain the ‘paramenters’, ‘model’ files.


#### pre_processing_predict(X, input_list, var_type)
Pre-processes the input data before prediction by scaling numerical features and creating dummy variables
for categorical features. Also handles missing and extra features in the input data.


* **Parameters**

    
    * **X** (*pandas.DataFrame*) – The input data to be pre-processed.


    * **input_list** (*list*) – A list of expected input features.


    * **var_type** (*dict*) – A dictionary with the types of the input features. The keys ‘cat’ and ‘num’ contain lists
    of categorical and numerical feature names respectively.



* **Returns**

    The pre-processed input data with scaled numerical features and dummy variables

        for categorical features. Any missing or extra features are handled accordingly.




* **Return type**

    pandas.DataFrame



#### pos_processing(y, output_list)
Post-processes the output of a model prediction to transform it into a more usable format.


* **Parameters**

    
    * **y** (*np.ndarray*) – The output of the model prediction, as a NumPy array.


    * **output_list** (*list*) – A list of column names representing the output variables.



* **Returns**

    A pandas DataFrame containing the post-processed output values.



* **Return type**

    pd.DataFrame


This function takes the output of a model prediction, which is typically a NumPy array of raw output values, and transforms it into a more usable format. The output variables are expected to have been one-hot encoded with the use of triple underscores (‘___’) as separator, and possibly have a random value added to the max value of each row. The function first separates the categorical and numerical variables, then processes the categorical variables by selecting the maximum value for each row and one-hot encoding them. Finally, it concatenates the categorical and numerical variables back together to produce a pandas DataFrame containing the post-processed output values.


#### predict_all(X, model_dict)
Apply all models in the model dictionary to the input data frame X and return the predictions.


* **Parameters**

    
    * **X** – A pandas DataFrame representing the input data.


    * **model_dict** – A dictionary containing the models and their associated metadata. The keys are the names of the
    models and the values are themselves dictionaries containing the following keys:
    - ‘model’: A trained machine learning model.
    - ‘input_list’: A list of the names of the input features used by the model.
    - ‘output_list’: A list of the names of the output features produced by the model.
    - ‘var_type’: A dictionary containing the types of the input and output features, with the keys

    > ’X’ and ‘y’, respectively, and the values being dictionaries themselves with
    > the following keys:
    > - ‘cat’: A list of the categorical input features.
    > - ‘num’: A list of the numerical input features.




* **Returns**

    A pandas DataFrame containing the predictions of all models in the model dictionary. The columns of the
    DataFrame are the names of the models, and the rows correspond to the input rows in X.



* **Raises**

    **ValueError** – If X is empty or None, or if the model dictionary is empty or None.



#### full_cycle(X_pred, load=False, \*\*kwargs)
Performs the full cycle of the machine learning pipeline: loads or trains the models, preprocesses the input data,
generates predictions, and post-processes the output data.


* **Parameters**

    
    * **X_pred** (*pandas.DataFrame*) – Input data to generate predictions for.


    * **load** (*bool**, **optional*) – If True, loads the trained models from disk instead of training new ones. Default is False.


    * **\*\*kwargs** – Additional keyword arguments passed to either load_model() or train_model() method.



* **Returns**

    Dataframe with the generated predictions.



* **Return type**

    pandas.DataFrame



* **Raises**

    **ValueError** – If load is True and path is not provided in kwargs.
