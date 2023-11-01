import pandas as pd
from .utils import *

class Preprocessing:
    @staticmethod
    def aliases(x):
        """
        Preprocessing the simulation aliases. Adjusts the index of the dataframe by creating a new column 'experiment' that
        receives the current indices of the DataFrame.

        Args:
            x (pandas.DataFrame): Input DataFrame

        Returns:
            pandas.DataFrame: DataFrame with modified index.
        """
        # Adds a new column named 'experiment' which receives the current indices of the DataFrame
        x['experiment'] = x.index

        # Sets the 'experiment' column as the new index for the DataFrame
        x.set_index('experiment', drop=True, inplace=True)

        # Returns the modified DataFrame
        return x
    
    @staticmethod
    def monitor_report(df):
        """
        Preprocesses the monitor report data.

        Args:
            y (pandas.DataFrame): Input DataFrame 

        Returns:
            pandas.DataFrame: The preprocessed dataframe.
        """
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Filters by 'asa_custom_type'
        df_copy = df_copy[df_copy['asa_custom_type'] == 'asa::recorder::AsaMonitorReport']
        
        # Converts the 'payload' column from nested strings to dictionaries
        df_copy.loc[:, 'payload'] = df_copy['payload'].apply(convert_nested_string_to_dict)

        # Extract values from 'payload' dictionary into new columns
        for key in ['side', 'fuel_consumed', 'time_of_flight']:
            df_copy.loc[:, key] = df_copy['payload'].apply(lambda x: find_key(x, key))

        # Identifying metric keys from 'payload' and creating columns for each metric
        metrics_keys = list(df_copy['payload'].iloc[0]['metrics']['last_state'].keys())
        for metric in metrics_keys:
            df_copy[metric] = df_copy['payload'].apply(lambda x: find_key(x, metric))

        # # Selects a subset of columns
        df_copy = df_copy[['experiment', 'side', 'fuel_consumed', 'time_of_flight', 'acft_standing', 'acft_damaged', 
        'acft_killed', 'aam_remaining', 'aam_hit', 'aam_frat', 'aam_lost', 'sam_remaining', 'sam_hit', 
        'sam_frat', 'sam_lost', 'bmb_remaining', 'bmb_released']]

        # Split the data into 'blue' and 'red'
        df_blue = df_copy[df_copy['side'] == 'blue']
        df_red = df_copy[df_copy['side'] == 'red']

        # Splits the data into 'blue' and 'red', merges both on 'experiment' and removes 'side_blue' and 'side_red' columns
        df_copy = pd.merge(df_blue, df_red, on='experiment', suffixes=('_blue', '_red')).set_index("experiment")
        
        return df_copy.drop(['side_blue', 'side_red'], axis=1)

    @staticmethod
    def weapon_detonation(df):
        """
        Function for pre-processing weapon detonation data.

        Args:
            df (pandas.DataFrame): The dataframe to preprocess.

        Returns:
            pandas.DataFrame: The preprocessed dataframe.
        """
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()
        
        # Filter by 'asa_type'
        df_copy = df_copy[df_copy['asa_type'] == 8]
        
        # Convert 'payload' column from nested strings to dictionaries
        df_copy.loc[:, 'payload'] = df_copy['payload'].apply(convert_nested_string_to_dict)

        # Extract values from 'payload' dictionary into new columns
        for key in ['side', 'miss_dist']:
            df_copy.loc[:, key] = df_copy['payload'].apply(lambda x: find_key(x, key))

        # Select a subset of columns
        df_copy = df_copy[['experiment', 'side', 'miss_dist']]
        
        # Split the data into 'blue' and 'red'
        df_blue = df_copy[df_copy['side'] == 'blue']
        df_red = df_copy[df_copy['side'] == 'red']
        
        # Group by 'experiment' and calculate mean 'miss_dist' for each side
        df_blue = df_blue.groupby('experiment')['miss_dist'].mean().reset_index().rename(columns={'miss_dist': 'miss_dist_blue'})
        df_red = df_red.groupby('experiment')['miss_dist'].mean().reset_index().rename(columns={'miss_dist': 'miss_dist_red'})

        # Merge both 'blue' and 'red' data on 'experiment'
        merged_df = pd.merge(df_blue, df_red, on='experiment').set_index("experiment")

        return merged_df

    @staticmethod
    def convert_categorical_to_dummies(df, column_name, prefix=None):
        """
        Convert a categorical column into dummy/indicator columns, and add these new columns 
        into the DataFrame at the same position of the original one.

        Args:
            df (pd.DataFrame): The DataFrame to process.
            column_name (str): The name of the categorical column to convert.
            prefix (str): The prefix to apply to the dummy column names.

        Returns:
            pd.DataFrame: The DataFrame with the original column replaced by dummy columns.
        """

        # Convert the column to categorical if it contains boolean values
        if df[column_name].dtype == bool:
            df[column_name] = df[column_name].astype(int)

        # If no prefix was given, use the column_name
        if prefix is None:
            prefix = column_name

        # Get the column index where the new columns should be inserted
        idx = list(df.columns).index(column_name)

        # Generate dummy columns
        dummies = pd.get_dummies(df[column_name], prefix=prefix)

        # Drop the original column from DataFrame
        df = df.drop(columns=[column_name])

        # Concatenate the dummy columns to the original DataFrame at the right location
        df = pd.concat([df.iloc[:, :idx], dummies, df.iloc[:, idx:]], axis=1)

        return df


