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
    def team_metrics(df):
        """
        Preprocesses the monitor report data with an additional filter for monitor_type.

        Args:
            df (pandas.DataFrame): Input DataFrame
            monitor_type (str): The specific monitor type to filter by. For example, 'AsaTeamMetrics@AsaModels' or 'AsaAirThreatMetric@AsaModels'.

        Returns:
            pandas.DataFrame: The preprocessed dataframe.
        """

        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Converts the 'payload' column from nested strings to dictionaries
        df_copy.loc[:, 'payload'] = df_copy['payload'].apply(convert_nested_string_to_dict)

        # Extract 'monitor_type' from 'payload' and create a new column
        df_copy['extracted_monitor_type'] = df_copy['payload'].apply(lambda x: x.get('monitor_type'))

        # Filters by 'asa_custom_type' and the extracted 'monitor_type'
        df_copy = df_copy[(df_copy['asa_custom_type'] == 'asa::recorder::AsaMonitorReport') & 
                        (df_copy['extracted_monitor_type'] == 'AsaTeamMetrics@AsaModels')]

        # Extract 'side', 'fuel_consumed', 'time_of_flight' values from 'payload'
        df_copy['side'] = df_copy['payload'].apply(lambda x: find_key(x.get('attributes', {}), 'side'))
        df_copy['fuel_consumed'] = df_copy['payload'].apply(lambda x: find_key(x.get('metrics', {}), 'fuel_consumed'))
        df_copy['time_of_flight'] = df_copy['payload'].apply(lambda x: find_key(x.get('metrics', {}), 'time_of_flight'))

        # Identifying metric keys from 'payload' and creating columns for each metric
        metrics_keys = set()
        for payload in df_copy['payload']:
            metrics_keys.update(payload.get('metrics', {}).get('last_state', {}).keys())
        
        for metric in metrics_keys:
            df_copy[metric] = df_copy['payload'].apply(lambda x: find_key(x['metrics'].get('last_state', {}), metric))

        # Selects a subset of columns that are available
        available_columns = ['experiment', 'side', 'fuel_consumed', 'time_of_flight'] + list(metrics_keys)
        available_columns = [col for col in available_columns if col in df_copy.columns]
        df_copy = df_copy[available_columns]

        unique_sides = df_copy['side'].dropna().unique()

        # Create a dictionary to hold dataframes for each side
        dfs = {}
        for side in unique_sides:
            dfs[side] = df_copy[df_copy['side'] == side]

        # Initialize merged DataFrame
        df_merged = None

        # Perform merging
        for side, df_side in dfs.items():
            if df_merged is None:
                df_merged = df_side
            else:
                df_merged = pd.merge(df_merged, df_side, on='experiment', suffixes=('', f'_{side}'))

        # Set index and drop redundant side columns if any
        df_merged = df_merged.set_index('experiment')
        side_columns = [col for col in df_merged.columns if col.startswith('side')]
        df_merged = df_merged.drop(side_columns, axis=1)

        # Reorder the dataframe by the index 'experiment'
        df_merged = df_merged.sort_index()

        return df_merged
    
    @staticmethod
    def air_threat_metric(df):
        """
        Preprocesses the monitor report data with an additional filter for monitor_type.

        Args:
            df (pandas.DataFrame): Input DataFrame

        Returns:
            pandas.DataFrame: The preprocessed dataframe.
        """

        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()

        # Converts the 'payload' column from nested strings to dictionaries
        df_copy.loc[:, 'payload'] = df_copy['payload'].apply(convert_nested_string_to_dict)

        # Extract 'monitor_type' from 'payload' and create a new column
        df_copy['extracted_monitor_type'] = df_copy['payload'].apply(lambda x: x.get('monitor_type'))

        # Filters by 'asa_custom_type' and the extracted 'monitor_type'
        df_copy = df_copy[(df_copy['asa_custom_type'] == 'asa::recorder::AsaMonitorReport') & 
                        (df_copy['extracted_monitor_type'] == 'AsaAirThreatMetric@AsaModels')]

        # Extract 'side' and 'threat_index' values from 'payload'
        df_copy['side'] = df_copy['payload'].apply(lambda x: find_key(x.get('attributes', {}), 'side'))
        df_copy['threat_index'] = df_copy['payload'].apply(lambda x: x.get('metrics', {}).get('threat_index'))

        # Selects a subset of columns that are available
        available_columns = ['experiment', 'side', 'threat_index']
        available_columns = [col for col in available_columns if col in df_copy.columns]
        df_copy = df_copy[available_columns]

        # Set index and drop redundant side columns if any
        df_copy = df_copy.set_index('experiment')
        side_columns = [col for col in df_copy.columns if col.startswith('side')]
        df_copy = df_copy.drop(side_columns, axis=1)

        # Reorder the dataframe by the index 'experiment'
        df_copy = df_copy.sort_index()

        return df_copy


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


